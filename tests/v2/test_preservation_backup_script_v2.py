from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "preservation" / "create_verified_backup.ps1"
POWERSHELL = Path(r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe")


class PreservationBackupScriptV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        if not POWERSHELL.exists():
            self.skipTest("Windows PowerShell is unavailable")
        if shutil.which("git") is None:
            self.skipTest("git is unavailable")

    def _git(self, repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def _init_repository(self, repository: Path) -> None:
        repository.mkdir(parents=True)
        self._git(repository, "init")
        self._git(repository, "config", "user.email", "preservation-test@example.invalid")
        self._git(repository, "config", "user.name", "Preservation Test")

    def _commit_all(self, repository: Path, message: str = "fixture") -> None:
        self._git(repository, "add", "-A")
        self._git(repository, "commit", "-m", message)

    def _run_script(
        self,
        *,
        root: Path,
        mexc: Path,
        backup_base: Path,
        mode: str,
        preflight_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(POWERSHELL),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT_PATH),
            "-Mode",
            mode,
            "-BackupBase",
            str(backup_base),
            "-RootPath",
            str(root),
            "-MexcPath",
            str(mexc),
            "-PythonPath",
            sys.executable,
        ]
        if preflight_only:
            command.append("-PreflightOnly")
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

    def _make_minimal_repositories(self, base: Path) -> tuple[Path, Path]:
        root = base / "root"
        mexc = base / "mexc"
        self._init_repository(root)
        self._init_repository(mexc)
        (root / "app.txt").write_text("root\n", encoding="utf-8")
        (mexc / "scanner.txt").write_text("mexc\n", encoding="utf-8")
        self._commit_all(root)
        self._commit_all(mexc)
        return root, mexc

    def test_windows_powershell_parser_reports_no_errors(self) -> None:
        escaped = str(SCRIPT_PATH).replace("'", "''")
        parser_command = (
            "$tokens=$null;$errors=$null;"
            f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped}',"
            "[ref]$tokens,[ref]$errors)|Out-Null;"
            "if($errors.Count -ne 0){$errors|ForEach-Object{$_.Message};exit 1}"
        )
        result = subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-Command", parser_command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_preflight_only_never_writes_and_security_modes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, mexc = self._make_minimal_repositories(base)

            checkpoint_base = base / "checkpoint"
            result = self._run_script(
                root=root,
                mexc=mexc,
                backup_base=checkpoint_base,
                mode="LocalCheckpoint",
                preflight_only=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(checkpoint_base.exists())
            payload = json.loads(result.stdout)
            self.assertTrue(payload["PreflightOnly"])
            self.assertEqual(payload["Mode"], "LocalCheckpoint")

            disaster_base = base / "same-disk-disaster"
            result = self._run_script(
                root=root,
                mexc=mexc,
                backup_base=disaster_base,
                mode="DisasterResilient",
                preflight_only=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source physical disk", result.stdout + result.stderr)
            self.assertFalse(disaster_base.exists())

            inside_repository = root / "forbidden-backup"
            result = self._run_script(
                root=root,
                mexc=mexc,
                backup_base=inside_repository,
                mode="LocalCheckpoint",
                preflight_only=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inside a source worktree", result.stdout + result.stderr)
            self.assertFalse(inside_repository.exists())

    def test_clean_repositories_allow_empty_untracked_file_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, mexc = self._make_minimal_repositories(base)
            backup_base = base / "checkpoint"

            result = self._run_script(
                root=root,
                mexc=mexc,
                backup_base=backup_base,
                mode="LocalCheckpoint",
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = list(backup_base.glob("koteika_preservation_*"))
            self.assertEqual(len(runs), 1)
            backup_root = runs[0]
            self.assertTrue((backup_root / "CHECKPOINT_VERIFIED.json").is_file())
            for name in ("root_untracked", "mexc_untracked"):
                source_manifest = backup_root / "manifests" / f"{name}_source_before.json"
                destination_manifest = backup_root / "manifests" / f"{name}_destination.json"
                self.assertEqual(json.loads(source_manifest.read_text(encoding="utf-8")), [])
                self.assertEqual(json.loads(destination_manifest.read_text(encoding="utf-8")), [])

    def test_local_checkpoint_is_verified_without_copying_forbidden_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            mexc = base / "mexc"
            backup_base = base / "backup"
            self._init_repository(root)
            self._init_repository(mexc)

            (root / "config").mkdir()
            (root / "config" / "secrets.env.example").write_text(
                "API_KEY=your_api_key_here\n", encoding="utf-8"
            )
            (root / "logs" / "observation").mkdir(parents=True)
            (root / "logs" / "system.log").write_text("runtime noise\n", encoding="utf-8")
            (root / "logs" / "observation" / "comparison.json").write_text(
                '{"verdict":"pause"}\n', encoding="utf-8"
            )
            (root / ".idea").mkdir()
            (root / ".idea" / "misc.xml").write_text("<project/>\n", encoding="utf-8")
            (root / "cache").mkdir()
            (root / "cache" / "cache.txt").write_text("cache\n", encoding="utf-8")
            (root / ".env.local").write_text("DO_NOT_COPY=fixture-only\n", encoding="utf-8")
            (root / "app.txt").write_text("base\n", encoding="utf-8")
            (root / "data" / "runtime" / "alert_locks").mkdir(parents=True)
            (root / "data" / "runtime" / "alert_locks" / "one.lock").write_text(
                '{"locked":true}\n', encoding="utf-8"
            )
            (root / "data" / "runtime" / "state.json").write_text(
                '{"state":"paper"}\n', encoding="utf-8"
            )
            (root / "data" / "runtime" / "bot_runtime.lock").write_text(
                '{"pid":999999}\n', encoding="utf-8"
            )
            root_db = root / "data" / "runtime" / "main.db"
            with closing(sqlite3.connect(root_db)) as connection:
                connection.execute("CREATE TABLE events(value TEXT)")
                connection.execute("INSERT INTO events VALUES ('checkpointed')")
                connection.commit()
            (root / "data" / "runtime" / "early.db").write_bytes(b"")
            self._commit_all(root)

            (mexc / ".env").write_text("MEXC_SECRET=fixture-only\n", encoding="utf-8")
            (mexc / "scanner.txt").write_text("base\n", encoding="utf-8")
            (mexc / "data" / "processed").mkdir(parents=True)
            (mexc / "data" / "history").mkdir()
            (mexc / "data" / "raw").mkdir()
            (mexc / "data" / "runtime").mkdir()
            (mexc / "data" / "processed" / "dataset.csv").write_text(
                "x,y\n1,2\n", encoding="utf-8"
            )
            (mexc / "data" / "history" / "AAA.csv").write_text(
                "ts,close\n1,2\n", encoding="utf-8"
            )
            (mexc / "data" / "raw" / "raw.csv").write_text(
                "raw\n", encoding="utf-8"
            )
            mexc_db = mexc / "data" / "runtime" / "runtime.db"
            with closing(sqlite3.connect(mexc_db)) as connection:
                connection.execute("CREATE TABLE decisions(value TEXT)")
                connection.execute("INSERT INTO decisions VALUES ('saved')")
                connection.commit()
            self._commit_all(mexc)

            (root / "app.txt").write_text("dirty root\n", encoding="utf-8")
            (root / "notes.txt").write_text("untracked root\n", encoding="utf-8")
            (root / "data" / "runtime" / "orphan.db-wal").write_bytes(b"raw-sidecar")
            (mexc / "scanner.txt").write_text("dirty mexc\n", encoding="utf-8")
            (mexc / "local_notes.txt").write_text("untracked mexc\n", encoding="utf-8")

            wal_connection = sqlite3.connect(root_db)
            try:
                wal_connection.execute("PRAGMA journal_mode=WAL")
                wal_connection.execute("PRAGMA wal_autocheckpoint=0")
                wal_connection.execute("INSERT INTO events VALUES ('from-wal')")
                wal_connection.commit()

                result = self._run_script(
                    root=root,
                    mexc=mexc,
                    backup_base=backup_base,
                    mode="LocalCheckpoint",
                )
            finally:
                wal_connection.close()

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = list(backup_base.glob("koteika_preservation_*"))
            self.assertEqual(len(runs), 1)
            backup_root = runs[0]
            checkpoint = backup_root / "CHECKPOINT_VERIFIED.json"
            self.assertTrue(checkpoint.is_file())
            self.assertFalse((backup_root / "VERIFIED_OK.json").exists())
            receipt = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(receipt["Mode"], "LocalCheckpoint")
            self.assertEqual(receipt["Status"], "verified")

            relative_files = {
                path.relative_to(backup_root).as_posix().lower()
                for path in backup_root.rglob("*")
                if path.is_file()
            }
            self.assertFalse(any(Path(path).name.startswith(".env") for path in relative_files))
            self.assertFalse(any(path.endswith("logs/system.log") for path in relative_files))
            self.assertFalse(any(path.endswith(".db-wal") for path in relative_files))
            self.assertFalse(any(path.endswith(".db-shm") for path in relative_files))
            self.assertFalse(any(path.endswith("data/runtime/bot_runtime.lock") for path in relative_files))
            self.assertTrue(
                (backup_root / "runtime" / "root" / "data" / "runtime" / "alert_locks" / "one.lock").is_file()
            )

            backed_up_root_db = backup_root / "sqlite" / "root" / "data" / "runtime" / "main.db"
            with closing(sqlite3.connect(backed_up_root_db)) as connection:
                values = [row[0] for row in connection.execute("SELECT value FROM events ORDER BY rowid")]
            self.assertEqual(values, ["checkpointed", "from-wal"])
            self.assertTrue(
                (backup_root / "sqlite" / "mexc" / "data" / "runtime" / "runtime.db").is_file()
            )

            manifest_path = backup_root / "MANIFEST_SHA256.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertGreater(len(manifest), 0)
            for row in manifest:
                payload_path = backup_root / Path(row["RelativePath"])
                self.assertEqual(payload_path.stat().st_size, row["Length"])
                digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
                self.assertEqual(digest, row["SHA256"])


if __name__ == "__main__":
    unittest.main()
