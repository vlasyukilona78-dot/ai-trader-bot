import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from queue import Empty, Queue

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".runtime_env" / "Lib" / "site-packages"
if PROJECT_RUNTIME_SITE_PACKAGES.exists():
    runtime_site_packages = str(PROJECT_RUNTIME_SITE_PACKAGES)
    if runtime_site_packages not in sys.path:
        sys.path.insert(0, runtime_site_packages)

from app.main import main

PROJECT_RUNTIME_PYTHON = PROJECT_ROOT / ".runtime_env" / "Scripts" / "python.exe"
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PROJECT_RUNTIME_CFG = PROJECT_ROOT / ".runtime_env" / "pyvenv.cfg"
PROJECT_RUNTIME_LOG_DIR = PROJECT_ROOT / "logs" / "runtime"
SUPERVISOR_STATE_PATH = PROJECT_RUNTIME_LOG_DIR / "active_supervisor.json"


def _select_base_python_from_runtime_cfg() -> str | None:
    if not PROJECT_RUNTIME_CFG.exists():
        return None
    try:
        for raw_line in PROJECT_RUNTIME_CFG.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if key.lower() == "executable" and value:
                candidate = Path(value)
                if candidate.exists():
                    return str(candidate)
            if key.lower() == "home" and value:
                candidate = Path(value) / "python.exe"
                if candidate.exists():
                    return str(candidate)
    except Exception:
        return None
    return None


def _select_child_python() -> str:
    runtime_base_python = _select_base_python_from_runtime_cfg()
    candidates = [
        runtime_base_python,
        PROJECT_RUNTIME_PYTHON,
        Path(sys.executable) if sys.executable else None,
        PROJECT_VENV_PYTHON,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if Path(candidate).exists():
                return str(candidate)
        except Exception:
            continue
    return sys.executable


def _stream_process_output(profile: str, process: subprocess.Popen[str], out_queue: Queue[tuple[str, str | None]]) -> None:
    stream = process.stdout
    if stream is None:
        out_queue.put((profile, None))
        return
    try:
        for raw_line in stream:
            out_queue.put((profile, raw_line.rstrip()))
    finally:
        out_queue.put((profile, None))


def _spawn_profile(profile: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["BOT_SIGNAL_PROFILE"] = profile
    env["PYTHONUNBUFFERED"] = "1"
    python_executable = _select_child_python()
    return subprocess.Popen(
        [python_executable, "-u", "-m", "app.main", "--loop", "--signal-profile", profile],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            return


def _read_active_supervisor_state() -> dict[str, object]:
    if not SUPERVISOR_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(SUPERVISOR_STATE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_active_supervisor_state(*, pid: int) -> None:
    PROJECT_RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": int(pid),
        "cwd": str(PROJECT_ROOT),
        "updated_at": int(time.time()),
    }
    SUPERVISOR_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_active_supervisor_state(*, pid: int) -> None:
    state = _read_active_supervisor_state()
    if int(state.get("pid") or 0) != int(pid):
        return
    try:
        SUPERVISOR_STATE_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _stop_previous_supervisor_if_needed() -> None:
    state = _read_active_supervisor_state()
    previous_pid = int(state.get("pid") or 0)
    if previous_pid <= 0 or previous_pid == os.getpid():
        return
    result = subprocess.run(
        ["taskkill", "/PID", str(previous_pid), "/T", "/F"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode == 0:
        print(f"[supervisor] stopped previous supervisor pid={previous_pid}")
        return
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout
    print(f"[supervisor] previous supervisor stop skipped pid={previous_pid} details={details}")


def _run_dual_profile_supervisor() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    except Exception:
        pass
    _stop_previous_supervisor_if_needed()
    _write_active_supervisor_state(pid=os.getpid())
    profiles = ("main", "early")
    out_queue: Queue[tuple[str, str | None]] = Queue()
    processes: dict[str, subprocess.Popen[str]] = {}
    threads: list[threading.Thread] = []

    print("[supervisor] Starting demo profiles: main + early")
    for profile in profiles:
        process = _spawn_profile(profile)
        processes[profile] = process
        thread = threading.Thread(
            target=_stream_process_output,
            args=(profile, process, out_queue),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        print(f"[supervisor] {profile} pid={process.pid}")

    active = set(profiles)
    interrupted = False
    try:
        while active:
            try:
                profile, line = out_queue.get(timeout=0.25)
            except Empty:
                for profile in list(active):
                    if processes[profile].poll() is not None:
                        print(f"[supervisor] {profile} exited code={processes[profile].returncode}")
                        active.discard(profile)
                continue

            if line is None:
                code = processes[profile].poll()
                print(f"[supervisor] {profile} exited code={code}")
                active.discard(profile)
                continue

            if line:
                print(f"[{profile}] {line}")
    except KeyboardInterrupt:
        interrupted = True
        print("[supervisor] Stopping child profiles...")
    finally:
        for process in processes.values():
            _terminate_process(process)
        for thread in threads:
            thread.join(timeout=1)
        _clear_active_supervisor_state(pid=os.getpid())

    if interrupted:
        return 0
    exit_codes = [process.returncode or 0 for process in processes.values()]
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    explicit_profile = str(os.getenv("BOT_SIGNAL_PROFILE", "")).strip().lower()
    if len(sys.argv) == 1:
        if explicit_profile in {"main", "early"}:
            sys.argv.extend(["--loop", "--signal-profile", explicit_profile])
            raise SystemExit(main())
        if explicit_profile in {"both", ""}:
            raise SystemExit(_run_dual_profile_supervisor())
        raise SystemExit(_run_dual_profile_supervisor())
    raise SystemExit(main())
