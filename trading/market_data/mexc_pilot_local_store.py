"""Local-only immutable storage boundary for the frozen MEXC P2 pilot contract.

The module is deliberately not an executor.  It contains no HTTP client, no U5
receipt factory and no default detached-evidence implementation.  A successful
intent claim returns an in-memory owner capability; persisted artifacts and
authoritative reconstruction can never recreate that capability.

Dependency identities in this foundation are declarative, self-reported
bindings, not code identity or cryptographic attestation.  They make fake/local
composition exact, but they do not open the real U5 path.  Real U5 remains
blocked until a reviewed coordinator constructs the concrete dependencies and
accepts no arbitrary plugins, or an independently reviewed pinned-key
cryptographic verifier supplies the missing attestation boundary.

That reviewed construction must supply a side-effect-free bound clock.  The
final permit gate intentionally samples that clock only after slow preflight,
replay and session-artifact validation; only a lightweight physical root/lock
check and pure arithmetic follow.  Before every actual HTTP attempt, a future
real network runner must gate on trusted now, the latest preflight and the
entire remaining worst-case run reservation, with no network call between that
gate and the attempt.  This local/fake foundation cannot substitute for that
coordinator-owned attempt gate.

The supported threat boundary rejects hostile filesystem state and all
caller-visible API substitutions.  Every supported/cooperating writer must
hold this run's lock.  The bounded pre/post checks and finite double scans give
stable-point evidence; they are not an atomic NTFS snapshot.  Arbitrary
external filesystem mutation during or after validation is explicitly out of
scope.  Real U5 therefore remains STOP unless the operator accepts this exact
boundary or a future handle-relative/snapshot implementation replaces it.
Arbitrary in-process private-memory tamper (`object.__setattr__`, `ctypes`,
debugger/process-memory writes) is likewise not a Python capability boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time
from typing import Any, Mapping, Protocol, TypeVar
import uuid

from trading.market_data.mexc_pilot_local_executor import PilotExecutorBindingsV1
from trading.market_data.mexc_pilot_run import (
    EndpointVerificationReceiptV1,
    MexcPublicQaPilotRunManifestV1,
    PilotDiskPreflightReceiptV1,
    PilotIntentDurabilityReceiptV1,
    PilotNetworkIntentV1,
    PilotRunArtifactError,
    PilotRunContractError,
    PilotRunStateV1,
    PilotShardResultV1,
    PilotStepFailureReceiptV1,
    U5PublicPilotAuthorizationReceiptV1,
    parse_pilot_run_manifest_v1,
    pilot_run_contract_hash,
)


PILOT_LOCAL_STORE_CONTRACT_VERSION = "mexc_public_qa_pilot_local_store_v1"
PILOT_DETACHED_EVIDENCE_REQUEST_VERSION = (
    "mexc_public_qa_pilot_detached_evidence_request_v1"
)
PILOT_DETACHED_EVIDENCE_RECEIPT_VERSION = (
    "mexc_public_qa_pilot_detached_evidence_receipt_v1"
)
PILOT_LOCAL_INVENTORY_VERSION = "mexc_public_qa_pilot_local_inventory_v1"
PILOT_LOCAL_RECOVERY_VERSION = "mexc_public_qa_pilot_local_recovery_v1"
PILOT_U5_VERIFICATION_REQUEST_VERSION = (
    "mexc_public_qa_pilot_u5_verification_request_v1"
)
PILOT_U5_VERIFICATION_EVIDENCE_VERSION = (
    "mexc_public_qa_pilot_u5_verification_evidence_v1"
)
PILOT_RUNTIME_AUTHORITY_BINDING_VERSION = (
    "mexc_public_qa_pilot_runtime_authority_binding_v1"
)
PILOT_RUNTIME_IDENTITY_ASSURANCE = (
    "declarative_self_reported_not_code_or_cryptographic_attestation"
)
PILOT_REAL_U5_CONSTRUCTION_POLICY = (
    "blocked_until_reviewed_coordinator_constructs_no_arbitrary_plugins_or_"
    "independent_pinned_key_crypto_verifier"
)
PILOT_FUTURE_HTTP_ATTEMPT_GATE_POLICY = (
    "trusted_now_latest_preflight_entire_remaining_worst_case_before_every_"
    "http_attempt_no_network_between_gate_and_attempt"
)
PILOT_FILESYSTEM_MUTATION_BOUNDARY = (
    "static_hostile_state_rejected_cooperating_writers_share_run_lock_finite_"
    "double_scan_not_atomic_external_mutation_out_of_scope_real_u5_requires_"
    "operator_acceptance_or_handle_relative_snapshot"
)

_PINNED_LOCAL_STORE_CONTRACT_HASH = (
    "21f27ec667d588ac254b893c5f25e44634cc2de1f8567efafc85d08fccca94ab"
)
_PINNED_RUNTIME_AUTHORITY_BINDING_CONTRACT_HASH = (
    "392aa908b30dac7d244a3efdd67887933ab54e5ec07168b184dbedc5870c1004"
)

_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CONTROL_ENTRY_BYTES = 512 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_TREE_DEPTH = 64
_MAX_SCAN_RUNTIME_US = 60 * 1_000_000
_SHA256_CHARS = frozenset("0123456789abcdef")
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PROCESS_LOCK_REGISTRY_GUARD = threading.Lock()
_PROCESS_LOCK_REGISTRY: dict[str, object] = {}
_RunnerResult = TypeVar("_RunnerResult")


class PilotLocalStoreError(PilotRunArtifactError):
    """Base class for local-store failures."""


class PilotLocalStoreLockError(PilotLocalStoreError):
    """The run lock is unavailable or is used by a non-owner."""


class PilotLocalStoreConflictError(PilotLocalStoreError):
    """An immutable slot already exists or has conflicting content."""


class PilotLocalStoreBoundsError(PilotLocalStoreError):
    """A bounded read or inventory limit was exceeded."""


class PilotLocalStoreRecoveryError(PilotLocalStoreError):
    """Authoritative artifacts cannot be reconstructed safely."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and frozenset(value) <= _SHA256_CHARS
    )


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotLocalStoreError("pilot_local_json_is_not_canonicalizable") from exc


def canonical_lf_bytes(payload: object) -> bytes:
    """Return the one accepted persisted representation: canonical JSON + LF."""

    return _canonical_json_bytes(payload) + b"\n"


def _reject_float(_: str) -> None:
    raise PilotLocalStoreError("pilot_local_json_float_is_forbidden")


def _reject_constant(_: str) -> None:
    raise PilotLocalStoreError("pilot_local_json_nonfinite_is_forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PilotLocalStoreError("pilot_local_json_duplicate_key")
        result[key] = value
    return result


def parse_canonical_lf_json(raw: bytes) -> object:
    """Parse exact canonical LF JSON, rejecting duplicates, floats and aliases."""

    if not isinstance(raw, bytes) or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise PilotLocalStoreError("pilot_local_json_lf_is_required")
    if b"\r" in raw:
        raise PilotLocalStoreError("pilot_local_json_cr_is_forbidden")
    try:
        text = raw[:-1].decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except PilotLocalStoreError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotLocalStoreError("pilot_local_json_decode_failed") from exc
    if canonical_lf_bytes(payload) != raw:
        raise PilotLocalStoreError("pilot_local_json_is_not_exact_canonical_lf")
    return payload


def _absolute_unresolved(path: str | os.PathLike[str]) -> Path:
    result = Path(os.path.abspath(os.fspath(path)))
    if not result.is_absolute():
        raise PilotLocalStoreError("pilot_local_path_must_be_absolute")
    return result


def _canonical_physical_target(path: Path) -> Path:
    """Resolve every existing ancestor, preserving only genuinely missing leaves."""

    missing: list[str] = []
    current = path
    while True:
        try:
            observed = current.lstat()
        except FileNotFoundError:
            if current.parent == current:
                raise PilotLocalStoreError(
                    "pilot_local_physical_root_cannot_be_resolved"
                )
            missing.append(current.name)
            current = current.parent
            continue
        except OSError as exc:
            raise PilotLocalStoreError(
                "pilot_local_physical_root_probe_failed"
            ) from exc
        if not _plain_mode(observed, directory=True):
            raise PilotLocalStoreError(
                "pilot_local_physical_root_chain_is_not_plain"
            )
        _validate_plain_existing_chain(current)
        resolved = Path(os.path.realpath(os.fspath(current)))
        break
    for name in reversed(missing):
        resolved = resolved / name
    return _absolute_unresolved(resolved)


def _path_key(path: Path) -> str:
    value = os.path.normcase(os.path.abspath(os.fspath(path)))
    return value.casefold() if os.name == "nt" else value


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        common = os.path.commonpath((os.fspath(left), os.fspath(right)))
    except ValueError:
        return False
    common_key = _path_key(Path(common))
    return common_key in {_path_key(left), _path_key(right)}


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = getattr(stat_result, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _plain_mode(stat_result: os.stat_result, *, directory: bool) -> bool:
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    return expected(stat_result.st_mode) and not _is_reparse(stat_result)


def _validate_plain_existing_chain(path: Path) -> None:
    current = path
    chain: list[Path] = []
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in reversed(chain):
        try:
            observed = item.lstat()
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_directory_chain_is_missing") from exc
        if not _plain_mode(observed, directory=True):
            raise PilotLocalStoreError("pilot_local_directory_chain_has_reparse")


def _ensure_plain_directory(path: Path) -> None:
    current = path
    chain: list[Path] = []
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in reversed(chain):
        if item.exists() or item.is_symlink():
            try:
                observed = item.lstat()
            except OSError as exc:
                raise PilotLocalStoreError(
                    "pilot_local_directory_chain_probe_failed"
                ) from exc
            if not _plain_mode(observed, directory=True):
                raise PilotLocalStoreError(
                    "pilot_local_directory_chain_has_reparse"
                )
            continue
        try:
            item.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_directory_create_failed") from exc
        try:
            observed = item.lstat()
        except OSError as exc:
            raise PilotLocalStoreError(
                "pilot_local_created_directory_probe_failed"
            ) from exc
        if not _plain_mode(observed, directory=True):
            raise PilotLocalStoreError("pilot_local_created_directory_has_reparse")


def _relative_locator(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise PilotLocalStoreError("pilot_local_locator_is_invalid")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise PilotLocalStoreError("pilot_local_locator_is_invalid")
    if candidate.as_posix() != value:
        raise PilotLocalStoreError("pilot_local_locator_is_not_canonical")
    return value


def _target_for(root: Path, locator: str) -> Path:
    parts = PurePosixPath(_relative_locator(locator)).parts
    target = root.joinpath(*parts)
    if os.path.commonpath((os.fspath(root), os.fspath(target))) != os.fspath(root):
        raise PilotLocalStoreError("pilot_local_locator_escapes_root")
    return target


def _stable_signature(
    observed: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    # Windows may advance ctime merely when the file is opened.  Identity,
    # length, mtime, mode and link count remain stable and are the portable
    # point-in-time checks relevant to replacement/content/alias detection.
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_mode,
        observed.st_nlink,
    )


def _file_identity(observed: os.stat_result) -> tuple[int, int]:
    return observed.st_dev, observed.st_ino


def _validate_open_plain_file_identity(handle: Any, path: Path) -> None:
    try:
        opened = os.fstat(handle.fileno())
        visible = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreLockError("pilot_run_lock_identity_probe_failed") from exc
    if (
        not _plain_mode(opened, directory=False)
        or not _plain_mode(visible, directory=False)
        or opened.st_nlink != 1
        or visible.st_nlink != 1
        or _file_identity(opened) != _file_identity(visible)
    ):
        raise PilotLocalStoreLockError("pilot_run_lock_leaf_is_aliased")


def _check_scan_deadline(deadline_ns: int | None, *, code: str) -> None:
    if deadline_ns is not None and time.monotonic_ns() > deadline_ns:
        raise PilotLocalStoreBoundsError(code)


def _reject_windows_named_streams(
    path: Path,
    *,
    deadline_ns: int | None,
) -> int:
    """Return enumerated stream count; reject every non-default NTFS stream."""

    if os.name != "nt":
        return 0
    _check_scan_deadline(deadline_ns, code="pilot_local_stream_scan_runtime_exceeded")
    import ctypes
    from ctypes import wintypes

    class _Win32FindStreamData(ctypes.Structure):
        _fields_ = [
            ("StreamSize", ctypes.c_longlong),
            ("cStreamName", wintypes.WCHAR * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(_Win32FindStreamData),
        wintypes.DWORD,
    ]
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Win32FindStreamData)]
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = [wintypes.HANDLE]
    find_close.restype = wintypes.BOOL
    data = _Win32FindStreamData()
    handle = find_first(os.fspath(path), 0, ctypes.byref(data), 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error == 38:  # ERROR_HANDLE_EOF: no data streams for this object.
            _check_scan_deadline(
                deadline_ns,
                code="pilot_local_stream_scan_runtime_exceeded",
            )
            return 0
        raise PilotLocalStoreError("pilot_local_stream_enumeration_failed")
    count = 0
    try:
        while True:
            _check_scan_deadline(
                deadline_ns,
                code="pilot_local_stream_scan_runtime_exceeded",
            )
            count += 1
            if data.cStreamName.casefold() != "::$data":
                raise PilotLocalStoreError("pilot_local_named_stream_is_forbidden")
            if count > 1:
                raise PilotLocalStoreError(
                    "pilot_local_default_stream_enumeration_is_not_unique"
                )
            if find_next(handle, ctypes.byref(data)):
                continue
            error = ctypes.get_last_error()
            if error != 38:  # ERROR_HANDLE_EOF
                raise PilotLocalStoreError("pilot_local_stream_enumeration_failed")
            break
    finally:
        find_close(handle)
    _check_scan_deadline(deadline_ns, code="pilot_local_stream_scan_runtime_exceeded")
    return count


def _open_plain_lock_file(path: Path) -> Any:
    """Open/create the lock leaf without ever following a precreated alias."""

    _ensure_plain_directory(path.parent)
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                before = path.lstat()
            except OSError as exc:
                raise PilotLocalStoreLockError(
                    "pilot_run_lock_leaf_probe_failed"
                ) from exc
            if not _plain_mode(before, directory=False) or before.st_nlink != 1:
                raise PilotLocalStoreLockError("pilot_run_lock_leaf_is_aliased")
            descriptor = os.open(path, flags)
        handle = os.fdopen(descriptor, "r+b")
        descriptor = None
        try:
            _validate_open_plain_file_identity(handle, path)
        except BaseException:
            handle.close()
            raise
        return handle
    except PilotLocalStoreError:
        raise
    except OSError as exc:
        raise PilotLocalStoreLockError("pilot_run_lock_open_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_exact_plain_file(
    path: Path,
    *,
    max_bytes: int,
    deadline_ns: int | None = None,
    deadline_code: str = "pilot_local_read_runtime_exceeded",
) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        raise PilotLocalStoreBoundsError("pilot_local_read_bound_is_invalid")
    _check_scan_deadline(deadline_ns, code=deadline_code)
    _validate_plain_existing_chain(path.parent)
    try:
        before = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_is_missing") from exc
    if not _plain_mode(before, directory=False):
        raise PilotLocalStoreError("pilot_local_artifact_is_not_plain_file")
    if before.st_nlink != 1:
        raise PilotLocalStoreError("pilot_local_artifact_hardlink_alias_is_forbidden")
    if before.st_size > max_bytes:
        raise PilotLocalStoreBoundsError("pilot_local_artifact_exceeds_read_bound")
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        _check_scan_deadline(deadline_ns, code=deadline_code)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_open_failed") from exc
    try:
        opened = os.fstat(descriptor)
        if not _plain_mode(opened, directory=False):
            raise PilotLocalStoreError("pilot_local_opened_artifact_is_not_plain")
        if _stable_signature(opened) != _stable_signature(before):
            raise PilotLocalStoreError("pilot_local_artifact_changed_before_read")
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            _check_scan_deadline(deadline_ns, code=deadline_code)
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1 - observed_bytes))
            _check_scan_deadline(deadline_ns, code=deadline_code)
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise PilotLocalStoreBoundsError("pilot_local_artifact_exceeds_read_bound")
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_vanished_after_read") from exc
    if (
        _stable_signature(after_open) != _stable_signature(opened)
        or _stable_signature(after_path) != _stable_signature(opened)
    ):
        raise PilotLocalStoreError("pilot_local_artifact_changed_during_read")
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise PilotLocalStoreError("pilot_local_artifact_short_read")
    # NTFS named streams are independent of the base stream's size/mtime and
    # may be added while the base stream is open.  Re-enumerate after the exact
    # read; callers performing a tree scan also run a final whole-tree pass.
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    try:
        final_path = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError(
            "pilot_local_artifact_vanished_after_stream_validation"
        ) from exc
    if _stable_signature(final_path) != _stable_signature(opened):
        raise PilotLocalStoreError(
            "pilot_local_artifact_changed_after_stream_validation"
        )
    _check_scan_deadline(deadline_ns, code=deadline_code)
    return raw


def _hash_exact_plain_file(
    path: Path,
    *,
    max_bytes: int,
    deadline_ns: int | None = None,
    deadline_code: str = "pilot_local_hash_runtime_exceeded",
) -> tuple[str, int, tuple[int, int, int, int, int, int]]:
    """Hash one bounded base stream with pre/post ADS and visible-leaf checks."""

    if type(max_bytes) is not int or max_bytes < 1:
        raise PilotLocalStoreBoundsError("pilot_local_hash_bound_is_invalid")
    _check_scan_deadline(deadline_ns, code=deadline_code)
    _validate_plain_existing_chain(path.parent)
    try:
        before = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_is_missing") from exc
    if not _plain_mode(before, directory=False):
        raise PilotLocalStoreError("pilot_local_artifact_is_not_plain_file")
    if before.st_nlink != 1:
        raise PilotLocalStoreError("pilot_local_artifact_hardlink_alias_is_forbidden")
    if before.st_size > max_bytes:
        raise PilotLocalStoreBoundsError("pilot_local_artifact_exceeds_hash_bound")
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        _check_scan_deadline(deadline_ns, code=deadline_code)
        descriptor = os.open(path, flags)
        _check_scan_deadline(deadline_ns, code=deadline_code)
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_open_failed") from exc
    digest = hashlib.sha256()
    counted = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not _plain_mode(opened, directory=False)
            or _stable_signature(opened) != _stable_signature(before)
        ):
            raise PilotLocalStoreError("pilot_local_artifact_changed_before_hash")
        while True:
            _check_scan_deadline(deadline_ns, code=deadline_code)
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            _check_scan_deadline(deadline_ns, code=deadline_code)
            if not chunk:
                break
            counted += len(chunk)
            if counted > max_bytes:
                raise PilotLocalStoreBoundsError(
                    "pilot_local_artifact_exceeds_hash_bound"
                )
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_artifact_vanished_after_hash") from exc
    if (
        counted != opened.st_size
        or _stable_signature(after_open) != _stable_signature(opened)
        or _stable_signature(after_path) != _stable_signature(opened)
    ):
        raise PilotLocalStoreError("pilot_local_artifact_changed_during_hash")
    _reject_windows_named_streams(path, deadline_ns=deadline_ns)
    try:
        final_path = path.lstat()
    except OSError as exc:
        raise PilotLocalStoreError(
            "pilot_local_artifact_vanished_after_stream_validation"
        ) from exc
    final_signature = _stable_signature(final_path)
    if final_signature != _stable_signature(opened):
        raise PilotLocalStoreError(
            "pilot_local_artifact_changed_after_stream_validation"
        )
    _check_scan_deadline(deadline_ns, code=deadline_code)
    return digest.hexdigest(), counted, final_signature


def _write_temp_and_link(
    target: Path,
    body: bytes,
    *,
    strict_create_new: bool,
) -> bool:
    """Publish with a same-directory hardlink; return True only for our winner."""

    _ensure_plain_directory(target.parent)
    if target.exists() or target.is_symlink():
        if strict_create_new:
            raise PilotLocalStoreConflictError("pilot_local_create_new_slot_preexists")
        existing = _read_exact_plain_file(target, max_bytes=len(body))
        if existing != body:
            raise PilotLocalStoreConflictError("pilot_local_immutable_artifact_conflict")
        return False
    temporary = target.with_name(f".pilot-{uuid.uuid4().hex}.tmp")
    cleanup_error: OSError | None = None
    won = True
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            if strict_create_new:
                raise PilotLocalStoreConflictError(
                    "pilot_local_create_new_slot_lost"
                ) from exc
            existing = _read_exact_plain_file(target, max_bytes=len(body))
            if existing != body:
                raise PilotLocalStoreConflictError(
                    "pilot_local_immutable_artifact_conflict"
                ) from exc
            won = False
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_hardlink_publication_failed") from exc
    except PilotLocalStoreError:
        raise
    except OSError as exc:
        raise PilotLocalStoreError("pilot_local_temp_publication_failed") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        raise PilotLocalStoreError("pilot_local_temp_cleanup_failed") from cleanup_error
    if _read_exact_plain_file(target, max_bytes=len(body)) != body:
        raise PilotLocalStoreError("pilot_local_publication_reload_mismatch")
    return won


@dataclass(frozen=True)
class PilotClockSampleV1:
    epoch_us: int
    monotonic_us: int
    clock_domain_id: str

    def __post_init__(self) -> None:
        if type(self.epoch_us) is not int or self.epoch_us < 1:
            raise PilotLocalStoreError("pilot_local_clock_epoch_is_invalid")
        if type(self.monotonic_us) is not int or self.monotonic_us < 0:
            raise PilotLocalStoreError("pilot_local_clock_monotonic_is_invalid")
        if not isinstance(self.clock_domain_id, str) or not _SAFE_IDENTIFIER_RE.fullmatch(
            self.clock_domain_id
        ):
            raise PilotLocalStoreError("pilot_local_clock_domain_is_invalid")


class PilotEvidenceClock(Protocol):
    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    @property
    def clock_domain_id(self) -> str: ...

    def sample(self) -> PilotClockSampleV1:
        """Return a same-domain epoch/monotonic observation."""


@dataclass(frozen=True)
class PilotDetachedEvidenceRequestV1:
    evidence_kind: str
    manifest_hash: str
    subject_hash: str
    artifact_sha256: str
    relative_locator: str
    publisher_instance_id: str
    observed_at_us: int
    observed_monotonic_us: int
    clock_domain_id: str
    contract_version: str = PILOT_DETACHED_EVIDENCE_REQUEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_DETACHED_EVIDENCE_REQUEST_VERSION:
            raise PilotLocalStoreError("pilot_detached_request_version_mismatch")
        if self.evidence_kind not in {
            "session_claim_reload",
            "intent_candidate_publication",
            "intent_candidate_reload",
            "intent_reservation_anchor",
        }:
            raise PilotLocalStoreError("pilot_detached_evidence_kind_is_invalid")
        for value in (self.manifest_hash, self.subject_hash, self.artifact_sha256):
            if not _is_sha256(value):
                raise PilotLocalStoreError("pilot_detached_evidence_hash_is_invalid")
        _relative_locator(self.relative_locator)
        if not isinstance(
            self.publisher_instance_id, str
        ) or not _SAFE_IDENTIFIER_RE.fullmatch(self.publisher_instance_id):
            raise PilotLocalStoreError("pilot_detached_publisher_is_invalid")
        PilotClockSampleV1(
            self.observed_at_us,
            self.observed_monotonic_us,
            self.clock_domain_id,
        )

    @property
    def request_hash(self) -> str:
        return _sha256(_canonical_json_bytes(self.as_dict()))

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

@dataclass(frozen=True)
class PilotDetachedEvidenceReceiptV1:
    request_hash: str
    evidence_hash: str
    anchored_at_us: int
    anchored_monotonic_us: int
    clock_domain_id: str
    contract_version: str = PILOT_DETACHED_EVIDENCE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_DETACHED_EVIDENCE_RECEIPT_VERSION:
            raise PilotLocalStoreError("pilot_detached_receipt_version_mismatch")
        if not _is_sha256(self.request_hash) or not _is_sha256(self.evidence_hash):
            raise PilotLocalStoreError("pilot_detached_receipt_hash_is_invalid")
        PilotClockSampleV1(
            self.anchored_at_us,
            self.anchored_monotonic_us,
            self.clock_domain_id,
        )

    def validate_for(self, request: PilotDetachedEvidenceRequestV1) -> None:
        if self.request_hash != request.request_hash:
            raise PilotLocalStoreError("pilot_detached_receipt_subject_mismatch")
        if self.clock_domain_id != request.clock_domain_id:
            raise PilotLocalStoreError("pilot_detached_receipt_clock_domain_mismatch")
        if (
            self.anchored_at_us < request.observed_at_us
            or self.anchored_monotonic_us < request.observed_monotonic_us
        ):
            raise PilotLocalStoreError("pilot_detached_receipt_precedes_observation")


class PilotDetachedEvidenceSink(Protocol):
    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    @property
    def domain_id(self) -> str: ...

    def anchor(
        self, request: PilotDetachedEvidenceRequestV1
    ) -> PilotDetachedEvidenceReceiptV1:
        """Persist evidence outside the subject inventory and return its receipt."""


@dataclass(frozen=True)
class PilotRuntimeAuthorityBindingV1:
    coordinator_implementation_contract_version: str
    coordinator_implementation_contract_hash: str
    clock_contract_version: str
    clock_contract_hash: str
    clock_domain_id: str
    detached_anchor_sink_contract_version: str
    detached_anchor_sink_contract_hash: str
    detached_anchor_sink_domain_id: str
    u5_verifier_contract_version: str
    u5_verifier_contract_hash: str
    u5_verifier_domain_id: str
    u5_verifier_trust_key_id: str
    u5_verifier_policy_version: str
    u5_verifier_policy_hash: str
    contract_version: str = PILOT_RUNTIME_AUTHORITY_BINDING_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_RUNTIME_AUTHORITY_BINDING_VERSION:
            raise PilotLocalStoreError(
                "pilot_runtime_authority_binding_version_mismatch"
            )
        for name in (
            "coordinator_implementation_contract_version",
            "clock_contract_version",
            "clock_domain_id",
            "detached_anchor_sink_contract_version",
            "detached_anchor_sink_domain_id",
            "u5_verifier_contract_version",
            "u5_verifier_domain_id",
            "u5_verifier_trust_key_id",
            "u5_verifier_policy_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SAFE_IDENTIFIER_RE.fullmatch(value):
                raise PilotLocalStoreError(
                    f"pilot_runtime_authority_{name}_is_invalid"
                )
        for name in (
            "coordinator_implementation_contract_hash",
            "clock_contract_hash",
            "detached_anchor_sink_contract_hash",
            "u5_verifier_contract_hash",
            "u5_verifier_policy_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise PilotLocalStoreError(
                    f"pilot_runtime_authority_{name}_is_invalid"
                )

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: object) -> "PilotRuntimeAuthorityBindingV1":
        if not isinstance(payload, dict) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise PilotLocalStoreError(
                "pilot_runtime_authority_binding_schema_mismatch"
            )
        return cls(**payload)

    @property
    def binding_hash(self) -> str:
        return _sha256(_canonical_json_bytes(self.as_dict()))


@dataclass(frozen=True)
class PilotU5VerificationRequestV1:
    manifest_hash: str
    authorization_receipt_hash: str
    external_authority_evidence_hash: str
    executor_bindings_hash: str
    runtime_authority_binding_hash: str
    orchestrator_session_id: str
    process_challenge_hash: str
    publisher_instance_id: str
    process_id: int
    requested_at_us: int
    requested_monotonic_us: int
    clock_contract_version: str
    clock_contract_hash: str
    clock_domain_id: str
    u5_verifier_contract_version: str
    u5_verifier_contract_hash: str
    u5_verifier_domain_id: str
    u5_verifier_trust_key_id: str
    u5_verifier_policy_version: str
    u5_verifier_policy_hash: str
    contract_version: str = PILOT_U5_VERIFICATION_REQUEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_U5_VERIFICATION_REQUEST_VERSION:
            raise PilotLocalStoreError("pilot_u5_verification_request_version_mismatch")
        for value in (
            self.manifest_hash,
            self.authorization_receipt_hash,
            self.external_authority_evidence_hash,
            self.executor_bindings_hash,
            self.runtime_authority_binding_hash,
            self.process_challenge_hash,
            self.clock_contract_hash,
            self.u5_verifier_contract_hash,
            self.u5_verifier_policy_hash,
        ):
            if not _is_sha256(value):
                raise PilotLocalStoreError("pilot_u5_verification_request_hash_is_invalid")
        for value in (
            self.orchestrator_session_id,
            self.publisher_instance_id,
            self.clock_contract_version,
            self.u5_verifier_contract_version,
            self.u5_verifier_domain_id,
            self.u5_verifier_trust_key_id,
            self.u5_verifier_policy_version,
        ):
            if not isinstance(value, str) or not _SAFE_IDENTIFIER_RE.fullmatch(value):
                raise PilotLocalStoreError(
                    "pilot_u5_verification_request_identifier_is_invalid"
                )
        if type(self.process_id) is not int or self.process_id < 1:
            raise PilotLocalStoreError("pilot_u5_verification_process_is_invalid")
        PilotClockSampleV1(
            self.requested_at_us,
            self.requested_monotonic_us,
            self.clock_domain_id,
        )

    @property
    def request_hash(self) -> str:
        return _sha256(_canonical_json_bytes(self.as_dict()))

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PilotU5VerificationEvidenceV1:
    request_hash: str
    manifest_hash: str
    authorization_receipt_hash: str
    external_authority_evidence_hash: str
    executor_bindings_hash: str
    runtime_authority_binding_hash: str
    process_challenge_hash: str
    verification_evidence_hash: str
    verified_at_us: int
    verified_monotonic_us: int
    clock_contract_version: str
    clock_contract_hash: str
    clock_domain_id: str
    u5_verifier_contract_version: str
    u5_verifier_contract_hash: str
    u5_verifier_domain_id: str
    u5_verifier_trust_key_id: str
    u5_verifier_policy_version: str
    u5_verifier_policy_hash: str
    offline_verification_passed: bool = True
    receipt_is_not_self_authorizing_acknowledged: bool = True
    contract_version: str = PILOT_U5_VERIFICATION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_U5_VERIFICATION_EVIDENCE_VERSION:
            raise PilotLocalStoreError("pilot_u5_verification_evidence_version_mismatch")
        for value in (
            self.request_hash,
            self.manifest_hash,
            self.authorization_receipt_hash,
            self.external_authority_evidence_hash,
            self.executor_bindings_hash,
            self.runtime_authority_binding_hash,
            self.process_challenge_hash,
            self.verification_evidence_hash,
            self.clock_contract_hash,
            self.u5_verifier_contract_hash,
            self.u5_verifier_policy_hash,
        ):
            if not _is_sha256(value):
                raise PilotLocalStoreError("pilot_u5_verification_evidence_hash_is_invalid")
        if (
            self.offline_verification_passed is not True
            or self.receipt_is_not_self_authorizing_acknowledged is not True
        ):
            raise PilotLocalStoreError("pilot_u5_verification_evidence_did_not_pass")
        for value in (
            self.clock_contract_version,
            self.u5_verifier_contract_version,
            self.u5_verifier_domain_id,
            self.u5_verifier_trust_key_id,
            self.u5_verifier_policy_version,
        ):
            if not isinstance(value, str) or not _SAFE_IDENTIFIER_RE.fullmatch(value):
                raise PilotLocalStoreError(
                    "pilot_u5_verification_evidence_identifier_is_invalid"
                )
        PilotClockSampleV1(
            self.verified_at_us,
            self.verified_monotonic_us,
            self.clock_domain_id,
        )

    def validate_for(self, request: PilotU5VerificationRequestV1) -> None:
        if (
            self.request_hash != request.request_hash
            or self.manifest_hash != request.manifest_hash
            or self.authorization_receipt_hash
            != request.authorization_receipt_hash
            or self.external_authority_evidence_hash
            != request.external_authority_evidence_hash
            or self.executor_bindings_hash != request.executor_bindings_hash
            or self.runtime_authority_binding_hash
            != request.runtime_authority_binding_hash
            or self.process_challenge_hash != request.process_challenge_hash
            or self.clock_contract_version != request.clock_contract_version
            or self.clock_contract_hash != request.clock_contract_hash
            or self.u5_verifier_contract_version
            != request.u5_verifier_contract_version
            or self.u5_verifier_contract_hash != request.u5_verifier_contract_hash
            or self.u5_verifier_domain_id != request.u5_verifier_domain_id
            or self.u5_verifier_trust_key_id != request.u5_verifier_trust_key_id
            or self.u5_verifier_policy_version
            != request.u5_verifier_policy_version
            or self.u5_verifier_policy_hash != request.u5_verifier_policy_hash
        ):
            raise PilotLocalStoreError("pilot_u5_verification_evidence_binding_mismatch")
        if self.clock_domain_id != request.clock_domain_id:
            raise PilotLocalStoreError("pilot_u5_verification_clock_domain_mismatch")
        if (
            self.verified_at_us < request.requested_at_us
            or self.verified_monotonic_us < request.requested_monotonic_us
        ):
            raise PilotLocalStoreError("pilot_u5_verification_precedes_request")


class PilotU5AuthorityVerifier(Protocol):
    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    @property
    def domain_id(self) -> str: ...

    @property
    def trust_key_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    @property
    def policy_hash(self) -> str: ...

    def verify(
        self, request: PilotU5VerificationRequestV1
    ) -> PilotU5VerificationEvidenceV1:
        """Offline-verify detached authority; the store provides no default."""


class PilotOwnedIntentRunner(Protocol):
    @property
    def contract_version(self) -> str: ...

    @property
    def contract_hash(self) -> str: ...

    def __call__(self, intent: PilotNetworkIntentV1) -> Any: ...


class _NonSerializableCapability:
    __slots__ = ()

    def __reduce__(self) -> object:
        raise TypeError("pilot_local_owner_capability_is_not_serializable")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("pilot_local_owner_capability_is_not_serializable")


class PilotRunSessionCapability(_NonSerializableCapability):
    __slots__ = (
        "_store_nonce",
        "_lease_nonce",
        "_binding_guard",
        "_session_nonce",
        "_owner_thread",
        "_process_id",
        "manifest_hash",
        "authorization_receipt_hash",
        "orchestrator_session_id",
        "claim_evidence_hash",
        "authority_verification_evidence_hash",
        "process_challenge_hash",
        "executor_bindings_hash",
        "runtime_authority_binding_hash",
        "clock_identity",
        "u5_verifier_identity",
        "output_root_identity",
        "session_claim_locator",
        "session_claim_artifact_sha256",
    )

    def __init__(
        self,
        *,
        store_nonce: object,
        lease_nonce: object,
        binding_guard: object,
        owner_thread: int,
        manifest_hash: str,
        authorization_receipt_hash: str,
        orchestrator_session_id: str,
        claim_evidence_hash: str,
        authority_verification_evidence_hash: str,
        process_challenge_hash: str,
        executor_bindings_hash: str,
        runtime_authority_binding_hash: str,
        clock_identity: tuple[str, str, str],
        u5_verifier_identity: tuple[str, str, str, str, str, str],
        output_root_identity: tuple[int, int],
        session_claim_locator: str,
        session_claim_artifact_sha256: str,
    ) -> None:
        self._store_nonce = store_nonce
        self._lease_nonce = lease_nonce
        self._binding_guard = binding_guard
        self._session_nonce = object()
        self._owner_thread = owner_thread
        self._process_id = os.getpid()
        self.manifest_hash = manifest_hash
        self.authorization_receipt_hash = authorization_receipt_hash
        self.orchestrator_session_id = orchestrator_session_id
        self.claim_evidence_hash = claim_evidence_hash
        self.authority_verification_evidence_hash = (
            authority_verification_evidence_hash
        )
        self.process_challenge_hash = process_challenge_hash
        self.executor_bindings_hash = executor_bindings_hash
        self.runtime_authority_binding_hash = runtime_authority_binding_hash
        self.clock_identity = clock_identity
        self.u5_verifier_identity = u5_verifier_identity
        self.output_root_identity = output_root_identity
        self.session_claim_locator = session_claim_locator
        self.session_claim_artifact_sha256 = session_claim_artifact_sha256


class PilotIntentOwnerCapability(_NonSerializableCapability):
    __slots__ = (
        "_store_nonce",
        "_lease_nonce",
        "_session_nonce",
        "_owner_thread",
        "_process_id",
        "manifest_hash",
        "intent_hash",
        "intent_slot_id",
        "stage",
        "ordinal",
        "_consumed",
        "_terminal",
        "_binding_guard",
        "executor_bindings_hash",
        "runtime_authority_binding_hash",
        "clock_identity",
        "u5_verifier_identity",
        "output_root_identity",
    )

    def __init__(
        self,
        *,
        store_nonce: object,
        lease_nonce: object,
        session_nonce: object,
        owner_thread: int,
        manifest_hash: str,
        intent_hash: str,
        intent_slot_id: str,
        stage: str,
        ordinal: int,
        binding_guard: object,
        executor_bindings_hash: str,
        runtime_authority_binding_hash: str,
        clock_identity: tuple[str, str, str],
        u5_verifier_identity: tuple[str, str, str, str, str, str],
        output_root_identity: tuple[int, int],
    ) -> None:
        self._store_nonce = store_nonce
        self._lease_nonce = lease_nonce
        self._session_nonce = session_nonce
        self._owner_thread = owner_thread
        self._process_id = os.getpid()
        self.manifest_hash = manifest_hash
        self.intent_hash = intent_hash
        self.intent_slot_id = intent_slot_id
        self.stage = stage
        self.ordinal = ordinal
        self._consumed = False
        self._terminal = False
        self._binding_guard = binding_guard
        self.executor_bindings_hash = executor_bindings_hash
        self.runtime_authority_binding_hash = runtime_authority_binding_hash
        self.clock_identity = clock_identity
        self.u5_verifier_identity = u5_verifier_identity
        self.output_root_identity = output_root_identity


class PilotRunLockLease(_NonSerializableCapability):
    __slots__ = ("_store", "_nonce", "_binding_guard", "_closed")

    def __init__(
        self,
        store: "MexcPilotLocalStoreV1",
        nonce: object,
        binding_guard: object,
    ) -> None:
        self._store = store
        self._nonce = nonce
        self._binding_guard = binding_guard
        self._closed = False

    def __enter__(self) -> "PilotRunLockLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._store._release_run_lock(self._nonce, self._binding_guard)
            self._closed = True


@dataclass(frozen=True)
class PilotIntentClaimResultV1:
    state: PilotRunStateV1
    intent: PilotNetworkIntentV1
    owner_capability: PilotIntentOwnerCapability


@dataclass(frozen=True)
class PilotInventoryEntryV1:
    relative_path: str
    artifact_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _relative_locator(self.relative_path)
        if not _is_sha256(self.artifact_sha256):
            raise PilotLocalStoreError("pilot_inventory_hash_is_invalid")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise PilotLocalStoreError("pilot_inventory_size_is_invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "artifact_sha256": self.artifact_sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class PilotInventoryScanV1:
    manifest_hash: str
    entries: tuple[PilotInventoryEntryV1, ...]
    total_bytes: int
    scanned_at_us: int
    scanned_monotonic_us: int
    clock_domain_id: str
    contract_version: str = PILOT_LOCAL_INVENTORY_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_LOCAL_INVENTORY_VERSION:
            raise PilotLocalStoreError("pilot_inventory_version_mismatch")
        if not _is_sha256(self.manifest_hash):
            raise PilotLocalStoreError("pilot_inventory_manifest_hash_is_invalid")
        if not isinstance(self.entries, tuple):
            raise PilotLocalStoreError("pilot_inventory_entries_are_not_immutable")
        paths = tuple(item.relative_path for item in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise PilotLocalStoreError("pilot_inventory_paths_are_not_canonical")
        if self.total_bytes != sum(item.byte_count for item in self.entries):
            raise PilotLocalStoreError("pilot_inventory_total_is_invalid")
        PilotClockSampleV1(
            self.scanned_at_us,
            self.scanned_monotonic_us,
            self.clock_domain_id,
        )

    @property
    def inventory_hash(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "domain": PILOT_LOCAL_INVENTORY_VERSION,
                    "manifest_hash": self.manifest_hash,
                    "entries": [item.as_dict() for item in self.entries],
                    "total_bytes": self.total_bytes,
                }
            )
        )


@dataclass(frozen=True)
class PilotRecoveryReportV1:
    state: PilotRunStateV1
    status: str
    stop_code: str | None
    stop_evidence_hash: str | None
    residue_paths: tuple[str, ...]
    restart_detected: bool
    network_permitted: bool = False
    contract_version: str = PILOT_LOCAL_RECOVERY_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PILOT_LOCAL_RECOVERY_VERSION:
            raise PilotLocalStoreError("pilot_recovery_version_mismatch")
        if self.status not in {"reconstructed_no_network", "stopped_no_network"}:
            raise PilotLocalStoreError("pilot_recovery_status_is_invalid")
        if self.network_permitted is not False:
            raise PilotLocalStoreError("pilot_recovery_cannot_grant_network")
        if tuple(sorted(self.residue_paths)) != self.residue_paths:
            raise PilotLocalStoreError("pilot_recovery_residue_is_not_canonical")


def _lock_file_nonblocking(handle: Any) -> None:
    descriptor = handle.fileno()
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(descriptor)
        handle.seek(0)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise PilotLocalStoreLockError("pilot_run_os_lock_is_held") from exc
    else:
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PilotLocalStoreLockError("pilot_run_os_lock_is_held") from exc


def _unlock_file(handle: Any) -> None:
    descriptor = handle.fileno()
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


class MexcPilotLocalStoreV1:
    """Pinned local persistence boundary; never an executor or authority source."""

    __slots__ = (
        "_manifest",
        "_executor_bindings",
        "_runtime_authority_binding",
        "_output_root",
        "_external_state_root",
        "_detached_evidence_sink",
        "_clock",
        "_publisher_instance_id",
        "_u5_authority_verifier",
        "_store_nonce",
        "_binding_guard",
        "_binding_snapshot",
        "_bound_output_root_identity",
        "_active_lease_nonce",
        "_active_lease_binding_guard",
        "_lock_owner_thread",
        "_lock_handle",
        "_lock_registry_key",
        "_session_capability",
        "_active_intent_capability",
        "_consumed_intent_hashes",
    )

    _WRITE_ONCE_SLOTS = frozenset(
        {
            "_manifest",
            "_executor_bindings",
            "_runtime_authority_binding",
            "_output_root",
            "_external_state_root",
            "_detached_evidence_sink",
            "_clock",
            "_publisher_instance_id",
            "_u5_authority_verifier",
            "_store_nonce",
            "_binding_guard",
            "_binding_snapshot",
            "_bound_output_root_identity",
            "_consumed_intent_hashes",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._WRITE_ONCE_SLOTS:
            try:
                object.__getattribute__(self, name)
            except AttributeError:
                pass
            else:
                raise AttributeError("pilot_local_store_binding_is_read_only")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        manifest: MexcPublicQaPilotRunManifestV1,
        executor_bindings: PilotExecutorBindingsV1,
        runtime_authority_binding: PilotRuntimeAuthorityBindingV1,
        output_root: str | os.PathLike[str],
        detached_evidence_sink: PilotDetachedEvidenceSink,
        clock: PilotEvidenceClock,
        publisher_instance_id: str,
        u5_authority_verifier: PilotU5AuthorityVerifier,
    ) -> None:
        if not isinstance(manifest, MexcPublicQaPilotRunManifestV1):
            raise PilotRunContractError("pilot_local_manifest_is_invalid")
        if not isinstance(executor_bindings, PilotExecutorBindingsV1):
            raise PilotLocalStoreError("pilot_local_executor_bindings_are_required")
        if not isinstance(
            runtime_authority_binding,
            PilotRuntimeAuthorityBindingV1,
        ):
            raise PilotLocalStoreError(
                "pilot_local_runtime_authority_binding_is_required"
            )
        if not isinstance(
            publisher_instance_id, str
        ) or not _SAFE_IDENTIFIER_RE.fullmatch(publisher_instance_id):
            raise PilotLocalStoreError("pilot_local_publisher_instance_is_invalid")
        if detached_evidence_sink is None or not callable(
            getattr(detached_evidence_sink, "anchor", None)
        ):
            raise PilotLocalStoreError("pilot_local_detached_sink_is_required")
        if clock is None or not callable(getattr(clock, "sample", None)):
            raise PilotLocalStoreError("pilot_local_clock_is_required")
        declared_output_root = _absolute_unresolved(output_root)
        physical_output_root = _canonical_physical_target(declared_output_root)
        if os.name == "nt" and _path_key(physical_output_root) != _path_key(
            declared_output_root
        ):
            raise PilotLocalStoreError("pilot_local_output_root_alias_is_forbidden")
        self._manifest = manifest
        self._executor_bindings = executor_bindings
        self._runtime_authority_binding = runtime_authority_binding
        self._output_root = physical_output_root
        if os.name == "nt":
            windows_path = self.output_root.as_posix()
            drive = self.output_root.drive.rstrip(":").upper()
            suffix = windows_path[2:] if len(windows_path) >= 2 and windows_path[1] == ":" else ""
            observed_locator = f"file:///{drive}:{suffix}"
            if observed_locator != self.manifest.output_root_locator:
                raise PilotLocalStoreError("pilot_local_output_root_locator_mismatch")
        # The lock namespace is keyed by the canonical output slot, not by a
        # manifest.  Competing manifests for one output root therefore contend
        # on one OS/process lock before either can inspect or publish control data.
        external_name = f".{self.output_root.name}.mexc-pilot-local-store-v1"
        self._external_state_root = self.output_root.parent / external_name
        if _paths_overlap(self.output_root, self.external_state_root):
            raise PilotLocalStoreError("pilot_local_external_state_derivation_failed")
        self._detached_evidence_sink = detached_evidence_sink
        self._clock = clock
        self._publisher_instance_id = publisher_instance_id
        if u5_authority_verifier is None or not callable(
            getattr(u5_authority_verifier, "verify", None)
        ):
            raise PilotLocalStoreError("pilot_local_u5_verifier_is_required")
        self._u5_authority_verifier = u5_authority_verifier
        self._assert_declared_contract_bindings()
        self._store_nonce = object()
        self._binding_guard = object()
        self._binding_snapshot = (
            self._manifest,
            self._manifest.manifest_hash,
            self._executor_bindings,
            self._executor_bindings.bindings_hash,
            self._runtime_authority_binding,
            self._runtime_authority_binding.binding_hash,
            self._output_root,
            self._external_state_root,
            self._detached_evidence_sink,
            self._clock,
            self._publisher_instance_id,
            self._u5_authority_verifier,
            self._binding_guard,
        )
        try:
            root_observed = self._output_root.lstat()
        except FileNotFoundError:
            self._bound_output_root_identity: tuple[int, int] | None = None
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_output_root_probe_failed") from exc
        else:
            if not _plain_mode(root_observed, directory=True):
                raise PilotLocalStoreError("pilot_local_output_root_is_not_plain")
            self._bound_output_root_identity = _file_identity(root_observed)
        self._active_lease_nonce: object | None = None
        self._active_lease_binding_guard: object | None = None
        self._lock_owner_thread: int | None = None
        self._lock_handle: Any | None = None
        self._lock_registry_key: str | None = None
        self._session_capability: PilotRunSessionCapability | None = None
        self._active_intent_capability: PilotIntentOwnerCapability | None = None
        self._consumed_intent_hashes: frozenset[str] = frozenset()

    @property
    def manifest(self) -> MexcPublicQaPilotRunManifestV1:
        return self._manifest

    @property
    def executor_bindings(self) -> PilotExecutorBindingsV1:
        return self._executor_bindings

    @property
    def runtime_authority_binding(self) -> PilotRuntimeAuthorityBindingV1:
        return self._runtime_authority_binding

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def external_state_root(self) -> Path:
        return self._external_state_root

    @property
    def detached_evidence_sink(self) -> PilotDetachedEvidenceSink:
        return self._detached_evidence_sink

    @property
    def clock(self) -> PilotEvidenceClock:
        return self._clock

    @property
    def publisher_instance_id(self) -> str:
        return self._publisher_instance_id

    @property
    def u5_authority_verifier(self) -> PilotU5AuthorityVerifier:
        return self._u5_authority_verifier

    def _clock_identity(self) -> tuple[str, str, str]:
        binding = self._runtime_authority_binding
        return (
            binding.clock_contract_version,
            binding.clock_contract_hash,
            binding.clock_domain_id,
        )

    def _u5_verifier_identity(self) -> tuple[str, str, str, str, str, str]:
        binding = self._runtime_authority_binding
        return (
            binding.u5_verifier_contract_version,
            binding.u5_verifier_contract_hash,
            binding.u5_verifier_domain_id,
            binding.u5_verifier_trust_key_id,
            binding.u5_verifier_policy_version,
            binding.u5_verifier_policy_hash,
        )

    def _current_output_root_identity(
        self,
        *,
        recheck_dependencies: bool = True,
    ) -> tuple[int, int]:
        self._assert_constructor_bindings(
            recheck_dependencies=recheck_dependencies,
        )
        identity = self._bound_output_root_identity
        if identity is None:
            raise PilotLocalStoreLockError("pilot_local_output_root_identity_is_unbound")
        return identity

    def _sample_clock(self) -> PilotClockSampleV1:
        self._assert_declared_contract_bindings()
        sample = self._clock.sample()
        if (
            not isinstance(sample, PilotClockSampleV1)
            or sample.clock_domain_id
            != self._runtime_authority_binding.clock_domain_id
        ):
            raise PilotLocalStoreError("pilot_local_clock_sample_binding_mismatch")
        return sample

    def _assert_declared_contract_bindings(self) -> None:
        executor = self._executor_bindings
        runtime = self._runtime_authority_binding
        if (
            executor.local_store_contract_version
            != PILOT_LOCAL_STORE_CONTRACT_VERSION
            or executor.local_store_contract_hash
            != mexc_pilot_local_store_contract_hash()
        ):
            raise PilotLocalStoreError("pilot_local_store_contract_binding_mismatch")
        if (
            executor.coordinator_contract_version != runtime.contract_version
            or executor.coordinator_contract_hash != runtime.binding_hash
            or executor.clock_contract_version != runtime.clock_contract_version
            or executor.clock_contract_hash != runtime.clock_contract_hash
            or executor.detached_anchor_sink_contract_version
            != runtime.detached_anchor_sink_contract_version
            or executor.detached_anchor_sink_contract_hash
            != runtime.detached_anchor_sink_contract_hash
        ):
            raise PilotLocalStoreError("pilot_local_runtime_binding_mismatch")
        if (
            self._manifest.endpoint_verification.verifier_contract_version
            != executor.endpoint_verifier_binding_version
            or self._manifest.endpoint_verification.verifier_contract_hash
            != executor.endpoint_verifier_binding_hash
            or self._manifest.shard_executor_contract_version
            != executor.shard_executor_binding_version
            or self._manifest.shard_executor_contract_hash
            != executor.shard_executor_binding_hash
        ):
            raise PilotLocalStoreError("pilot_local_manifest_composite_binding_mismatch")
        if (
            getattr(self._clock, "contract_version", None)
            != runtime.clock_contract_version
            or getattr(self._clock, "contract_hash", None)
            != runtime.clock_contract_hash
            or getattr(self._clock, "clock_domain_id", None)
            != runtime.clock_domain_id
            or getattr(self._detached_evidence_sink, "contract_version", None)
            != runtime.detached_anchor_sink_contract_version
            or getattr(self._detached_evidence_sink, "contract_hash", None)
            != runtime.detached_anchor_sink_contract_hash
            or getattr(self._detached_evidence_sink, "domain_id", None)
            != runtime.detached_anchor_sink_domain_id
            or getattr(self._u5_authority_verifier, "contract_version", None)
            != runtime.u5_verifier_contract_version
            or getattr(self._u5_authority_verifier, "contract_hash", None)
            != runtime.u5_verifier_contract_hash
            or getattr(self._u5_authority_verifier, "domain_id", None)
            != runtime.u5_verifier_domain_id
            or getattr(self._u5_authority_verifier, "trust_key_id", None)
            != runtime.u5_verifier_trust_key_id
            or getattr(self._u5_authority_verifier, "policy_version", None)
            != runtime.u5_verifier_policy_version
            or getattr(self._u5_authority_verifier, "policy_hash", None)
            != runtime.u5_verifier_policy_hash
        ):
            raise PilotLocalStoreError("pilot_local_runtime_dependency_identity_mismatch")

    def _assert_constructor_bindings(
        self,
        *,
        recheck_dependencies: bool = True,
    ) -> None:
        snapshot = self._binding_snapshot
        if (
            self._manifest is not snapshot[0]
            or self._manifest.manifest_hash != snapshot[1]
            or self._executor_bindings is not snapshot[2]
            or self._executor_bindings.bindings_hash != snapshot[3]
            or self._runtime_authority_binding is not snapshot[4]
            or self._runtime_authority_binding.binding_hash != snapshot[5]
            or self._output_root != snapshot[6]
            or self._external_state_root != snapshot[7]
            or self._detached_evidence_sink is not snapshot[8]
            or self._clock is not snapshot[9]
            or self._publisher_instance_id != snapshot[10]
            or self._u5_authority_verifier is not snapshot[11]
            or self._binding_guard is not snapshot[12]
            or self._external_state_root
            != self._output_root.parent
            / f".{self._output_root.name}.mexc-pilot-local-store-v1"
            or _path_key(_canonical_physical_target(self._output_root))
            != _path_key(self._output_root)
        ):
            raise PilotLocalStoreLockError("pilot_local_constructor_binding_changed")
        if recheck_dependencies:
            try:
                self._assert_declared_contract_bindings()
            except PilotLocalStoreError as exc:
                raise PilotLocalStoreLockError(
                    "pilot_local_runtime_dependency_binding_changed"
                ) from exc
        try:
            root_observed = self._output_root.lstat()
        except FileNotFoundError:
            if self._bound_output_root_identity is not None:
                raise PilotLocalStoreLockError(
                    "pilot_local_output_root_identity_disappeared"
                )
        except OSError as exc:
            raise PilotLocalStoreLockError(
                "pilot_local_output_root_identity_probe_failed"
            ) from exc
        else:
            if not _plain_mode(root_observed, directory=True):
                raise PilotLocalStoreLockError(
                    "pilot_local_output_root_identity_is_not_plain"
                )
            observed_identity = _file_identity(root_observed)
            if self._bound_output_root_identity is None:
                object.__setattr__(
                    self,
                    "_bound_output_root_identity",
                    observed_identity,
                )
            elif observed_identity != self._bound_output_root_identity:
                raise PilotLocalStoreLockError(
                    "pilot_local_output_root_identity_changed"
                )

    @property
    def run_control_root(self) -> Path:
        return self.output_root / "run-control"

    @property
    def _run_lock_path(self) -> Path:
        return self.external_state_root / "locks" / "run.lock"

    def acquire_run_lock(self) -> PilotRunLockLease:
        self._assert_constructor_bindings()
        if self._active_lease_nonce is not None:
            raise PilotLocalStoreLockError("pilot_run_lock_is_not_reentrant")
        _ensure_plain_directory(self._run_lock_path.parent)
        registry_key = _path_key(self._run_lock_path)
        registry_token = object()
        with _PROCESS_LOCK_REGISTRY_GUARD:
            if registry_key in _PROCESS_LOCK_REGISTRY:
                raise PilotLocalStoreLockError("pilot_run_process_lock_is_held")
            _PROCESS_LOCK_REGISTRY[registry_key] = registry_token
        handle: Any | None = None
        try:
            handle = _open_plain_lock_file(self._run_lock_path)
            _lock_file_nonblocking(handle)
            _validate_open_plain_file_identity(handle, self._run_lock_path)
        except BaseException:
            if handle is not None:
                handle.close()
            with _PROCESS_LOCK_REGISTRY_GUARD:
                if _PROCESS_LOCK_REGISTRY.get(registry_key) is registry_token:
                    del _PROCESS_LOCK_REGISTRY[registry_key]
            raise
        nonce = object()
        self._active_lease_nonce = nonce
        self._active_lease_binding_guard = self._binding_guard
        self._lock_owner_thread = threading.get_ident()
        self._lock_handle = handle
        self._lock_registry_key = registry_key
        return PilotRunLockLease(self, nonce, self._binding_guard)

    def _release_run_lock(self, nonce: object, binding_guard: object) -> None:
        self._require_owner(lease_nonce=nonce, binding_guard=binding_guard)
        assert self._lock_handle is not None
        handle = self._lock_handle
        registry_key = self._lock_registry_key
        if self._active_intent_capability is not None:
            self._active_intent_capability._terminal = True
        self._active_intent_capability = None
        self._session_capability = None
        self._active_lease_nonce = None
        self._active_lease_binding_guard = None
        self._lock_owner_thread = None
        self._lock_handle = None
        self._lock_registry_key = None
        try:
            _unlock_file(handle)
        finally:
            handle.close()
            if registry_key is not None:
                with _PROCESS_LOCK_REGISTRY_GUARD:
                    _PROCESS_LOCK_REGISTRY.pop(registry_key, None)

    def _require_owner(
        self,
        *,
        lease_nonce: object | None = None,
        binding_guard: object | None = None,
        recheck_dependencies: bool = True,
    ) -> object:
        self._assert_constructor_bindings(
            recheck_dependencies=recheck_dependencies,
        )
        active = self._active_lease_nonce
        if active is None:
            raise PilotLocalStoreLockError("pilot_run_lock_is_required")
        if self._lock_owner_thread != threading.get_ident():
            raise PilotLocalStoreLockError("pilot_run_lock_foreign_thread")
        if lease_nonce is not None and lease_nonce is not active:
            raise PilotLocalStoreLockError("pilot_run_lock_capability_mismatch")
        if (
            self._active_lease_binding_guard is not self._binding_guard
            or binding_guard is not None
            and binding_guard is not self._binding_guard
        ):
            raise PilotLocalStoreLockError("pilot_run_lock_binding_mismatch")
        if self._lock_handle is None:
            raise PilotLocalStoreLockError("pilot_run_lock_handle_is_missing")
        _validate_open_plain_file_identity(self._lock_handle, self._run_lock_path)
        return active

    def _path(self, locator: str) -> Path:
        return _target_for(self.output_root, locator)

    def _publish_json(
        self,
        locator: str,
        payload: object,
        *,
        max_bytes: int = _MAX_CONTROL_ENTRY_BYTES,
        strict_create_new: bool = False,
    ) -> bool:
        self._require_owner()
        body = canonical_lf_bytes(payload)
        if len(body) > max_bytes:
            raise PilotLocalStoreBoundsError("pilot_local_publication_exceeds_bound")
        return _write_temp_and_link(
            self._path(locator),
            body,
            strict_create_new=strict_create_new,
        )

    def _reload_json(self, locator: str, *, max_bytes: int) -> object:
        self._require_owner()
        raw = _read_exact_plain_file(self._path(locator), max_bytes=max_bytes)
        return parse_canonical_lf_json(raw)

    def publish_manifest(self) -> PilotRunStateV1:
        self._publish_json(
            "run-control/manifest.json",
            self.manifest.as_dict(),
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        reloaded = parse_pilot_run_manifest_v1(
            self._reload_json(
                "run-control/manifest.json",
                max_bytes=_MAX_MANIFEST_BYTES,
            )
        )
        if reloaded != self.manifest:
            raise PilotLocalStoreConflictError("pilot_local_manifest_reload_conflict")
        projected = PilotRunStateV1(self.manifest)
        self._require_expected_state(projected)
        return projected

    def publish_authorization(
        self, receipt: U5PublicPilotAuthorizationReceiptV1
    ) -> PilotRunStateV1:
        durable = self._require_expected_state(PilotRunStateV1(self.manifest))
        state = durable.with_authorization(
            receipt,
            now_us=receipt.authorized_at_us,
        )
        self._publish_json("run-control/authorization.json", receipt.as_dict())
        reloaded = U5PublicPilotAuthorizationReceiptV1.from_dict(
            self._reload_json(
                "run-control/authorization.json",
                max_bytes=_MAX_CONTROL_ENTRY_BYTES,
            )
        )
        if reloaded != receipt:
            raise PilotLocalStoreConflictError("pilot_local_authorization_reload_conflict")
        self._require_expected_state(state)
        return state

    def _probe_volume_identity(self) -> str:
        _ensure_plain_directory(self.run_control_root)
        if os.name != "nt":
            raise PilotLocalStoreError(
                "pilot_local_preflight_requires_windows_fixed_ntfs"
            )
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        volume_path = ctypes.create_unicode_buffer(32768)
        if not kernel32.GetVolumePathNameW(
            wintypes.LPCWSTR(os.fspath(self.output_root)),
            volume_path,
            len(volume_path),
        ):
            raise PilotLocalStoreError("pilot_local_volume_path_probe_failed")
        drive_type = kernel32.GetDriveTypeW(wintypes.LPCWSTR(volume_path.value))
        if drive_type != 3:  # DRIVE_FIXED
            raise PilotLocalStoreError("pilot_local_volume_is_not_fixed")
        serial = wintypes.DWORD()
        maximum_component = wintypes.DWORD()
        flags = wintypes.DWORD()
        filesystem = ctypes.create_unicode_buffer(64)
        if not kernel32.GetVolumeInformationW(
            wintypes.LPCWSTR(volume_path.value),
            None,
            0,
            ctypes.byref(serial),
            ctypes.byref(maximum_component),
            ctypes.byref(flags),
            filesystem,
            len(filesystem),
        ):
            raise PilotLocalStoreError("pilot_local_volume_information_probe_failed")
        if filesystem.value.upper() != "NTFS":
            raise PilotLocalStoreError("pilot_local_volume_is_not_ntfs")
        device = self.output_root.stat().st_dev
        return f"ntfs_{serial.value:08x}_{device:x}"[:128]

    def _probe_hardlink_create_new(self) -> None:
        _ensure_plain_directory(self.run_control_root)
        source = self.run_control_root / f".preflight-{uuid.uuid4().hex}.source"
        target = source.with_suffix(".link")
        cleanup_error: OSError | None = None
        try:
            with source.open("xb") as handle:
                handle.write(b"pilot-local-hardlink-probe-v1\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(source, target)
            source_stat = source.lstat()
            target_stat = target.lstat()
            if (
                not _plain_mode(source_stat, directory=False)
                or not _plain_mode(target_stat, directory=False)
                or source_stat.st_dev != target_stat.st_dev
                or source_stat.st_ino != target_stat.st_ino
                or source_stat.st_nlink < 2
                or target_stat.st_nlink < 2
            ):
                raise PilotLocalStoreError("pilot_local_hardlink_probe_binding_failed")
        except PilotLocalStoreError:
            raise
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_hardlink_probe_failed") from exc
        finally:
            for path in (target, source):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise PilotLocalStoreError(
                "pilot_local_hardlink_probe_cleanup_failed"
            ) from cleanup_error

    def _require_fresh_roots(self, roots: tuple[str, ...]) -> None:
        _validate_plain_existing_chain(self.output_root)
        for locator in roots:
            target = _target_for(self.output_root, locator)
            current = self.output_root
            for part in PurePosixPath(locator).parts:
                current = current / part
                try:
                    observed = current.lstat()
                except FileNotFoundError:
                    break
                except OSError as exc:
                    raise PilotLocalStoreError(
                        "pilot_local_preflight_root_chain_probe_failed"
                    ) from exc
                if current == target:
                    raise PilotLocalStoreError(
                        "pilot_local_preflight_fresh_root_preexists"
                    )
                if not _plain_mode(observed, directory=True):
                    raise PilotLocalStoreError(
                        "pilot_local_preflight_root_chain_has_reparse"
                    )

    def _scan_tree_identity_snapshot(
        self,
        *,
        root: Path,
        max_entries: int,
        max_total_bytes: int,
        deadline_ns: int,
        deadline_code: str,
    ) -> dict[
        str,
        tuple[str, tuple[int, int, int, int, int, int], str | None],
    ]:
        """Bound one structure/signature/digest pass without charging inventory."""

        if (
            type(max_entries) is not int
            or max_entries < 1
            or max_entries > self.manifest.budgets.max_inventory_entries
        ):
            raise PilotLocalStoreBoundsError(
                "pilot_local_tree_verification_entry_bound_is_invalid"
            )
        if type(max_total_bytes) is not int or max_total_bytes < 1:
            raise PilotLocalStoreBoundsError(
                "pilot_local_tree_verification_byte_bound_is_invalid"
            )
        try:
            root_observed = root.lstat()
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_stream_root_probe_failed") from exc
        if not _plain_mode(root_observed, directory=True):
            raise PilotLocalStoreError("pilot_local_stream_root_is_not_plain")
        _validate_plain_existing_chain(root)
        _check_scan_deadline(deadline_ns, code=deadline_code)
        _reject_windows_named_streams(root, deadline_ns=deadline_ns)
        structure = {
            ".": ("directory", _stable_signature(root_observed), None),
        }
        visited = 0
        total_bytes = 0
        stack = [(root, 0)]
        while stack:
            directory, depth = stack.pop()
            _check_scan_deadline(deadline_ns, code=deadline_code)
            try:
                iterator = os.scandir(directory)
            except OSError as exc:
                raise PilotLocalStoreError(
                    "pilot_local_stream_tree_scan_failed"
                ) from exc
            with iterator:
                for child in iterator:
                    _check_scan_deadline(deadline_ns, code=deadline_code)
                    visited += 1
                    if visited > max_entries:
                        raise PilotLocalStoreBoundsError(
                            "pilot_local_stream_visited_entry_budget_exceeded"
                        )
                    path = Path(child.path)
                    try:
                        observed = path.lstat()
                    except OSError as exc:
                        raise PilotLocalStoreError(
                            "pilot_local_stream_tree_entry_probe_failed"
                        ) from exc
                    if child.is_symlink() or _is_reparse(observed):
                        raise PilotLocalStoreError(
                            "pilot_local_stream_tree_reparse_is_forbidden"
                        )
                    relative = path.relative_to(root).as_posix()
                    if stat.S_ISDIR(observed.st_mode):
                        _reject_windows_named_streams(
                            path, deadline_ns=deadline_ns
                        )
                        structure[relative] = (
                            "directory",
                            _stable_signature(observed),
                            None,
                        )
                        if depth + 1 > _MAX_TREE_DEPTH:
                            raise PilotLocalStoreBoundsError(
                                "pilot_local_stream_depth_budget_exceeded"
                            )
                        stack.append((path, depth + 1))
                    elif stat.S_ISREG(observed.st_mode):
                        if observed.st_nlink != 1:
                            raise PilotLocalStoreError(
                                "pilot_local_stream_tree_hardlink_alias_is_forbidden"
                            )
                        remaining_bytes = max_total_bytes - total_bytes
                        if observed.st_size > remaining_bytes:
                            raise PilotLocalStoreBoundsError(
                                "pilot_local_tree_verification_byte_budget_exceeded"
                            )
                        digest, byte_count, final_signature = (
                            _hash_exact_plain_file(
                                path,
                                max_bytes=max(1, remaining_bytes),
                                deadline_ns=deadline_ns,
                                deadline_code=deadline_code,
                            )
                        )
                        total_bytes += byte_count
                        structure[relative] = (
                            "file",
                            final_signature,
                            digest,
                        )
                    else:
                        raise PilotLocalStoreError(
                            "pilot_local_stream_tree_special_file_is_forbidden"
                        )
        _check_scan_deadline(deadline_ns, code=deadline_code)
        return structure

    def _require_no_named_streams_tree(self) -> None:
        try:
            self.output_root.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise PilotLocalStoreError("pilot_local_stream_root_probe_failed") from exc
        deadline_ns = time.monotonic_ns() + _MAX_SCAN_RUNTIME_US * 1_000
        first = self._scan_tree_identity_snapshot(
            root=self.output_root,
            max_entries=self.manifest.budgets.max_inventory_entries,
            max_total_bytes=self.manifest.budgets.max_total_output_bytes,
            deadline_ns=deadline_ns,
            deadline_code="pilot_local_stream_scan_runtime_exceeded",
        )
        second = self._scan_tree_identity_snapshot(
            root=self.output_root,
            max_entries=self.manifest.budgets.max_inventory_entries,
            max_total_bytes=self.manifest.budgets.max_total_output_bytes,
            deadline_ns=deadline_ns,
            deadline_code="pilot_local_stream_scan_runtime_exceeded",
        )
        if second != first:
            raise PilotLocalStoreError(
                "pilot_local_stream_tree_changed_during_verification"
            )

    def _measure_preflight_receipt(
        self,
        state: PilotRunStateV1,
        *,
        step: int,
    ) -> PilotDiskPreflightReceiptV1:
        if state.authorization is None:
            raise PilotLocalStoreError("pilot_local_preflight_lacks_durable_authorization")
        sample = self._sample_clock()
        state.authorization.validate_for(self.manifest, now_us=sample.epoch_us)
        volume_identity = self._probe_volume_identity()
        roots = self.manifest.remaining_fresh_roots(step)
        self._require_fresh_roots(roots)
        self._probe_hardlink_create_new()
        self._require_no_named_streams_tree()
        free_before = shutil.disk_usage(self.output_root).free
        reserved = self.manifest.remaining_storage_reservation(step)
        if (
            free_before < reserved
            or free_before < self.manifest.budgets.min_free_disk_bytes_before_run
            and step == -1
            or free_before - reserved
            < self.manifest.budgets.required_free_disk_bytes_after_reservation
        ):
            raise PilotLocalStoreBoundsError(
                "pilot_local_preflight_insufficient_free_disk"
            )
        request = self.manifest.endpoint_verification.probe_request
        return PilotDiskPreflightReceiptV1(
            manifest_hash=self.manifest.manifest_hash,
            authorization_receipt_hash=state.authorization.receipt_hash,
            step_ordinal=step,
            checked_at_us=sample.epoch_us,
            valid_until_us=(
                sample.epoch_us + self.manifest.budgets.max_preflight_age_us
            ),
            output_root_locator=self.manifest.output_root_locator,
            volume_identity=volume_identity,
            storage_profile=request.storage_profile,
            storage_profile_hash=request.storage_profile_hash,
            free_bytes_before=free_before,
            reserved_bytes=reserved,
            free_bytes_after_reservation=free_before - reserved,
            fresh_relative_roots=roots,
            path_chain_reparse_free=True,
            local_fixed_volume=True,
            same_volume_publication=True,
            hardlink_create_new_supported=True,
        )

    def measure_and_publish_preflight(
        self,
        expected_state: PilotRunStateV1,
    ) -> PilotRunStateV1:
        state = self._require_expected_state(expected_state)
        action = state.next_action
        if not action.startswith("run_local_preflight:"):
            raise PilotLocalStoreConflictError("pilot_local_preflight_is_out_of_order")
        step = int(action.rsplit(":", 1)[1])
        receipt = self._measure_preflight_receipt(state, step=step)
        candidate = state.with_preflight(receipt, now_us=receipt.checked_at_us)
        step_name = "endpoint" if step == -1 else f"shard-{step:04d}"
        locator = f"run-control/preflights/{step_name}.json"
        self._publish_json(locator, receipt.as_dict())
        reloaded = PilotDiskPreflightReceiptV1.from_dict(
            self._reload_json(locator, max_bytes=_MAX_CONTROL_ENTRY_BYTES)
        )
        if reloaded != receipt:
            raise PilotLocalStoreConflictError("pilot_local_preflight_reload_conflict")
        self._require_expected_state(candidate)
        return candidate

    def _recheck_active_preflight(self, state: PilotRunStateV1) -> PilotClockSampleV1:
        self._require_owner()
        if state.authorization is None or not state.preflight_receipts:
            raise PilotLocalStoreError("pilot_local_active_preflight_is_missing")
        receipt = state.preflight_receipts[-1]
        sample = self._sample_clock()
        state.authorization.validate_for(self.manifest, now_us=sample.epoch_us)
        receipt.validate_for(
            self.manifest,
            state.authorization,
            now_us=sample.epoch_us,
        )
        if self._probe_volume_identity() != receipt.volume_identity:
            raise PilotLocalStoreError("pilot_local_preflight_volume_changed")
        self._require_fresh_roots(receipt.fresh_relative_roots)
        self._probe_hardlink_create_new()
        self._require_no_named_streams_tree()
        free_now = shutil.disk_usage(self.output_root).free
        if (
            free_now < receipt.reserved_bytes
            or free_now - receipt.reserved_bytes
            < self.manifest.budgets.required_free_disk_bytes_after_reservation
        ):
            raise PilotLocalStoreBoundsError(
                "pilot_local_preflight_recheck_insufficient_free_disk"
            )
        self._require_owner()
        final_sample = self._sample_clock()
        self._require_owner()
        if (
            final_sample.clock_domain_id != sample.clock_domain_id
            or final_sample.monotonic_us < sample.monotonic_us
            or final_sample.epoch_us < sample.epoch_us
        ):
            raise PilotLocalStoreError("pilot_local_preflight_probe_clock_order_mismatch")
        # The final time sample is deliberately after every potentially slow IO
        # probe.  Nothing that can expire is checked only against the pre-probe
        # timestamp.
        state.authorization.validate_for(
            self.manifest,
            now_us=final_sample.epoch_us,
        )
        receipt.validate_for(
            self.manifest,
            state.authorization,
            now_us=final_sample.epoch_us,
        )
        return final_sample

    def _session_claim_locator(
        self, authorization: U5PublicPilotAuthorizationReceiptV1
    ) -> str:
        identity = _sha256(
            _canonical_json_bytes(
                {
                    "domain": "mexc_public_qa_pilot_process_session_slot_v1",
                    "manifest_hash": self.manifest.manifest_hash,
                    "authorization_receipt_hash": authorization.receipt_hash,
                    "orchestrator_session_id": authorization.orchestrator_session_id,
                }
            )
        )
        return f"session-claims/{identity}.claim.json"

    def _anchor_detached(
        self,
        *,
        kind: str,
        subject_hash: str,
        artifact_sha256: str,
        locator: str,
        observed: PilotClockSampleV1,
    ) -> PilotDetachedEvidenceReceiptV1:
        self._assert_declared_contract_bindings()
        request = PilotDetachedEvidenceRequestV1(
            evidence_kind=kind,
            manifest_hash=self.manifest.manifest_hash,
            subject_hash=subject_hash,
            artifact_sha256=artifact_sha256,
            relative_locator=locator,
            publisher_instance_id=self.publisher_instance_id,
            observed_at_us=observed.epoch_us,
            observed_monotonic_us=observed.monotonic_us,
            clock_domain_id=observed.clock_domain_id,
        )
        receipt = self.detached_evidence_sink.anchor(request)
        self._assert_declared_contract_bindings()
        if not isinstance(receipt, PilotDetachedEvidenceReceiptV1):
            raise PilotLocalStoreError("pilot_detached_sink_returned_invalid_receipt")
        receipt.validate_for(request)
        return receipt

    def _require_live_session_claim(
        self,
        capability: PilotRunSessionCapability,
        *,
        expected_locator: str,
    ) -> None:
        if (
            capability.session_claim_locator != expected_locator
            or not _is_sha256(capability.session_claim_artifact_sha256)
        ):
            raise PilotLocalStoreLockError(
                "pilot_local_live_session_claim_binding_mismatch"
            )
        target = _target_for(self.external_state_root, expected_locator)
        raw = _read_exact_plain_file(
            target,
            max_bytes=_MAX_CONTROL_ENTRY_BYTES,
        )
        if _sha256(raw) != capability.session_claim_artifact_sha256:
            raise PilotLocalStoreLockError(
                "pilot_local_live_session_claim_artifact_changed"
            )

    def claim_process_session(
        self,
        expected_state: PilotRunStateV1,
    ) -> PilotRunSessionCapability:
        lease_nonce = self._require_owner()
        state = self._require_expected_state(expected_state)
        authorization = state.authorization
        if authorization is None:
            raise PilotLocalStoreError(
                "pilot_local_session_requires_durable_authorization"
            )
        if self._session_capability is not None:
            existing = self._session_capability
            if existing.authorization_receipt_hash != authorization.receipt_hash:
                raise PilotLocalStoreConflictError("pilot_local_live_session_claim_conflict")
            self._require_live_session_claim(
                existing,
                expected_locator=self._session_claim_locator(authorization),
            )
            return existing
        verifier = self.u5_authority_verifier
        if verifier is None:
            raise PilotLocalStoreError("pilot_local_verified_u5_capability_is_required")
        observed = self._sample_clock()
        authorization.validate_for(self.manifest, now_us=observed.epoch_us)
        challenge_hash = _sha256(
            os.urandom(32)
            + self.manifest.manifest_hash.encode("ascii")
            + authorization.receipt_hash.encode("ascii")
            + str(os.getpid()).encode("ascii")
            + str(threading.get_ident()).encode("ascii")
        )
        verification_request = PilotU5VerificationRequestV1(
            manifest_hash=self.manifest.manifest_hash,
            authorization_receipt_hash=authorization.receipt_hash,
            external_authority_evidence_hash=(
                authorization.external_authority_evidence_hash
            ),
            executor_bindings_hash=self.executor_bindings.bindings_hash,
            runtime_authority_binding_hash=(
                self.runtime_authority_binding.binding_hash
            ),
            orchestrator_session_id=authorization.orchestrator_session_id,
            process_challenge_hash=challenge_hash,
            publisher_instance_id=self.publisher_instance_id,
            process_id=os.getpid(),
            requested_at_us=observed.epoch_us,
            requested_monotonic_us=observed.monotonic_us,
            clock_contract_version=self.runtime_authority_binding.clock_contract_version,
            clock_contract_hash=self.runtime_authority_binding.clock_contract_hash,
            clock_domain_id=observed.clock_domain_id,
            u5_verifier_contract_version=(
                self.runtime_authority_binding.u5_verifier_contract_version
            ),
            u5_verifier_contract_hash=(
                self.runtime_authority_binding.u5_verifier_contract_hash
            ),
            u5_verifier_domain_id=(
                self.runtime_authority_binding.u5_verifier_domain_id
            ),
            u5_verifier_trust_key_id=(
                self.runtime_authority_binding.u5_verifier_trust_key_id
            ),
            u5_verifier_policy_version=(
                self.runtime_authority_binding.u5_verifier_policy_version
            ),
            u5_verifier_policy_hash=(
                self.runtime_authority_binding.u5_verifier_policy_hash
            ),
        )
        verification = verifier.verify(verification_request)
        self._assert_declared_contract_bindings()
        if not isinstance(verification, PilotU5VerificationEvidenceV1):
            raise PilotLocalStoreError("pilot_local_u5_verifier_returned_invalid_evidence")
        verification.validate_for(verification_request)
        claimed = self._sample_clock()
        if (
            claimed.clock_domain_id != verification.clock_domain_id
            or claimed.epoch_us < verification.verified_at_us
            or claimed.monotonic_us < verification.verified_monotonic_us
        ):
            raise PilotLocalStoreError("pilot_local_u5_verification_clock_order_mismatch")
        authorization.validate_for(self.manifest, now_us=claimed.epoch_us)
        payload = {
            "domain": "mexc_public_qa_pilot_process_session_claim_v1",
            "manifest_hash": self.manifest.manifest_hash,
            "authorization_receipt_hash": authorization.receipt_hash,
            "orchestrator_session_id": authorization.orchestrator_session_id,
            "publisher_instance_id": self.publisher_instance_id,
            "process_id": os.getpid(),
            "claimed_at_us": claimed.epoch_us,
            "claimed_monotonic_us": claimed.monotonic_us,
            "clock_domain_id": claimed.clock_domain_id,
            "process_challenge_hash": challenge_hash,
            "u5_verification_request_hash": verification_request.request_hash,
            "u5_verification_evidence_hash": verification.verification_evidence_hash,
            "executor_bindings_hash": self.executor_bindings.bindings_hash,
            "runtime_authority_binding_hash": (
                self.runtime_authority_binding.binding_hash
            ),
            "clock_identity": list(self._clock_identity()),
            "u5_verifier_identity": list(self._u5_verifier_identity()),
            "output_root_identity": list(self._current_output_root_identity()),
            "restart_policy": "preexisting_claim_forbids_network",
        }
        body = canonical_lf_bytes(payload)
        locator = self._session_claim_locator(authorization)
        target = _target_for(self.external_state_root, locator)
        _write_temp_and_link(target, body, strict_create_new=True)
        reloaded = _read_exact_plain_file(target, max_bytes=_MAX_CONTROL_ENTRY_BYTES)
        if reloaded != body:
            raise PilotLocalStoreError("pilot_local_session_claim_reload_mismatch")
        reloaded_at = self._sample_clock()
        if reloaded_at.clock_domain_id != claimed.clock_domain_id:
            raise PilotLocalStoreError("pilot_local_clock_domain_changed")
        subject_hash = _sha256(_canonical_json_bytes(payload))
        evidence = self._anchor_detached(
            kind="session_claim_reload",
            subject_hash=subject_hash,
            artifact_sha256=_sha256(body),
            locator=locator,
            observed=reloaded_at,
        )
        finalized = self._sample_clock()
        if (
            finalized.epoch_us < evidence.anchored_at_us
            or finalized.monotonic_us < evidence.anchored_monotonic_us
            or finalized.clock_domain_id != evidence.clock_domain_id
        ):
            raise PilotLocalStoreError(
                "pilot_local_session_anchor_clock_order_mismatch"
            )
        authorization.validate_for(self.manifest, now_us=finalized.epoch_us)
        capability = PilotRunSessionCapability(
            store_nonce=self._store_nonce,
            lease_nonce=lease_nonce,
            binding_guard=self._binding_guard,
            owner_thread=threading.get_ident(),
            manifest_hash=self.manifest.manifest_hash,
            authorization_receipt_hash=authorization.receipt_hash,
            orchestrator_session_id=authorization.orchestrator_session_id,
            claim_evidence_hash=evidence.evidence_hash,
            authority_verification_evidence_hash=(
                verification.verification_evidence_hash
            ),
            process_challenge_hash=challenge_hash,
            executor_bindings_hash=self.executor_bindings.bindings_hash,
            runtime_authority_binding_hash=(
                self.runtime_authority_binding.binding_hash
            ),
            clock_identity=self._clock_identity(),
            u5_verifier_identity=self._u5_verifier_identity(),
            output_root_identity=self._current_output_root_identity(),
            session_claim_locator=locator,
            session_claim_artifact_sha256=_sha256(body),
        )
        self._session_capability = capability
        return capability

    def _validate_session_capability(
        self,
        capability: PilotRunSessionCapability,
        state: PilotRunStateV1,
    ) -> None:
        lease_nonce = self._require_owner()
        if not isinstance(capability, PilotRunSessionCapability):
            raise PilotLocalStoreLockError("pilot_local_session_capability_is_invalid")
        if (
            capability is not self._session_capability
            or capability._store_nonce is not self._store_nonce
            or capability._lease_nonce is not lease_nonce
            or capability._binding_guard is not self._binding_guard
            or capability._owner_thread != threading.get_ident()
            or capability._process_id != os.getpid()
            or capability.manifest_hash != self.manifest.manifest_hash
            or state.authorization is None
            or capability.authorization_receipt_hash
            != state.authorization.receipt_hash
            or capability.orchestrator_session_id
            != state.authorization.orchestrator_session_id
            or capability.executor_bindings_hash
            != self.executor_bindings.bindings_hash
            or capability.runtime_authority_binding_hash
            != self.runtime_authority_binding.binding_hash
            or capability.clock_identity != self._clock_identity()
            or capability.u5_verifier_identity != self._u5_verifier_identity()
            or capability.output_root_identity
            != self._current_output_root_identity()
        ):
            raise PilotLocalStoreLockError("pilot_local_session_capability_mismatch")
        assert state.authorization is not None
        self._require_live_session_claim(
            capability,
            expected_locator=self._session_claim_locator(state.authorization),
        )

    def _intent_fields(
        self,
        state: PilotRunStateV1,
        issued: PilotClockSampleV1,
    ) -> dict[str, object]:
        if state.authorization is None or not state.preflight_receipts:
            raise PilotLocalStoreError("pilot_local_intent_prerequisites_are_missing")
        action = state.next_action
        if action == "run_endpoint_verification_stage":
            stage = "endpoint_verification"
            ordinal = -1
            plan = self.manifest.endpoint_verification
            binding = plan.plan_hash
            root = plan.relative_artifact_root
            attempts = plan.max_network_attempts
            raw_bytes = plan.max_total_raw_body_bytes
            storage_bytes = plan.max_total_storage_bytes
            runtime_us = plan.max_runtime_us
        elif action.startswith("collect_shard:"):
            ordinal = int(action.rsplit(":", 1)[1])
            stage = "shard_acquisition"
            plan = self.manifest.shards[ordinal]
            request = plan.request
            binding = plan.plan_id
            root = plan.relative_artifact_root
            attempts = request.required_pages * request.resource_limits.max_attempts_per_page
            raw_bytes = min(
                request.resource_limits.max_total_raw_body_bytes,
                attempts * request.resource_limits.max_raw_body_bytes_per_attempt,
            )
            storage_bytes = request.resource_limits.max_logical_storage_bytes
            runtime_us = request.resource_limits.max_collection_runtime_us
        else:
            raise PilotLocalStoreError("pilot_local_state_does_not_request_network_intent")
        return {
            "manifest_hash": self.manifest.manifest_hash,
            "authorization_receipt_hash": state.authorization.receipt_hash,
            "preflight_receipt_hash": state.preflight_receipts[-1].receipt_hash,
            "stage": stage,
            "ordinal": ordinal,
            "step_binding_hash": binding,
            "relative_artifact_root": root,
            "clock_domain_id": issued.clock_domain_id,
            "orchestrator_session_id": state.authorization.orchestrator_session_id,
            "publisher_instance_id": self.publisher_instance_id,
            "issued_at_us": issued.epoch_us,
            "issued_monotonic_us": issued.monotonic_us,
            "reserved_network_attempts": attempts,
            "reserved_raw_body_bytes": raw_bytes,
            "reserved_storage_bytes": storage_bytes,
            "reserved_runtime_us": runtime_us,
        }

    def claim_and_seal_next_intent(
        self,
        state: PilotRunStateV1,
        session_capability: PilotRunSessionCapability,
    ) -> PilotIntentClaimResultV1:
        state = self._require_expected_state(state)
        self._validate_session_capability(session_capability, state)
        if self._active_intent_capability is not None:
            raise PilotLocalStoreConflictError("pilot_local_live_intent_already_exists")
        self._recheck_active_preflight(state)
        issued = self._sample_clock()
        fields = self._intent_fields(state, issued)
        candidate_payload = PilotNetworkIntentV1.candidate_payload_for(**fields)
        candidate_hash = PilotNetworkIntentV1.candidate_hash_for(**fields)
        candidate_body = canonical_lf_bytes(candidate_payload)
        candidate_artifact_sha256 = _sha256(candidate_body)
        stage = str(fields["stage"])
        ordinal = int(fields["ordinal"])
        slot_id = PilotNetworkIntentV1.slot_id_for(
            manifest_hash=self.manifest.manifest_hash,
            stage=stage,
            ordinal=ordinal,
        )
        candidate_locator = PilotNetworkIntentV1.slot_locator_for(
            manifest_hash=self.manifest.manifest_hash,
            stage=stage,
            ordinal=ordinal,
        )
        self._publish_json(
            candidate_locator,
            candidate_payload,
            strict_create_new=True,
        )
        published = self._sample_clock()
        publication_evidence = self._anchor_detached(
            kind="intent_candidate_publication",
            subject_hash=candidate_hash,
            artifact_sha256=candidate_artifact_sha256,
            locator=candidate_locator,
            observed=published,
        )
        reloaded_payload = self._reload_json(
            candidate_locator,
            max_bytes=_MAX_CONTROL_ENTRY_BYTES,
        )
        if canonical_lf_bytes(reloaded_payload) != candidate_body:
            raise PilotLocalStoreError("pilot_local_intent_candidate_reload_mismatch")
        reloaded = self._sample_clock()
        reload_evidence = self._anchor_detached(
            kind="intent_candidate_reload",
            subject_hash=candidate_hash,
            artifact_sha256=candidate_artifact_sha256,
            locator=candidate_locator,
            observed=reloaded,
        )
        reservation_observed = self._sample_clock()
        reservation_evidence = self._anchor_detached(
            kind="intent_reservation_anchor",
            subject_hash=candidate_hash,
            artifact_sha256=candidate_artifact_sha256,
            locator=candidate_locator,
            observed=reservation_observed,
        )
        samples = (issued, published, reloaded, reservation_observed)
        if len({item.clock_domain_id for item in samples}) != 1:
            raise PilotLocalStoreError("pilot_local_clock_domain_changed")
        durability = PilotIntentDurabilityReceiptV1(
            intent_candidate_hash=candidate_hash,
            intent_slot_id=slot_id,
            intent_candidate_locator=candidate_locator,
            intent_candidate_artifact_sha256=candidate_artifact_sha256,
            publisher_instance_id=self.publisher_instance_id,
            durable_publication_receipt_hash=publication_evidence.evidence_hash,
            fresh_reload_receipt_hash=reload_evidence.evidence_hash,
            detached_reservation_anchor_hash=reservation_evidence.evidence_hash,
            published_at_us=published.epoch_us,
            reloaded_at_us=reloaded.epoch_us,
            anchored_at_us=reservation_evidence.anchored_at_us,
            published_monotonic_us=published.monotonic_us,
            reloaded_monotonic_us=reloaded.monotonic_us,
            anchored_monotonic_us=reservation_evidence.anchored_monotonic_us,
        )
        intent = PilotNetworkIntentV1(**fields, durability_receipt=durability)
        projected = state.with_network_intent(intent)
        sealed_locator = intent.sealed_intent_locator
        self._publish_json(
            sealed_locator,
            intent.as_dict(),
            strict_create_new=True,
        )
        sealed_reload = PilotNetworkIntentV1.from_dict(
            self._reload_json(sealed_locator, max_bytes=_MAX_CONTROL_ENTRY_BYTES)
        )
        if sealed_reload != intent:
            raise PilotLocalStoreError("pilot_local_sealed_intent_reload_mismatch")
        projected_reload = state.with_network_intent(sealed_reload)
        if projected_reload != projected:
            raise PilotLocalStoreError("pilot_local_sealed_intent_projection_mismatch")
        capability = PilotIntentOwnerCapability(
            store_nonce=self._store_nonce,
            lease_nonce=self._active_lease_nonce,
            session_nonce=session_capability._session_nonce,
            owner_thread=threading.get_ident(),
            manifest_hash=self.manifest.manifest_hash,
            intent_hash=intent.intent_hash,
            intent_slot_id=intent.intent_slot_id,
            stage=intent.stage,
            ordinal=intent.ordinal,
            binding_guard=self._binding_guard,
            executor_bindings_hash=self.executor_bindings.bindings_hash,
            runtime_authority_binding_hash=(
                self.runtime_authority_binding.binding_hash
            ),
            clock_identity=self._clock_identity(),
            u5_verifier_identity=self._u5_verifier_identity(),
            output_root_identity=self._current_output_root_identity(),
        )
        self._require_expected_state(
            projected_reload,
            allow_active_intent=True,
        )
        self._active_intent_capability = capability
        return PilotIntentClaimResultV1(projected_reload, sealed_reload, capability)

    def _validate_intent_owner_capability(
        self,
        capability: PilotIntentOwnerCapability,
        intent: PilotNetworkIntentV1,
        *,
        recheck_dependencies: bool = True,
        reload_session_claim: bool = True,
    ) -> None:
        lease_nonce = self._require_owner(
            recheck_dependencies=recheck_dependencies,
        )
        session = self._session_capability
        if (
            not isinstance(capability, PilotIntentOwnerCapability)
            or capability is not self._active_intent_capability
            or session is None
            or session._store_nonce is not self._store_nonce
            or session._lease_nonce is not lease_nonce
            or session._binding_guard is not self._binding_guard
            or session._owner_thread != threading.get_ident()
            or session._process_id != os.getpid()
            or session.manifest_hash != self.manifest.manifest_hash
            or session.authorization_receipt_hash
            != intent.authorization_receipt_hash
            or session.orchestrator_session_id != intent.orchestrator_session_id
            or session.executor_bindings_hash
            != self.executor_bindings.bindings_hash
            or session.runtime_authority_binding_hash
            != self.runtime_authority_binding.binding_hash
            or session.clock_identity != self._clock_identity()
            or session.u5_verifier_identity != self._u5_verifier_identity()
            or session.output_root_identity
            != self._current_output_root_identity(
                recheck_dependencies=recheck_dependencies,
            )
            or capability._store_nonce is not self._store_nonce
            or capability._lease_nonce is not lease_nonce
            or capability._binding_guard is not self._binding_guard
            or capability._session_nonce is not session._session_nonce
            or capability._owner_thread != threading.get_ident()
            or capability._process_id != os.getpid()
            or capability.manifest_hash != self.manifest.manifest_hash
            or capability.intent_hash != intent.intent_hash
            or capability.intent_slot_id != intent.intent_slot_id
            or capability.stage != intent.stage
            or capability.ordinal != intent.ordinal
            or capability.executor_bindings_hash
            != self.executor_bindings.bindings_hash
            or capability.runtime_authority_binding_hash
            != self.runtime_authority_binding.binding_hash
            or capability.clock_identity != self._clock_identity()
            or capability.u5_verifier_identity != self._u5_verifier_identity()
            or capability.output_root_identity
            != self._current_output_root_identity(
                recheck_dependencies=recheck_dependencies,
            )
            or capability._terminal
            or intent.intent_hash in self._consumed_intent_hashes
        ):
            raise PilotLocalStoreLockError("pilot_local_intent_owner_capability_mismatch")
        if reload_session_claim:
            self._require_live_session_claim(
                session,
                expected_locator=session.session_claim_locator,
            )

    def run_owned_intent_once(
        self,
        expected_state: PilotRunStateV1,
        capability: PilotIntentOwnerCapability,
        runner: PilotOwnedIntentRunner,
    ) -> _RunnerResult:
        """Consume the live grant before crossing an injected runner boundary."""

        if not callable(runner):
            raise PilotLocalStoreError("pilot_local_owned_intent_runner_is_invalid")
        # Resolve every caller-controlled property before authoritative replay,
        # capability validation and the final physical/U5 gate.  The resulting
        # pair is plain immutable data; the runner is not consulted again until
        # after permission has been irreversibly consumed.
        runner_identity = (
            getattr(runner, "contract_version", None),
            getattr(runner, "contract_hash", None),
        )
        state = self._require_expected_state(
            expected_state,
            allow_active_intent=True,
        )
        if not state.network_intents:
            raise PilotLocalStoreConflictError("pilot_local_active_intent_is_missing")
        intent = state.network_intents[-1]
        if intent.stage == "endpoint_verification":
            expected_runner = (
                self.executor_bindings.endpoint_runner_contract_version,
                self.executor_bindings.endpoint_runner_contract_hash,
            )
        else:
            expected_runner = (
                self.executor_bindings.shard_runner_contract_version,
                self.executor_bindings.shard_runner_contract_hash,
            )
        if runner_identity != expected_runner:
            raise PilotLocalStoreError("pilot_local_owned_intent_runner_binding_mismatch")
        self._recheck_active_preflight(state)
        # Finish every potentially slow disk operation before the final clock
        # sample: exact replay, session-claim reload and capability validation.
        state = self._require_expected_state(
            expected_state,
            allow_active_intent=True,
            recheck_dependencies=False,
        )
        if not state.network_intents:
            raise PilotLocalStoreConflictError("pilot_local_active_intent_is_missing")
        intent = state.network_intents[-1]
        self._validate_intent_owner_capability(
            capability,
            intent,
            recheck_dependencies=False,
        )
        assert state.authorization is not None
        final_sample = self._sample_clock()
        # The bound clock is a reviewed, side-effect-free coordinator dependency.
        # This last non-injected check catches physical root/lock replacement; no
        # full replay follows the sample, avoiding a stale-time replay window.
        self._validate_intent_owner_capability(
            capability,
            intent,
            recheck_dependencies=False,
            reload_session_claim=False,
        )
        state.authorization.validate_for(
            self.manifest,
            now_us=final_sample.epoch_us,
        )
        state.preflight_receipts[-1].validate_for(
            self.manifest,
            state.authorization,
            now_us=final_sample.epoch_us,
        )
        if intent.ordinal == -1:
            remaining = self.manifest.remaining_run_elapsed_reservation(
                -1,
                intent_anchor_us=final_sample.epoch_us,
            )
        else:
            previous_completed = (
                state.endpoint_verification.completed_at_us
                if intent.ordinal == 0
                else state.shard_results[intent.ordinal - 1].step_completed_at_us
            )
            remaining = self.manifest.remaining_run_elapsed_reservation(
                intent.ordinal,
                intent_anchor_us=final_sample.epoch_us,
                previous_completed_at_us=previous_completed,
            )
        if final_sample.epoch_us + remaining >= state.authorization.expires_at_us:
            raise PilotLocalStoreError(
                "pilot_local_u5_window_cannot_cover_remaining_run_at_callback"
            )
        # Detach first and retain an append-only store-side consume record before
        # crossing the callback.  Mutating fields on the caller-held object
        # cannot reattach it or erase the store's nonce record.
        object.__setattr__(
            self,
            "_consumed_intent_hashes",
            self._consumed_intent_hashes | frozenset({intent.intent_hash}),
        )
        self._active_intent_capability = None
        capability._consumed = True
        capability._terminal = True
        return runner(intent)

    def _allowed_control_locators(self) -> frozenset[str]:
        result = {
            "run-control/manifest.json",
            "run-control/authorization.json",
            "run-control/preflights/endpoint.json",
            "run-control/endpoint-verification.json",
            "run-control/terminal-failure-candidate.json",
            "run-control/terminal-failure.json",
            "run-control/result-candidate.json",
        }
        stages = (("endpoint_verification", -1),) + tuple(
            ("shard_acquisition", ordinal)
            for ordinal in range(len(self.manifest.shards))
        )
        for stage, ordinal in stages:
            candidate = PilotNetworkIntentV1.slot_locator_for(
                manifest_hash=self.manifest.manifest_hash,
                stage=stage,
                ordinal=ordinal,
            )
            result.add(candidate)
            result.add(candidate.removesuffix(".candidate.json") + ".sealed.json")
            if ordinal >= 0:
                result.add(f"run-control/preflights/shard-{ordinal:04d}.json")
                result.add(f"run-control/shard-results/{ordinal:04d}.json")
        return frozenset(result)

    def _scan_control_once(
        self,
        *,
        recheck_dependencies: bool = True,
    ) -> tuple[dict[str, bytes], tuple[str, ...]]:
        self._require_owner(recheck_dependencies=recheck_dependencies)
        if not self.run_control_root.exists():
            return {}, ()
        _validate_plain_existing_chain(self.run_control_root)
        allowed = self._allowed_control_locators()
        allowed_directories = frozenset(
            {
                "run-control/preflights",
                "run-control/network-intents",
                "run-control/shard-results",
            }
        )
        artifacts: dict[str, bytes] = {}
        residue: list[str] = []
        seen_file_ids: set[tuple[int, int]] = set()
        total_bytes = 0
        visited = 0
        scan_started_ns = time.monotonic_ns()
        deadline_ns = scan_started_ns + _MAX_SCAN_RUNTIME_US * 1_000
        try:
            root_observed = self.run_control_root.lstat()
        except OSError as exc:
            raise PilotLocalStoreRecoveryError(
                "pilot_control_root_probe_failed"
            ) from exc
        if not _plain_mode(root_observed, directory=True):
            raise PilotLocalStoreRecoveryError("pilot_control_root_is_not_plain")
        first_structure: dict[
            str,
            tuple[str, tuple[int, int, int, int, int, int], str | None],
        ] = {
            ".": ("directory", _stable_signature(root_observed), None),
        }
        _reject_windows_named_streams(
            self.run_control_root, deadline_ns=deadline_ns
        )
        stack = [(self.run_control_root, 0)]
        while stack:
            directory, depth = stack.pop()
            _check_scan_deadline(
                deadline_ns,
                code="pilot_control_scan_runtime_exceeded",
            )
            try:
                iterator = os.scandir(directory)
            except OSError as exc:
                raise PilotLocalStoreRecoveryError("pilot_control_scan_failed") from exc
            with iterator:
                for child in iterator:
                    _check_scan_deadline(
                        deadline_ns,
                        code="pilot_control_scan_runtime_exceeded",
                    )
                    visited += 1
                    if visited > self.manifest.budgets.max_inventory_entries:
                        raise PilotLocalStoreBoundsError(
                            "pilot_control_visited_entry_budget_exceeded"
                        )
                    path = Path(child.path)
                    observed = path.lstat()
                    relative = path.relative_to(self.output_root).as_posix()
                    if _is_reparse(observed) or child.is_symlink():
                        raise PilotLocalStoreRecoveryError(
                            "pilot_control_reparse_is_forbidden"
                        )
                    if stat.S_ISDIR(observed.st_mode):
                        _reject_windows_named_streams(path, deadline_ns=deadline_ns)
                        first_structure[
                            path.relative_to(self.run_control_root).as_posix()
                        ] = ("directory", _stable_signature(observed), None)
                        if relative not in allowed_directories:
                            residue.append(relative + "/")
                        if depth + 1 > _MAX_TREE_DEPTH:
                            raise PilotLocalStoreBoundsError(
                                "pilot_control_depth_budget_exceeded"
                            )
                        stack.append((path, depth + 1))
                        continue
                    if not stat.S_ISREG(observed.st_mode):
                        raise PilotLocalStoreRecoveryError(
                            "pilot_control_special_file_is_forbidden"
                        )
                    if observed.st_nlink != 1:
                        raise PilotLocalStoreRecoveryError(
                            "pilot_control_hardlink_alias_is_forbidden"
                        )
                    identity = _file_identity(observed)
                    if identity in seen_file_ids:
                        raise PilotLocalStoreRecoveryError(
                            "pilot_control_duplicate_file_identity_is_forbidden"
                        )
                    seen_file_ids.add(identity)
                    limit = (
                        _MAX_MANIFEST_BYTES
                        if relative == "run-control/manifest.json"
                        else _MAX_CONTROL_ENTRY_BYTES
                    )
                    raw = _read_exact_plain_file(
                        path,
                        max_bytes=limit,
                        deadline_ns=deadline_ns,
                        deadline_code="pilot_control_scan_runtime_exceeded",
                    )
                    first_structure[
                        path.relative_to(self.run_control_root).as_posix()
                    ] = ("file", _stable_signature(observed), _sha256(raw))
                    _check_scan_deadline(
                        deadline_ns,
                        code="pilot_control_scan_runtime_exceeded",
                    )
                    total_bytes += len(raw)
                    if total_bytes > self.manifest.budgets.max_run_control_bytes:
                        raise PilotLocalStoreBoundsError(
                            "pilot_control_byte_budget_exceeded"
                        )
                    artifacts[relative] = raw
                    if relative not in allowed:
                        residue.append(relative)
        canonical_residue = tuple(sorted(residue))
        verified_structure = self._scan_tree_identity_snapshot(
            root=self.run_control_root,
            max_entries=self.manifest.budgets.max_inventory_entries,
            max_total_bytes=self.manifest.budgets.max_run_control_bytes,
            deadline_ns=deadline_ns,
            deadline_code="pilot_control_scan_runtime_exceeded",
        )
        if verified_structure != first_structure:
            raise PilotLocalStoreRecoveryError(
                "pilot_control_tree_changed_during_scan"
            )
        _check_scan_deadline(
            deadline_ns,
            code="pilot_control_scan_runtime_exceeded",
        )
        return artifacts, canonical_residue

    @staticmethod
    def _parse_receipt(
        artifacts: Mapping[str, bytes],
        locator: str,
        receipt_type: Any,
    ) -> object | None:
        raw = artifacts.get(locator)
        if raw is None:
            return None
        payload = parse_canonical_lf_json(raw)
        receipt = receipt_type.from_dict(payload)
        if canonical_lf_bytes(receipt.as_dict()) != raw:
            raise PilotLocalStoreRecoveryError("pilot_control_receipt_round_trip_mismatch")
        return receipt

    def _replay_intent(
        self,
        artifacts: Mapping[str, bytes],
        state: PilotRunStateV1,
        *,
        stage: str,
        ordinal: int,
    ) -> tuple[PilotRunStateV1, str | None]:
        candidate_locator = PilotNetworkIntentV1.slot_locator_for(
            manifest_hash=self.manifest.manifest_hash,
            stage=stage,
            ordinal=ordinal,
        )
        sealed_locator = candidate_locator.removesuffix(".candidate.json") + ".sealed.json"
        candidate_raw = artifacts.get(candidate_locator)
        sealed_raw = artifacts.get(sealed_locator)
        if candidate_raw is None and sealed_raw is None:
            return state, None
        if candidate_raw is None or sealed_raw is None:
            return state, "unresolved_network_intent_after_restart"
        candidate_payload = parse_canonical_lf_json(candidate_raw)
        sealed_payload = parse_canonical_lf_json(sealed_raw)
        intent = PilotNetworkIntentV1.from_dict(sealed_payload)
        if (
            intent.stage != stage
            or intent.ordinal != ordinal
            or canonical_lf_bytes(intent.intent_candidate_payload) != candidate_raw
            or candidate_payload != intent.intent_candidate_payload
            or canonical_lf_bytes(intent.as_dict()) != sealed_raw
        ):
            raise PilotLocalStoreRecoveryError("pilot_control_intent_binding_mismatch")
        return state.with_network_intent(intent), None

    def _restart_claim_exists(
        self, authorization: U5PublicPilotAuthorizationReceiptV1 | None
    ) -> bool:
        if authorization is None:
            return False
        target = _target_for(
            self.external_state_root,
            self._session_claim_locator(authorization),
        )
        try:
            target.lstat()
        except FileNotFoundError:
            if self._session_capability is not None:
                raise PilotLocalStoreRecoveryError(
                    "pilot_local_live_session_claim_is_missing"
                )
            return False
        except OSError as exc:
            raise PilotLocalStoreRecoveryError(
                "pilot_local_restart_claim_probe_failed"
            ) from exc
        # A capability exists only in the process that successfully published,
        # reloaded and anchored this exact claim.  Any occupied slot without that
        # live capability—including a dangling junction—is restart evidence.
        capability = self._session_capability
        if capability is None:
            return True
        expected_locator = self._session_claim_locator(authorization)
        if capability.authorization_receipt_hash != authorization.receipt_hash:
            raise PilotLocalStoreRecoveryError(
                "pilot_local_live_session_claim_authorization_mismatch"
            )
        try:
            self._require_live_session_claim(
                capability,
                expected_locator=expected_locator,
            )
        except PilotLocalStoreError as exc:
            raise PilotLocalStoreRecoveryError(
                "pilot_local_live_session_claim_is_not_durable"
            ) from exc
        return False

    def _reconstruct_authoritative_state(
        self,
        *,
        stop_unresolved: bool,
        recheck_dependencies: bool = True,
    ) -> PilotRecoveryReportV1:
        """Replay fixed control artifacts once without trusting caller state."""

        artifacts, residue = self._scan_control_once(
            recheck_dependencies=recheck_dependencies,
        )
        manifest_raw = artifacts.get("run-control/manifest.json")
        if manifest_raw is None:
            raise PilotLocalStoreRecoveryError("pilot_control_manifest_is_missing")
        manifest = parse_pilot_run_manifest_v1(parse_canonical_lf_json(manifest_raw))
        if manifest != self.manifest or canonical_lf_bytes(manifest.as_dict()) != manifest_raw:
            raise PilotLocalStoreRecoveryError("pilot_control_manifest_binding_mismatch")
        state = PilotRunStateV1(manifest)
        auth = self._parse_receipt(
            artifacts,
            "run-control/authorization.json",
            U5PublicPilotAuthorizationReceiptV1,
        )
        if auth is not None:
            assert isinstance(auth, U5PublicPilotAuthorizationReceiptV1)
            state = state.with_authorization(auth, now_us=auth.authorized_at_us)
        stop_code: str | None = None
        if auth is not None:
            endpoint_preflight = self._parse_receipt(
                artifacts,
                "run-control/preflights/endpoint.json",
                PilotDiskPreflightReceiptV1,
            )
            if endpoint_preflight is not None:
                assert isinstance(endpoint_preflight, PilotDiskPreflightReceiptV1)
                state = state.with_preflight(
                    endpoint_preflight,
                    now_us=endpoint_preflight.checked_at_us,
                )
                state, stop_code = self._replay_intent(
                    artifacts,
                    state,
                    stage="endpoint_verification",
                    ordinal=-1,
                )
                if stop_code is None and state.network_intents:
                    endpoint = self._parse_receipt(
                        artifacts,
                        "run-control/endpoint-verification.json",
                        EndpointVerificationReceiptV1,
                    )
                    failure = self._parse_receipt(
                        artifacts,
                        "run-control/terminal-failure.json",
                        PilotStepFailureReceiptV1,
                    )
                    if endpoint is not None and failure is not None:
                        raise PilotLocalStoreRecoveryError(
                            "pilot_control_endpoint_result_conflict"
                        )
                    if endpoint is not None:
                        assert isinstance(endpoint, EndpointVerificationReceiptV1)
                        state = state.with_endpoint_verification(endpoint)
                    elif failure is not None:
                        assert isinstance(failure, PilotStepFailureReceiptV1)
                        state = state.with_step_failure(failure)
                    else:
                        stop_code = "unresolved_network_intent_after_restart"
        if stop_code is None and state.endpoint_verification is not None:
            for ordinal in range(len(self.manifest.shards)):
                preflight = self._parse_receipt(
                    artifacts,
                    f"run-control/preflights/shard-{ordinal:04d}.json",
                    PilotDiskPreflightReceiptV1,
                )
                if preflight is None:
                    break
                assert isinstance(preflight, PilotDiskPreflightReceiptV1)
                state = state.with_preflight(preflight, now_us=preflight.checked_at_us)
                state, stop_code = self._replay_intent(
                    artifacts,
                    state,
                    stage="shard_acquisition",
                    ordinal=ordinal,
                )
                if stop_code is not None:
                    break
                if len(state.network_intents) != ordinal + 2:
                    break
                result = self._parse_receipt(
                    artifacts,
                    f"run-control/shard-results/{ordinal:04d}.json",
                    PilotShardResultV1,
                )
                failure = self._parse_receipt(
                    artifacts,
                    "run-control/terminal-failure.json",
                    PilotStepFailureReceiptV1,
                )
                if result is not None and failure is not None:
                    raise PilotLocalStoreRecoveryError(
                        "pilot_control_shard_result_conflict"
                    )
                if result is not None:
                    assert isinstance(result, PilotShardResultV1)
                    state = state.with_shard_result(result)
                    continue
                if failure is not None:
                    assert isinstance(failure, PilotStepFailureReceiptV1)
                    state = state.with_step_failure(failure)
                else:
                    stop_code = "unresolved_network_intent_after_restart"
                break
        failure_raw = artifacts.get("run-control/terminal-failure.json")
        failure_candidate_raw = artifacts.get(
            "run-control/terminal-failure-candidate.json"
        )
        if (failure_raw is None) != (failure_candidate_raw is None):
            stop_code = stop_code or "unresolved_terminal_failure_publication"
        if state.failure_receipt is not None and failure_candidate_raw is not None:
            if (
                canonical_lf_bytes(state.failure_receipt.failure_candidate_payload)
                != failure_candidate_raw
            ):
                raise PilotLocalStoreRecoveryError(
                    "pilot_control_failure_candidate_binding_mismatch"
                )
        result_raw = artifacts.get("run-control/result-candidate.json")
        if result_raw is not None:
            if state.next_action != "publish_detached_result_anchor":
                stop_code = stop_code or "out_of_order_result_candidate"
            elif canonical_lf_bytes(state.result_candidate_payload) != result_raw:
                raise PilotLocalStoreRecoveryError(
                    "pilot_control_result_candidate_binding_mismatch"
                )
        recognized = {entry[1] for entry in state.run_control_inventory}
        if result_raw is not None and state.next_action == "publish_detached_result_anchor":
            recognized.add("run-control/result-candidate.json")
        residue = tuple(sorted(set(residue) | (set(artifacts) - recognized)))
        if residue:
            stop_code = stop_code or "unexpected_run_control_residue"
        restart = self._restart_claim_exists(auth)
        if restart:
            stop_code = stop_code or "preexisting_process_session_claim_after_restart"
        evidence_hash: str | None = None
        if stop_code is not None:
            evidence_hash = _sha256(
                _canonical_json_bytes(
                    {
                        "domain": "mexc_public_qa_pilot_recovery_stop_evidence_v1",
                        "manifest_hash": self.manifest.manifest_hash,
                        "stop_code": stop_code,
                        "residue_paths": list(residue),
                        "artifact_hashes": {
                            locator: _sha256(raw)
                            for locator, raw in sorted(artifacts.items())
                        },
                    }
                )
            )
            if stop_unresolved and state.next_action != "stopped":
                state = state.stopped(reason=stop_code, evidence_hash=evidence_hash)
        reported_stop_code = stop_code or state.stop_reason
        reported_evidence_hash = evidence_hash or state.stop_evidence_hash
        return PilotRecoveryReportV1(
            state=state,
            status=(
                "stopped_no_network"
                if reported_stop_code is not None
                else "reconstructed_no_network"
            ),
            stop_code=reported_stop_code,
            stop_evidence_hash=reported_evidence_hash,
            residue_paths=residue,
            restart_detected=restart,
            network_permitted=False,
        )

    def reconstruct_authoritative_state(self) -> PilotRecoveryReportV1:
        """Recover for restart: unresolved work is STOP and permission is always zero."""

        return self._reconstruct_authoritative_state(stop_unresolved=True)

    @staticmethod
    def _state_digest(state: PilotRunStateV1) -> str:
        if not isinstance(state, PilotRunStateV1):
            raise PilotLocalStoreConflictError("pilot_local_expected_state_is_invalid")
        return _sha256(_canonical_json_bytes(state.as_dict()))

    def _require_expected_state(
        self,
        expected: PilotRunStateV1,
        *,
        allow_active_intent: bool = False,
        recheck_dependencies: bool = True,
    ) -> PilotRunStateV1:
        report = self._reconstruct_authoritative_state(
            stop_unresolved=False,
            recheck_dependencies=recheck_dependencies,
        )
        if report.stop_code is not None:
            allowed = (
                allow_active_intent
                and report.stop_code == "unresolved_network_intent_after_restart"
                and report.state.network_intents
                and report.state.next_action.startswith("await_")
            )
            if not allowed:
                raise PilotLocalStoreRecoveryError(
                    f"pilot_local_authoritative_prefix_is_stopped:{report.stop_code}"
                )
        if self._state_digest(report.state) != self._state_digest(expected):
            raise PilotLocalStoreConflictError("pilot_local_expected_state_cas_mismatch")
        return report.state

    def scan_inventory(
        self,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> PilotInventoryScanV1:
        """Hash a fresh point-in-time file inventory without parsing its contents."""

        self._require_owner()
        entry_limit = (
            self.manifest.budgets.max_inventory_entries
            if max_entries is None
            else max_entries
        )
        byte_limit = (
            self.manifest.budgets.max_total_output_bytes
            if max_bytes is None
            else max_bytes
        )
        if type(entry_limit) is not int or entry_limit < 1:
            raise PilotLocalStoreBoundsError("pilot_inventory_entry_bound_is_invalid")
        if type(byte_limit) is not int or byte_limit < 1:
            raise PilotLocalStoreBoundsError("pilot_inventory_byte_bound_is_invalid")
        if entry_limit > self.manifest.budgets.max_inventory_entries:
            raise PilotLocalStoreBoundsError(
                "pilot_inventory_entry_bound_exceeds_manifest"
            )
        if byte_limit > self.manifest.budgets.max_total_output_bytes:
            raise PilotLocalStoreBoundsError(
                "pilot_inventory_byte_bound_exceeds_manifest"
            )
        if not self.output_root.exists():
            raise PilotLocalStoreError("pilot_inventory_output_root_is_missing")
        _validate_plain_existing_chain(self.output_root)
        entries: list[PilotInventoryEntryV1] = []
        seen_file_ids: set[tuple[int, int]] = set()
        total_bytes = 0
        visited = 0
        scan_started_ns = time.monotonic_ns()
        deadline_ns = scan_started_ns + _MAX_SCAN_RUNTIME_US * 1_000
        try:
            root_observed = self.output_root.lstat()
        except OSError as exc:
            raise PilotLocalStoreError("pilot_inventory_root_probe_failed") from exc
        if not _plain_mode(root_observed, directory=True):
            raise PilotLocalStoreError("pilot_inventory_root_is_not_plain")
        first_structure: dict[
            str,
            tuple[str, tuple[int, int, int, int, int, int], str | None],
        ] = {
            ".": ("directory", _stable_signature(root_observed), None),
        }
        _reject_windows_named_streams(self.output_root, deadline_ns=deadline_ns)
        stack = [(self.output_root, 0)]
        while stack:
            directory, depth = stack.pop()
            _check_scan_deadline(
                deadline_ns,
                code="pilot_inventory_scan_runtime_exceeded",
            )
            try:
                iterator = os.scandir(directory)
            except OSError as exc:
                raise PilotLocalStoreError("pilot_inventory_directory_scan_failed") from exc
            with iterator:
                for child in iterator:
                    _check_scan_deadline(
                        deadline_ns,
                        code="pilot_inventory_scan_runtime_exceeded",
                    )
                    visited += 1
                    if visited > entry_limit:
                        raise PilotLocalStoreBoundsError(
                            "pilot_inventory_visited_entry_budget_exceeded"
                        )
                    path = Path(child.path)
                    observed = path.lstat()
                    if child.is_symlink() or _is_reparse(observed):
                        raise PilotLocalStoreError("pilot_inventory_reparse_is_forbidden")
                    if stat.S_ISDIR(observed.st_mode):
                        _reject_windows_named_streams(
                            path, deadline_ns=deadline_ns
                        )
                        first_structure[path.relative_to(self.output_root).as_posix()] = (
                            "directory",
                            _stable_signature(observed),
                            None,
                        )
                        if depth + 1 > _MAX_TREE_DEPTH:
                            raise PilotLocalStoreBoundsError(
                                "pilot_inventory_depth_budget_exceeded"
                            )
                        stack.append((path, depth + 1))
                        continue
                    if not stat.S_ISREG(observed.st_mode):
                        raise PilotLocalStoreError(
                            "pilot_inventory_special_file_is_forbidden"
                        )
                    identity = (observed.st_dev, observed.st_ino)
                    if observed.st_nlink != 1 or identity in seen_file_ids:
                        raise PilotLocalStoreError(
                            "pilot_inventory_hardlink_alias_is_forbidden"
                        )
                    seen_file_ids.add(identity)
                    if total_bytes + observed.st_size > byte_limit:
                        raise PilotLocalStoreBoundsError(
                            "pilot_inventory_byte_budget_exceeded"
                        )
                    digest, counted, final_signature = _hash_exact_plain_file(
                        path,
                        max_bytes=max(1, byte_limit - total_bytes),
                        deadline_ns=deadline_ns,
                        deadline_code="pilot_inventory_scan_runtime_exceeded",
                    )
                    first_structure[path.relative_to(self.output_root).as_posix()] = (
                        "file",
                        final_signature,
                        digest,
                    )
                    total_bytes += counted
                    entries.append(
                        PilotInventoryEntryV1(
                            relative_path=path.relative_to(
                                self.output_root
                            ).as_posix(),
                            artifact_sha256=digest,
                            byte_count=counted,
                        )
                    )
        verified_structure = self._scan_tree_identity_snapshot(
            root=self.output_root,
            max_entries=entry_limit,
            max_total_bytes=byte_limit,
            deadline_ns=deadline_ns,
            deadline_code="pilot_inventory_scan_runtime_exceeded",
        )
        if verified_structure != first_structure:
            raise PilotLocalStoreError(
                "pilot_inventory_tree_changed_during_scan"
            )
        canonical_entries = tuple(
            sorted(entries, key=lambda item: item.relative_path)
        )
        sample = self._sample_clock()
        result = PilotInventoryScanV1(
            manifest_hash=self.manifest.manifest_hash,
            entries=canonical_entries,
            total_bytes=total_bytes,
            scanned_at_us=sample.epoch_us,
            scanned_monotonic_us=sample.monotonic_us,
            clock_domain_id=sample.clock_domain_id,
        )
        _check_scan_deadline(
            deadline_ns,
            code="pilot_inventory_scan_runtime_exceeded",
        )
        return result


_RUNTIME_AUTHORITY_BINDING_SCHEMA = {
    "contract_version": PILOT_RUNTIME_AUTHORITY_BINDING_VERSION,
    "field_set": list(PilotRuntimeAuthorityBindingV1.__dataclass_fields__),
    "binding_hash": "sha256_exact_canonical_as_dict",
    "identities": {
        "coordinator_implementation": "contract_version_and_hash",
        "clock": "contract_version_hash_and_domain",
        "detached_anchor_sink": "contract_version_hash_and_domain",
        "u5_verifier": (
            "contract_version_hash_domain_trust_key_and_policy_version_hash"
        ),
    },
}


def pilot_runtime_authority_binding_contract_hash() -> str:
    digest = _sha256(_canonical_json_bytes(_RUNTIME_AUTHORITY_BINDING_SCHEMA))
    if digest != _PINNED_RUNTIME_AUTHORITY_BINDING_CONTRACT_HASH:
        raise PilotLocalStoreError(
            "pilot_runtime_authority_binding_contract_changed_without_version_bump"
        )
    return digest


_LOCAL_STORE_SCHEMA = {
    "contract_version": PILOT_LOCAL_STORE_CONTRACT_VERSION,
    "pilot_run_contract_hash": pilot_run_contract_hash(),
    "runtime_authority_binding_contract_hash": (
        pilot_runtime_authority_binding_contract_hash()
    ),
    "authority": {
        "network_or_u5_factory_present": False,
        "u5_receipt_is_self_authorizing": False,
        "injected_offline_u5_verifier_has_default": False,
        "verified_u5_evidence_binds_fresh_process_challenge": True,
        "persisted_artifact_grants_network": False,
        "authoritative_reconstruction_grants_network": False,
        "live_nonserializable_owner_capability_required": True,
        "owner_capability_consumed_before_injected_runner_callback": True,
        "consumed_permission_key": "persisted_intent_hash_store_side_append_only",
        "callback_gate_rechecks_u5_and_remaining_run_window": True,
        "executor_and_runtime_authority_bindings_required": True,
        "clock_sink_and_u5_verifier_identities_rechecked_each_mutator": True,
        "runner_identity_rechecked_at_callback_gate": True,
        "runtime_identity_assurance": PILOT_RUNTIME_IDENTITY_ASSURANCE,
        "real_u5_construction_policy": PILOT_REAL_U5_CONSTRUCTION_POLICY,
        "bound_clock_construction": "reviewed_coordinator_owned_side_effect_free",
        "final_gate_order": (
            "slow_preflight_then_exact_replay_and_session_validation_then_final_"
            "clock_sample_then_lightweight_physical_check_then_pure_arithmetic_"
            "consume_callback"
        ),
        "future_real_http_attempt_gate": PILOT_FUTURE_HTTP_ATTEMPT_GATE_POLICY,
        "filesystem_mutation_boundary": PILOT_FILESYSTEM_MUTATION_BOUNDARY,
        "real_u5_filesystem_boundary": (
            "stop_until_operator_acceptance_or_handle_relative_snapshot"
        ),
        "in_process_private_memory_tamper": "explicitly_out_of_scope",
        "preexisting_process_session_claim": "restart_stop_zero_permit",
    },
    "publication": {
        "encoding": "utf8_sorted_keys_compact_no_nan_single_lf",
        "ordinary_same_bytes": "idempotent",
        "ordinary_conflict": "reject",
        "intent_candidate": "hardlink_create_new_preexisting_identical_or_conflict_loses",
        "sealed_intent": "hardlink_create_new_fresh_reload_before_capability",
        "temporary_and_target_same_directory": True,
        "caller_authoritative_terminal_receipt_publishers_present": False,
    },
    "locking": {
        "process_registry": True,
        "os_lock": "msvcrt_locking_or_fcntl_flock_nonblocking",
        "owner_process_and_thread_guard": True,
        "namespace_identity": "canonical_output_root_not_manifest",
        "lexical_or_short_name_alias": "reject",
        "lock_leaf_reparse_or_hardlink": "reject_before_mutation",
        "opened_lock_handle_identity_rechecked": True,
        "opened_lock_identity_rechecked_each_mutator": True,
        "output_root_volume_file_id_bound_after_creation": True,
        "constructor_bindings": "private_read_only_and_rechecked",
        "external_state_root": "deterministic_sibling_not_caller_selected",
        "persistent_session_claim_outside_subject_inventory": True,
    },
    "preflight": {
        "caller_receipt_accepted": False,
        "store_measures_free_space_and_fresh_roots": True,
        "windows_fixed_ntfs_volume_probed": True,
        "same_volume_create_new_hardlink_probed": True,
        "fresh_recheck_before_intent_cas_and_runner_callback": True,
        "fresh_root_probe": "lstat_only_file_not_found_means_absent",
        "ntfs_named_streams_before_permission": (
            "bounded_pre_post_file_and_final_root_directory_file_reject"
        ),
        "stable_tree_before_permission": (
            "two_bounded_exact_locator_signature_digest_passes"
        ),
    },
    "recovery": {
        "source": "fixed_run_control_artifact_prefix_only",
        "state_json_authoritative": False,
        "public_pilot_state_transitions_only": True,
        "every_permission_mutation_cas_matches_fresh_replay": True,
        "control_file_stream_validation": "pre_post_plus_final_visible_signature",
        "terminal_control_tree_verification": (
            "second_bounded_exact_locator_kind_signature_digest_pass"
        ),
        "unresolved_intent": "stop_without_network",
        "result": "always_zero_network_permission",
    },
    "inventory": {
        "bounded_entries_and_bytes": True,
        "caller_bounds_cannot_exceed_manifest_caps": True,
        "visited_files_and_directories_are_bounded": True,
        "depth_and_monotonic_runtime_are_bounded": True,
        "deadline_checks": "before_after_chunk_post_read_and_pre_return",
        "kernel_blocking_call_preemption": False,
        "streaming_no_content_parse": True,
        "ntfs_named_streams": (
            "bounded_pre_post_read_hash_and_final_root_directory_file_reject"
        ),
        "post_stream_leaf_check": "visible_identity_signature_rechecked",
        "post_stream_entry_accounting": (
            "separate_rejection_only_not_inventory_entry_charge"
        ),
        "terminal_tree_verification": (
            "second_bounded_exact_locator_kind_signature_digest_pass"
        ),
        "terminal_tree_verification_counter": (
            "separate_and_clamped_to_manifest_cap_shared_deadline"
        ),
        "atomic_whole_tree_snapshot": False,
        "cooperating_writers_share_run_lock": True,
        "concurrent_external_filesystem_mutation": "explicitly_out_of_scope",
        "reparse_or_symlink": "reject",
        "hardlink_alias": "reject",
        "before_after_identity_and_metadata_stable": True,
        "not_a_final_detached_anchor": True,
    },
    "limits": {
        "manifest_bytes": _MAX_MANIFEST_BYTES,
        "control_entry_bytes": _MAX_CONTROL_ENTRY_BYTES,
        "read_chunk_bytes": _READ_CHUNK_BYTES,
        "tree_depth": _MAX_TREE_DEPTH,
        "scan_runtime_us": _MAX_SCAN_RUNTIME_US,
    },
    "field_sets": {
        "clock_sample": list(PilotClockSampleV1.__dataclass_fields__),
        "detached_request": list(PilotDetachedEvidenceRequestV1.__dataclass_fields__),
        "detached_receipt": list(PilotDetachedEvidenceReceiptV1.__dataclass_fields__),
        "u5_verification_request": list(
            PilotU5VerificationRequestV1.__dataclass_fields__
        ),
        "u5_verification_evidence": list(
            PilotU5VerificationEvidenceV1.__dataclass_fields__
        ),
        "runtime_authority_binding": list(
            PilotRuntimeAuthorityBindingV1.__dataclass_fields__
        ),
        "inventory_entry": list(PilotInventoryEntryV1.__dataclass_fields__),
        "inventory_scan": list(PilotInventoryScanV1.__dataclass_fields__),
        "recovery_report": list(PilotRecoveryReportV1.__dataclass_fields__),
    },
}


def mexc_pilot_local_store_contract_hash() -> str:
    digest = _sha256(_canonical_json_bytes(_LOCAL_STORE_SCHEMA))
    if _PINNED_LOCAL_STORE_CONTRACT_HASH and digest != _PINNED_LOCAL_STORE_CONTRACT_HASH:
        raise PilotLocalStoreError("pilot_local_store_contract_changed_without_version_bump")
    return digest


__all__ = [
    "MexcPilotLocalStoreV1",
    "PILOT_DETACHED_EVIDENCE_RECEIPT_VERSION",
    "PILOT_DETACHED_EVIDENCE_REQUEST_VERSION",
    "PILOT_LOCAL_INVENTORY_VERSION",
    "PILOT_LOCAL_RECOVERY_VERSION",
    "PILOT_LOCAL_STORE_CONTRACT_VERSION",
    "PILOT_FUTURE_HTTP_ATTEMPT_GATE_POLICY",
    "PILOT_FILESYSTEM_MUTATION_BOUNDARY",
    "PILOT_RUNTIME_AUTHORITY_BINDING_VERSION",
    "PILOT_RUNTIME_IDENTITY_ASSURANCE",
    "PILOT_REAL_U5_CONSTRUCTION_POLICY",
    "PILOT_U5_VERIFICATION_EVIDENCE_VERSION",
    "PILOT_U5_VERIFICATION_REQUEST_VERSION",
    "PilotClockSampleV1",
    "PilotDetachedEvidenceReceiptV1",
    "PilotDetachedEvidenceRequestV1",
    "PilotDetachedEvidenceSink",
    "PilotEvidenceClock",
    "PilotIntentClaimResultV1",
    "PilotIntentOwnerCapability",
    "PilotInventoryEntryV1",
    "PilotInventoryScanV1",
    "PilotLocalStoreBoundsError",
    "PilotLocalStoreConflictError",
    "PilotLocalStoreError",
    "PilotLocalStoreLockError",
    "PilotLocalStoreRecoveryError",
    "PilotOwnedIntentRunner",
    "PilotRecoveryReportV1",
    "PilotRuntimeAuthorityBindingV1",
    "PilotRunLockLease",
    "PilotRunSessionCapability",
    "PilotU5AuthorityVerifier",
    "PilotU5VerificationEvidenceV1",
    "PilotU5VerificationRequestV1",
    "canonical_lf_bytes",
    "mexc_pilot_local_store_contract_hash",
    "pilot_runtime_authority_binding_contract_hash",
    "parse_canonical_lf_json",
]
