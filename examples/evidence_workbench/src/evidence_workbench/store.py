# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed storage with locally verifiable snapshot hashes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .schema import (
    APP_VERSION,
    SCHEMA_VERSION,
    EvidenceError,
    IntegrityError,
    PageInput,
    canonical_json,
    chunk_text,
    normalize_text,
    safe_filename,
    sha256_bytes,
    sha256_file,
    sha256_text,
    slugify,
    utc_now,
    validate_case_id,
)

try:  # pragma: no cover - Windows fallback is covered through the same lock API.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_HEAD_NAME = "HEAD"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_HASH_NAME = "manifest.sha256"
_INDEX_NAME = "index.sqlite"
_AUDIT_HEAD_NAME = "AUDIT_HEAD.json"
_INDEX_COLUMNS = (
    "chunk_id",
    "citation_id",
    "document_id",
    "filename",
    "source_sha256",
    "page_number",
    "chunk_number",
    "start_offset",
    "end_offset",
    "text",
    "extraction_method",
    "needs_review",
    "evidence_image",
)


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


class Workspace:
    """Manage cases without mutating any committed evidence snapshot."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.cases_root = self.root / "cases"
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    def create_case(self, name: str, case_id: str | None = None) -> dict[str, Any]:
        display_name = name.strip()
        if not display_name:
            raise EvidenceError("case name must not be empty")
        chosen = validate_case_id(case_id or slugify(display_name))
        case_root = self._case_root(chosen)
        try:
            case_root.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise EvidenceError(f"case already exists: {chosen}") from exc
        for directory in (
            "objects",
            "page_records",
            "page_images",
            "snapshots",
            "exports",
            "staging",
        ):
            (case_root / directory).mkdir(mode=0o700)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "id": chosen,
            "name": display_name,
            "created_at": utc_now(),
        }
        _atomic_write(case_root / "case.json", _json_bytes(metadata))
        self.record_event(chosen, "case_created", {"name": display_name})
        return metadata

    def list_cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for metadata_path in sorted(self.cases_root.glob("*/case.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                head = self.head_snapshot_id(str(metadata["id"]), required=False)
                metadata["head_snapshot_id"] = head
                if head:
                    case_id = str(metadata["id"])
                    manifest = self.load_manifest(case_id, head)
                    # Listing is intentionally cheap; authoritative operations
                    # pin and verify one snapshot before trusting it.
                    metadata["integrity_verified"] = None
                    metadata["coverage"] = self._coverage_with_exclusions(
                        manifest["sources"], manifest.get("excluded_sources", [])
                    )
                cases.append(metadata)
            except (OSError, json.JSONDecodeError, KeyError, EvidenceError):
                continue
        return cases

    def get_case(self, case_id: str) -> dict[str, Any]:
        path = self._case_root(case_id) / "case.json"
        if not path.is_file():
            raise EvidenceError(f"unknown case: {case_id}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"case metadata is unreadable: {case_id}") from exc
        value["head_snapshot_id"] = self.head_snapshot_id(case_id, required=False)
        return value

    def head_snapshot_id(self, case_id: str, *, required: bool = True) -> str:
        path = self._case_root(case_id) / _HEAD_NAME
        if not path.is_file():
            if required:
                raise EvidenceError(f"case has no evidence snapshot: {case_id}")
            return ""
        snapshot_id = path.read_text(encoding="ascii").strip()
        if len(snapshot_id) != 64 or any(char not in "0123456789abcdef" for char in snapshot_id):
            raise IntegrityError(f"case HEAD is invalid: {case_id}")
        return snapshot_id

    def load_manifest(self, case_id: str, snapshot_id: str | None = None) -> dict[str, Any]:
        chosen = snapshot_id or self.head_snapshot_id(case_id)
        manifest_path = self._snapshot_root(case_id, chosen) / _MANIFEST_NAME
        if not manifest_path.is_file():
            raise IntegrityError(f"snapshot manifest is missing: {chosen}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"snapshot manifest is unreadable: {chosen}") from exc
        if manifest.get("snapshot_id") != chosen:
            raise IntegrityError(f"snapshot id does not match manifest: {chosen}")
        return manifest

    def index_path(self, case_id: str, snapshot_id: str | None = None) -> Path:
        chosen = snapshot_id or self.head_snapshot_id(case_id)
        return self._snapshot_root(case_id, chosen) / _INDEX_NAME

    def latest_index_path(self, case_id: str) -> Path:
        return self.index_path(case_id)

    def source_path(self, case_id: str, source_sha256: str) -> Path:
        _require_sha256(source_sha256, "source")
        return self._case_root(case_id) / "objects" / source_sha256

    def page_record(self, case_id: str, record_sha256: str) -> dict[str, Any]:
        _require_sha256(record_sha256, "page record")
        path = self._case_root(case_id) / "page_records" / f"{record_sha256}.json"
        if not path.is_file() or sha256_file(path) != record_sha256:
            raise IntegrityError(f"page record failed integrity verification: {record_sha256}")
        return json.loads(path.read_text(encoding="utf-8"))

    def archive_source(self, case_id: str, source: Path) -> tuple[str, Path]:
        """Copy one regular, non-symlink source into the content-addressed store."""

        self.get_case(case_id)
        objects_root = self._case_root(case_id) / "objects"
        if source.is_symlink():
            raise EvidenceError(f"symlinked inputs are not accepted: {source}")
        resolved = source.resolve()
        if resolved.parent == objects_root and len(resolved.name) == 64:
            if sha256_file(resolved) != resolved.name:
                raise IntegrityError(f"existing source object is corrupt: {resolved.name}")
            return resolved.name, resolved
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            input_descriptor = os.open(source, flags)
        except OSError as exc:
            raise EvidenceError(f"could not open source without following links: {source}") from exc
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".incoming.", suffix=".tmp", dir=objects_root
        )
        temporary = Path(temporary_name)
        try:
            source_stat = os.fstat(input_descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise EvidenceError(f"input is not a regular file: {source}")
            digest = hashlib.sha256()
            with (
                os.fdopen(input_descriptor, "rb") as input_stream,
                os.fdopen(descriptor, "wb") as output,
            ):
                while block := input_stream.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            source_sha256 = digest.hexdigest()
            destination = self.source_path(case_id, source_sha256)
            os.chmod(temporary, 0o400)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if not destination.is_file() or sha256_file(destination) != source_sha256:
                    raise IntegrityError(f"existing source object is corrupt: {source_sha256}")
        except BaseException:
            for open_descriptor in (input_descriptor, descriptor):
                try:
                    os.close(open_descriptor)
                except OSError:
                    pass
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink(missing_ok=True)
        return source_sha256, destination

    def archive_page_image(self, case_id: str, image_path: Path) -> tuple[str, str]:
        if image_path.is_symlink() or not image_path.is_file():
            raise EvidenceError(f"page evidence image is not a regular file: {image_path}")
        image_hash = sha256_file(image_path)
        suffix = image_path.suffix.lower() or ".png"
        relative = f"page_images/{image_hash}{suffix}"
        destination = self._case_root(case_id) / relative
        if not destination.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{image_hash}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            try:
                with image_path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if sha256_file(temporary) != image_hash:
                    raise IntegrityError("page image changed while it was archived")
                os.chmod(temporary, 0o400)
                os.replace(temporary, destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        elif sha256_file(destination) != image_hash:
            raise IntegrityError(f"existing page image is corrupt: {relative}")
        return image_hash, relative

    def commit_document(
        self,
        case_id: str,
        *,
        source_path: Path,
        filename: str,
        pages: Iterable[PageInput],
        extraction: dict[str, Any],
        document_status: str = "indexed",
        document_error: str = "",
    ) -> dict[str, Any]:
        """Add one source and create or reuse the resulting immutable snapshot."""

        raw_basename = Path(filename).name.replace("\x00", "").strip()
        if not raw_basename or raw_basename in {".", ".."}:
            raise EvidenceError("filename is empty or unsafe")
        clean_filename = safe_filename(filename)
        source_sha256, archived = self.archive_source(case_id, source_path)
        filename_identity_sha256 = sha256_text(normalize_text(raw_basename))
        document_id = sha256_text(f"{source_sha256}\0{filename_identity_sha256}")
        page_entries: list[dict[str, Any]] = []
        for page in sorted(pages, key=lambda item: item.page_number):
            if page.page_number < 1:
                raise EvidenceError("page numbers start at 1")
            image_hash = ""
            image_relative = ""
            if page.evidence_image:
                image_hash, image_relative = self.archive_page_image(
                    case_id, Path(page.evidence_image)
                )
            page_record = {
                "schema_version": SCHEMA_VERSION,
                "source_sha256": source_sha256,
                "page_number": page.page_number,
                "text": page.text,
                "text_sha256": sha256_text(page.text),
                "extraction_method": page.extraction_method,
                "status": page.status,
                "quality_score": round(float(page.quality_score), 6),
                "quality_score_kind": "deterministic_heuristic",
                "needs_review": bool(page.needs_review),
                "error": page.error,
                "evidence_image_sha256": image_hash,
                "evidence_image": image_relative,
                "metadata": page.metadata,
            }
            page_data = _json_bytes(page_record)
            record_sha256 = sha256_bytes(page_data)
            record_path = self._case_root(case_id) / "page_records" / f"{record_sha256}.json"
            if not record_path.exists():
                _atomic_write(record_path, page_data, mode=0o400)
            elif sha256_file(record_path) != record_sha256:
                raise IntegrityError(f"existing page record is corrupt: {record_sha256}")
            page_entries.append(
                {
                    key: value
                    for key, value in page_record.items()
                    if key not in {"schema_version", "text"}
                }
                | {"record_sha256": record_sha256}
            )

        document = {
            # Bytes are deduplicated by source_sha256, while document_id keeps
            # distinct evidence aliases such as Exhibit-A and Exhibit-B.
            "document_id": document_id,
            "filename": clean_filename,
            "filename_identity_sha256": filename_identity_sha256,
            "source_sha256": source_sha256,
            "size_bytes": archived.stat().st_size,
            "media_type": _media_type(clean_filename),
            "status": document_status,
            "error": document_error,
            "page_count": len(page_entries),
            "pages": page_entries,
            "extraction": extraction,
        }
        snapshot = self._commit_snapshot(case_id, [document])
        return {"document": document, "snapshot": snapshot}

    def coverage(self, case_id: str, snapshot_id: str | None = None) -> dict[str, Any]:
        return dict(self.load_manifest(case_id, snapshot_id)["coverage"])

    def verify(
        self,
        case_id: str,
        snapshot_id: str | None = None,
        *,
        require_audit_binding: bool = True,
    ) -> dict[str, Any]:
        chosen = snapshot_id or self.head_snapshot_id(case_id)
        snapshot_root = self._snapshot_root(case_id, chosen)
        failures: list[str] = []
        manifest_path = snapshot_root / _MANIFEST_NAME
        hash_path = snapshot_root / _MANIFEST_HASH_NAME
        if not manifest_path.is_file() or not hash_path.is_file():
            failures.append("snapshot manifest or manifest hash is missing")
            return {"ok": False, "snapshot_id": chosen, "failures": failures}

        manifest_hash = sha256_file(manifest_path)
        expected_line = f"{manifest_hash}  {_MANIFEST_NAME}"
        if hash_path.read_text(encoding="ascii").strip() != expected_line:
            failures.append("manifest.sha256 does not match manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append("manifest.json is not valid JSON")
            return {"ok": False, "snapshot_id": chosen, "failures": failures}

        sources = manifest.get("sources")
        excluded_sources = manifest.get("excluded_sources", [])
        writer_version = manifest.get("app_version")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("case_id") != case_id
            or not isinstance(writer_version, str)
            or not writer_version
            or len(writer_version) > 64
            or not isinstance(sources, list)
            or any(not isinstance(source, dict) for source in sources)
        ):
            failures.append("manifest schema, writer version, case id, or sources are invalid")
            sources = []
        if not isinstance(excluded_sources, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("document"), dict)
            for item in excluded_sources
        ):
            failures.append("manifest excluded-source tombstones are invalid")
            excluded_sources = []
        for item in excluded_sources:
            expected_tombstone = sha256_text(
                canonical_json(
                    {
                        key: item.get(key)
                        for key in (
                            "document",
                            "reviewer",
                            "reason",
                            "excluded_at",
                            "excluded_from_snapshot_id",
                        )
                    }
                )
            )
            if item.get("tombstone_id") != expected_tombstone:
                failures.append("excluded-source tombstone hash mismatch")
        inventory = self._snapshot_inventory(
            case_id,
            sources,
            excluded_sources,
            writer_version=writer_version if isinstance(writer_version, str) else "",
        )
        if sha256_text(canonical_json(inventory)) != chosen:
            failures.append("snapshot id does not match its canonical evidence inventory")
        if manifest.get("inventory_sha256") != chosen:
            failures.append("manifest inventory_sha256 does not match snapshot id")
        recomputed_coverage = self._coverage_with_exclusions(sources, excluded_sources)
        if manifest.get("coverage") != recomputed_coverage:
            failures.append("manifest coverage does not match authenticated source/page state")
        index_path = snapshot_root / _INDEX_NAME
        if not index_path.is_file():
            failures.append("search index is missing")
        elif sha256_file(index_path) != manifest.get("index_sha256"):
            failures.append("search index hash mismatch")

        evidence_sources = [*sources, *(item["document"] for item in excluded_sources)]
        for source in evidence_sources:
            source_hash = str(source.get("source_sha256", ""))
            try:
                object_path = self.source_path(case_id, source_hash)
            except EvidenceError:
                failures.append(f"source object hash mismatch: {source_hash}")
                continue
            if not object_path.is_file() or sha256_file(object_path) != source_hash:
                failures.append(f"source object hash mismatch: {source_hash}")
            pages = source.get("pages")
            if not isinstance(pages, list) or any(not isinstance(page, dict) for page in pages):
                failures.append(f"source page inventory is invalid: {source_hash}")
                continue
            for page in pages:
                record_hash = str(page.get("record_sha256", ""))
                try:
                    _require_sha256(record_hash, "page record")
                except EvidenceError:
                    failures.append(f"page record hash mismatch: {record_hash}")
                    continue
                record_path = self._case_root(case_id) / "page_records" / f"{record_hash}.json"
                if not record_path.is_file() or sha256_file(record_path) != record_hash:
                    failures.append(f"page record hash mismatch: {record_hash}")
                    continue
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    failures.append(f"page record is invalid JSON: {record_hash}")
                    continue
                if sha256_text(str(record.get("text", ""))) != record.get("text_sha256"):
                    failures.append(f"page text hash mismatch: {record_hash}")
                image_relative = str(record.get("evidence_image", ""))
                image_hash = str(record.get("evidence_image_sha256", ""))
                if image_relative:
                    image_path = (self._case_root(case_id) / image_relative).resolve()
                    if (
                        not image_path.is_relative_to(self._case_root(case_id))
                        or not image_path.is_file()
                        or sha256_file(image_path) != image_hash
                    ):
                        failures.append(f"page evidence image hash mismatch: {image_relative}")

        if index_path.is_file():
            try:
                expected_index_content = self._expected_index_content_sha256(case_id, sources)
                actual_index_content = self._actual_index_content_sha256(index_path)
                if manifest.get("index_content_sha256") != expected_index_content:
                    failures.append(
                        "manifest index-content hash does not match authenticated page records"
                    )
                if actual_index_content != expected_index_content:
                    failures.append(
                        "search index content does not match authenticated page records"
                    )
            except (EvidenceError, sqlite3.Error, json.JSONDecodeError) as exc:
                failures.append(f"search index content verification failed: {exc}")

        audit = self.verify_audit(case_id)
        failures.extend(audit["failures"])
        if require_audit_binding:
            if audit["ok"]:
                try:
                    snapshot_events = [
                        event
                        for event in self.audit_events(case_id)
                        if event.get("event_type") == "snapshot_committed"
                        and event.get("payload", {}).get("snapshot_id") == chosen
                    ]
                except IntegrityError as exc:
                    snapshot_events = []
                    failures.append(str(exc))
                if not snapshot_events:
                    failures.append("audit log does not bind this snapshot manifest")
                else:
                    committed = snapshot_events[-1].get("payload", {})
                    expected_lineage = {
                        "parent_snapshot_id": manifest.get("parent_snapshot_id", ""),
                        "manifest_created_at": manifest.get("created_at", ""),
                        "manifest_sha256": manifest_hash,
                    }
                    for field, expected in expected_lineage.items():
                        if committed.get(field, "") != expected:
                            failures.append(f"snapshot audit binding mismatch: {field}")
        result = {
            "ok": not failures,
            "snapshot_id": chosen,
            "manifest_sha256": manifest_hash,
            "failures": failures,
            "coverage": recomputed_coverage,
            "audit_head": audit.get("head", ""),
        }
        if not failures and audit["ok"]:
            result["effective_coverage"] = self.effective_coverage(case_id, manifest)
        return result

    def record_event(self, case_id: str, event_type: str, payload: dict[str, Any]) -> str:
        case_root = self._case_root(case_id)
        case_root.mkdir(parents=True, exist_ok=True)
        lock_path = case_root / ".audit.lock"
        with self._audit_lock(lock_path):
            return self._record_event_locked(case_id, event_type, payload)

    def atomic_audit_event(
        self,
        case_id: str,
        event_type: str,
        build: Callable[[], tuple[dict[str, Any], Any]],
    ) -> tuple[Any, str]:
        """Build state and append its event under one cross-process audit lock."""

        with self._audit_lock(self._case_root(case_id) / ".audit.lock"):
            payload, result = build()
            event_hash = self._record_event_locked(case_id, event_type, payload)
            return result, event_hash

    def _record_event_locked(self, case_id: str, event_type: str, payload: dict[str, Any]) -> str:
        case_root = self._case_root(case_id)
        audit_path = case_root / "audit.jsonl"
        previous_hash = "0" * 64
        sequence = 1
        if audit_path.is_file():
            lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line]
            if lines:
                last = json.loads(lines[-1])
                previous_hash = str(last["event_hash"])
                sequence = int(last["sequence"]) + 1
        event = {
            "sequence": sequence,
            "case_id": case_id,
            "event_type": event_type,
            "created_at": utc_now(),
            "previous_hash": previous_hash,
            "payload": payload,
        }
        event_hash = sha256_text(canonical_json(event))
        event["event_hash"] = event_hash
        with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(audit_path, 0o600)
        _atomic_write(
            case_root / _AUDIT_HEAD_NAME,
            _json_bytes({"sequence": sequence, "event_hash": event_hash}),
            mode=0o400,
        )
        return event_hash

    def audit_events(self, case_id: str) -> list[dict[str, Any]]:
        path = self._case_root(case_id) / "audit.jsonl"
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"audit line {line_number} is invalid JSON") from exc
        return events

    def verify_audit(self, case_id: str) -> dict[str, Any]:
        failures: list[str] = []
        previous_hash = "0" * 64
        expected_sequence = 1
        try:
            events = self.audit_events(case_id)
        except IntegrityError as exc:
            return {"ok": False, "head": "", "failures": [str(exc)]}
        if not events:
            failures.append("audit log is missing or empty")
        for event in events:
            observed_hash = str(event.get("event_hash", ""))
            unhashed = {key: value for key, value in event.items() if key != "event_hash"}
            expected_hash = sha256_text(canonical_json(unhashed))
            if event.get("sequence") != expected_sequence:
                failures.append(f"audit sequence gap at {expected_sequence}")
            if event.get("previous_hash") != previous_hash:
                failures.append(f"audit previous hash mismatch at {expected_sequence}")
            if observed_hash != expected_hash:
                failures.append(f"audit event hash mismatch at {expected_sequence}")
            previous_hash = observed_hash
            expected_sequence += 1
        anchor_path = self._case_root(case_id) / _AUDIT_HEAD_NAME
        if not anchor_path.is_file():
            failures.append("audit head anchor is missing")
        else:
            try:
                anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
                if anchor.get("sequence") != len(events):
                    failures.append("audit head sequence does not match the log")
                if anchor.get("event_hash") != previous_hash:
                    failures.append("audit head hash does not match the log")
            except (OSError, json.JSONDecodeError):
                failures.append("audit head anchor is unreadable")
        head_snapshot = self.head_snapshot_id(case_id, required=False)
        if head_snapshot and not any(
            event.get("event_type") == "snapshot_committed"
            and event.get("payload", {}).get("snapshot_id") == head_snapshot
            for event in events
        ):
            failures.append("audit log does not contain the current snapshot commit")
        return {"ok": not failures, "head": previous_hash, "failures": failures}

    def record_review(
        self,
        case_id: str,
        *,
        snapshot_id: str,
        target_type: str,
        target_id: str,
        status: str,
        reviewer: str,
        notes: str = "",
        expected_target_sha256: str | None = None,
    ) -> dict[str, Any]:
        with self._audit_lock(self._case_root(case_id) / ".snapshot.lock"):
            return self._record_review_locked(
                case_id,
                snapshot_id=snapshot_id,
                target_type=target_type,
                target_id=target_id,
                status=status,
                reviewer=reviewer,
                notes=notes,
                expected_target_sha256=expected_target_sha256,
            )

    def _record_review_locked(
        self,
        case_id: str,
        *,
        snapshot_id: str,
        target_type: str,
        target_id: str,
        status: str,
        reviewer: str,
        notes: str = "",
        expected_target_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Append one human review decision without rewriting evidence."""

        if snapshot_id != self.head_snapshot_id(case_id):
            raise EvidenceError("reviews may only be recorded against the current snapshot")
        if target_type not in {"page", "chronology_event"}:
            raise EvidenceError("review target_type must be page or chronology_event")
        if status not in {"accepted", "rejected", "unreviewed"}:
            raise EvidenceError("review status must be accepted, rejected, or unreviewed")
        clean_reviewer = reviewer.strip()
        clean_notes = notes.strip()
        if not clean_reviewer or len(clean_reviewer) > 120:
            raise EvidenceError("reviewer must contain 1 to 120 characters")
        if not target_id or len(target_id) > 160 or len(clean_notes) > 4_000:
            raise EvidenceError("review target or notes exceed the supported length")
        if target_type == "page":
            if expected_target_sha256 is None:
                raise EvidenceError("page review requires the exact inspected page-record SHA-256")
            _require_sha256(expected_target_sha256, "page record")
            valid_pages = {
                f"{source['document_id']}:p{page['page_number']}": page["record_sha256"]
                for source in self.load_manifest(case_id, snapshot_id)["sources"]
                for page in source["pages"]
            }
            if target_id not in valid_pages:
                raise EvidenceError(f"unknown page review target: {target_id}")
            target_sha256 = valid_pages[target_id]
            if expected_target_sha256 != target_sha256:
                raise EvidenceError(
                    "the page record changed; refresh and inspect the current record"
                )
        else:
            target_sha256 = target_id
        decision = {
            "snapshot_id": snapshot_id,
            "target_type": target_type,
            "target_id": target_id,
            "target_sha256": target_sha256,
            "status": status,
            "reviewer": clean_reviewer,
            "notes": clean_notes,
            "reviewed_at": utc_now(),
        }
        event_hash = self.record_event(case_id, "review_recorded", decision)
        return decision | {"audit_event_hash": event_hash}

    def reviews(
        self, case_id: str, snapshot_id: str, target_type: str
    ) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        current_page_hashes: dict[str, str] = {}
        if target_type == "page":
            current_page_hashes = {
                f"{source['document_id']}:p{page['page_number']}": page["record_sha256"]
                for source in self.load_manifest(case_id, snapshot_id)["sources"]
                for page in source["pages"]
            }
        for event in self.audit_events(case_id):
            payload = event.get("payload", {})
            if (
                event.get("event_type") == "review_recorded"
                and payload.get("target_type") == target_type
            ):
                target_id = str(payload.get("target_id", ""))
                content_matches = (
                    payload.get("target_sha256") == current_page_hashes.get(target_id)
                    if target_type == "page"
                    else payload.get("target_sha256") == target_id
                )
                if not content_matches:
                    continue
                latest[target_id] = dict(payload) | {
                    "audit_event_hash": event.get("event_hash", "")
                }
        return latest

    def effective_coverage(
        self, case_id: str, manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        chosen = manifest or self.load_manifest(case_id)
        coverage = self._coverage_with_exclusions(
            chosen["sources"], chosen.get("excluded_sources", [])
        )
        reviews = self.reviews(case_id, chosen["snapshot_id"], "page")
        pending: list[dict[str, Any]] = []
        accepted_count = 0
        for source in chosen["sources"]:
            for page in source["pages"]:
                if not page.get("needs_review"):
                    continue
                citation_id = f"{source['document_id']}:p{page['page_number']}"
                review = reviews.get(citation_id, {})
                if page.get("status") == "readable" and review.get("status") == "accepted":
                    accepted_count += 1
                else:
                    pending.append(
                        {
                            "citation_id": citation_id,
                            "filename": source["filename"],
                            "page_number": page["page_number"],
                            "page_status": page.get("status", ""),
                            "review_status": review.get("status", "unreviewed"),
                            "error": page.get("error", ""),
                        }
                    )
        coverage["pages_review_accepted"] = accepted_count
        coverage["review_pending_pages"] = pending
        coverage["complete_for_negative_assertions"] = (
            coverage["documents_failed"] == 0
            and coverage["pages_failed"] == 0
            and coverage["pages_total"] > 0
            and not pending
        )
        return coverage

    def exclude_document(
        self,
        case_id: str,
        document_id: str,
        *,
        reviewer: str,
        reason: str,
    ) -> dict[str, Any]:
        """Create a new snapshot without one source alias and audit the decision."""

        _require_sha256(document_id, "document")
        clean_reviewer = reviewer.strip()
        clean_reason = reason.strip()
        if not clean_reviewer or len(clean_reviewer) > 120:
            raise EvidenceError("reviewer must contain 1 to 120 characters")
        if not clean_reason or len(clean_reason) > 2_000:
            raise EvidenceError("exclusion reason must contain 1 to 2000 characters")
        return self._commit_snapshot(
            case_id,
            [],
            exclusion_requests=[
                {
                    "document_id": document_id,
                    "reviewer": clean_reviewer,
                    "reason": clean_reason,
                    "excluded_at": utc_now(),
                }
            ],
        )

    @contextmanager
    def audit_lock(self, case_id: str) -> Iterator[None]:
        """Exclude concurrent audit/review writes across processes."""

        self.get_case(case_id)
        with self._audit_lock(self._case_root(case_id) / ".audit.lock"):
            yield

    @contextmanager
    def publication_lock(self, case_id: str) -> Iterator[None]:
        """Pin HEAD and review state in the repository-wide lock order."""

        self.get_case(case_id)
        with self._audit_lock(self._case_root(case_id) / ".snapshot.lock"):
            with self._audit_lock(self._case_root(case_id) / ".audit.lock"):
                yield

    def _commit_snapshot(
        self,
        case_id: str,
        new_documents: list[dict[str, Any]],
        *,
        exclusion_requests: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        commit_lock = self._case_root(case_id) / ".snapshot.lock"
        # Hold a case-wide file lock from parent read through HEAD update.
        with self._audit_lock(commit_lock):
            parent_id = self.head_snapshot_id(case_id, required=False)
            existing: dict[str, dict[str, Any]] = {}
            if parent_id:
                parent = self.load_manifest(case_id, parent_id)
                existing = {str(source["document_id"]): source for source in parent["sources"]}
                excluded_existing = {
                    str(item["document"]["document_id"]): item
                    for item in parent.get("excluded_sources", [])
                }
            else:
                excluded_existing = {}
            requests = exclusion_requests or []
            requested_ids = {str(request["document_id"]) for request in requests}
            missing = requested_ids - existing.keys()
            if missing:
                raise EvidenceError(
                    "cannot exclude unknown document ids: " + ", ".join(sorted(missing))
                )
            for request in requests:
                document_id = str(request["document_id"])
                document = existing.pop(document_id)
                tombstone = {
                    "tombstone_id": sha256_text(
                        canonical_json(
                            {
                                "document": document,
                                "reviewer": request["reviewer"],
                                "reason": request["reason"],
                                "excluded_at": request["excluded_at"],
                                "excluded_from_snapshot_id": parent_id,
                            }
                        )
                    ),
                    "document": document,
                    "reviewer": request["reviewer"],
                    "reason": request["reason"],
                    "excluded_at": request["excluded_at"],
                    "excluded_from_snapshot_id": parent_id,
                }
                excluded_existing[document_id] = tombstone
            for document in new_documents:
                document_id = str(document["document_id"])
                existing[document_id] = document
                excluded_existing.pop(document_id, None)
            sources = sorted(
                existing.values(),
                key=lambda item: (str(item["document_id"]), str(item["filename"]).casefold()),
            )
            excluded_sources = sorted(
                excluded_existing.values(),
                key=lambda item: str(item["document"]["document_id"]),
            )
            inventory = self._snapshot_inventory(case_id, sources, excluded_sources)
            snapshot_id = sha256_text(canonical_json(inventory))
            final_root = self._snapshot_root(case_id, snapshot_id)
            if final_root.exists():
                verification = self.verify(case_id, snapshot_id, require_audit_binding=False)
                if not verification["ok"]:
                    raise IntegrityError(
                        f"existing snapshot is corrupt: {snapshot_id}: "
                        + "; ".join(verification["failures"])
                    )
                manifest = self.load_manifest(case_id, snapshot_id)
                manifest_hash = sha256_file(final_root / _MANIFEST_NAME)
                has_binding = any(
                    event.get("event_type") == "snapshot_committed"
                    and event.get("payload", {}).get("snapshot_id") == snapshot_id
                    for event in self.audit_events(case_id)
                )
                if not has_binding:
                    self.record_event(
                        case_id,
                        "snapshot_committed",
                        self._snapshot_commit_payload(
                            manifest,
                            manifest_hash,
                            requested_ids,
                        ),
                    )
                self._write_head(case_id, snapshot_id)
                return manifest

            snapshots_root = self._case_root(case_id) / "snapshots"
            stage_root = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=snapshots_root))
            try:
                index_path = stage_root / _INDEX_NAME
                self._build_index(case_id, sources, index_path)
                coverage = self._coverage_with_exclusions(sources, excluded_sources)
                index_content_sha256 = self._expected_index_content_sha256(case_id, sources)
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "app_version": APP_VERSION,
                    "case_id": case_id,
                    "snapshot_id": snapshot_id,
                    "parent_snapshot_id": parent_id,
                    "created_at": utc_now(),
                    "inventory_sha256": snapshot_id,
                    "index_sha256": sha256_file(index_path),
                    "index_content_sha256": index_content_sha256,
                    "coverage": coverage,
                    "sources": sources,
                    "excluded_sources": excluded_sources,
                }
                manifest_data = _json_bytes(manifest)
                manifest_hash = sha256_bytes(manifest_data)
                _atomic_write(stage_root / _MANIFEST_NAME, manifest_data, mode=0o400)
                _atomic_write(
                    stage_root / _MANIFEST_HASH_NAME,
                    f"{manifest_hash}  {_MANIFEST_NAME}\n".encode("ascii"),
                    mode=0o400,
                )
                os.replace(stage_root, final_root)
            except BaseException:
                shutil.rmtree(stage_root, ignore_errors=True)
                raise
            self.record_event(
                case_id,
                "snapshot_committed",
                self._snapshot_commit_payload(manifest, manifest_hash, requested_ids),
            )
            self._write_head(case_id, snapshot_id)
            return manifest

    @staticmethod
    def _snapshot_commit_payload(
        manifest: dict[str, Any], manifest_hash: str, requested_ids: set[str]
    ) -> dict[str, Any]:
        return {
            "snapshot_id": manifest["snapshot_id"],
            "parent_snapshot_id": manifest["parent_snapshot_id"],
            "manifest_created_at": manifest["created_at"],
            "manifest_sha256": manifest_hash,
            "source_count": len(manifest["sources"]),
            "excluded_document_ids": sorted(requested_ids),
            "coverage": manifest["coverage"],
        }

    def _build_index(self, case_id: str, sources: list[dict[str, Any]], index_path: Path) -> None:
        connection = sqlite3.connect(index_path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    citation_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_number INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    evidence_image TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            for row in self._expected_index_rows(case_id, sources):
                values = tuple(row[column] for column in _INDEX_COLUMNS)
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
                connection.execute(
                    "INSERT INTO chunk_fts(chunk_id, text) VALUES (?, ?)",
                    (row["chunk_id"], row["text"]),
                )
            connection.commit()
            connection.execute("PRAGMA optimize")
            connection.commit()
        except sqlite3.Error as exc:
            raise EvidenceError(f"could not build SQLite FTS5 index: {exc}") from exc
        finally:
            connection.close()

    def _expected_index_rows(
        self, case_id: str, sources: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        for source in sources:
            for page in source["pages"]:
                record = self.page_record(case_id, str(page["record_sha256"]))
                text = str(record["text"])
                for chunk_number, (start, end, chunk) in enumerate(chunk_text(text), 1):
                    citation_id = f"{source['document_id']}:p{page['page_number']}"
                    yield {
                        "chunk_id": f"{citation_id}:c{chunk_number}",
                        "citation_id": citation_id,
                        "document_id": source["document_id"],
                        "filename": source["filename"],
                        "source_sha256": source["source_sha256"],
                        "page_number": page["page_number"],
                        "chunk_number": chunk_number,
                        "start_offset": start,
                        "end_offset": end,
                        "text": chunk,
                        "extraction_method": page["extraction_method"],
                        "needs_review": 1 if page["needs_review"] else 0,
                        "evidence_image": page.get("evidence_image", ""),
                    }

    def _expected_index_content_sha256(self, case_id: str, sources: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for row in self._expected_index_rows(case_id, sources):
            digest.update(canonical_json(row).encode("utf-8"))
            digest.update(b"\n")
        digest.update(b"--fts--\n")
        for row in self._expected_index_rows(case_id, sources):
            digest.update(
                canonical_json({"chunk_id": row["chunk_id"], "text": row["text"]}).encode("utf-8")
            )
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _actual_index_content_sha256(index_path: Path) -> str:
        connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        digest = hashlib.sha256()
        try:
            metadata = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if metadata is None or metadata["value"] != str(SCHEMA_VERSION):
                raise IntegrityError("search index schema version is missing or unsupported")
            columns = ", ".join(_INDEX_COLUMNS)
            for row in connection.execute(
                f"SELECT {columns} FROM chunks "  # noqa: S608
                "ORDER BY document_id, page_number, chunk_number"
            ):
                value = {column: row[column] for column in _INDEX_COLUMNS}
                digest.update(canonical_json(value).encode("utf-8"))
                digest.update(b"\n")
            digest.update(b"--fts--\n")
            for row in connection.execute(
                "SELECT chunk_fts.chunk_id, chunk_fts.text "
                "FROM chunk_fts JOIN chunks ON chunks.chunk_id = chunk_fts.chunk_id "
                "ORDER BY chunks.document_id, chunks.page_number, chunks.chunk_number"
            ):
                value = {"chunk_id": row["chunk_id"], "text": row["text"]}
                digest.update(canonical_json(value).encode("utf-8"))
                digest.update(b"\n")
        finally:
            connection.close()
        return digest.hexdigest()

    def _snapshot_inventory(
        self,
        case_id: str,
        sources: list[dict[str, Any]],
        excluded_sources: list[dict[str, Any]] | None = None,
        *,
        writer_version: str = APP_VERSION,
    ) -> dict[str, Any]:
        normalized_sources: list[dict[str, Any]] = []
        for source in sorted(sources, key=lambda item: str(item.get("document_id", ""))):
            normalized_sources.append(
                {key: value for key, value in source.items() if key not in {"created_at"}}
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "app_version": writer_version,
            "case_id": case_id,
            "sources": normalized_sources,
            "excluded_sources": sorted(
                excluded_sources or [], key=lambda item: str(item.get("tombstone_id", ""))
            ),
        }

    @staticmethod
    def _coverage_with_exclusions(
        sources: list[dict[str, Any]], excluded_sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        coverage = Workspace._coverage_for(sources)
        coverage["documents_excluded"] = len(excluded_sources)
        coverage["excluded_documents"] = [
            {
                "document_id": item.get("document", {}).get("document_id", ""),
                "filename": item.get("document", {}).get("filename", ""),
                "source_sha256": item.get("document", {}).get("source_sha256", ""),
                "reason": item.get("reason", ""),
                "reviewer": item.get("reviewer", ""),
                "excluded_at": item.get("excluded_at", ""),
                "tombstone_id": item.get("tombstone_id", ""),
            }
            for item in excluded_sources
        ]
        return coverage

    @staticmethod
    def _coverage_for(sources: list[dict[str, Any]]) -> dict[str, Any]:
        pages = [
            page
            for source in sources
            if isinstance(source.get("pages"), list)
            for page in source["pages"]
            if isinstance(page, dict)
        ]
        readable = [page for page in pages if page.get("status") == "readable"]
        failed = [page for page in pages if page.get("status") == "failed"]
        incomplete = [
            page for page in pages if page.get("status") != "readable" or page.get("needs_review")
        ]
        methods: dict[str, int] = {}
        for page in pages:
            method = str(page.get("extraction_method", "unknown"))
            methods[method] = methods.get(method, 0) + 1
        failed_documents = [
            source
            for source in sources
            if source.get("status") != "indexed"
            or not isinstance(source.get("pages"), list)
            or not source.get("pages")
        ]
        return {
            "documents_total": len(sources),
            "documents_failed": len(failed_documents),
            "pages_total": len(pages),
            "pages_readable": len(readable),
            "pages_failed": len(failed),
            "pages_needing_review": sum(1 for page in pages if page.get("needs_review")),
            "complete_for_negative_assertions": (
                bool(sources) and bool(pages) and not failed_documents and not incomplete
            ),
            "extraction_methods": dict(sorted(methods.items())),
            "incomplete_documents": [
                {
                    "document_id": source.get("document_id", ""),
                    "filename": source.get("filename", ""),
                    "status": source.get("status", ""),
                    "error": source.get("error", ""),
                }
                for source in failed_documents
            ],
            "incomplete_pages": [
                {
                    "source_sha256": page.get("source_sha256", ""),
                    "page_number": page.get("page_number", 0),
                    "status": page.get("status", ""),
                    "quality_score": page.get("quality_score", 0),
                    "error": page.get("error", ""),
                }
                for page in incomplete
            ],
        }

    def _write_head(self, case_id: str, snapshot_id: str) -> None:
        _atomic_write(
            self._case_root(case_id) / _HEAD_NAME,
            f"{snapshot_id}\n".encode("ascii"),
            mode=0o600,
        )

    def _case_root(self, case_id: str) -> Path:
        chosen = validate_case_id(case_id)
        root = (self.cases_root / chosen).resolve()
        if not root.is_relative_to(self.cases_root):
            raise EvidenceError("case path escapes the workspace")
        return root

    def _snapshot_root(self, case_id: str, snapshot_id: str) -> Path:
        _require_sha256(snapshot_id, "snapshot")
        return self._case_root(case_id) / "snapshots" / snapshot_id

    @contextmanager
    def _audit_lock(self, lock_path: Path) -> Iterator[None]:
        with self._thread_lock:
            lock_path.touch(mode=0o600, exist_ok=True)
            with lock_path.open("r+") as stream:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvidenceError(f"{label} id must be a SHA-256 digest")
    return value
