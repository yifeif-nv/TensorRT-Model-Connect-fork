# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-protected loopback HTTP API and static browser workbench."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

from .chronology import ChronologyBuilder
from .export import EvidenceExporter
from .ingest import Ingestor
from .schema import EvidenceError, safe_filename
from .search import EvidenceSearch
from .store import Workspace


_CASE_ROUTE = re.compile(r"^/api/cases/(?P<case>[a-z0-9][a-z0-9-]{0,62})(?P<tail>/.*)?$")
_SOURCE_ROUTE = re.compile(r"^/source/(?P<document_id>[0-9a-f]{64})$")
_PAGE_IMAGE_ROUTE = re.compile(r"^/page-image/(?P<sha>[0-9a-f]{64})(?P<suffix>\.[A-Za-z0-9]+)$")
_PAGE_RECORD_ROUTE = re.compile(
    r"^/page-record/(?P<document_id>[0-9a-f]{64})/p(?P<page>[1-9][0-9]*)$"
)
_EXPORT_ROUTE = re.compile(r"^/export-file/(?P<sha>[0-9a-f]{64})[.]zip$")


def _open_verified_file(path: Path, expected_sha256: str) -> tuple[BinaryIO, int]:
    """Open, hash, and rewind one regular file without reopening its pathname."""

    if path.is_symlink():
        raise EvidenceError(f"refusing to serve a symlinked file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise EvidenceError(f"could not open download safely: {path}") from exc
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        initial = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial.st_mode):
            raise EvidenceError(f"download is not a regular file: {path}")
        fingerprint = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        )
        digest = hashlib.sha256()
        observed_size = 0
        while block := stream.read(1024 * 1024):
            digest.update(block)
            observed_size += len(block)
        final = os.fstat(stream.fileno())
        final_fingerprint = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if final_fingerprint != fingerprint or observed_size != initial.st_size:
            raise EvidenceError(f"download changed while it was verified: {path}")
        if digest.hexdigest() != expected_sha256:
            raise EvidenceError(f"download failed SHA-256 verification: {path}")
        stream.seek(0)
        return stream, initial.st_size
    except BaseException:
        stream.close()
        raise


class WorkbenchServer:
    def __init__(
        self,
        workspace: Workspace,
        ingestor: Ingestor,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        max_upload_bytes: int = 250 * 1024 * 1024,
        allow_remote: bool = False,
    ):
        if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
            raise EvidenceError(
                "non-loopback binding requires the explicit --allow-remote acknowledgement"
            )
        self.workspace = workspace
        self.ingestor = ingestor
        self.host = host
        self.port = int(port)
        self.token = token or secrets.token_urlsafe(32)
        self.max_upload_bytes = int(max_upload_bytes)
        self.allow_remote = allow_remote
        self._model_lock = threading.Lock()
        handler = _handler_factory(self)
        self.httpd = ThreadingHTTPServer((host, self.port), handler)
        self.httpd.daemon_threads = True

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _handler_factory(application: WorkbenchServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvidenceWorkbench/0.1"

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

        def log_message(self, format: str, *args: object) -> None:
            # Request paths may contain sensitive case information. Operators can
            # add a reverse-proxy access log deliberately when appropriate.
            del format, args

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            try:
                self._require_safe_host_and_origin(method)
                if parsed.path.startswith("/api/"):
                    self._require_auth()
                    self._handle_api(method, parsed.path)
                else:
                    self._handle_static(method, parsed.path)
            except _ResponseSent:
                return
            except EvidenceError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "resource not found"})
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary.
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"internal workbench error: {type(exc).__name__}"},
                )

        def _require_safe_host_and_origin(self, method: str) -> None:
            host = self.headers.get("Host", "")
            address_host, address_port = application.address
            allowed_hosts = {
                f"{address_host}:{address_port}",
                f"127.0.0.1:{address_port}",
                f"localhost:{address_port}",
                f"[::1]:{address_port}",
            }
            if application.allow_remote:
                allowed_hosts.add(f"{application.host}:{address_port}")
            if host not in allowed_hosts:
                self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "unrecognized Host header"})
                raise _ResponseSent
            origin = self.headers.get("Origin", "")
            if method == "POST" and origin:
                parsed_origin = urlparse(origin)
                if parsed_origin.scheme != "http" or parsed_origin.netloc != host:
                    self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin writes are refused"})
                    raise _ResponseSent

        def _require_auth(self) -> None:
            authorization = self.headers.get("Authorization", "")
            expected = f"Bearer {application.token}"
            if not secrets.compare_digest(authorization, expected):
                self._json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "missing or invalid Evidence Workbench token"},
                )
                raise _ResponseSent

        def _handle_api(self, method: str, path: str) -> None:
            if path == "/api/health" and method == "GET":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/cases":
                if method == "GET":
                    self._json(HTTPStatus.OK, {"cases": application.workspace.list_cases()})
                    return
                if method == "POST":
                    payload = self._json_body(32 * 1024)
                    case = application.workspace.create_case(
                        str(payload.get("name", "")),
                        str(payload["id"]) if payload.get("id") else None,
                    )
                    self._json(HTTPStatus.CREATED, case)
                    return

            route = _CASE_ROUTE.fullmatch(path)
            if route is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown API endpoint"})
                return
            case_id = route.group("case")
            tail = route.group("tail") or ""
            if tail == "" and method == "GET":
                case = application.workspace.get_case(case_id)
                if case["head_snapshot_id"]:
                    snapshot_id = case["head_snapshot_id"]
                    case["manifest"] = application.workspace.load_manifest(case_id, snapshot_id)
                    case["verification"] = application.workspace.verify(case_id, snapshot_id)
                    case["effective_coverage"] = application.workspace.effective_coverage(
                        case_id, case["manifest"]
                    )
                    case["page_reviews"] = application.workspace.reviews(
                        case_id, case["head_snapshot_id"], "page"
                    )
                self._json(HTTPStatus.OK, case)
                return
            if tail == "/verify" and method == "GET":
                self._json(HTTPStatus.OK, application.workspace.verify(case_id))
                return
            if tail == "/search" and method == "POST":
                payload = self._json_body(32 * 1024)
                try:
                    limit = int(payload.get("limit", 20))
                except (TypeError, ValueError) as exc:
                    raise EvidenceError("search limit must be an integer") from exc
                result = EvidenceSearch(application.workspace).search(
                    case_id,
                    str(payload.get("query", "")),
                    mode=str(payload.get("mode", "all")),
                    limit=limit,
                )
                self._json(HTTPStatus.OK, result)
                return
            if tail == "/chronology" and method == "POST":
                self._json(HTTPStatus.OK, ChronologyBuilder(application.workspace).build(case_id))
                return
            if tail == "/reviews/page" and method == "POST":
                payload = self._json_body(16 * 1024)
                result = application.workspace.record_review(
                    case_id,
                    snapshot_id=str(payload.get("snapshot_id", "")),
                    target_type="page",
                    target_id=str(payload.get("target_id", "")),
                    status=str(payload.get("status", "")),
                    reviewer=str(payload.get("reviewer", "")),
                    notes=str(payload.get("notes", "")),
                    expected_target_sha256=str(payload.get("record_sha256", "")),
                )
                self._json(HTTPStatus.CREATED, result)
                return
            if tail == "/reviews/event" and method == "POST":
                payload = self._json_body(16 * 1024)
                event_id = str(payload.get("target_id", ""))
                snapshot_id = application.workspace.head_snapshot_id(case_id)
                chronology = ChronologyBuilder(application.workspace).build(case_id, snapshot_id)
                if event_id not in {event["event_id"] for event in chronology["events"]}:
                    raise EvidenceError(f"unknown chronology event: {event_id}")
                result = application.workspace.record_review(
                    case_id,
                    snapshot_id=snapshot_id,
                    target_type="chronology_event",
                    target_id=event_id,
                    status=str(payload.get("status", "")),
                    reviewer=str(payload.get("reviewer", "")),
                    notes=str(payload.get("notes", "")),
                )
                self._json(HTTPStatus.CREATED, result)
                return
            if tail == "/documents" and method == "POST":
                self._upload(case_id)
                return
            if tail == "/documents/exclude" and method == "POST":
                payload = self._json_body(16 * 1024)
                result = application.workspace.exclude_document(
                    case_id,
                    str(payload.get("document_id", "")),
                    reviewer=str(payload.get("reviewer", "")),
                    reason=str(payload.get("reason", "")),
                )
                self._json(HTTPStatus.CREATED, result)
                return
            if tail == "/export" and method == "POST":
                payload = self._json_body(64 * 1024)
                manifest = application.workspace.load_manifest(case_id)
                output = (
                    application.workspace.root
                    / "cases"
                    / case_id
                    / "exports"
                    / f"{manifest['snapshot_id'][:16]}-{secrets.token_hex(6)}"
                )
                result = EvidenceExporter(application.workspace).export_all(
                    case_id,
                    output,
                    include_originals=bool(payload.get("include_originals", True)),
                    draft=bool(payload.get("draft", False)),
                )
                result["download_url"] = (
                    f"/api/cases/{case_id}/export-file/{result['audit_bundle_sha256']}.zip"
                )
                self._json(HTTPStatus.OK, result)
                return
            export_route = _EXPORT_ROUTE.fullmatch(tail)
            if export_route and method == "GET":
                expected_hash = export_route.group("sha")
                exports_root = application.workspace.root / "cases" / case_id / "exports"
                opened: tuple[BinaryIO, int] | None = None
                for candidate in exports_root.glob("*/evidence-audit-bundle.zip"):
                    try:
                        if not candidate.resolve().is_relative_to(exports_root.resolve()):
                            continue
                        opened = _open_verified_file(candidate, expected_hash)
                        break
                    except (EvidenceError, FileNotFoundError):
                        continue
                if opened is None:
                    raise FileNotFoundError(expected_hash)
                stream, size = opened
                try:
                    self._stream_file(
                        stream,
                        size,
                        content_type="application/zip",
                        download_name=f"{case_id}-{expected_hash[:12]}.zip",
                        attachment=True,
                    )
                finally:
                    stream.close()
                return
            source_route = _SOURCE_ROUTE.fullmatch(tail)
            if source_route and method == "GET":
                document_id = source_route.group("document_id")
                manifest = application.workspace.load_manifest(case_id)
                all_sources = [
                    *manifest["sources"],
                    *(tombstone["document"] for tombstone in manifest.get("excluded_sources", [])),
                ]
                source = next(
                    (item for item in all_sources if item["document_id"] == document_id),
                    None,
                )
                if source is None:
                    raise FileNotFoundError(document_id)
                source_hash = str(source["source_sha256"])
                source_path = application.workspace.source_path(case_id, source_hash)
                active_type = str(source["media_type"]) in {
                    "text/html",
                    "text/csv",
                    "application/json",
                }
                self._file(
                    source_path,
                    expected_sha256=source_hash,
                    content_type=(
                        "application/octet-stream" if active_type else str(source["media_type"])
                    ),
                    download_name=str(source["filename"]),
                    attachment=active_type,
                )
                return
            image_route = _PAGE_IMAGE_ROUTE.fullmatch(tail)
            if image_route and method == "GET":
                relative = (
                    Path("page_images") / f"{image_route.group('sha')}{image_route.group('suffix')}"
                )
                image_path = application.workspace.root / "cases" / case_id / relative
                if not image_path.is_file():
                    raise FileNotFoundError(str(relative))
                self._file(
                    image_path,
                    expected_sha256=image_route.group("sha"),
                    content_type=mimetypes.guess_type(image_path.name)[0],
                )
                return
            record_route = _PAGE_RECORD_ROUTE.fullmatch(tail)
            if record_route and method == "GET":
                document_id = record_route.group("document_id")
                page_number = int(record_route.group("page"))
                manifest = application.workspace.load_manifest(case_id)
                documents = [
                    *manifest["sources"],
                    *(item["document"] for item in manifest.get("excluded_sources", [])),
                ]
                source = next(
                    (item for item in documents if item["document_id"] == document_id),
                    None,
                )
                if source is None:
                    raise FileNotFoundError(document_id)
                page = next(
                    (item for item in source["pages"] if item["page_number"] == page_number),
                    None,
                )
                if page is None:
                    raise FileNotFoundError(f"{document_id}:p{page_number}")
                record = application.workspace.page_record(case_id, page["record_sha256"])
                self._json(
                    HTTPStatus.OK,
                    {
                        "document_id": document_id,
                        "filename": source["filename"],
                        "citation_id": f"{document_id}:p{page_number}",
                        "snapshot_id": manifest["snapshot_id"],
                        "record_sha256": page["record_sha256"],
                        "record": record,
                    },
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown API endpoint"})

        def _upload(self, case_id: str) -> None:
            application.workspace.get_case(case_id)
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise EvidenceError("upload requires a valid Content-Length") from exc
            if length < 1 or length > application.max_upload_bytes:
                raise EvidenceError(
                    f"upload length must be between 1 and {application.max_upload_bytes} bytes"
                )
            raw_filename = self.headers.get("X-Filename", "")
            if len(raw_filename) > 512:
                raise EvidenceError("upload filename exceeds 512 characters")
            filename = safe_filename(raw_filename)
            staging = application.workspace.root / "cases" / case_id / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="upload-", suffix=Path(filename).suffix, dir=staging
            )
            temporary = Path(temporary_name)
            remaining = length
            try:
                with os.fdopen(descriptor, "wb") as output:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise EvidenceError("upload ended before Content-Length bytes arrived")
                        output.write(chunk)
                        remaining -= len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                with application._model_lock:
                    result = application.ingestor.ingest(
                        case_id, temporary, display_filename=raw_filename
                    )
                self._json(HTTPStatus.CREATED, result)
            finally:
                temporary.unlink(missing_ok=True)

        def _json_body(self, maximum: int) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise EvidenceError("JSON request requires Content-Length") from exc
            if length < 0 or length > maximum:
                raise EvidenceError(f"JSON request exceeds {maximum} bytes")
            try:
                value = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                raise EvidenceError("request body is not valid JSON") from exc
            if not isinstance(value, dict):
                raise EvidenceError("request body must be a JSON object")
            return value

        def _handle_static(self, method: str, path: str) -> None:
            if method != "GET":
                self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
                return
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            if relative not in {"index.html", "app.js", "styles.css"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            static_root = resources.files("evidence_workbench").joinpath("static")
            asset = static_root.joinpath(relative)
            data = asset.read_bytes()
            content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self._security_headers(
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' blob: data:; frame-src blob:; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self';"
            )
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: HTTPStatus, value: Any) -> None:
            data = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
            self.send_response(status)
            self._security_headers("default-src 'none'; frame-ancestors 'none'")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _file(
            self,
            path: Path,
            *,
            expected_sha256: str,
            content_type: str | None = None,
            download_name: str | None = None,
            attachment: bool = False,
        ) -> None:
            stream, size = _open_verified_file(path, expected_sha256)
            try:
                self._stream_file(
                    stream,
                    size,
                    content_type=content_type,
                    download_name=download_name,
                    attachment=attachment,
                )
            finally:
                stream.close()

        def _stream_file(
            self,
            stream: BinaryIO,
            size: int,
            *,
            content_type: str | None = None,
            download_name: str | None = None,
            attachment: bool = False,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self._security_headers("default-src 'none'; frame-ancestors 'none'")
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            if download_name:
                self.send_header(
                    "Content-Disposition",
                    f"{'attachment' if attachment else 'inline'}; "
                    f"filename={json.dumps(safe_filename(download_name))}",
                )
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

        def _security_headers(self, content_security_policy: str) -> None:
            self.send_header("Content-Security-Policy", content_security_policy)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")

    return Handler


class _ResponseSent(Exception):
    pass
