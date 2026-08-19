# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the standalone Evidence Workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .chronology import ChronologyBuilder
from .export import EvidenceExporter
from .ingest import Ingestor
from .schema import APP_VERSION, EvidenceError
from .search import EvidenceSearch, SEARCH_MODES
from .server import WorkbenchServer
from .store import Workspace
from .trtmc import TrtmcRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence-workbench",
        description=(
            "Build content-addressed document snapshots, run Model Connect OCR, "
            "and produce page-cited search, chronology, and review exports."
        ),
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("EVIDENCE_WORKBENCH_HOME", "./evidence-workspace"),
        help="Local workspace directory (default: ./evidence-workspace)",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-case", help="Create an empty local case")
    create.add_argument("name", help="Human-readable case name")
    create.add_argument("--id", dest="case_id", help="Stable lowercase case id")

    subparsers.add_parser("cases", help="List local cases")

    ingest = subparsers.add_parser("ingest", help="Archive and index local files")
    ingest.add_argument("case_id")
    ingest.add_argument("files", nargs="+")
    _add_ocr_options(ingest)
    ingest.add_argument(
        "--max-source-mib", type=int, default=250, help="Per-file upload safety limit"
    )
    ingest.add_argument(
        "--pdf-render-scale", type=float, default=2.0, help="PDFium OCR render scale"
    )

    verify = subparsers.add_parser("verify", help="Verify hashes and the audit chain")
    verify.add_argument("case_id")
    verify.add_argument("--snapshot", default=None)

    search = subparsers.add_parser("search", help="Search verified indexed text")
    search.add_argument("case_id")
    search.add_argument("query")
    search.add_argument("--mode", choices=sorted(SEARCH_MODES), default="all")
    search.add_argument("--limit", type=int, default=20)

    chronology = subparsers.add_parser("chronology", help="Extract dates with exact page context")
    chronology.add_argument("case_id")

    review_page = subparsers.add_parser(
        "review-page", help="Record a human decision for a review-required page"
    )
    _add_review_options(review_page, "citation_id")
    review_page.add_argument(
        "--record-sha256",
        required=True,
        help="Exact page-record SHA-256 inspected by the reviewer",
    )

    review_event = subparsers.add_parser(
        "review-event", help="Accept or reject a deterministic chronology event"
    )
    _add_review_options(review_event, "event_id")

    exclude = subparsers.add_parser(
        "exclude-document",
        help="Create a new snapshot that excludes one mistaken or superseded source",
    )
    exclude.add_argument("case_id")
    exclude.add_argument("document_id")
    exclude.add_argument("--reviewer", required=True)
    exclude.add_argument("--reason", required=True)

    export = subparsers.add_parser(
        "export", help="Create HTML, CSV, DOCX, XLSX, JSON, and ZIP artifacts"
    )
    export.add_argument("case_id")
    export.add_argument("--output", required=True, dest="output_directory")
    export.add_argument("--no-originals", action="store_true")
    export.add_argument(
        "--draft",
        action="store_true",
        help="Allow an explicitly marked draft export while reviews remain pending",
    )

    serve = subparsers.add_parser("serve", help="Run the token-protected local browser app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token", default=os.environ.get("EVIDENCE_WORKBENCH_TOKEN", ""))
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="Acknowledge exposure when binding outside loopback",
    )
    serve.add_argument("--max-upload-mib", type=int, default=250)
    _add_ocr_options(serve)
    return parser


def _add_ocr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ocr-bundle",
        help="DeepSeek-OCR or compatible Model Connect .bundle artifact",
    )
    parser.add_argument("--trtmc-binary", help="Path to the native trtmc executable")
    parser.add_argument("--hf-python", help="Tokenizer/helper Python interpreter")
    parser.add_argument("--model-timeout", type=float, default=300.0)
    parser.add_argument("--ocr-max-new-tokens", type=int, default=800)


def _add_review_options(parser: argparse.ArgumentParser, target_name: str) -> None:
    parser.add_argument("case_id")
    parser.add_argument(target_name)
    parser.add_argument("--status", choices=["accepted", "rejected", "unreviewed"], required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Workspace(args.workspace)
    try:
        result = _dispatch(args, workspace)
    except KeyboardInterrupt:
        print("Evidence Workbench stopped.", file=sys.stderr)
        return 130
    except EvidenceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if result is not None:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=None if args.compact else 2,
            )
        )
    if args.command == "verify" and result is not None and not result.get("ok", False):
        return 1
    return 0


def _dispatch(args: argparse.Namespace, workspace: Workspace) -> Any:
    if args.command == "create-case":
        return workspace.create_case(args.name, args.case_id)
    if args.command == "cases":
        return {"cases": workspace.list_cases()}
    if args.command == "verify":
        return workspace.verify(args.case_id, args.snapshot)
    if args.command == "search":
        return EvidenceSearch(workspace).search(
            args.case_id, args.query, mode=args.mode, limit=args.limit
        )
    if args.command == "chronology":
        return ChronologyBuilder(workspace).build(args.case_id)
    if args.command == "review-page":
        snapshot_id = workspace.head_snapshot_id(args.case_id)
        return workspace.record_review(
            args.case_id,
            snapshot_id=snapshot_id,
            target_type="page",
            target_id=args.citation_id,
            status=args.status,
            reviewer=args.reviewer,
            notes=args.notes,
            expected_target_sha256=args.record_sha256,
        )
    if args.command == "review-event":
        snapshot_id = workspace.head_snapshot_id(args.case_id)
        chronology = ChronologyBuilder(workspace).build(args.case_id, snapshot_id)
        if args.event_id not in {event["event_id"] for event in chronology["events"]}:
            raise EvidenceError(f"unknown chronology event: {args.event_id}")
        return workspace.record_review(
            args.case_id,
            snapshot_id=snapshot_id,
            target_type="chronology_event",
            target_id=args.event_id,
            status=args.status,
            reviewer=args.reviewer,
            notes=args.notes,
        )
    if args.command == "exclude-document":
        return workspace.exclude_document(
            args.case_id,
            args.document_id,
            reviewer=args.reviewer,
            reason=args.reason,
        )
    if args.command == "export":
        return EvidenceExporter(workspace).export_all(
            args.case_id,
            args.output_directory,
            include_originals=not args.no_originals,
            draft=args.draft,
        )
    if args.command == "ingest":
        runner = _ocr_runner(args)
        ingestor = Ingestor(
            workspace,
            ocr_runner=runner,
            max_source_bytes=args.max_source_mib * 1024 * 1024,
            pdf_render_scale=args.pdf_render_scale,
            ocr_max_new_tokens=args.ocr_max_new_tokens,
        )
        results = ingestor.ingest_many(args.case_id, [Path(item) for item in args.files])
        return {
            "case_id": args.case_id,
            "documents": [result["document"] for result in results],
            "snapshot_id": results[-1]["snapshot"]["snapshot_id"],
            "coverage": results[-1]["snapshot"]["coverage"],
        }
    if args.command == "serve":
        runner = _ocr_runner(args)
        ingestor = Ingestor(
            workspace,
            ocr_runner=runner,
            max_source_bytes=args.max_upload_mib * 1024 * 1024,
            ocr_max_new_tokens=args.ocr_max_new_tokens,
        )
        server = WorkbenchServer(
            workspace,
            ingestor,
            host=args.host,
            port=args.port,
            token=args.token or None,
            max_upload_bytes=args.max_upload_mib * 1024 * 1024,
            allow_remote=args.allow_remote,
        )
        host, port = server.address
        print(f"Evidence Workbench: http://{host}:{port}/", file=sys.stderr)
        print(f"Bearer token: {server.token}", file=sys.stderr)
        print("Documents and inference remain local to this process.", file=sys.stderr)
        try:
            server.serve_forever()
        finally:
            server.shutdown()
        return None
    raise EvidenceError(f"unknown command: {args.command}")


def _ocr_runner(args: argparse.Namespace) -> TrtmcRunner | None:
    if not args.ocr_bundle:
        return None
    return TrtmcRunner(
        args.ocr_bundle,
        binary=args.trtmc_binary,
        hf_python=args.hf_python,
        timeout=args.model_timeout,
    )
