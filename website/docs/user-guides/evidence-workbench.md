---
title: Build a Local Evidence Workbench
description: Ingest PDFs and images with Model Connect OCR, verify content-addressed snapshots, search exact cited text, and export review artifacts.
---

Evidence Workbench is a standalone reference application under
`examples/evidence_workbench/`. It demonstrates how a product can use a Model
Connect OCR bundle without making generated text the authority for evidence or
coverage decisions.

## What the example proves

- PDF pages prefer native text and use Model Connect OCR only as a fallback.
- Original documents, page records, rendered OCR pages, and indexes are hashed.
- Search results point to an exact source, page, quote, and character range.
- Negative exact-phrase results fail closed when any page is incomplete or
  review-required.
- Chronology dates come from indexed text; ambiguous numeric dates are not
  silently normalized.
- Human page and chronology decisions are appended to the local audit chain;
  final exports are blocked while required review remains unresolved.
- Mistaken or superseded sources can be excluded only by creating a new
  snapshot with an authenticated reviewer/reason tombstone; reports retain and
  disclose excluded evidence separately.
- Exports include reviewable HTML, CSV, DOCX, XLSX, JSON, exact page records,
  and an unsigned SHA-256 ZIP receipt.

The example does not claim that a citation proves a legal or medical
conclusion. OCR pages remain review-required because the current image runtime
does not expose calibrated OCR confidence or token-budget completion metadata.

## Build the OCR model

```bash
trtmc build deepseek-ai/DeepSeek-OCR-2 \
  -o deepseek-ocr.bundle \
  --trust-remote-code \
  --precision fp16 \
  --max-cache-length 4096 \
  --fp32-layers 6,7,8,9,10,11,12
```

Review the remote checkpoint code before enabling `--trust-remote-code`.

## Install and run the example

From the repository root:

```bash
python3 -m venv .venv-evidence
.venv-evidence/bin/python -m pip install -e \
  'examples/evidence_workbench[test]'

.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data \
  create-case "Example matter" --id example-matter

.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data \
  ingest example-matter ./records.pdf \
  --ocr-bundle ./deepseek-ocr.bundle \
  --trtmc-binary ./build/trtmc
```

Verify and search:

```bash
.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data verify example-matter

.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data search example-matter \
  "follow-up imaging" --mode phrase
```

Build the chronology and record the review IDs returned by the commands:

```bash
.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data chronology example-matter

.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data review-page example-matter \
  <document-id>:p1 --record-sha256 <inspected-page-record-sha256> \
  --status accepted --reviewer "A. Reviewer"

.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data review-event example-matter \
  <event-id> --status accepted --reviewer "A. Reviewer"
```

The `--record-sha256` value comes from the page entry in the ingest JSON or
current snapshot manifest and binds acceptance to the inspected indexed text.

Use the browser page-review dialog to compare the rendered evidence, full
indexed text, deterministic quality signals, and exact page-record hash before
accepting an OCR page. An unchanged record keeps its review decision when an
unrelated document creates a later snapshot.

Exclude a wrong or superseded upload without deleting it from history:

```bash
.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data exclude-document example-matter \
  <document-id> --reviewer "A. Reviewer" \
  --reason "Superseded by the corrected signed exhibit"
```

Search and final reports disclose excluded-source counts and identities. The
export retains authenticated tombstones, page records, and (when originals are
included) excluded source objects under a separate path.

Use `export --draft` while reviews remain pending. A final export fails closed
until review-required readable pages are accepted and every chronology event
has an accepted or rejected decision.

Run the local browser interface:

```bash
.venv-evidence/bin/evidence-workbench \
  --workspace ./evidence-data serve \
  --ocr-bundle ./deepseek-ocr.bundle \
  --trtmc-binary ./build/trtmc
```

The process prints a loopback URL and random bearer token. The API requires
that token even on localhost. Search uses POST bodies so sensitive queries do
not enter access logs, and the UI downloads a fresh draft or final audit ZIP.

## Interpret negative results

| Status | Contract |
| --- | --- |
| `NOT_PRESENT_IN_INDEXED_TEXT` | An exact normalized phrase was absent and every page supported a negative assertion. |
| `NO_VERIFIED_MATCH` | A broader lexical query found no deterministic match; this is not an absence claim. |
| `COVERAGE_INCOMPLETE` | At least one page is failed, empty, or still awaiting required acceptance, so absence cannot be asserted. |

The local audit head detects accidental truncation but is not a digital
signature or WORM store. Retain or sign exported heads externally when a
filesystem writer is inside the threat model.

Read the example's [README](https://github.com/NVIDIA/TensorRT-Model-Connect/tree/main/examples/evidence_workbench)
for the full storage, export, security, and performance boundaries.
