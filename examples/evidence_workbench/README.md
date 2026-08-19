# Evidence Workbench

Evidence Workbench is a local reference application for turning PDFs, text
files, and page images into content-addressed evidence snapshots. It uses a
TensorRT Model Connect vision-language bundle for OCR when a PDF page has no
usable native text, then provides deterministic search, cited chronology, hash
verification, and reviewable exports.

It is deliberately not a generic “chat with PDF” wrapper. Model-generated OCR
can add text to the index, but it cannot override integrity or coverage state.

## Evidence contract

The application provides these guarantees:

- Original sources, extracted page records, and rendered OCR pages are stored
  by SHA-256.
- Every cumulative evidence inventory receives a content-addressed snapshot ID;
  published snapshot directories are never overwritten.
- Search citations contain the source hash, page number, exact stored quote,
  and character offsets.
- An exact-phrase negative result is returned only when the snapshot verifies
  and every page is readable and either did not require review or has an
  accepted page-review decision in the audit chain.
- Ambiguous numeric dates remain ambiguous in the chronology.
- HTML, CSV, JSON, DOCX, XLSX, and ZIP exports carry the source snapshot and
  verification state. The ZIP includes a per-file hash receipt.
- Audit events form a locally anchored hash chain. `AUDIT_HEAD.json` detects
  accidental deletion and prefix truncation, but a filesystem writer can
  rewrite both files. Retain or sign exported audit heads externally when an
  adversarial tamper boundary is required.

The application does **not** claim that OCR is error-free, that a cited quote
semantically proves a legal or medical conclusion, or that lexical search can
prove semantic absence. Human review remains required for OCR-derived pages
and consequential work product.

Multi-frame TIFF files are expanded into distinct pages. Per-frame,
per-document, page-count, source-byte, and cumulative rendered-pixel limits
fail closed before unbounded OCR work is scheduled.

If a mistaken, corrupt, or superseded upload would otherwise block coverage,
an authorized reviewer can exclude that source in a new snapshot. Exclusion
does not erase evidence: the new manifest authenticates a tombstone with the
source identity, reason, reviewer, and prior snapshot, and exports retain the
excluded records in a visibly separate section.

## Prerequisites

1. Build and install TensorRT Model Connect by following the repository
   [source-build guide](../../website/docs/getting-started/source-build.md).
2. Use a supported NVIDIA GPU/runtime combination for the model bundle.
3. Use Python 3.10 or newer for the standalone application.

Create the application environment:

```bash
python3 -m venv .venv-evidence
.venv-evidence/bin/python -m pip install -e 'examples/evidence_workbench[test]'
```

The application dependencies are isolated from the TensorRT Model Connect
builder package. PDF rendering uses the liberal-licensed PDFium binding
`pypdfium2`; Office exports use `python-docx` and `openpyxl`.

## Build the OCR bundle

The checked-in DeepSeek-OCR model contract uses this configuration:

```bash
trtmc build deepseek-ai/DeepSeek-OCR-2 \
  -o deepseek-ocr.bundle \
  --trust-remote-code \
  --precision fp16 \
  --max-cache-length 4096 \
  --fp32-layers 6,7,8,9,10,11,12
```

Review remote checkpoint code before enabling `--trust-remote-code`. The
exact qualified model and test contract live in
`tests/e2e/models/deepseek_ocr/manifests/deepseek-ocr.json`.

## Command workflow

Create and populate a case:

```bash
evidence-workbench --workspace ./evidence-data \
  create-case "Example matter" --id example-matter

evidence-workbench --workspace ./evidence-data ingest example-matter \
  ./records/intake.pdf ./records/notes.txt \
  --ocr-bundle ./deepseek-ocr.bundle \
  --trtmc-binary ./build/trtmc
```

Verify before using the index:

```bash
evidence-workbench --workspace ./evidence-data verify example-matter
```

Search exact indexed text:

```bash
evidence-workbench --workspace ./evidence-data search example-matter \
  "follow-up imaging" --mode phrase
```

Search returns one of these explicit states:

| Status | Meaning |
| --- | --- |
| `MATCHES_FOUND` | Cited matches were found and full-page coverage is complete. |
| `MATCHES_FOUND_COVERAGE_INCOMPLETE` | Matches exist, but some other pages remain failed or review-required. |
| `NOT_PRESENT_IN_INDEXED_TEXT` | An exact normalized phrase is absent from a fully verified text index. |
| `NO_VERIFIED_MATCH` | A broader all/any-token query has no deterministic match. This is not an absence claim. |
| `COVERAGE_INCOMPLETE` | No match was found and at least one page cannot support a negative assertion. |

Build the source-derived chronology. Its JSON contains stable `event_id`
values and every review-required page has a stable citation ID such as
`<document-id>:p1`:

```bash
evidence-workbench --workspace ./evidence-data chronology example-matter
```

Record review decisions without rewriting the evidence snapshot:

```bash
evidence-workbench --workspace ./evidence-data review-page example-matter \
  <document-id>:p1 --record-sha256 <inspected-page-record-sha256> \
  --status accepted --reviewer "A. Reviewer" \
  --notes "Compared with the rendered page"

evidence-workbench --workspace ./evidence-data review-event example-matter \
  <event-id> --status accepted --reviewer "A. Reviewer"
```

Use the page's `record_sha256` from the ingest JSON or current
`snapshots/<snapshot-id>/manifest.json`; that binds the decision to the exact
text the reviewer inspected.

For OCR pages, compare the rendered evidence against the full indexed text and
record hash before accepting the page. The browser interface presents this
side-by-side and will not save a page decision until that exact record has been
opened. Reviews for unchanged record hashes carry forward when unrelated
documents create a new snapshot.

Exclude a mistaken or superseded source without rewriting prior snapshots:

```bash
evidence-workbench --workspace ./evidence-data exclude-document example-matter \
  <document-id> --reviewer "A. Reviewer" \
  --reason "Superseded by the corrected signed exhibit"
```

The active index no longer searches an excluded source. Every search response
states the excluded-source count, and final reports and bundles disclose and
retain the authenticated tombstones. Re-ingesting the same source alias
reactivates it in a later snapshot.

Final exports are blocked until all failed/incomplete pages are resolved, every
review-required readable page is accepted, and every chronology event is
accepted or rejected. Create an explicitly watermarked draft when review is
still in progress:

```bash
evidence-workbench --workspace ./evidence-data export example-matter \
  --output ./review/example-matter-draft --draft

evidence-workbench --workspace ./evidence-data export example-matter \
  --output ./review/example-matter-final
```

The export contains:

- `report.html` and `report.docx`
- `chronology.csv`, `chronology.json`, and `chronology.xlsx`
- `coverage.csv` and `coverage.json`
- `excluded-sources.csv` and prominent excluded-source tables
- the exact snapshot manifest and verification result
- exact content-addressed page records and the original `manifest.sha256`
- original source files by default
- an unsigned SHA-256 `bundle-receipt.json` and
  `evidence-audit-bundle.zip`

Use `--no-originals` only when the recipient already has separately verified
source objects.

## Local browser application

```bash
evidence-workbench --workspace ./evidence-data serve \
  --ocr-bundle ./deepseek-ocr.bundle \
  --trtmc-binary ./build/trtmc
```

The server prints its loopback URL and a random bearer token. Paste that token
into the browser page. API routes require the token even on localhost. The UI
shows rendered OCR evidence beside the exact indexed text and record hash,
supports page/event review and audited source exclusion, and downloads fresh
draft or final ZIP exports directly. Search requests use authenticated JSON
bodies rather than URLs so queries do not enter access logs. Binding outside
loopback is refused unless `--allow-remote` is present; that flag is an
acknowledgement, not TLS or an enterprise authentication system.

## Local storage

```text
evidence-data/
  cases/<case-id>/
    objects/<source-sha256>
    page_records/<record-sha256>.json
    page_images/<image-sha256>.png
    snapshots/<snapshot-id>/
      manifest.json
      manifest.sha256
      index.sqlite
    audit.jsonl
    AUDIT_HEAD.json
    exports/
```

Snapshots are never overwritten. Re-ingesting an identical evidence inventory
reuses and verifies the existing snapshot.

## Current scaling boundary

The public Python and CLI inference surfaces load a bundle for each subprocess.
Evidence Workbench caches extracted pages permanently, but initial OCR of many
scanned pages still pays that process/load cost per page. A persistent native
JSONL worker is the appropriate future throughput optimization. This reference
application does not hide that limitation or make unmeasured throughput claims.

See [VALIDATION.md](VALIDATION.md) for the tested evidence tiers and unrun
boundaries.
