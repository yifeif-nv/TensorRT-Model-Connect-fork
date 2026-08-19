# Evidence Workbench validation

Validation is split into evidence tiers so CPU orchestration tests are not
mistaken for GPU/model-quality proof.

## Tier 1: deterministic CPU contracts

```bash
.venv-evidence/bin/python -m pytest examples/evidence_workbench/tests -q
ruff check examples/evidence_workbench
ruff format --check examples/evidence_workbench
.venv-evidence/bin/python -m compileall -q \
  examples/evidence_workbench/src \
  examples/evidence_workbench/tests
```

The tests cover:

- native and OCR snapshot determinism, duplicate reuse, and byte-identical
  source aliases with distinct evidence identities and human citations
- source, page-record, derived-coverage, SQLite/FTS index, and audit
  modification/deletion/truncation detection
- symlink rejection, page/frame/pixel bounds, and upload-size enforcement
- exact Model Connect subprocess arguments, errors, and timeouts
- OCR failure retention and coverage gating
- exhaustive phrase search beyond the former candidate cap, whole-token
  all/any search, and coverage-aware negative-result boundaries
- ambiguous-date chronology with exact character-offset quotes, stable event
  IDs, alias separation, and append-only review decisions
- PDFium scanned-PDF rendering into the Model Connect OCR path
- mixed native-text/scanned PDF detection (including small embedded images)
  and multi-frame TIFF handling
- HTML escaping, active-original download behavior, spreadsheet-formula
  neutralization, private file modes, exact manifest/page-record export, final
  review gating, and unsigned ZIP hash receipts
- bearer-token, Host/Origin, POST-search, repeated export/download, and
  end-to-end loopback upload/search/source behavior
- review carry-forward for unchanged page records, audited source exclusion,
  crash recovery, and retained/disclosed exclusion tombstones

The test executable that emulates `trtmc` proves orchestration only. It is not
model accuracy or GPU runtime evidence.

## Tier 2: repository checks

```bash
python3 tools/legal_headers.py --check
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
git diff --check
```

Run the website build after changing the user guide or sidebar:

```bash
npm --prefix website ci
npm --prefix website run build
```

## Tier 3: real Model Connect OCR

Build the exact bundle from [README.md](README.md), then ingest a PDF containing
both native-text and scanned pages. Acceptance requires:

- `trtmc inspect` identifies the DeepSeek-OCR family/runtime.
- Native-text pages use `pdf_native_text` and do not invoke OCR.
- Scanned pages use `model_connect_ocr`, retain a rendered page image, and stay
  marked `needs_review`.
- A failed or timed-out OCR page appears in coverage and prevents
  `NOT_PRESENT_IN_INDEXED_TEXT`.
- Search citations open the correct original PDF page or rendered OCR page.
- `verify` passes after ingestion and after export.
- A final export remains blocked until OCR pages are accepted and chronology
  events are accepted or rejected; `--draft` remains clearly marked.

Record the GPU, TensorRT/CUDA versions, bundle SHA-256, source fixture hashes,
commands, elapsed time, and output artifact path. Do not promote a single
document smoke to broad OCR accuracy or throughput evidence.

## Explicitly unproven without Tier 3

- DeepSeek-OCR accuracy on a customer corpus
- dense-page legibility after the model's fixed image preprocessing
- multi-thousand-page ingestion throughput
- portability to an unqualified GPU, TensorRT, CUDA, or plugin combination
- legal, medical, or regulatory fitness of generated work product
