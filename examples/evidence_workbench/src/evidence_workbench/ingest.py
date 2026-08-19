# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document ingestion with native-text preference and Model Connect OCR fallback."""

from __future__ import annotations

import html
import importlib.metadata
import json
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from .schema import EvidenceError, OptionalDependencyError, PageInput, normalize_text
from .store import Workspace
from .trtmc import TrtmcRunner, runtime_result_dict


_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".html", ".htm"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


class Ingestor:
    """Convert supported local files into immutable evidence pages."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        ocr_runner: TrtmcRunner | None = None,
        max_source_bytes: int = 250 * 1024 * 1024,
        pdf_render_scale: float = 2.0,
        ocr_max_new_tokens: int = 800,
        max_pdf_pages: int = 10_000,
        max_rendered_pixels: int = 40_000_000,
        max_image_frames: int = 1_000,
        max_document_rendered_pixels: int = 200_000_000,
    ):
        self.workspace = workspace
        self.ocr_runner = ocr_runner
        self.max_source_bytes = int(max_source_bytes)
        self.pdf_render_scale = float(pdf_render_scale)
        self.ocr_max_new_tokens = int(ocr_max_new_tokens)
        self.max_pdf_pages = int(max_pdf_pages)
        self.max_rendered_pixels = int(max_rendered_pixels)
        self.max_image_frames = int(max_image_frames)
        self.max_document_rendered_pixels = int(max_document_rendered_pixels)
        if self.max_source_bytes < 1:
            raise EvidenceError("max_source_bytes must be positive")
        if self.pdf_render_scale <= 0:
            raise EvidenceError("pdf_render_scale must be positive")
        if (
            min(
                self.max_pdf_pages,
                self.max_rendered_pixels,
                self.max_image_frames,
                self.max_document_rendered_pixels,
            )
            < 1
        ):
            raise EvidenceError("page, frame, and rendered-pixel limits must be positive")

    def ingest(
        self,
        case_id: str,
        source: str | Path,
        *,
        display_filename: str | None = None,
    ) -> dict[str, Any]:
        source_path = Path(source).expanduser()
        if source_path.is_symlink():
            raise EvidenceError(f"symlinked inputs are not accepted: {source_path}")
        if not source_path.is_file():
            raise EvidenceError(f"input is not a regular file: {source_path}")
        size = source_path.stat().st_size
        if size > self.max_source_bytes:
            raise EvidenceError(
                f"input exceeds the {self.max_source_bytes}-byte safety limit: {source_path}"
            )

        suffix = source_path.suffix.lower()
        # Freeze the exact bytes first. All extraction below reads only from the
        # content-addressed object, so page text cannot race a changing caller path.
        _source_sha256, archived_source = self.workspace.archive_source(case_id, source_path)
        if archived_source.stat().st_size > self.max_source_bytes:
            raise EvidenceError("archived source exceeds the configured safety limit")
        document_error = ""
        document_status = "indexed"
        with tempfile.TemporaryDirectory(prefix="evidence-ingest-") as temporary_directory:
            try:
                if suffix == ".pdf":
                    pages, extraction = self._pdf_pages(archived_source, Path(temporary_directory))
                elif suffix in _IMAGE_SUFFIXES:
                    pages, extraction = self._image_pages(
                        archived_source, Path(temporary_directory), suffix
                    )
                elif suffix in _TEXT_SUFFIXES:
                    pages, extraction = self._text_pages(archived_source, suffix)
                else:
                    raise EvidenceError(
                        "unsupported document type; supported extensions are PDF, TXT, MD, "
                        "CSV, JSON, HTML, PNG, JPEG, WEBP, and TIFF"
                    )
            except (EvidenceError, OptionalDependencyError) as exc:
                pages = []
                extraction = {"provider": "none", "error": str(exc)}
                document_status = "failed"
                document_error = str(exc)

            if pages and all(page.status == "failed" for page in pages):
                document_status = "failed"
                document_error = "every page failed extraction"
            result = self.workspace.commit_document(
                case_id,
                source_path=archived_source,
                filename=display_filename or source_path.name,
                pages=pages,
                extraction=extraction,
                document_status=document_status,
                document_error=document_error,
            )
        self.workspace.record_event(
            case_id,
            "document_ingested",
            {
                "source_sha256": result["document"]["source_sha256"],
                "filename": result["document"]["filename"],
                "status": document_status,
                "page_count": len(pages),
                "snapshot_id": result["snapshot"]["snapshot_id"],
            },
        )
        return result

    def ingest_many(self, case_id: str, sources: Iterable[str | Path]) -> list[dict[str, Any]]:
        # Each committed file remains independently reviewable. The final result
        # points to the cumulative immutable snapshot.
        return [self.ingest(case_id, source) for source in sources]

    def _text_pages(self, source: Path, suffix: str) -> tuple[list[PageInput], dict[str, Any]]:
        text = _decode_text(source.read_bytes())
        if suffix == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2, sort_keys=True)
            except json.JSONDecodeError:
                # Preserve invalid JSON as evidence rather than rewriting or dropping it.
                pass
        elif suffix in {".html", ".htm"}:
            parser = _VisibleTextParser()
            parser.feed(text)
            parser.close()
            text = html.unescape("".join(parser.parts))
        text = normalize_text(text)
        quality = _native_text_quality(text)
        status = "readable" if text.strip() else "failed"
        return [
            PageInput(
                page_number=1,
                text=text,
                extraction_method="native_text",
                status=status,
                quality_score=quality,
                needs_review=status != "readable" or quality < 0.65,
                error="" if status == "readable" else "document contains no readable text",
                metadata={"quality_signals": _text_quality_signals(text)},
            )
        ], {"provider": "python_standard_library", "format": suffix.lstrip(".")}

    def _image_pages(
        self, source: Path, temporary_root: Path, original_suffix: str
    ) -> tuple[list[PageInput], dict[str, Any]]:
        canonical_images = _canonicalize_images(
            source,
            temporary_root,
            self.max_rendered_pixels,
            self.max_image_frames,
            self.max_document_rendered_pixels,
        )
        extraction = (
            self.ocr_runner.extraction_receipt()
            if self.ocr_runner is not None
            else {"provider": "TensorRT-Model-Connect", "configured": False}
        )
        pages: list[PageInput] = []
        for page_number, (canonical_image, image_metadata) in enumerate(canonical_images, 1):
            if self.ocr_runner is None:
                pages.append(
                    PageInput(
                        page_number=page_number,
                        text="",
                        extraction_method="model_connect_ocr",
                        status="incomplete",
                        quality_score=0.0,
                        needs_review=True,
                        error="OCR bundle was not configured",
                        evidence_image=str(canonical_image),
                        metadata={"image": image_metadata},
                    )
                )
                continue
            try:
                result = self.ocr_runner.ocr(
                    canonical_image, max_new_tokens=self.ocr_max_new_tokens
                )
            except EvidenceError as exc:
                pages.append(
                    PageInput(
                        page_number=page_number,
                        text="",
                        extraction_method="model_connect_ocr",
                        status="failed",
                        quality_score=0.0,
                        needs_review=True,
                        error=str(exc),
                        evidence_image=str(canonical_image),
                        metadata={"image": image_metadata},
                    )
                )
                continue
            pages.append(
                PageInput(
                    page_number=page_number,
                    text=normalize_text(result.text),
                    extraction_method="model_connect_ocr",
                    status=result.status,
                    quality_score=result.quality_score,
                    needs_review=True,
                    error=("" if result.status == "readable" else "OCR output failed quality gate"),
                    evidence_image=str(canonical_image),
                    metadata={
                        "image": image_metadata,
                        "original_suffix": original_suffix,
                        "quality_signals": result.quality_signals,
                        "runtime": runtime_result_dict(result.runtime),
                    },
                )
            )
        if self.ocr_runner is not None:
            self.ocr_runner.verify_identity()
        return pages, extraction

    def _pdf_pages(
        self, source: Path, temporary_root: Path
    ) -> tuple[list[PageInput], dict[str, Any]]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OptionalDependencyError(
                "PDF support requires the evidence-workbench PDF dependencies; "
                "install this example with `pip install -e .`"
            ) from exc

        try:
            renderer_version = importlib.metadata.version("pypdfium2")
        except importlib.metadata.PackageNotFoundError:
            renderer_version = "unknown"
        extraction: dict[str, Any] = {
            "provider": "PDFium native text with TensorRT-Model-Connect OCR fallback",
            "pdf_renderer": "pypdfium2",
            "pdf_renderer_version": renderer_version,
            "pdf_render_scale": self.pdf_render_scale,
            "ocr": (
                self.ocr_runner.extraction_receipt()
                if self.ocr_runner is not None
                else {"provider": "TensorRT-Model-Connect", "configured": False}
            ),
        }
        pages: list[PageInput] = []
        try:
            document = pdfium.PdfDocument(str(source))
        except Exception as exc:
            raise EvidenceError(f"PDF could not be opened: {exc}") from exc
        if len(document) > self.max_pdf_pages:
            close = getattr(document, "close", None)
            if callable(close):
                close()
            raise EvidenceError(
                f"PDF has {len(document)} pages, exceeding the {self.max_pdf_pages}-page limit"
            )

        try:
            total_rendered_pixels = 0
            for page_index in range(len(document)):
                page_number = page_index + 1
                page = None
                text_page = None
                bitmap = None
                try:
                    page = document[page_index]
                    text_page = page.get_textpage()
                    native_text = normalize_text(text_page.get_text_range())
                    native_quality = _native_text_quality(native_text)
                    has_substantial_image = _pdf_page_has_substantial_image(page)
                    if (
                        len(native_text.strip()) >= 40
                        and native_quality >= 0.62
                        and not has_substantial_image
                    ):
                        pages.append(
                            PageInput(
                                page_number=page_number,
                                text=native_text,
                                extraction_method="pdf_native_text",
                                status="readable",
                                quality_score=native_quality,
                                needs_review=False,
                                metadata={
                                    "quality_signals": _text_quality_signals(native_text),
                                    "substantial_image_detected": False,
                                },
                            )
                        )
                        continue

                    image_path = temporary_root / f"page-{page_number:06d}.png"
                    width, height = page.get_size()
                    rendered_pixels = int(
                        width * self.pdf_render_scale * height * self.pdf_render_scale
                    )
                    if rendered_pixels > self.max_rendered_pixels:
                        raise EvidenceError(
                            f"rendered PDF page exceeds the {self.max_rendered_pixels}-pixel limit"
                        )
                    total_rendered_pixels += rendered_pixels
                    if total_rendered_pixels > self.max_document_rendered_pixels:
                        raise EvidenceError(
                            "PDF rendering exceeds the cumulative document-pixel limit"
                        )
                    bitmap = page.render(scale=self.pdf_render_scale)
                    image = bitmap.to_pil()
                    try:
                        image.save(image_path, format="PNG")
                    finally:
                        image.close()
                    if self.ocr_runner is None:
                        pages.append(
                            PageInput(
                                page_number=page_number,
                                text=native_text,
                                extraction_method="model_connect_ocr",
                                status="incomplete",
                                quality_score=native_quality,
                                needs_review=True,
                                error="page requires OCR but no OCR bundle was configured",
                                evidence_image=str(image_path),
                                metadata={
                                    "native_quality_signals": _text_quality_signals(native_text),
                                    "substantial_image_detected": has_substantial_image,
                                },
                            )
                        )
                        continue
                    ocr = self.ocr_runner.ocr(image_path, max_new_tokens=self.ocr_max_new_tokens)
                    pages.append(
                        PageInput(
                            page_number=page_number,
                            text=normalize_text(ocr.text),
                            extraction_method="model_connect_ocr",
                            status=ocr.status,
                            quality_score=ocr.quality_score,
                            needs_review=True,
                            error=(
                                "" if ocr.status == "readable" else "OCR output failed quality gate"
                            ),
                            evidence_image=str(image_path),
                            metadata={
                                "quality_signals": ocr.quality_signals,
                                "native_quality_signals": _text_quality_signals(native_text),
                                "substantial_image_detected": has_substantial_image,
                                "runtime": runtime_result_dict(ocr.runtime),
                            },
                        )
                    )
                except Exception as exc:
                    pages.append(
                        PageInput(
                            page_number=page_number,
                            text="",
                            extraction_method="pdf_page_extraction",
                            status="failed",
                            quality_score=0.0,
                            needs_review=True,
                            error=str(exc),
                        )
                    )
                finally:
                    for resource in (bitmap, text_page, page):
                        close = getattr(resource, "close", None)
                        if callable(close):
                            close()
        finally:
            close = getattr(document, "close", None)
            if callable(close):
                close()
        if self.ocr_runner is not None:
            self.ocr_runner.verify_identity()
        return pages, extraction


def _decode_text(data: bytes) -> str:
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise EvidenceError("text document is not valid UTF-8 or UTF-16")


def _text_quality_signals(text: str) -> dict[str, Any]:
    if not text:
        return {
            "length": 0,
            "printable_ratio": 0.0,
            "word_character_ratio": 0.0,
            "replacement_characters": 0,
        }
    printable = sum(character.isprintable() or character.isspace() for character in text)
    word_characters = sum(character.isalnum() for character in text)
    return {
        "length": len(text),
        "printable_ratio": round(printable / len(text), 6),
        "word_character_ratio": round(word_characters / len(text), 6),
        "replacement_characters": text.count("\ufffd"),
    }


def _native_text_quality(text: str) -> float:
    signals = _text_quality_signals(text)
    if not text.strip():
        return 0.0
    length_score = min(1.0, len(text.strip()) / 100.0)
    quality = (
        0.5 * float(signals["printable_ratio"])
        + 0.3 * float(signals["word_character_ratio"])
        + 0.2 * length_score
        - min(0.4, int(signals["replacement_characters"]) * 0.05)
    )
    return round(max(0.0, min(1.0, quality)), 6)


def _canonicalize_images(
    source: Path,
    temporary_root: Path,
    max_pixels: int,
    max_frames: int,
    max_total_pixels: int,
) -> list[tuple[Path, dict[str, Any]]]:
    """Decode every frame and write the PNG contract consumed by trtmc."""

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise OptionalDependencyError(
            "image ingestion requires Pillow; install the standalone application dependencies"
        ) from exc
    try:
        with Image.open(source) as opened:
            frame_count = int(getattr(opened, "n_frames", 1) or 1)
            if frame_count > max_frames:
                raise EvidenceError(
                    f"image contains {frame_count} frames, exceeding the {max_frames}-frame limit"
                )
            outputs: list[tuple[Path, dict[str, Any]]] = []
            total_pixels = 0
            for frame_index in range(frame_count):
                opened.seek(frame_index)
                width, height = opened.size
                if width < 1 or height < 1 or width * height > max_pixels:
                    raise EvidenceError(
                        f"image frame {frame_index + 1} exceeds the {max_pixels}-pixel safety limit"
                    )
                total_pixels += width * height
                if total_pixels > max_total_pixels:
                    raise EvidenceError("image frames exceed the cumulative document-pixel limit")
                opened.load()
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                converted = oriented.convert("RGB")
                output = temporary_root / f"canonical-image-{frame_index + 1:06d}.png"
                try:
                    converted.save(output, format="PNG", optimize=False)
                finally:
                    converted.close()
                    if oriented is not opened:
                        oriented.close()
                outputs.append(
                    (
                        output,
                        {
                            "decoder": "Pillow",
                            "source_format": str(opened.format or "unknown"),
                            "frame_number": frame_index + 1,
                            "frame_count": frame_count,
                            "width": width,
                            "height": height,
                            "canonical_format": "PNG-RGB",
                        },
                    )
                )
            return outputs
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError(f"image could not be decoded safely: {exc}") from exc


def _pdf_page_has_substantial_image(page: Any) -> bool:
    """Fail closed when any raster image may contain text outside the text layer."""

    try:
        import pypdfium2.raw as pdfium_c

        return (
            next(
                iter(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])),
                None,
            )
            is not None
        )
    except Exception:
        # Failure to inspect page composition cannot authorize a negative result.
        return True
