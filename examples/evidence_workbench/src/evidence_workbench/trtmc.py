# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-free adapter to public Model Connect runtime commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import EvidenceError, ModelConnectError, sha256_file, sha256_text


DEFAULT_OCR_PROMPT = "Extract the text from this image."


@dataclass(frozen=True)
class RuntimeResult:
    stdout: str
    stderr: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class OcrResult:
    text: str
    quality_score: float
    needs_review: bool
    status: str
    quality_signals: dict[str, Any]
    runtime: RuntimeResult


class TrtmcRunner:
    """Invoke one Model Connect bundle through argument-vector subprocesses."""

    def __init__(
        self,
        bundle: str | Path,
        *,
        binary: str | Path | None = None,
        hf_python: str | Path | None = None,
        timeout: float = 300.0,
    ):
        self.binary = self._resolve_binary(binary)
        self.bundle = self._regular_file(bundle, "bundle")
        self.hf_python = str(self._regular_file(hf_python, "hf_python")) if hf_python else ""
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise EvidenceError("Model Connect timeout must be positive")
        self._identity: dict[str, Any] | None = None
        self._initial_bundle_fingerprint = _file_fingerprint(self.bundle)
        self._initial_binary_fingerprint = _file_fingerprint(self.binary)
        self._initial_hf_python_fingerprint = (
            _file_fingerprint(Path(self.hf_python)) if self.hf_python else None
        )

    @staticmethod
    def _resolve_binary(binary: str | Path | None) -> Path:
        candidate = str(binary) if binary else shutil.which("trtmc")
        if not candidate:
            raise EvidenceError("trtmc was not found; build Model Connect or pass --trtmc-binary")
        path = Path(candidate).expanduser()
        if path.is_symlink():
            path = path.resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EvidenceError(f"trtmc binary is not executable: {path}")
        return path.resolve()

    @staticmethod
    def _regular_file(path_value: str | Path, label: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_symlink():
            path = path.resolve()
        if not path.is_file():
            raise EvidenceError(f"{label} is not a regular file: {path}")
        return path.resolve()

    def identity(self) -> dict[str, Any]:
        """Return content hashes and inspect output for the loaded artifact."""

        if self._identity is not None:
            return dict(self._identity)
        self._assert_artifacts_unchanged()
        bundle_sha256 = sha256_file(self.bundle)
        binary_sha256 = sha256_file(self.binary)
        hf_python_sha256 = sha256_file(Path(self.hf_python)) if self.hf_python else ""
        inspection = self._run(["inspect", str(self.bundle)], timeout=min(30.0, self.timeout))
        self._assert_artifacts_unchanged()
        self._identity = {
            "bundle_sha256": bundle_sha256,
            "bundle_size_bytes": self.bundle.stat().st_size,
            "binary_sha256": binary_sha256,
            "inspect_output": inspection.stdout,
            "hf_python_sha256": hf_python_sha256,
        }
        return dict(self._identity)

    def ocr(
        self,
        image: str | Path,
        *,
        prompt: str = DEFAULT_OCR_PROMPT,
        max_new_tokens: int = 600,
    ) -> OcrResult:
        image_path = self._regular_file(image, "OCR image")
        self.identity()
        if max_new_tokens < 1:
            raise EvidenceError("OCR max_new_tokens must be positive")
        argv = [
            "run",
            str(self.bundle),
            "--prompt",
            prompt,
            "--image",
            str(image_path),
            "--max-new-tokens",
            str(max_new_tokens),
            "--greedy",
            "--chat-template",
        ]
        if self.hf_python:
            argv.extend(["--hf-python", self.hf_python])
        runtime = self._run(argv)
        text = runtime.stdout.strip()
        signals = _ocr_quality_signals(text, max_new_tokens=max_new_tokens)
        status = "readable" if text and signals["quality_score"] >= 0.35 else "failed"
        return OcrResult(
            text=text,
            quality_score=float(signals["quality_score"]),
            # The current image run surface exposes no token count or calibrated
            # confidence. Every OCR page therefore remains human-review required.
            needs_review=True,
            status=status,
            quality_signals=signals,
            runtime=runtime,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 600,
        chat_template: bool = True,
        no_thinking: bool = True,
    ) -> RuntimeResult:
        if not prompt.strip():
            raise EvidenceError("generation prompt must not be empty")
        self.identity()
        argv = [
            "run",
            str(self.bundle),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--greedy",
        ]
        if chat_template:
            argv.append("--chat-template")
        if no_thinking:
            argv.append("--no-thinking")
        if self.hf_python:
            argv.extend(["--hf-python", self.hf_python])
        return self._run(argv)

    def embed(self, text: str) -> tuple[list[float], RuntimeResult]:
        if not text.strip():
            raise EvidenceError("embedding input must not be empty")
        self.identity()
        argv = ["embed", str(self.bundle), "--prompt", text]
        if self.hf_python:
            argv.extend(["--hf-python", self.hf_python])
        runtime = self._run(argv)
        try:
            payload = json.loads(runtime.stdout)
            embedding = payload["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("embedding is empty")
            values = [float(value) for value in embedding]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ModelConnectError("trtmc embed did not return strict embedding JSON") from exc
        return values, runtime

    def extraction_receipt(self) -> dict[str, Any]:
        return {
            "provider": "TensorRT-Model-Connect",
            "runner": "trtmc argv subprocess",
            "model": self.identity(),
            "ocr_contract": {
                "prompt": DEFAULT_OCR_PROMPT,
                "confidence_boundary": (
                    "quality_score is an application heuristic; the runtime does not expose "
                    "OCR log probabilities or image-run token counts"
                ),
            },
        }

    def verify_identity(self) -> dict[str, Any]:
        """Re-hash runtime artifacts before evidence is committed."""

        identity = self.identity()
        self._assert_artifacts_unchanged()
        observed = {
            "bundle_sha256": sha256_file(self.bundle),
            "binary_sha256": sha256_file(self.binary),
            "hf_python_sha256": (sha256_file(Path(self.hf_python)) if self.hf_python else ""),
        }
        for field, value in observed.items():
            if identity.get(field, "") != value:
                raise ModelConnectError(
                    f"{field.removesuffix('_sha256')} changed during evidence extraction"
                )
        self._assert_artifacts_unchanged()
        return identity

    def _run(self, arguments: list[str], *, timeout: float | None = None) -> RuntimeResult:
        self._assert_artifacts_unchanged()
        command = [str(self.binary), *arguments]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                shell=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ModelConnectError(
                f"Model Connect timed out after {timeout or self.timeout:.1f}s"
            ) from exc
        except OSError as exc:
            raise ModelConnectError(f"could not launch Model Connect: {exc}") from exc
        finally:
            self._assert_artifacts_unchanged()
        if result.returncode != 0:
            diagnostic = (result.stderr or result.stdout).strip()[-4_000:]
            raise ModelConnectError(
                f"Model Connect failed with exit {result.returncode}: {diagnostic}"
            )
        return RuntimeResult(
            stdout=result.stdout,
            stderr=result.stderr[-16_000:],
            argv=tuple(command),
        )

    def _assert_artifacts_unchanged(self) -> None:
        checks = [
            (self.bundle, self._initial_bundle_fingerprint, "bundle"),
            (self.binary, self._initial_binary_fingerprint, "trtmc binary"),
        ]
        if self.hf_python and self._initial_hf_python_fingerprint is not None:
            checks.append((Path(self.hf_python), self._initial_hf_python_fingerprint, "hf_python"))
        for path, expected, label in checks:
            try:
                observed = _file_fingerprint(path)
            except OSError as exc:
                raise ModelConnectError(f"{label} became unavailable: {path}") from exc
            if observed != expected:
                raise ModelConnectError(
                    f"{label} changed after runner initialization; create a new runner "
                    "so provenance cannot become stale"
                )


def _ocr_quality_signals(text: str, *, max_new_tokens: int) -> dict[str, Any]:
    if not text:
        return {
            "quality_score": 0.0,
            "length": 0,
            "printable_ratio": 0.0,
            "replacement_characters": 0,
            "maximum_repeated_character_run": 0,
            "possible_token_budget_truncation": False,
            "max_new_tokens": max_new_tokens,
        }
    printable = sum(character.isprintable() or character.isspace() for character in text)
    replacement = text.count("\ufffd")
    maximum_run = 1
    current_run = 1
    previous = text[0]
    for character in text[1:]:
        if character == previous and not character.isspace():
            current_run += 1
            maximum_run = max(maximum_run, current_run)
        else:
            current_run = 1
            previous = character
    printable_ratio = printable / len(text)
    length_score = min(1.0, len(text.strip()) / 120.0)
    replacement_penalty = min(0.4, replacement / max(1, len(text)) * 8.0)
    repetition_penalty = 0.35 if maximum_run >= 12 else 0.0
    quality = max(
        0.0,
        min(
            1.0,
            0.55 * printable_ratio + 0.45 * length_score - replacement_penalty - repetition_penalty,
        ),
    )
    stripped = text.rstrip()
    possible_truncation = len(stripped) > 200 and stripped[-1:] not in ".!?;:)]}\"'"
    return {
        "quality_score": round(quality, 6),
        "length": len(text),
        "printable_ratio": round(printable_ratio, 6),
        "replacement_characters": replacement,
        "maximum_repeated_character_run": maximum_run,
        "possible_token_budget_truncation": possible_truncation,
        "max_new_tokens": max_new_tokens,
    }


def runtime_result_dict(result: RuntimeResult) -> dict[str, Any]:
    """Return a stable logical receipt without ephemeral paths or timings."""

    argv = list(result.argv)
    if argv:
        argv[0] = "<trtmc-binary>"
    if len(argv) > 2 and argv[1] in {"run", "embed", "inspect"}:
        argv[2] = "<model-bundle>"
    for flag, replacement in (
        ("--image", "<evidence-image>"),
        ("--hf-python", "<hf-python>"),
    ):
        if flag in argv:
            position = argv.index(flag) + 1
            if position < len(argv):
                argv[position] = replacement
    return {
        "argv_contract": argv,
        "stdout_sha256": sha256_text(result.stdout),
    }


def _file_fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
