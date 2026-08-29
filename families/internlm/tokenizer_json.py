# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM tokenizer artifact requirement."""

from __future__ import annotations

from pathlib import Path


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    """Require the checkpoint to provide its native tokenizer.json."""

    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        detail = f": {previous_error}" if previous_error else ""
        raise FileNotFoundError(f"InternLM checkpoint is missing tokenizer.json{detail}")
    return True
