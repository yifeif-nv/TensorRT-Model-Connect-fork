# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal command-line entrypoint for family-owned builds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .build import BuildRequest, build


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trtmc")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build one model-family bundle")
    build_parser.add_argument("model_dir", type=Path)
    build_parser.add_argument("-o", "--output", type=Path, required=True)
    build_parser.add_argument("--family", required=True)
    build_parser.add_argument("--task", required=True)
    build_parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp32")
    build_parser.add_argument("--max-sequence-length", type=int)
    build_parser.add_argument("--image-height", type=int)
    build_parser.add_argument("--image-width", type=int)
    build_parser.add_argument("--video-num-frames", type=int)
    build_parser.add_argument("--max-batch-size", type=int, default=1)
    build_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    build_parser.add_argument("--context-parallel-size", type=int, default=1)
    build_parser.add_argument("--quantization")
    build_parser.add_argument("--fp32-layer", type=int, action="append", default=[])
    build_parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "build":
        raise AssertionError(f"unhandled command: {args.command}")
    build(
        BuildRequest(
            model_dir=args.model_dir,
            output_path=args.output,
            precision=args.precision,
            family=args.family,
            task=args.task,
            max_sequence_length=args.max_sequence_length,
            image_height=args.image_height,
            image_width=args.image_width,
            video_num_frames=args.video_num_frames,
            max_batch_size=args.max_batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            context_parallel_size=args.context_parallel_size,
            quantization=args.quantization,
            fp32_layers=tuple(args.fp32_layer),
            verbose=args.verbose,
        )
    )
    return 0
