# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect import build_cli


def test_build_command_forwards_only_direct_inputs(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(build_cli, "build", captured.append)
    model = tmp_path / "model"
    output = tmp_path / "model.bundle"

    assert (
        build_cli.main(
            [
                "build",
                str(model),
                "--output",
                str(output),
                "--family",
                "patchtsmixer",
                "--task",
                "time_series_forecast",
                "--precision",
                "fp16",
                "--max-sequence-length",
                "1024",
                "--image-height",
                "512",
                "--image-width",
                "768",
                "--video-num-frames",
                "17",
                "--max-batch-size",
                "3",
                "--tensor-parallel-size",
                "2",
                "--context-parallel-size",
                "4",
                "--quantization",
                "fp8",
                "--fp32-layer",
                "2",
                "--fp32-layer",
                "5",
                "--verbose",
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.model_dir == model
    assert request.output_path == output
    assert request.family == "patchtsmixer"
    assert request.task == "time_series_forecast"
    assert request.precision == "fp16"
    assert request.max_sequence_length == 1024
    assert request.image_height == 512
    assert request.image_width == 768
    assert request.video_num_frames == 17
    assert request.max_batch_size == 3
    assert request.tensor_parallel_size == 2
    assert request.context_parallel_size == 4
    assert request.quantization == "fp8"
    assert request.fp32_layers == (2, 5)
    assert request.verbose is True
