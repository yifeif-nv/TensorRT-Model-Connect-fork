# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream already-qualified MiniMax-H3 plans into a runnable TRTMC bundle."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    write_bundle,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    SOL_ENGINE_1344X768_124_TO_345F,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    validate_build_receipt,
    validate_source_revision,
)
PLAN_SECTIONS = {
    "text_encoder_plan": "text_encoder.plan",
    "vision_encoder_plan": "vision_encoder.plan",
    "adaln_precompute_plan": "adaln_precompute.plan",
    "denoiser_head_plan": "denoiser_head.plan",
    "denoiser_tail_plan": "denoiser_tail.plan",
    "denoiser_finish_plan": "denoiser_finish.plan",
    "fl2va_keyframe_vae_encoder_plan": "fl2va_keyframe_vae_encoder.plan",
    "vae_tile_decoder_plan": "vae_tile_decoder.plan",
    "audio_vae_decoder_plan": "audio_vae_decoder.plan",
}
EAGER_BUNDLE_SECTIONS = ("tokenizer.json", "config.json")
LAZY_BUNDLE_SECTIONS = tuple(PLAN_SECTIONS)


def _target_metadata() -> tuple[str, str, str]:
    """Bind a bundle to the TensorRT ABI and GPU that built its plans."""

    trt_version = engine_builder._get_trt_version()
    trt_abi = engine_builder._trt_abi_from_version(trt_version)
    gpu_name = engine_builder._get_gpu_name()
    if trt_version == "unknown" or not trt_abi or not gpu_name:
        raise RuntimeError(
            "MiniMax-H3 bundle packaging requires a detected TensorRT version and GPU"
        )
    return trt_version, trt_abi, gpu_name


def _bundle_loading_policy() -> dict[str, object]:
    """Keep only metadata resident; H3 loads one large plan at a time."""

    return {
        "mode": "staged",
        "eager_sections": list(EAGER_BUNDLE_SECTIONS),
        "lazy_sections": list(PLAN_SECTIONS),
    }


def _audio_vae_metadata(model: Path, profile) -> dict[str, object]:
    path = model / "audio_vae" / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MiniMax-H3 AudioVAE config: {path}")
    config = json.loads(path.read_text())
    rates = config.get("decoder_rates")
    latent_mean = config.get("latents_mean")
    latent_std = config.get("latents_std")
    if (
        not isinstance(rates, list)
        or not rates
        or not isinstance(latent_mean, list)
        or not isinstance(latent_std, list)
        or len(latent_mean) != profile.audio_in_channels
        or len(latent_std) != profile.audio_in_channels
    ):
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")
    try:
        hop_length = math.prod(int(value) for value in rates)
        sampling_rate = int(config["sampling_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata") from error
    if hop_length <= 0 or sampling_rate <= 0 or profile.audio_rows % 2:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")
    return {
        "audio_latent_frames": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_min": AUDIO_LATENT_FRAMES_MIN,
        "audio_latent_frames_opt": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_max": AUDIO_LATENT_FRAMES_MAX,
        "audio_sample_rate": sampling_rate,
        "audio_hop_length": hop_length,
        "audio_channels": 2,
        "audio_vae_precision": "fp32",
        "audio_vae_input_normalized": False,
        "audio_latents_mean": [float(value) for value in latent_mean],
        "audio_latents_std": [float(value) for value in latent_std],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    plans = Path(args.plans_dir)
    model = Path(args.model_path)
    output = Path(args.output)
    source_revision = validate_source_revision(args.source_revision)
    profile = SOL_ENGINE_1344X768_124_TO_345F
    trt_version, trt_abi, gpu_name = _target_metadata()
    receipt_path = plans / "build_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing native build receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    tokenizer = (model / "tokenizer" / "tokenizer.json").resolve(strict=True)
    audio_vae_metadata = _audio_vae_metadata(model, profile)
    expected_source_sha, recorded, tokenizer_record, snapshot_record = validate_build_receipt(
        receipt,
        plans_dir=plans,
        snapshot=model,
        tokenizer=tokenizer,
        build_helper=Path(__file__).with_name("build_native_components.py"),
        source_revision=source_revision,
        profile=profile,
        hash_files=False,
    )
    if receipt.get("denoiser_mode") != "first_block":
        raise ValueError("MiniMax-H3 build receipt denoiser mode does not match packaging mode")
    if receipt.get("transformer_ref") is not None:
        raise ValueError("Non-Ref2VA packaging rejects transformer_ref receipt metadata")

    sections: list[BundleSection] = []
    for section_name, filename in PLAN_SECTIONS.items():
        path = plans / filename
        sections.append(
            _bundle_section_from_file(
                section_name,
                path,
                expected_sha256=recorded[filename]["sha256"],
            )
        )
    sections.append(
        _bundle_section_from_file(
            "tokenizer.json", tokenizer, expected_sha256=tokenizer_record["sha256"]
        )
    )
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "precision": "bf16",
        "engine_backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "bundle_loading": _bundle_loading_policy(),
        "tokenizer_add_special_tokens": 0,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": expected_source_sha,
        "build_helper_sha256": receipt["build_helper_sha256"],
        "checkpoint_inventory_sha256": snapshot_record["inventory_sha256"],
        "workspace_limit_bytes": dict(receipt["workspace_limit_bytes"]),
        "plan_sha256": {
            filename: recorded[filename]["sha256"] for filename in PLAN_SECTIONS.values()
        },
        "first_block_cache": True,
        "denoiser_cache_mode": "first_block",
        "first_block_cache_threshold": 0.08,
        "height": 768,
        "width": 1344,
        "canvas_multiple": CANVAS_MULTIPLE,
        "canvas_short_edge": CANVAS_SHORT_EDGE,
        "canvas_max_pixels": CANVAS_MAX_PIXELS,
        "explicit_canvas_sizes": [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES],
        "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
        "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
        "public_workflows": ["t2va", "fl2va"],
        "conditioning": {
            "implementation": "shared_native_qwen3_vl",
            "text_encoder_section": "text_encoder_plan",
            "vision_encoder_section": "vision_encoder_plan",
            "keyframe_vae_encoder_section": "fl2va_keyframe_vae_encoder_plan",
            "text_sequence_profile": [1, 1144, 2641],
            "vision_patch_profile": [2040, 4032, 4176],
            "vision_row_profile": [1, 1008, 2088],
            "t2va_dummy_vision_rows": 1,
            "t2va_vision_count": 0,
            "t2va_vision_mask_nonzero": 0,
            "keyframe_vae_tile_batch_profile": [1, 28, 33],
            "reachable_canvas_count": 95,
            "max_rounded_canvas": [576, 1856],
            "max_condition_video_rows": 2088,
            "mode_coupled_profile_required": True,
        },
        "num_frames": VIDEO_NUM_FRAMES_OPT,
        "num_frames_min": VIDEO_NUM_FRAMES_MIN,
        "num_frames_opt": VIDEO_NUM_FRAMES_OPT,
        "num_frames_max": VIDEO_NUM_FRAMES_MAX,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 1.0,
        "scheduler_grid_points": 50,
        "transformer_forwards": 49,
        "attention_mode": "dense",
        "text_rows": profile.text_rows,
        "text_rows_min": profile.min_text_rows,
        "text_rows_opt": profile.opt_text_rows,
        "text_rows_max": profile.text_rows,
        "audio_rows": profile.opt_audio_rows,
        "audio_rows_min": profile.min_audio_rows,
        "audio_rows_opt": profile.opt_audio_rows,
        "audio_rows_max": profile.audio_rows,
        **audio_vae_metadata,
        "video_rows": profile.opt_video_rows,
        "video_rows_min": profile.min_video_rows,
        "video_rows_opt": profile.opt_video_rows,
        "video_rows_max": profile.video_rows,
        "packed_sequence_length_min": profile.min_sequence_length,
        "packed_sequence_length_opt": profile.opt_sequence_length,
        "packed_sequence_length_max": profile.sequence_length,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": 4,
        "context_parallel_size": 1,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 15,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }
    sections.append(BundleSection("config.json", json.dumps(config, indent=2).encode()))
    info = BundleInfo(
        model_id="MiniMaxAI/MiniMax-H3",
        model_type="minimax_h3",
        family="minimax_h3",
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy="diffusion_minimax_h3",
        precision="bf16",
        tokenizer_add_special_tokens=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(output, info, sections)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
