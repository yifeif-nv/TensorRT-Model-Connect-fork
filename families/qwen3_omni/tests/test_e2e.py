# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native Task API, and official-reference E2E for Qwen3-Omni."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "qwen3_omni"
TASKS = frozenset({"audio_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
TEXT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names
    if not enabled:
        parameters = [
            pytest.param(
                name,
                marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"
                ),
            )
            for name in names
        ]
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(raw: str | None, label: str) -> Path:
    assert raw, f"selected {FAMILY} E2E requires {label}"
    path = Path(raw)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_QWEN3_OMNI_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_QWEN3_OMNI_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest["hf_revision"],
                local_files_only=True,
            )
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires exact cached checkpoint {manifest['hf_id']}"
        ) from error


def _runtime_paths() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_qwen3_omni.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, output: Path) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "generate-audio",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            str(case["prompt"]),
            "--output",
            str(output),
            "--max-new-tokens",
            str(int(case["max_new_tokens"])),
            "--talker-max-new-tokens",
            str(int(case["talker_max_new_tokens"])),
            "--seed",
            str(int(case["seed"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(case["runtime_timeout_s"]),
    )
    assert output.is_file(), completed.stderr[-2000:]
    return json.loads(completed.stdout)


def _native_text(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "run",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            str(case["prompt"]),
            "--max-new-tokens",
            str(int(case["max_new_tokens"])),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(case["runtime_timeout_s"]),
    )
    return json.loads(completed.stdout)


def _official_reference(
    model_dir: Path, manifest: dict, case: dict
) -> tuple[np.ndarray, int, np.ndarray, str, list[int]]:
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_dir, revision=manifest["hf_revision"], local_files_only=True
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_dir,
        revision=manifest["hf_revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        enable_audio_output=True,
    ).eval()
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": str(case["prompt"])}]},
    ]
    inputs = processor.apply_chat_template(
        conversation,
        chat_template=TEXT_CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    code_inputs = []

    def capture_code2wav_input(_module, args) -> None:
        code_inputs.append(args[0].detach().cpu())

    hook = model.code2wav.register_forward_pre_hook(capture_code2wav_input)
    torch.manual_seed(int(case["seed"]))
    torch.cuda.manual_seed_all(int(case["seed"]))
    try:
        with torch.inference_mode():
            text_ids, audio = model.generate(
                **inputs,
                thinker_max_new_tokens=int(case["max_new_tokens"]),
                talker_max_new_tokens=int(case["talker_max_new_tokens"]),
                thinker_do_sample=False,
                talker_do_sample=False,
                speaker=str(case["speaker"]),
            )
    finally:
        hook.remove()
    assert len(code_inputs) == 1
    codes = np.asarray(code_inputs[0], dtype=np.int32)
    generated = text_ids[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    if generated and generated[-1] == int(model.config.thinker_config.eos_token_id):
        generated.pop()
    text = processor.tokenizer.decode(generated).strip()
    return np.asarray(audio.detach().float().cpu()).reshape(-1), 24000, codes, text, generated


def _bundle_section(bundle: Path, name: str) -> bytes:
    with bundle.open("rb") as file:
        assert file.read(8) == b"BUNDLE\x01\x00"
        header_length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_length))
        descriptor = header["sections"][name]
        file.seek(16 + header_length + int(descriptor["offset"]))
        data = file.read(int(descriptor["length"]))
    assert len(data) == int(descriptor["length"])
    return data


def _teacher_forced_code2wav(bundle: Path, codes: np.ndarray) -> np.ndarray:
    import tensorrt as trt
    import torch

    runtime_config = json.loads(_bundle_section(bundle, "runtime.json"))
    quantizers = int(runtime_config["code2wav_num_quantizers"])
    max_frames = int(runtime_config["code2wav_max_frames"])
    upsample_factor = int(runtime_config["code2wav_upsample_factor"])
    output_delay = int(runtime_config["code2wav_output_delay"])
    assert codes.ndim == 3 and codes.shape[:2] == (1, quantizers)
    frames = int(codes.shape[2])
    assert 0 < frames <= max_frames
    padded = np.zeros((1, quantizers, max_frames), dtype=np.int32)
    padded[:, :, :frames] = codes

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(_bundle_section(bundle, "code2wav.plan"))
    assert engine is not None
    context = engine.create_execution_context()
    device_codes = torch.from_numpy(padded).to(device="cuda")
    waveform_shape = tuple(context.get_tensor_shape("waveform"))
    waveform = torch.empty(waveform_shape, device="cuda", dtype=torch.float32)
    assert context.set_tensor_address("codec_tokens", device_codes.data_ptr())
    assert context.set_tensor_address("waveform", waveform.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    samples = frames * upsample_factor - output_delay
    assert 0 < samples <= waveform.numel()
    return np.asarray(waveform.detach().cpu()).reshape(-1)[:samples]


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _assert_native_audio(
    native_path: Path, reference: np.ndarray, sample_rate: int, thresholds: dict
) -> None:
    import soundfile as sf

    actual, actual_rate = sf.read(native_path, dtype="float32")
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    assert int(actual_rate) == sample_rate
    assert actual.size / sample_rate >= float(thresholds["duration_s_min"])
    assert float(np.sqrt(np.mean(actual**2))) >= float(thresholds["rms_min"])
    assert float(np.max(np.abs(actual))) >= float(thresholds["peak_min"])
    ratio = actual.size / max(reference.size, 1)
    assert ratio >= float(thresholds["duration_ratio_min"])
    assert ratio <= float(thresholds["duration_ratio_max"])


def _assert_teacher_forced_code2wav(
    actual: np.ndarray, reference: np.ndarray, thresholds: dict
) -> None:
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    assert actual.size == reference.size
    denominator = float(
        np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(reference.astype(np.float64))
    )
    assert denominator > 0.0
    cosine = float(np.dot(actual.astype(np.float64), reference.astype(np.float64)) / denominator)
    assert cosine >= float(thresholds["waveform_cosine_min"])


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime_paths()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    native_audio = tmp_path / "native.wav"
    native_audio_result = _native(binary, runtime_root, bundle, case, native_audio)
    assert int(native_audio_result["sample_rate"]) == 24000
    assert int(native_audio_result["num_samples"]) > 0
    native_text = _native_text(binary, runtime_root, bundle, case)
    reference, sample_rate, codes, reference_text, reference_token_ids = _official_reference(
        model_dir, manifest, case
    )
    assert native_text["text"] == reference_text
    assert native_text["token_ids"] == reference_token_ids
    thresholds = _thresholds(case_name)
    _assert_teacher_forced_code2wav(_teacher_forced_code2wav(bundle, codes), reference, thresholds)
    _assert_native_audio(native_audio, reference, sample_rate, thresholds)
