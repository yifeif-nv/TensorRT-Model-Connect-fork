# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_model_connect.runtime_config import Layer
from tensorrt_model_connect.families.minimax_h3 import (
    checkpoint,
    provenance,
    quantized_checkpoint,
    staged_build,
)
from tensorrt_model_connect.families.minimax_h3.plugin import (
    _quantized_transformer_build_input,
    plugin,
    write_path_free_effective_build_config,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    QUANTIZED_TRANSFORMER_CONFIG,
    validate_native_bundle_config,
)
from tensorrt_model_connect.families.minimax_h3.runtime_config_schema import SCHEMA
from tests.builder.conftest import read_bundle_file


plugin_module = importlib.import_module("tensorrt_model_connect.families.minimax_h3.plugin")
SOURCE_REVISION = "a" * 40


def _model(root: Path) -> Path:
    model = root / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    audio_config = model / "audio_vae" / "config.json"
    audio_config.parent.mkdir(parents=True)
    audio_config.write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32_000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    transformer = model / "transformer" / "base.safetensors"
    transformer.parent.mkdir(parents=True)
    transformer.write_bytes(b"base")
    return model


def _snapshot(model: Path) -> dict:
    files = {}
    for path in sorted(model.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model).as_posix()
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[relative] = {
            "blob_id": digest,
            "bytes": len(payload),
            "sha256": digest,
        }
    payload = {
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "files": files,
    }
    return {
        **payload,
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _write_plan(path: Path, payload: bytes) -> dict[str, int | str]:
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _quantized_file(root: Path) -> Path:
    path = root / "minimax_h3_fl2va_int8_convrot.safetensors"
    path.write_bytes(b"synthetic")
    return path


def test_quantized_transformer_config_is_build_only() -> None:
    field = next(field for field in SCHEMA.fields if field.name == "quantized_transformer")
    assert field.default == ""
    assert field.type_tag == "string"
    assert field.allowed_layers == frozenset({Layer.BUILD_TIME, Layer.SESSION_REQUEST})


def test_plugin_forwards_only_an_explicit_quantized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized = _quantized_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        staged_build,
        "build_staged_bundle",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Path(args[1]),
    )
    model = tmp_path / "model"
    output = tmp_path / "model.bundle"

    assert _quantized_transformer_build_input({"quantized_transformer": quantized}) == (
        quantized.absolute()
    )
    assert _quantized_transformer_build_input({"quantized_transformer_path": quantized}) is None
    assert (
        plugin.build_staged_bundle(
            str(model),
            str(output),
            SimpleNamespace(raw={"quantized_transformer": str(quantized)}),
            {"_model_dir": str(model)},
            precision="bf16",
            parallel_config=SimpleNamespace(mode="single"),
        )
        == output
    )
    assert calls == [
        (
            (model, str(output)),
            {"verbose": False, "quantized_transformer": quantized.absolute()},
        )
    ]
    with pytest.raises(ValueError, match="explicit .safetensors file"):
        _quantized_transformer_build_input({"quantized_transformer": True})


def test_effective_config_keeps_transformer_ref_strict_and_quant_field_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "checkpoint_revision": "revision",
        "checkpoint_inventory_sha256": "inventory",
        "source_revision": "source",
        "builder_source_sha256": "builder",
    }
    monkeypatch.setattr(plugin_module, "load_bundle_config", lambda _path: config)
    artifact = tmp_path / "model.bundle"

    missing_ref = SimpleNamespace(to_effective_dict=lambda: {"minimax_h3": {}})
    with pytest.raises(ValueError, match="effective config is missing transformer_ref"):
        write_path_free_effective_build_config(missing_ref, artifact)

    legacy_bf16 = SimpleNamespace(
        to_effective_dict=lambda: {
            "minimax_h3": {
                "transformer_ref": {"value": "", "source": "schema_default"},
            }
        }
    )
    sidecar = write_path_free_effective_build_config(legacy_bf16, artifact)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "quantized_transformer" not in payload["minimax_h3"]


def test_staged_quantized_build_is_scoped_resumable_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", SOURCE_REVISION)
    model = _model(tmp_path)
    quantized = _quantized_file(tmp_path)
    output = tmp_path / "h3-int8.bundle"
    snapshot = _snapshot(model)
    source_file_identity = quantized_checkpoint.QuantizedSourceFileIdentity(
        device=1,
        inode=2,
        size_bytes=quantized.stat().st_size,
        mtime_ns=3,
        ctime_ns=4,
    )
    quantized_identity = replace(
        quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY,
        source_file_identity=source_file_identity,
    )
    validations = []
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        staged_build,
        "checkpoint_snapshot_record",
        lambda _model, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        staged_build,
        "validate_checkpoint_snapshot_record",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        quantized_checkpoint,
        "validate_quantized_transformer_checkpoint",
        lambda path: validations.append(path) or quantized_identity,
    )

    def build(component: str, _model: Path, plan: Path, **kwargs):
        calls.append((component, kwargs))
        return _write_plan(plan, component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    assert (
        staged_build.build_staged_bundle(model, output, quantized_transformer=quantized) == output
    )
    assert validations == [quantized.absolute(), quantized.absolute()]
    assert [component for component, _kwargs in calls] == [
        component for component, _filename, _section in staged_build._COMPONENTS
    ]
    for component, kwargs in calls:
        if component in staged_build._QUANTIZED_TRANSFORMER_COMPONENTS:
            assert kwargs["quantized_transformer_path"] == quantized.absolute()
        else:
            assert "quantized_transformer_path" not in kwargs

    header, sections = read_bundle_file(str(output))
    config = json.loads(sections["config.json"])
    metadata = quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY.bundle_metadata()
    assert header["quantization"] == metadata["quantization"]
    assert config["precision"] == "bf16"
    assert config["quantization"] == QUANTIZED_TRANSFORMER_CONFIG
    assert config["quantization"]["checkpoint_scope"] == "full_transformer_overlay"
    assert config["quantization"]["quantized_weight_count"] == 250
    assert config["quantized_transformer"] == metadata
    assert str(tmp_path).lower() not in json.dumps(config).lower()
    assert (
        validate_native_bundle_config(output, source_revision=SOURCE_REVISION)[
            "quantized_transformer"
        ]
        == metadata
    )
    original_load_bundle_config = provenance.load_bundle_config
    for orphaned_field in ("quantized_transformer", "quantization"):
        orphaned = dict(config)
        orphaned.pop(orphaned_field)
        monkeypatch.setattr(provenance, "load_bundle_config", lambda _path, value=orphaned: value)
        with pytest.raises(ValueError, match="must be present together"):
            validate_native_bundle_config(output, source_revision=SOURCE_REVISION)
    monkeypatch.setattr(provenance, "load_bundle_config", original_load_bundle_config)

    receipt_path = output.with_name(f"{output.name}.plans") / staged_build._RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["build_identity"]["quantized_transformer"] == metadata
    assert receipt["build_identity"]["quantized_transformer_source_file_identity"] == (
        source_file_identity.receipt_metadata()
    )
    assert str(tmp_path).lower() not in json.dumps(receipt).lower()

    effective = SimpleNamespace(
        to_effective_dict=lambda: {
            "minimax_h3": {
                "transformer_ref": {"value": "", "source": "schema_default"},
                "quantized_transformer": {
                    "value": str(quantized.absolute()),
                    "source": "session_request",
                },
            }
        }
    )
    sidecar = write_path_free_effective_build_config(effective, output)
    sidecar_text = sidecar.read_text(encoding="utf-8").lower()
    assert str(quantized.absolute()).lower() not in sidecar_text
    quantized_summary = json.loads(sidecar_text)["minimax_h3"]["quantized_transformer"]["value"]
    assert quantized_summary["logical_role"] == "quantized_transformer"
    assert quantized_summary["sha256"] == metadata["sha256"]

    calls.clear()
    staged_build.build_staged_bundle(model, output, quantized_transformer=quantized)
    assert calls == []

    changed_identity = replace(
        quantized_identity,
        source_file_identity=replace(source_file_identity, mtime_ns=5),
    )
    monkeypatch.setattr(
        quantized_checkpoint,
        "validate_quantized_transformer_checkpoint",
        lambda _path: changed_identity,
    )
    with pytest.raises(ValueError, match="staged plans belong to a different checkpoint"):
        staged_build.build_staged_bundle(model, output, quantized_transformer=quantized)


def test_quantized_path_is_forwarded_to_the_selected_child_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized = _quantized_file(tmp_path)
    output = tmp_path / "tail.plan"
    observed = []

    def run(command, *, check):
        assert check is True
        observed.append(command)
        plan = Path(command[command.index("--output") + 1])
        record_path = Path(command[command.index("--record-output") + 1])
        record = _write_plan(plan, b"tail")
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    staged_build._run_component(
        "denoiser_tail",
        tmp_path,
        output,
        verbose=False,
        quantized_transformer_path=quantized.absolute(),
    )
    command = observed[0]
    assert command[command.index("--quantized-transformer") + 1] == str(quantized.absolute())
    with pytest.raises(ValueError, match="only build AdaLN and denoiser"):
        staged_build._run_component(
            "text_encoder",
            tmp_path,
            tmp_path / "text.plan",
            verbose=False,
            quantized_transformer_path=quantized.absolute(),
        )


def test_quantized_child_loads_only_dense_transformer_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized = _quantized_file(tmp_path)
    loads = []
    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(
        quantized_checkpoint,
        "validate_quantized_transformer_checkpoint",
        lambda _path: quantized_checkpoint.QUANTIZED_CHECKPOINT_IDENTITY,
    )
    monkeypatch.setattr(
        quantized_checkpoint,
        "load_selected_quantized_transformer_weights",
        lambda path, keys: loads.append((path, tuple(keys))) or {key: object() for key in keys},
    )
    monkeypatch.setattr(
        checkpoint,
        "load_selected_component_state_dict",
        lambda *_args, **_kwargs: pytest.fail("official transformer must not be loaded"),
    )

    family = "tensorrt_model_connect.families.minimax_h3"
    adaln = ModuleType(f"{family}.adaln_builder")
    adaln.checkpoint_keys = lambda _profile: ("quantized.adaln",)

    def build(weights, _profile, **options):
        assert set(weights) == {"quantized.adaln"}
        return _write_plan(Path(options["output_path"]), b"quantized-adaln")

    adaln.build_adaln_precompute_engine = build
    monkeypatch.setitem(sys.modules, adaln.__name__, adaln)
    output = tmp_path / "adaln.plan"
    assert staged_build._build_component(
        "adaln_precompute",
        tmp_path,
        output,
        verbose=False,
        quantized_transformer_path=quantized.absolute(),
    ) == _write_plan(output, b"quantized-adaln")
    assert loads == [(quantized.absolute(), ("quantized.adaln",))]

    with pytest.raises(ValueError, match="only build AdaLN and denoiser"):
        staged_build._build_component(
            "text_encoder",
            tmp_path,
            tmp_path / "text.plan",
            verbose=False,
            quantized_transformer_path=quantized.absolute(),
        )
