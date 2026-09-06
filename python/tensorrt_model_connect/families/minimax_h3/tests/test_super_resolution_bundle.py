# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.minimax_h3 import provenance as provenance_module
from tensorrt_model_connect.families.minimax_h3.plugin import (
    _super_resolution_build_inputs,
    write_path_free_effective_build_config,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    super_resolution_bundle_config,
    super_resolution_source_identity,
    validate_super_resolution_bundle_config,
    validate_super_resolution_source_identity,
)
from tensorrt_model_connect.families.minimax_h3.runtime_config_schema import SCHEMA
from tensorrt_model_connect.runtime_config import Layer


plugin_module = importlib.import_module("tensorrt_model_connect.families.minimax_h3.plugin")


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "realesr-general-x4v3.pth"
    weak = tmp_path / "realesr-general-wdn-x4v3.pth"
    primary.write_bytes(b"primary")
    weak.write_bytes(b"weak")
    return primary, weak


def _bind_test_checkpoint_digests(
    monkeypatch: pytest.MonkeyPatch, primary: Path, weak: Path
) -> None:
    monkeypatch.setattr(provenance_module, "SUPER_RESOLUTION_PRIMARY_BYTES", primary.stat().st_size)
    monkeypatch.setattr(
        provenance_module,
        "SUPER_RESOLUTION_PRIMARY_SHA256",
        hashlib.sha256(primary.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(provenance_module, "SUPER_RESOLUTION_WEAK_BYTES", weak.stat().st_size)
    monkeypatch.setattr(
        provenance_module,
        "SUPER_RESOLUTION_WEAK_SHA256",
        hashlib.sha256(weak.read_bytes()).hexdigest(),
    )


def test_super_resolution_checkpoint_options_are_build_only(tmp_path: Path) -> None:
    fields = {field.name: field for field in SCHEMA.fields}
    for name in ("super_resolution_model", "super_resolution_weak_model"):
        field = fields[name]
        assert field.type_tag == "string"
        assert field.default == ""
        assert Layer.BUILD_TIME in field.allowed_layers
        assert Layer.SESSION_REQUEST in field.allowed_layers
        assert Layer.BUNDLE_DEFAULT not in field.allowed_layers
        assert Layer.PLATFORM_PROFILE not in field.allowed_layers

    primary, weak = _checkpoints(tmp_path)
    assert _super_resolution_build_inputs(
        {
            "super_resolution_model": primary,
            "super_resolution_weak_model": weak,
        }
    ) == (primary.absolute(), weak.absolute(), 0.5)
    assert _super_resolution_build_inputs({"super_resolution_model": primary}) == (
        primary.absolute(),
        None,
        1.0,
    )
    with pytest.raises(ValueError, match="requires super_resolution_model"):
        _super_resolution_build_inputs({"super_resolution_weak_model": weak})


def test_super_resolution_bundle_contract_is_exact_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, weak = _checkpoints(tmp_path)
    _bind_test_checkpoint_digests(monkeypatch, primary, weak)
    identity = super_resolution_source_identity(
        primary,
        weak,
        denoise_strength=0.5,
    )
    config = super_resolution_bundle_config(identity)

    assert validate_super_resolution_bundle_config(config) == config
    assert config["section"] == "video_super_resolution_plan"
    assert config["source_shape"] == [480, 864]
    assert config["target_shape"] == [720, 1296]
    assert config["batch_profile"] == [1, 4, 8]
    assert config["layout"] == "NHWC"
    assert config["io_dtype"] == "float32"
    assert config["scale"] == config["model_upscale"] == 4
    assert config["delivery_scale"] == 1.5
    assert config["denoise_strength"] == 0.5
    assert config["learned_residual_strength"] == 0.25
    assert config["dni"] == {
        "enabled": True,
        "primary_weight": 0.5,
        "weak_denoise_weight": 0.5,
    }
    assert str(tmp_path).lower() not in json.dumps(config).lower()

    malformed = dict(config)
    malformed["layout"] = "NCHW"
    with pytest.raises(ValueError, match="mismatch for layout"):
        validate_super_resolution_bundle_config(malformed)

    unsupported_blend = dict(identity)
    unsupported_blend["denoise_strength"] = 0.75
    with pytest.raises(ValueError, match="unsupported DNI blend"):
        validate_super_resolution_source_identity(unsupported_blend)


def test_super_resolution_rejects_primary_checkpoint_renamed_as_weak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, weak = _checkpoints(tmp_path)
    _bind_test_checkpoint_digests(monkeypatch, primary, weak)
    weak.write_bytes(primary.read_bytes())

    with pytest.raises(ValueError, match="does not match the official digest"):
        super_resolution_source_identity(primary, weak, denoise_strength=0.5)


def test_effective_config_replaces_both_super_resolution_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary, weak = _checkpoints(tmp_path)
    _bind_test_checkpoint_digests(monkeypatch, primary, weak)
    runtime_sr = super_resolution_bundle_config(
        super_resolution_source_identity(primary, weak, denoise_strength=0.5)
    )
    artifact = tmp_path / "h3.bundle"
    monkeypatch.setattr(
        plugin_module,
        "load_bundle_config",
        lambda _path: {
            "checkpoint_revision": "a" * 40,
            "checkpoint_inventory_sha256": "b" * 64,
            "source_revision": "c" * 40,
            "builder_source_sha256": "d" * 64,
            "super_resolution": runtime_sr,
        },
    )
    effective = SimpleNamespace(
        to_effective_dict=lambda: {
            "minimax_h3": {
                "transformer_ref": {"value": "", "source": "schema_default"},
                "quantized_transformer": {"value": "", "source": "schema_default"},
                "super_resolution_model": {
                    "value": str(primary.absolute()),
                    "source": "session_request",
                },
                "super_resolution_weak_model": {
                    "value": str(weak.absolute()),
                    "source": "session_request",
                },
            }
        }
    )

    sidecar = write_path_free_effective_build_config(effective, artifact)
    text = sidecar.read_text(encoding="utf-8")
    payload = json.loads(text)["minimax_h3"]
    assert str(primary.absolute()).lower() not in text.lower()
    assert str(weak.absolute()).lower() not in text.lower()
    assert payload["super_resolution_model"]["value"]["logical_role"] == (
        "super_resolution_primary"
    )
    assert payload["super_resolution_weak_model"]["value"]["logical_role"] == (
        "super_resolution_weak_denoise"
    )
