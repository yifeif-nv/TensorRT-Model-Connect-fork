# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build or reuse the single current TRTMC development image."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .process import CiError, CommandRunner, GitHubFiles


class DockerImageManager:
    def __init__(self, repository: Path, env: dict[str, str]):
        self.repository = repository.resolve()
        self.env = env
        self.commands = CommandRunner(cwd=self.repository, env=env)
        self.github = GitHubFiles(env)

    def ensure(self) -> str:
        image = self.env.get("TRTMC_CI_IMAGE", "trtmc-ci:trt11.1")
        self.commands.run(
            [
                "docker",
                "build",
                "--file",
                "Dockerfile",
                "--tag",
                image,
                ".",
            ]
        )
        if self._inspect(image).returncode:
            raise CiError(f"Docker image was not created: {image}")
        self.github.environment("TRTMC_CI_IMAGE", image)
        print(f"TRTMC_CI_IMAGE={image}")
        return image

    def source_contract_json(
        self,
        *,
        tensorrt_version: str | None = None,
        tensorrt_apt_version: str | None = None,
    ) -> str:
        contract = self._source_contract()
        requested = {
            "tensorrt_version": tensorrt_version,
            "tensorrt_apt_version": tensorrt_apt_version,
        }
        for field, value in requested.items():
            if value is not None and value != contract[field]:
                raise CiError(f"Dockerfile supports only {field}={contract[field]}, got {value}")
        return json.dumps(contract, sort_keys=True)

    def _source_contract(self) -> dict[str, str]:
        source = (self.repository / "Dockerfile").read_text(encoding="utf-8")
        trt = re.search(r'"tensorrt==([0-9.]+)"', source)
        apt = re.search(r'"libnvinfer-dev=([^"\\]+)"', source)
        if trt is None or apt is None:
            raise CiError("Dockerfile does not declare one exact TensorRT contract")
        return {
            "dockerfile": "Dockerfile",
            "tensorrt_version": trt.group(1),
            "tensorrt_apt_version": apt.group(1),
        }

    def _inspect(self, image: str):
        return self.commands.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
        )
