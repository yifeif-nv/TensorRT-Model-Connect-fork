# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run family-owned E2E tests directly, without a central harness."""

from __future__ import annotations

import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from tools import test_impact

from .context import CiContext
from .process import CiError


class E2ERunner:
    def __init__(self, context: CiContext):
        self.context = context

    def selective(self) -> None:
        impact_path = self.context.repository / "impact.json"
        if not impact_path.is_file():
            raise CiError("selective E2E requires impact.json")
        payload = json.loads(impact_path.read_text(encoding="utf-8"))
        self._run(
            tuple(str(value) for value in payload["families"]),
            tuple(str(value) for value in payload.get("testcases", ())),
        )

    def full(self) -> None:
        self._run(test_impact.inventory(self.context.repository))

    def _run(self, families: tuple[str, ...], testcases: tuple[str, ...] = ()) -> None:
        if not families:
            print("No family E2E tests selected")
            return
        binary = self._required_path("TRTMC_BINARY")
        runtime_root = self._required_path("TRTMC_RUNTIME_ROOT")
        native_build = self._required_path("TRTMC_NATIVE_BUILD_DIR")
        if not binary.is_file():
            raise CiError(f"TRTMC_BINARY is not a file: {binary}")
        if not runtime_root.is_dir():
            raise CiError(f"TRTMC_RUNTIME_ROOT is not a directory: {runtime_root}")
        if not native_build.is_dir() or not (native_build / "CTestTestfile.cmake").is_file():
            raise CiError("TRTMC_NATIVE_BUILD_DIR is not a configured CTest build tree")
        if not (runtime_root / "libtrtmc_backend_trt.so").is_file():
            raise CiError("TRTMC_RUNTIME_ROOT has no libtrtmc_backend_trt.so")
        known = set(test_impact.inventory(self.context.repository))
        unknown = sorted(set(families) - known)
        if unknown:
            raise CiError("unknown E2E families: " + ", ".join(unknown))
        tests = [f"families/{family}/tests/test_e2e.py" for family in sorted(set(families))]
        self._run_family_ctests(native_build, tuple(sorted(set(families))))
        for family, test in zip(sorted(set(families)), tests):
            with self._isolated_runtime_root(runtime_root, family) as isolated:
                command = [
                    "python",
                    "-m",
                    "pytest",
                    test,
                    "--e2e-model",
                    family,
                ]
                if testcases:
                    command.extend(("--e2e-testcase", ",".join(sorted(set(testcases)))))
                command.extend(("-q", "-x", "-p", "no:cacheprovider"))
                self.context.run(
                    command,
                    updates={
                        "TRTMC_BINARY": str(binary),
                        "TRTMC_RUNTIME_ROOT": str(isolated),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    limit=self.context.env.get("TRTMC_E2E_TIMEOUT", "12h"),
                )

    @contextmanager
    def _isolated_runtime_root(self, runtime_root: Path, family: str):
        required = (
            "libtrtmc_core.so",
            "libtrtmc_backend_trt.so",
            f"libtrtmc_model_{family}.so",
        )
        for name in required:
            if not (runtime_root / name).is_file():
                raise CiError(f"TRTMC_RUNTIME_ROOT has no {name}")

        with tempfile.TemporaryDirectory(prefix=f"trtmc-{family}-runtime-") as directory:
            root = Path(directory)
            isolated = root / "tensorrt_model_connect/bin"
            isolated.mkdir(parents=True)
            for name in required:
                (isolated / name).symlink_to((runtime_root / name).resolve())

            # Preserve only non-family wheel dependencies expected by RUNPATH.
            site_packages = runtime_root.parent.parent
            for package in ("tensorrt_libs", "torch"):
                source = site_packages / package
                if source.is_dir():
                    (root / package).symlink_to(source.resolve(), target_is_directory=True)
            yield isolated

    def _run_family_ctests(self, build: Path, families: tuple[str, ...]) -> None:
        listing = self.context.run(
            ["ctest", "--test-dir", build, "--show-only=json-v1"],
            capture_output=True,
        )
        tests = json.loads(listing.stdout).get("tests", [])
        selected = []
        targets = []
        owners = {f"/families/{family}" for family in families}
        for test in tests:
            command = test.get("command") or []
            properties = {
                str(item["name"]): str(item["value"])
                for item in test.get("properties", [])
                if "name" in item and "value" in item
            }
            location = properties.get("WORKING_DIRECTORY", "")
            if not any(location.endswith(owner) or f"{owner}/" in location for owner in owners):
                continue
            name = str(test["name"])
            selected.append(name)
            targets.append(
                Path(str(command[0])).name
                if command
                else name
                if name.startswith("test_")
                else f"test_{name}"
            )
        if not selected:
            print("No family-owned C++ tests selected")
            return
        self.context.run(
            [
                "cmake",
                "--build",
                build,
                "--parallel",
                "8",
                "--target",
                *sorted(set(targets)),
            ],
            limit=self.context.env.get("CPP_BUILD_TIMEOUT", "30m"),
        )
        expression = "^(" + "|".join(re.escape(name) for name in sorted(selected)) + ")$"
        self.context.run(
            [
                "ctest",
                "--test-dir",
                build,
                "--output-on-failure",
                "-R",
                expression,
            ],
            limit=self.context.env.get("CPP_UNIT_TIMEOUT", "20m"),
        )

    def _required_path(self, name: str) -> Path:
        value = self.context.env.get(name, "")
        if not value:
            raise CiError(f"{name} is required")
        path = Path(value)
        if not path.exists():
            raise CiError(f"{name} does not exist: {path}")
        return path
