# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only the three explicit selectors shared by family-owned E2E tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-model",
        action="append",
        default=[],
        help="Select a family or manifest; repeat or comma-separate values",
    )
    parser.addoption(
        "--e2e-testcase",
        action="append",
        default=[],
        help="Select an exact family-owned testcase",
    )
    parser.addoption(
        "--e2e-models-file",
        default=None,
        help="Select names listed one per line in a file",
    )
