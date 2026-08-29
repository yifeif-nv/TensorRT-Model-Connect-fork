<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bring your own kernel

BYOK is an optional one-way extension: an application or family builder depends
on ModelConnect's public BYOK API; ModelConnect never depends on an example.

Install the optional runtime and build the identity example:

```bash
python -m pip install -e .
cmake -S . -B build -DTRTMC_BUILD_EXAMPLES=ON -DTRTMC_ENABLE_BYOK=ON
cmake --build build --target trtmc_byok_identity_copy test_byok_tvm_ffi
ctest --test-dir build -R '^byok_tvm_ffi$' --output-on-failure
cmake --build build --parallel
cmake --install build --prefix "$PWD/build/install"
```

The test loads `identity_copy_kernel.so` through `trtmc::load_byok_kernel`,
builds and serializes a TensorRT engine containing `TvmFfiKernel`, deserializes
it, runs it on CUDA, and checks the output.

Install `.[cutedsl]` to compile the CuTe DSL residual-add example.

A family-owned builder adds an external kernel directly to its TensorRT graph:

```python
from tensorrt_model_connect.byok import add_kernel

output, = add_kernel(
    network,
    plugin_library="/absolute/path/build/install/lib/libtrtmc_byok_tvm_ffi.so",
    kernel_name="my_family.residual_add",
    inputs=[hidden, attention_projection],
    output_specs=[{"dims": [256, 768], "dtype": "float16"}],
)
```

Load the matching module before the bundle:

```bash
trtmc run model.bundle \
  --runtime-root build/install/lib \
  --byok-library ./residual_add.so \
  --byok-function run \
  --byok-name my_family.residual_add \
  --prompt "Hello"
```

`export_cutedsl_residual_add.py` compiles the fixed-shape FP16 residual-add DSO.
The kernel name, function, tensor shape, and dtype are explicit; no graph,
source, or ABI hash is generated or checked.
