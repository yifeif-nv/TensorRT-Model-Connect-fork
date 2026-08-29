---
title: Python Build API
---

The shared Python API contains only `BuildRequest`, `build()`, and
`BundleWriter`.

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(BuildRequest(
    model_dir=Path("/models/gpt2"),
    output_path=Path("gpt2.bundle"),
    family="gpt2",
    task="text_generation",
    precision="fp16",
))
```

The core resolves one exact family module and calls its plain
`build(request, writer)` function. It does not retry another family.

`model_dir` is a local path. The selected family alone decides whether that
directory is a Hugging Face snapshot or a prepared checkpoint; the shared API
does not infer a source or impose a checkpoint schema.

`tensor_parallel_size` and `context_parallel_size` are direct request fields,
not an options bag. Every family must either implement the requested value or
reject it. Cosmos3 accepts context parallel size 1 or 2. FLUX and Wan accept
1, 2, 4, or 8; their family-owned builders and runtimes reject simultaneous
tensor and context parallelism. Other families require the default value 1.
