---
title: Add a Model Family
---

Create one directory:

```text
families/my_family/
├── __init__.py
├── model.py
├── runtime/
│   ├── CMakeLists.txt
│   └── plugin.cpp
└── tests/
    ├── test_e2e.py
    ├── manifests/<case>.json
    └── thresholds/<testcase>.json
```

`model.py` must expose exactly one plain function:

```python
def build(request, writer):
    if request.context_parallel_size != 1:
        raise ValueError("my_family does not support context parallelism")
    if request.task != "text_generation":
        raise ValueError("my_family supports only text_generation")
    writer.set_header(family="my_family", task=request.task, backend="trt")
    writer.add_json("runtime.json", {"tensor_parallel_size": 1})
    writer.add_bytes("engine.plan", build_engine(request))
```

The builder must not inherit from a base class. It may import the shared
`BuildRequest` and `BundleWriter` contracts, but it must not import another
family or shared model implementation.

The runtime CMake file creates `trtmc_model_my_family`. Its factory exports
`trtmc_create_family`, reads only sections owned by this family, and returns a
concrete implementation of an abstract interface in `trtmc/task.h`.

`FamilyContext.reader` is read-only. A factory normally consumes its sections
before returning. If a pipeline needs deferred section loading, copy the
`BundleReader` value into that pipeline; do not retain a reference to the
factory context.

Adding the directory must not require editing a core registry, source list, or
strategy map. Run:

```bash
python tools/test_impact.py --validate
```

`test_e2e.py` directly builds the bundle, invokes the native CLI through
`TRTMC_BINARY` and `TRTMC_RUNTIME_ROOT`, runs the official reference, and
applies only thresholds it actually reads. It must not import a central runner,
comparator, or sibling-family fixture.

An E2E manifest may declare an exact `hf_id` (and, when available,
`hf_revision`) or omit `hf_id` for a prepared local checkpoint supplied through
the family-specific model-directory environment variable. Do not invent an HF
ID for a checkpoint that is not published there.
