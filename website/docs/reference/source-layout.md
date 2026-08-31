---
title: Source Layout
---

```text
families/<family>/             complete model-owned vertical slice; optional requirements.txt
requirements/base.txt         thin build-tool delta over the pinned GPU base
python/tensorrt_model_connect/ minimal Python build and bundle contracts
include/trtmc/                 abstract Task, Engine, bundle, and loader APIs
src/bundle/                    reader-only bundle container implementation
src/runtime/core/              libtrtmc_core.so device and engine primitives
src/runtime/loader/            libtrtmc_runtime.so exact DSO loader
src/runtime/backend/           libtrtmc_backend_trt.so implementation
src/cli/                       CLI and its private image/audio file I/O
tests/core/                    shared-boundary tests only
tools/model_ci.py              family inventory and impact
website/                       documentation generated from family ownership
```

No production family source lives under `python/`, `src/runtime/models/`, or
`tests/e2e/models/`.
