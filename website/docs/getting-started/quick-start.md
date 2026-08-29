---
title: Quick Start
---

Install the wheel produced by the release or local package stage:

```bash
python -m pip install /path/to/tensorrt_model_connect-0.1.0-*.whl
```

Families with extra build dependencies expose explicit extras. For example,
use `tensorrt-model-connect[sana-wm]` when building SANA-WM; unrelated families
do not require Torch.

Build a bundle from a local checkpoint directory:

```bash
python -m tensorrt_model_connect build /models/gpt2 \
  --family gpt2 \
  --task text_generation \
  --precision fp16 \
  --output gpt2.bundle
```

`--family` and `--task` are always explicit. The build core imports only
`families.gpt2.model` and calls `build(request, writer)` once. The selected
family owns the directory layout: it may accept a Hugging Face snapshot or a
prepared local checkpoint, but the core never guesses or downloads a source.

For a wheel install, resolve its native runtime directory directly from the
installed package:

```bash
TRTMC_RUNTIME_ROOT="$(python -c 'import pathlib, tensorrt_model_connect as m; print(pathlib.Path(m.__file__).parent / "bin")')"
trtmc run gpt2.bundle \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --prompt "Hello" \
  --max-new-tokens 32
```

For a native CMake install, point the loader at the directory containing the matching
`libtrtmc_core.so`, `libtrtmc_runtime.so`, `libtrtmc_backend_trt.so`, and
`libtrtmc_model_gpt2.so`. The loader reads the bundle header, loads exactly
those DSOs, and returns the abstract task interface declared by the bundle.

```bash
trtmc run gpt2.bundle \
  --runtime-root /opt/trtmc/lib \
  --prompt "Hello" \
  --max-new-tokens 32
```

The shell variable above is only a convenient explicit argument. The CLI never
searches environment variables, the current directory, or an installed
fallback runtime.
