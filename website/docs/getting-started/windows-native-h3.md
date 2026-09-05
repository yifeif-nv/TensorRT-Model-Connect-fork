---
title: Native Windows MiniMax H3
description: Build and run H3-Base with ModelConnect C++ and TensorRT-RTX.
---

This integration runs MiniMax H3-Base through the native ModelConnect C++
runtime and TensorRT-RTX. Python is used only to build the bundle; generation
does not invoke Python, PyTorch, FastVideo, Triton, FFmpeg, or a subprocess.
Windows Media Foundation reads reference media and writes H.264/AAC MP4 files.

A Ref2VA-enabled 13-plan bundle supports:

- T2VA from a prompt;
- FL2VA from a prompt plus a first frame, last frame, or both; and
- Ref2VA from a prompt plus ordered image, video, and/or audio references.

A smaller 9-plan base bundle supports T2VA and FL2VA only.

All workflows use the official BF16 weights and dense attention graph. T2VA and
FL2VA share one dynamic transformer plan; a Ref2VA-enabled bundle adds its
task-specific transformer, AdaLN, and reference encoders. TensorRT optimization
profile 0 specializes the tested five-second shape, while profile 1 handles
other supported prompt lengths, canvases, and durations. There is no separate
15-second model.

H3 aligns frame counts to `17 * n + 5` at 24 fps. Consequently, a request for
120 frames produces 124 frames (5.167 seconds), while 345 frames produces
14.375 seconds. Prompts are tokenized per request and are not fixed to the
example prompt.

H3-Context-IR and H3-Regenerate-2K are separate services and are not included.

## Build the native runtime

Run the helper from a clean Git checkout in an x64 Visual Studio 2022 developer
PowerShell with Git, Ninja, CMake, 64-bit CPython 3.10 or newer, CUDA 12.9,
and TensorRT-RTX 1.6.1.120 installed. The SDK must include a TensorRT-RTX
`win_amd64` wheel matching that CPython version. That is the tested Windows/SM120
toolchain; use the matching Python wheel and runtime DLL from the same SDK:

```powershell
$RepoRoot = (Resolve-Path '<ModelConnect-checkout>').Path
$CudaRoot = '<CUDA-12.9-root>'
$RtxRoot = '<TensorRT-RTX-root>'
$ArtifactRoot = (New-Item -ItemType Directory -Force `
    '<artifact-directory-outside-the-checkout>').FullName
$BuildRoot = Join-Path $ArtifactRoot 'build'
$InstallRoot = Join-Path $ArtifactRoot 'install'
$OutputRoot = New-Item -ItemType Directory -Force `
    (Join-Path $ArtifactRoot 'outputs') | Select-Object -ExpandProperty FullName

& (Join-Path $RepoRoot 'src\runtime\models\minimax_h3\build_windows.ps1') `
    -CudaRoot $CudaRoot `
    -TensorRtRtxRoot $RtxRoot `
    -BuildDirectory $BuildRoot
```

The helper builds only the runtime CLI, core library, TensorRT-RTX backend, and
MiniMax-H3 plugin. CUDA and MSVC runtimes are linked statically. The matching
TensorRT-RTX runtime DLL is copied beside `trtmc.exe`.
Pass `-BuildTests` when developing the integration to build and run its native
test set as well.

Install the runtime when consuming it from another C++ project:

```powershell
cmake --install $BuildRoot --prefix $InstallRoot --config Release
```

## Build the bundle

Install this repository's build-only Python package and the checkpoint-reading
dependencies. The builder can download the pinned checkpoint directly from
Hugging Face. ModelConnect consumes the official root Diffusers-format
conversion supplied alongside the original `FL2VA/` and `Ref2VA/` trees; the
allowlist below avoids downloading those duplicate original-format weights:

```powershell
$H3Revision = '48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc'
$CheckpointRoot = Join-Path $ArtifactRoot 'checkpoint'
$Bundle = Join-Path $ArtifactRoot 'MiniMax-H3.bundle'
$env:PATH = @((Join-Path $RtxRoot 'bin'), (Join-Path $RtxRoot 'lib'), `
    (Join-Path $CudaRoot 'bin'), $env:PATH) `
    -join [IO.Path]::PathSeparator

$PythonTag = & python -c `
    "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$RtxWheels = @(Get-ChildItem -LiteralPath (Join-Path $RtxRoot 'python') `
    -Filter "tensorrt_rtx-*-$PythonTag-none-win_amd64.whl" -File)
if ($RtxWheels.Count -ne 1) {
    throw "Expected exactly one TensorRT-RTX wheel for $PythonTag"
}

python -m pip install `
    "torch>=2.0" "safetensors>=0.4" "numpy>=1.24" `
    "ml_dtypes>=0.4" "huggingface_hub>=0.23" `
    "tomli>=2.0; python_version < '3.11'" `
    $RtxWheels[0].FullName
python -m pip install --no-deps -e $RepoRoot -C py-only=true
# Run hf auth login first if Hugging Face asks you to authenticate.
$RootPatterns = @(
    'model_index.json', 'modular_model_index.json',
    'processor/**', 'scheduler/**', 'audio_scheduler/**',
    'text_encoder/**', 'tokenizer/**', 'transformer/**',
    'transformer_ref/**', 'vae/**', 'audio_vae/**'
)
$Checkpoint = (& python -c `
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download('MiniMaxAI/MiniMax-H3', revision=sys.argv[2], local_dir=sys.argv[1], allow_patterns=sys.argv[3:]))" `
    $CheckpointRoot $H3Revision $RootPatterns).Trim()
```

Build the full T2VA, FL2VA, and Ref2VA bundle:

```powershell
$TransformerRef = Join-Path $Checkpoint 'transformer_ref'

python -m tensorrt_model_connect build $Checkpoint `
    --rtx --precision bf16 `
    --output $Bundle `
    --set "minimax_h3.transformer_ref=$TransformerRef"
```

For a 9-plan T2VA/FL2VA-only bundle, omit `transformer_ref/**` from
`$RootPatterns` and omit the final `--set` argument.

Large plans are written to `$Bundle.plans` as they complete. Repeating the same
command resumes an interrupted build. The largest engines use TensorRT-RTX's
default workspace limit; no H3-specific workspace cap is imposed.

The complete checkpoint with `transformer_ref` is about 196 GiB and the
resulting 13-plan bundle is about 178 GiB. Allow at least 450 GiB of free disk
for the checkpoint, resumable plans, and finalized bundle. The checkpoint is
covered by the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
in addition to this repository's license.

## Generate video

Use the runtime built above:

```powershell
$Trtmc = Join-Path $BuildRoot 'trtmc.exe'
```

T2VA, nominal five seconds:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'A cinematic sunrise over a mountain lake with synchronized birds and wind.' `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --output (Join-Path $OutputRoot 't2va-5s.mp4')
```

T2VA, longest aligned output:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'A continuous documentary shot with synchronized dialogue and ambience.' `
    --num-frames 345 --height 768 --width 1344 --seed 0 `
    --output (Join-Path $OutputRoot 't2va-14.375s.mp4')
```

FL2VA:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'Continue naturally between the supplied endpoints.' `
    --first-frame .\first.png --last-frame .\last.png `
    --num-frames 120 --seed 7 `
    --output (Join-Path $OutputRoot 'fl2va.mp4')
```

Ref2VA preserves the order of reference flags:

```powershell
& $Trtmc generate-video $Bundle `
    --prompt 'Use <Picture 1> as the subject and <Audio 1> as the voice reference.' `
    --reference-image .\subject.png `
    --reference-audio .\voice.wav `
    --num-frames 120 --height 768 --width 1344 --seed 11 `
    --output (Join-Path $OutputRoot 'ref2va.mp4')
```

Reference videos and explicit audio references must be 2--15 seconds. Ref2VA
accepts at most 9 images, 3 videos, 3 explicit audio files, and 12 files total.
Audio can be the sole Ref2VA input; references remain ordered exactly as supplied.
Prompt and dialogue text is UTF-8. The published stable dialogue languages are
Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese,
Russian, and Spanish.

## Call the bundle from C++

The CLI decodes media files and writes MP4. Native applications call the same
bundle through `trtmc::IPipeline` and pass already-decoded, host-resident value
types. For example:

```cmake
cmake_minimum_required(VERSION 3.24)
project(h3_consumer LANGUAGES CXX)

find_package(trtmc CONFIG REQUIRED)
add_executable(h3_consumer main.cpp)
target_link_libraries(h3_consumer PRIVATE trtmc::trtmc_core)
target_compile_features(h3_consumer PRIVATE cxx_std_17)
set_property(TARGET h3_consumer PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
```

Configure with `-DCMAKE_PREFIX_PATH=$InstallRoot`. Use the same Visual Studio
2022 toolset as ModelConnect and keep `$InstallRoot\bin` on `PATH` while the
application runs.

```cpp
#include <trtmc/pipeline.h>

#include <utility>

// Application-owned decoders. See the field layout comments in pipeline.h.
trtmc::VideoImageInput decode_image(const char* path);
trtmc::VideoClipInput decode_video(const char* path);
trtmc::AudioResult decode_audio(const char* path);

int main() {
    const std::string install_bin = R"(<install-root>\bin)";
    trtmc::LoadOptions options;
    options.runtime_cache_path = "minimax-h3.rtxcache";
    options.backend_search_paths = {install_bin};
    options.model_plugin_search_paths = {install_bin + R"(\trtmc\models)"};
    options.set_tokens = {
        "minimax_h3.retain_engines=true",
        "minimax_h3.retained_tail_weight_budget_gib=24",
    };
    auto pipeline = trtmc::load("MiniMax-H3.bundle", options);

    trtmc::VideoGenerationRequest fl;
    fl.prompt = "Continue naturally between the supplied endpoints.";
    fl.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    fl.first_frame = decode_image("first.png");
    fl.last_frame = decode_image("last.png");
    fl.config.video_num_frames = 120;
    fl.config.height = 768;
    fl.config.width = 1344;
    fl.config.seed = 7;
    trtmc::VideoResult fl_result = pipeline->generate_video(fl);

    trtmc::VideoGenerationRequest ref;
    ref.prompt = "Use <Picture 1> as the subject and <Audio 1> as the voice reference.";
    ref.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    ref.config.video_num_frames = 120;
    ref.config.height = 768;
    ref.config.width = 1344;
    ref.config.seed = 11;

    trtmc::VideoReferenceInput image;
    image.kind = trtmc::VideoReferenceKind::kImage;
    image.image = decode_image("subject.png");
    ref.references.push_back(std::move(image));

    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    audio.audio = decode_audio("voice.wav");
    ref.references.push_back(std::move(audio));
    trtmc::VideoResult ref_result = pipeline->generate_video(ref);
}
```

Keep `references` in semantic prompt order. A video entry uses
`VideoReferenceKind::kVideo` and `VideoReferenceInput::video`; its optional
soundtrack stays attached in `VideoClipInput::soundtrack`. `VideoResult` owns
the generated RGB frames in contiguous THWC order and interleaved stereo audio,
so the application can write them with its preferred container library without
adding a dependency to the model runtime.

## Reproduce the five-second performance result

Use the checked-in benchmark prompt, retain the five hot engines, and keep
a persistent TensorRT-RTX runtime cache:

```powershell
$Prompt = (Get-Content -Raw `
    (Join-Path $RepoRoot `
        'tests\e2e\models\minimax_h3\prompts\t2va-example-1.json') |
    ConvertFrom-Json).prompt
$RuntimeCache = Join-Path $ArtifactRoot 'minimax-h3-dense-fbc.rtxcache'

& $Trtmc generate-video $Bundle `
    --prompt $Prompt `
    --num-frames 120 --height 768 --width 1344 --seed 0 `
    --num-inference-steps 50 --guidance-scale 1 `
    --runtime-cache $RuntimeCache `
    --set "minimax_h3.retain_engines=true" `
    --set "minimax_h3.retained_tail_weight_budget_gib=24" `
    --set "minimax_h3.first_block_cache_threshold=0.30" `
    --warmup 1 --benchmark 1 `
    --output (Join-Path $OutputRoot 'minimax-h3-t2va-124f.mp4')
```

On the tested Spark system, the measured request completed in
542,663.046 ms (9:02.663). It used profile `0/2`, ran 49 transformer forwards,
and evaluated the tail 6 times while reusing it 43 times. The output contains
124 frames at 1344x768 and 24 fps plus stereo AAC audio at 32 kHz.

That 9:02.663 number is the measured iteration after one same-process warmup.
The command above intentionally generates twice; its cold load, warmup, measured
generation, and MP4 writes take substantially longer wall-clock time.

The `0.30` FirstBlockCache threshold is a measured preset, not a universal
default. Lower values recompute more tail steps; higher values can affect
quality. The first and final tail evaluations always run.

`generation_ms` measures the native `generate_video` call, including
conditioning, denoising, both VAEs, and device-to-host copies. It excludes
bundle loading and MP4 encoding, which the CLI reports separately. Hardware,
prompt length, and runtime-cache state can change the result.
