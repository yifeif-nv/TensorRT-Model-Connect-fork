/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

trtmc::cli::Command parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
}

bool parse_throws(std::vector<std::string> arguments) {
    try {
        (void)parse(std::move(arguments));
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

class FakeText final : public trtmc::ITextGeneration,
                       public trtmc::IEmbedding,
                       public trtmc::ILoraAdapterManager {
  public:
    trtmc::TextGenerationConfig seen;
    std::string loaded_adapter_id;
    std::string loaded_adapter_path;
    const char* task() const noexcept override { return trtmc::ITextGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 7; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        seen = config;
        return {prompt + ":" + std::to_string(config.max_new_tokens), {1, 2}};
    }

    trtmc::EmbeddingResult embed(const std::string& text) override {
        return {{static_cast<float>(text.size())}, 1};
    }

    void load_lora_adapter(const std::string& adapter_id,
                           const std::string& adapter_path) override {
        loaded_adapter_id = adapter_id;
        loaded_adapter_path = adapter_path;
    }

    void unload_lora_adapter(const std::string& adapter_id) override {
        if (loaded_adapter_id == adapter_id) {
            loaded_adapter_id.clear();
            loaded_adapter_path.clear();
        }
    }

    std::vector<std::string> loaded_lora_adapters() const override {
        return loaded_adapter_id.empty() ? std::vector<std::string>{}
                                         : std::vector<std::string>{loaded_adapter_id};
    }
};

class BareTextTask final : public trtmc::ITask {
  public:
    const char* task() const noexcept override { return trtmc::ITextGeneration::kTask; }
};

class EmptyImageWorker final : public trtmc::IImageGeneration {
  public:
    trtmc::ImageResult generate_image(const std::string&,
                                      const trtmc::ImageGenerationConfig&) override {
        trtmc::ImageResult result;
        result.num_frames = 0;
        return result;
    }
};

class FakeStreamingAudio final : public trtmc::IAudioGeneration,
                                 public trtmc::IStreamingAudioGeneration {
  public:
    trtmc::AudioGenerationConfig seen;
    std::int32_t seen_chunk_frames{0};

    trtmc::AudioResult generate_audio(const std::string&,
                                      const trtmc::AudioGenerationConfig&) override {
        throw std::logic_error("synchronous audio path was selected");
    }

    std::int32_t generate_audio_streaming(const std::string&,
                                          const trtmc::AudioGenerationConfig& config,
                                          trtmc::AudioChunkCallback callback,
                                          std::int32_t chunk_frames) override {
        seen = config;
        seen_chunk_frames = chunk_frames;
        const float samples[] = {0.25F, -0.5F, 0.75F};
        callback(samples, 3, 24000);
        return 3;
    }
};

class FakeForecast final : public trtmc::ITimeSeriesForecast {
  public:
    std::int32_t seen_frequency{-1};

    trtmc::ForecastResult forecast(const trtmc::ForecastRequest& request) override {
        seen_frequency = request.frequency;
        trtmc::ForecastResult result;
        result.values.assign(request.past_values.begin(), request.past_values.end());
        result.shape = {1, static_cast<std::int64_t>(request.past_values.size())};
        return result;
    }
};

bool dispatch_throws(const trtmc::cli::Command& command, trtmc::ITask& task) {
    try {
        std::ostringstream output;
        (void)trtmc::cli::dispatch(command, task, output);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    const std::vector<std::string> execution_commands{
        "run",
        "encode",
        "embed",
        "rerank",
        "classify",
        "extract-features",
        "disparity",
        "segment",
        "segment-prompted",
        "video-segment",
        "generate-audio",
        "transcribe",
        "transcribe-batch",
        "transcribe-streaming",
        "speak",
        "speech-session",
        "generate-image",
        "generate-image-batch",
        "generate-video",
        "solve",
        "forecast",
        "generate-world",
    };
    for (const auto& name : execution_commands) {
        const auto command = parse({"trtmc", name, "model.bundle", "--runtime-root", "lib"});
        check(command.name == name, "execution command parses");
        check(command.runtime_root == "lib", "runtime root is retained");
    }

    check(parse_throws({"trtmc", "run", "model.bundle"}),
          "execution command requires runtime root");
    check(parse_throws(
              {"trtmc", "run", "model.bundle", "--runtime-root", "a", "--runtime-root", "b"}),
          "duplicate runtime root rejected");
    check(
        parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--bogus", "value"}),
        "unknown command option rejected");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt", "a",
                        "--prompt", "b"}),
          "duplicate command option rejected");
    const auto byok = parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt",
                             "hello", "--byok-library", "kernel.so", "--byok-function", "run",
                             "--byok-name", "example.kernel"});
    check(byok.options.at("--byok-library") == "kernel.so" &&
              byok.options.at("--byok-function") == "run" &&
              byok.options.at("--byok-name") == "example.kernel",
          "BYOK options are retained");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt",
                        "hello", "--byok-library", "kernel.so"}),
          "partial BYOK options are rejected");
    const auto image =
        parse({"trtmc", "generate-image", "model.bundle", "--runtime-root", "lib", "--prompt",
               "cat", "--output", "cat.png", "--guidance-scale", "4.5", "--cfg-scale", "2.0"});
    check(image.options.at("--guidance-scale") == "4.5" && image.options.at("--cfg-scale") == "2.0",
          "image guidance options are retained");
    const auto batch =
        parse({"trtmc", "generate-image-batch", "model.bundle", "--runtime-root", "lib",
               "--prompts", "prompts.txt", "--seeds", "1,2", "--output", "images"});
    check(batch.kind == trtmc::cli::CommandKind::kGenerateImageBatch,
          "batch image command parses directly");

    const auto streaming_audio =
        parse({"trtmc", "generate-audio", "model.bundle", "--runtime-root", "lib", "--prompt",
               "hello", "--output", "/tmp/audio.raw", "--stream", "true", "--chunk-frames", "4"});
    check(streaming_audio.options.at("--stream") == "true" &&
              streaming_audio.options.at("--chunk-frames") == "4",
          "streaming audio options are retained");

    const auto video = parse({"trtmc", "video-segment", "model.bundle", "--runtime-root", "lib",
                              "--frame", "a.png", "--frame", "b.png", "--prompt", "car"});
    check(video.frames == std::vector<std::string>({"a.png", "b.png"}),
          "video frames preserve repeated option order");
    const auto transcription_batch =
        parse({"trtmc", "transcribe-batch", "model.bundle", "--runtime-root", "lib", "--input",
               "a.wav", "--input", "b.wav"});
    check(transcription_batch.inputs == std::vector<std::string>({"a.wav", "b.wav"}),
          "batch transcription inputs preserve repeated option order");
    check(parse({"trtmc", "inspect", "model.bundle"}).kind == trtmc::cli::CommandKind::kInspect,
          "inspect does not require runtime root");
    check(parse({"trtmc", "version"}).kind == trtmc::cli::CommandKind::kVersion,
          "version parses without bundle");
    check(parse_throws({"trtmc", "--version"}), "version flag alias is not accepted");
    check(parse_throws({"trtmc", "-h"}), "help flag alias is not accepted");

    trtmc::cli::Command run_command;
    run_command.kind = trtmc::cli::CommandKind::kRun;
    run_command.name = "run";
    run_command.options.emplace("--prompt", "hello");
    run_command.options.emplace("--max-new-tokens", "3");
    run_command.options.emplace("--temperature", "0.7");
    run_command.options.emplace("--top-k", "9");
    run_command.options.emplace("--top-p", "0.8");
    run_command.options.emplace("--min-p", "0.1");
    run_command.options.emplace("--seed", "42");
    run_command.options.emplace("--repetition-penalty", "1.1");
    run_command.options.emplace("--use-chat-template", "true");
    run_command.options.emplace("--enable-thinking", "false");
    run_command.options.emplace("--lora-adapter", "/tmp/adapter");
    run_command.options.emplace("--lora-adapter-id", "demo");

    FakeText text;
    std::ostringstream output;
    check(trtmc::cli::dispatch(run_command, text, output) == 0, "text task dispatch succeeds");
    check(output.str().find("hello:3") != std::string::npos,
          "text task dispatch writes result JSON");
    check(text.seen.temperature == 0.7F && text.seen.top_k == 9 && text.seen.top_p == 0.8F &&
              text.seen.min_p == 0.1F && text.seen.seed == 42 &&
              text.seen.repetition_penalty == 1.1F && text.seen.use_chat_template &&
              !text.seen.enable_thinking && text.seen.lora_adapter_id == "demo" &&
              text.loaded_adapter_id == "demo" && text.loaded_adapter_path == "/tmp/adapter",
          "text sampling options reach the Task API");

    trtmc::cli::Command embed_command;
    embed_command.kind = trtmc::cli::CommandKind::kEmbed;
    embed_command.name = "embed";
    embed_command.options.emplace("--text", "hello");
    std::ostringstream embed_output;
    check(trtmc::cli::dispatch(embed_command, text, embed_output) == 0,
          "secondary Task API dispatch succeeds");
    check(embed_output.str().find("5.0") != std::string::npos,
          "secondary Task API result is returned");

    BareTextTask bare;
    check(dispatch_throws(run_command, bare), "active task without exact interface is rejected");

    trtmc::cli::Command worker_command;
    worker_command.kind = trtmc::cli::CommandKind::kGenerateVideo;
    worker_command.name = "generate-video";
    worker_command.options.emplace("--prompt", "hello");
    worker_command.options.emplace("--output", "unused-worker-output");
    EmptyImageWorker worker;
    std::ostringstream worker_output;
    check(trtmc::cli::dispatch(worker_command, worker, worker_output) == 0,
          "empty context-parallel worker result succeeds");
    check(worker_output.str().find("\"worker\":true") != std::string::npos,
          "empty context-parallel worker result is explicit");
    worker_command.kind = trtmc::cli::CommandKind::kGenerateImage;
    worker_command.name = "generate-image";
    worker_output.str("");
    worker_output.clear();
    check(trtmc::cli::dispatch(worker_command, worker, worker_output) == 0,
          "empty context-parallel image worker succeeds");
    check(worker_output.str().find("\"worker\":true") != std::string::npos,
          "empty context-parallel image worker result is explicit");

    trtmc::cli::Command audio_command;
    audio_command.kind = trtmc::cli::CommandKind::kGenerateAudio;
    audio_command.name = "generate-audio";
    const std::filesystem::path audio_path = "/tmp/trtmc-cli-stream-test.raw";
    audio_command.options.emplace("--prompt", "hello");
    audio_command.options.emplace("--output", audio_path.string());
    audio_command.options.emplace("--max-new-tokens", "9");
    audio_command.options.emplace("--seed", "17");
    audio_command.options.emplace("--stream", "true");
    audio_command.options.emplace("--chunk-frames", "4");
    FakeStreamingAudio audio;
    std::ostringstream audio_output;
    check(trtmc::cli::dispatch(audio_command, audio, audio_output) == 0,
          "streaming audio task dispatch succeeds");
    check(std::filesystem::file_size(audio_path) == 3 * sizeof(float),
          "streaming audio writes raw float32 samples");
    check(audio.seen.max_new_tokens == 9 && audio.seen.seed == 17 && audio.seen_chunk_frames == 4,
          "streaming audio options reach the Task API");
    check(audio_output.str().find("\"format\":\"float32le\"") != std::string::npos,
          "streaming audio output format is explicit");
    std::filesystem::remove(audio_path);

    const std::filesystem::path forecast_values_path = "/tmp/trtmc-cli-forecast-values.f32";
    const std::filesystem::path forecast_mask_path = "/tmp/trtmc-cli-forecast-mask.f32";
    const float forecast_values[] = {1.0F, 2.0F};
    const float forecast_mask[] = {1.0F, 1.0F};
    {
        std::ofstream values_file(forecast_values_path, std::ios::binary);
        values_file.write(reinterpret_cast<const char*>(forecast_values), sizeof(forecast_values));
        std::ofstream mask_file(forecast_mask_path, std::ios::binary);
        mask_file.write(reinterpret_cast<const char*>(forecast_mask), sizeof(forecast_mask));
    }
    trtmc::cli::Command forecast_command;
    forecast_command.kind = trtmc::cli::CommandKind::kForecast;
    forecast_command.name = "forecast";
    forecast_command.options.emplace("--input", forecast_values_path.string());
    forecast_command.options.emplace("--mask", forecast_mask_path.string());
    forecast_command.options.emplace("--frequency", "7");
    FakeForecast forecast;
    std::ostringstream forecast_output;
    check(trtmc::cli::dispatch(forecast_command, forecast, forecast_output) == 0,
          "forecast task dispatch succeeds");
    check(forecast.seen_frequency == 7, "forecast frequency reaches the Task API");
    std::filesystem::remove(forecast_values_path);
    std::filesystem::remove(forecast_mask_path);

    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
