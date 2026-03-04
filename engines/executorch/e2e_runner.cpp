/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

// POSIX headers for stderr suppression
#include <fcntl.h>
#include <unistd.h>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "multimodal.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(data_path, "", "Path to data file.");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer stuff.");
DEFINE_string(prompt, "What is in this image?", "Text prompt.");
DEFINE_string(image_path, "", "Path to input image file.");

DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0. 0 = greedy argmax sampling (deterministic).");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads. Defaults to -1 (auto-detect performant cores).");

DEFINE_int32(target_size, 896, "Target image size for resizing.");
DEFINE_bool(warmup, false, "Whether to run a warmup run.");

// ---- Benchmark flags --------------------------------------------------------
DEFINE_bool(benchmark, false, "Run PP/TG benchmark instead of normal inference.");

DEFINE_string(
    pp,
    "2,4,8,16,32,64,128,256,512",
    "Comma-separated prefill token counts to benchmark, e.g. \"2,4,8\".");

DEFINE_string(
    tg,
    "32,64,128,256",
    "Comma-separated generation token counts to benchmark, e.g. \"32,64\".");

DEFINE_int32(n_runs, 3, "Number of runs per benchmark config (results averaged).");
// -----------------------------------------------------------------------------

namespace {

using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::extension::llm::Stats;

// Silence stdout+stderr during benchmark runs to suppress runner internal noise
// (PyTorchObserver JSON, prompt echoes, re2 warnings, etc.)
struct OutputSuppressor {
  int saved_stdout = -1;
  int saved_stderr = -1;
  int devnull_fd   = -1;

  void suppress() {
    fflush(stdout);
    fflush(stderr);
    devnull_fd   = open("/dev/null", O_WRONLY);
    saved_stdout = dup(STDOUT_FILENO);
    saved_stderr = dup(STDERR_FILENO);
    dup2(devnull_fd, STDOUT_FILENO);
    dup2(devnull_fd, STDERR_FILENO);
  }

  void restore() {
    if (saved_stdout >= 0) {
      dup2(saved_stdout, STDOUT_FILENO);
      dup2(saved_stderr, STDERR_FILENO);
      close(saved_stdout);
      close(saved_stderr);
      close(devnull_fd);
      saved_stdout = -1;
      saved_stderr = -1;
    }
  }

  ~OutputSuppressor() { restore(); }
};

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
      str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      result.push_back(std::stoi(token));
    }
  }
  return result;
}

// Build a text-only multimodal input with Gemma3 chat template (normal inference / TG)
std::vector<MultimodalInput> make_text_inputs(const std::string& prompt_str) {
  return {
      make_text_input(
          "<start_of_turn>user\n" + prompt_str +
          "<end_of_turn>\n<start_of_turn>model\n"),
  };
}

// Raw input WITHOUT chat template, used for PP benchmark so
// actual token count matches the target as closely as possible.
std::vector<MultimodalInput> make_raw_inputs(const std::string& text) {
  return { make_text_input(text) };
}

// Generate a repeated-word dummy prompt targeting exactly n_tokens total.
// Gemma tokenizer prepends 1 BOS token automatically, so we generate
// (n_tokens - 1) words. "hello " = 1 token in Gemma vocab.
// Actual count is confirmed via Stats.num_prompt_tokens.
std::string make_dummy_prompt(int n_tokens) {
  int n_words = std::max(1, n_tokens - 1);  // subtract 1 for BOS
  std::string result;
  result.reserve(n_words * 6);
  for (int i = 0; i < n_words; ++i) {
    result += "hello ";
  }
  return result;
}

MultimodalInput loadImage(const std::string& image_path) {
  if (!ends_with(image_path, ".jpg") && !ends_with(image_path, ".jpeg") &&
      !ends_with(image_path, ".png") && !ends_with(image_path, ".bmp")) {
    ET_LOG(
        Error,
        "Unsupported image file format: %s (only .jpg, .jpeg, .png, .bmp are supported)",
        image_path.c_str());
    throw std::runtime_error("Unsupported image file format");
  }

  int width, height, channels;
  unsigned char* data =
      stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    ET_LOG(Error, "Failed to load image: %s", image_path.c_str());
    throw std::runtime_error("Failed to load image");
  }

  ET_LOG(Info, "Loaded image: %s, size: %dx%d, channels: %d",
         image_path.c_str(), width, height, channels);

  const int target_size = FLAGS_target_size;
  std::vector<uint8_t> resized_data(target_size * target_size * channels);
  int resize_result = stbir_resize_uint8(
      data, width, height, 0,
      resized_data.data(), target_size, target_size, 0, channels);

  if (!resize_result) {
    stbi_image_free(data);
    throw std::runtime_error("Failed to resize image");
  }

  std::vector<float> chw_data(channels * target_size * target_size);
  for (int h = 0; h < target_size; ++h) {
    for (int w = 0; w < target_size; ++w) {
      for (int c = 0; c < channels; ++c) {
        uint8_t px = resized_data[h * target_size * channels + w * channels + c];
        chw_data[c * target_size * target_size + h * target_size + w] =
            static_cast<float>(px) / 255.0f;
      }
    }
  }

  Image image(std::move(chw_data), target_size, target_size, channels);
  stbi_image_free(data);
  return make_image_input(std::move(image));
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

struct BenchResult {
  int    target;       // requested token count
  int    actual;       // actual token count from Stats
  double speed_tok_s;  // tokens / second
  double time_ms;      // wall time in ms
};

void print_table_header(const char* title) {
  printf("\n====== %s ======\n", title);
  printf("%-14s  %-12s  %-12s  %-14s\n",
         "Target tok", "Actual tok", "Time (ms)", "Speed (tok/s)");
  printf("%-14s  %-12s  %-12s  %-14s\n",
         "--------------", "------------", "------------", "--------------");
}

void print_result(const BenchResult& r) {
  printf("%-14d  %-12d  %-12.1f  %-14.2f\n",
         r.target, r.actual, r.time_ms, r.speed_tok_s);
  fflush(stdout);
}

// PP: feed n_tokens prompt, max_new_tokens=1, measure prefill time
// PP time = first_token_ms - inference_start_ms
// (covers tokenize + full prefill pass, up to first generated token)
BenchResult run_pp_once(
    ::executorch::extension::llm::MultimodalRunner* runner,
    int n_tokens,
    float temperature) {
  // Use raw text (no chat template) so token count matches target precisely
  auto inputs = make_raw_inputs(make_dummy_prompt(n_tokens));

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = 1;
  config.temperature = temperature;

  BenchResult result{n_tokens, 0, 0.0, 0.0};

  // Suppress token output during benchmark
  auto token_cb = [](const std::string&) {};

  auto stats_cb = [&](const Stats& s) {
    // prefill = from inference start to first token emitted
    long prefill_ms = s.first_token_ms - s.inference_start_ms;
    if (prefill_ms <= 0) {
      // fallback: use prompt_eval_end_ms if first_token_ms not set
      prefill_ms = s.prompt_eval_end_ms - s.inference_start_ms;
    }
    result.actual      = static_cast<int>(s.num_prompt_tokens);
    result.time_ms     = static_cast<double>(prefill_ms);
    result.speed_tok_s = prefill_ms > 0
        ? static_cast<double>(s.num_prompt_tokens) / prefill_ms * 1000.0
        : 0.0;
  };

  OutputSuppressor suppress;
  suppress.suppress();
  auto err = runner->generate(inputs, config, token_cb, stats_cb);
  suppress.restore();

  if (err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "PP run failed for n_tokens=%d", n_tokens);
  }
  runner->reset();
  return result;
}

// TG: feed short prompt, generate n_tokens, measure decode throughput
// TG time = inference_end_ms - first_token_ms
BenchResult run_tg_once(
    ::executorch::extension::llm::MultimodalRunner* runner,
    int n_tokens,
    float temperature) {
  auto inputs = make_text_inputs("Hello, please count from one:");

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = n_tokens;
  config.temperature = temperature;

  BenchResult result{n_tokens, 0, 0.0, 0.0};

  auto token_cb = [](const std::string&) {};

  auto stats_cb = [&](const Stats& s) {
    // decode = from first token to end of generation
    long tg_ms     = s.inference_end_ms - s.first_token_ms;
    result.actual  = static_cast<int>(s.num_generated_tokens);
    result.time_ms = static_cast<double>(tg_ms);
    result.speed_tok_s = (tg_ms > 0 && result.actual > 0)
        ? static_cast<double>(result.actual) / tg_ms * 1000.0
        : 0.0;
  };

  OutputSuppressor suppress;
  suppress.suppress();
  auto err = runner->generate(inputs, config, token_cb, stats_cb);
  suppress.restore();

  if (err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "TG run failed for n_tokens=%d", n_tokens);
  }
  runner->reset();
  return result;
}

BenchResult average_runs(
    std::function<BenchResult()> run_fn,
    int n_runs,
    int target) {
  std::vector<double> speeds, times;
  int actual = 0;
  for (int i = 0; i < n_runs; ++i) {
    auto r = run_fn();
    speeds.push_back(r.speed_tok_s);
    times.push_back(r.time_ms);
    actual = r.actual;
    printf("  run %d/%d: actual=%d  time=%.1fms  speed=%.2f tok/s\n",
           i + 1, n_runs, r.actual, r.time_ms, r.speed_tok_s);
    fflush(stdout);
  }
  double avg_speed = std::accumulate(speeds.begin(), speeds.end(), 0.0) / n_runs;
  double avg_time  = std::accumulate(times.begin(),  times.end(),  0.0) / n_runs;
  return {target, actual, avg_speed, avg_time};
}

} // namespace

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path     = FLAGS_model_path.c_str();
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt         = FLAGS_prompt.c_str();
  const char* image_path     = FLAGS_image_path.c_str();
  const char* data_path      = FLAGS_data_path.c_str();
  float  temperature         = static_cast<float>(FLAGS_temperature);
  int32_t cpu_threads        = FLAGS_cpu_threads;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer from: %s", tokenizer_path);
    return 1;
  }

  std::unique_ptr<::executorch::extension::llm::MultimodalRunner> runner =
      ::executorch::extension::llm::create_multimodal_runner(
          model_path, std::move(tokenizer), data_path);
  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create multimodal runner");
    return 1;
  }

  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load multimodal runner");
    return 1;
  }

  // =========================================================================
  // Benchmark mode
  // =========================================================================
  if (FLAGS_benchmark) {
    int n_runs = FLAGS_n_runs;
    printf("\nModel : %s\n", model_path);
    printf("Runs  : %d per config (averaged)\n", n_runs);

    // Warmup
    printf("\n[Warmup...]\n");
    fflush(stdout);
    run_pp_once(runner.get(), 8, temperature);
    printf("[Warmup done]\n");

    // PP benchmark
    auto pp_list = parse_int_list(FLAGS_pp);
    print_table_header("Prefill (PP) Benchmark");
    for (int n : pp_list) {
      printf("\nPP target=%d tok:\n", n);
      fflush(stdout);
      auto result = average_runs(
          [&]() { return run_pp_once(runner.get(), n, temperature); },
          n_runs, n);
      printf("  AVG -> actual=%-5d  time=%-8.1fms  speed=%.2f tok/s\n",
             result.actual, result.time_ms, result.speed_tok_s);
    }

    // TG benchmark
    auto tg_list = parse_int_list(FLAGS_tg);
    print_table_header("Token Generation (TG) Benchmark");
    for (int n : tg_list) {
      printf("\nTG target=%d tok:\n", n);
      fflush(stdout);
      auto result = average_runs(
          [&]() { return run_tg_once(runner.get(), n, temperature); },
          n_runs, n);
      printf("  AVG -> actual=%-5d  time=%-8.1fms  speed=%.2f tok/s\n",
             result.actual, result.time_ms, result.speed_tok_s);
    }

    printf("\n");
    return 0;
  }

  // =========================================================================
  // Normal inference mode
  // =========================================================================
  if (FLAGS_warmup) {
    ET_LOG(Info, "Running warmup...");
    ::executorch::extension::llm::GenerationConfig warmup_config;
    warmup_config.max_new_tokens = 1;
    warmup_config.temperature = temperature;
    auto wi = make_text_inputs("Hello");
    runner->generate(wi, warmup_config, [](const std::string&){});
    runner->reset();
  }

  std::vector<MultimodalInput> inputs;
  if (std::string(image_path).empty()) {
    ET_LOG(Info, "Running in text-only mode");
    inputs = make_text_inputs(prompt);
  } else {
    ET_LOG(Info, "Running in multimodal mode with image: %s", image_path);
    inputs = {
        make_text_input("<start_of_turn>user\n<start_of_image>"),
        loadImage(image_path),
        make_text_input(
            std::string(prompt) + "<end_of_turn>\n<start_of_turn>model\n"),
    };
  }

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = 200;
  config.temperature = temperature;

  // Print tokens as they stream out
  auto token_cb = [](const std::string& tok) {
    printf("%s", tok.c_str());
    fflush(stdout);
  };

  // Print stats after generation
  auto stats_cb = [](const Stats& s) {
    long prefill_ms = s.first_token_ms - s.inference_start_ms;
    if (prefill_ms <= 0) {
      prefill_ms = s.prompt_eval_end_ms - s.inference_start_ms;
    }
    long tg_ms = s.inference_end_ms - s.first_token_ms;

    double pp_speed = (prefill_ms > 0 && s.num_prompt_tokens > 0)
        ? static_cast<double>(s.num_prompt_tokens) / prefill_ms * 1000.0 : 0.0;
    double tg_speed = (tg_ms > 0 && s.num_generated_tokens > 0)
        ? static_cast<double>(s.num_generated_tokens) / tg_ms * 1000.0 : 0.0;

    printf("\n\n=== Stats ===\n");
    printf("Prompt tokens    : %lld\n",  (long long)s.num_prompt_tokens);
    printf("Generated tokens : %lld\n",  (long long)s.num_generated_tokens);
    printf("Prefill          : %ld ms  (%.2f tok/s)\n", prefill_ms, pp_speed);
    printf("Decode           : %ld ms  (%.2f tok/s)\n", tg_ms, tg_speed);
    printf("Model load       : %ld ms\n",
           s.model_load_end_ms - s.model_load_start_ms);
  };

  auto error = runner->generate(inputs, config, token_cb, stats_cb);
  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate");
    return 1;
  }

  return 0;
}
