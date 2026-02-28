// model_bench.cpp
// onnxruntime-genai C++ inference + llama.cpp-style PP/TG benchmark
//
// Build:
//   mkdir build && cd build
//   cmake .. -DOGA_DIR=/path/to/onnxruntime-genai -DCMAKE_BUILD_TYPE=Release
//   make -j$(nproc)
//
// Usage:
//   ./model_bench -m /path/to/model -e cpu --bench --pp 8 128 512 --tg 128 --reps 5
//   ./model_bench -m /path/to/model -e cpu -p "Hello, who are you?"
//   ./model_bench -m /path/to/model -e cpu          # interactive

#include "ort_genai.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double elapsed_ms(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Args
// ─────────────────────────────────────────────────────────────────────────────
struct Args {
    std::string      model_path;
    std::string      execution_provider = "";   // empty = follow_config
    std::string      prompt             = "";   // empty = interactive
    int              max_length         = 4096;
    // bench
    bool             bench = false;
    std::vector<int> pp    = {8, 128, 512, 1024};
    int              tg    = 128;
    int              reps  = 5;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " -m <model_path> [options]\n"
        << "\n  -m, --model        Model directory (required)\n"
        << "  -e, --ep           Execution provider: cpu | cuda | dml (default: follow_config)\n"
        << "  -p, --prompt       Prompt text (omit for interactive mode)\n"
        << "  -l, --max-length   Max generation length (default: 4096)\n"
        << "\nBenchmark:\n"
        << "  --bench            Enable benchmark mode\n"
        << "  --pp N [N ...]     Prefill token counts   (default: 8 128 512 1024)\n"
        << "  --tg N             Decode steps per run   (default: 128, 0=PP only)\n"
        << "  --reps N           Repetitions            (default: 5)\n";
}

static Args parse_args(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); exit(1); }
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nxt = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; exit(1); }
            return argv[++i];
        };
        if      (arg == "-m" || arg == "--model")      a.model_path          = nxt();
        else if (arg == "-e" || arg == "--ep")         a.execution_provider  = nxt();
        else if (arg == "-p" || arg == "--prompt")     a.prompt              = nxt();
        else if (arg == "-l" || arg == "--max-length") a.max_length          = std::stoi(nxt());
        else if (arg == "--bench")                     a.bench               = true;
        else if (arg == "--tg")                        a.tg                  = std::stoi(nxt());
        else if (arg == "--reps")                      a.reps                = std::stoi(nxt());
        else if (arg == "--pp") {
            a.pp.clear();
            while (i + 1 < argc && argv[i+1][0] != '-') a.pp.push_back(std::stoi(argv[++i]));
            if (a.pp.empty()) { std::cerr << "--pp needs at least one value\n"; exit(1); }
        }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); exit(0); }
        else { std::cerr << "Unknown arg: " << arg << "\n"; print_usage(argv[0]); exit(1); }
    }
    if (a.model_path.empty()) { std::cerr << "-m/--model is required\n"; exit(1); }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// llama.cpp-style perf output
// ─────────────────────────────────────────────────────────────────────────────
static void print_perf(int n_pp, double pp_ms, int n_tg, double tg_ms) {
    double total = pp_ms + tg_ms;
    double pp_tps = (pp_ms > 0) ? (n_pp * 1000.0 / pp_ms) : 0.0;
    double tg_tps = (tg_ms > 0 && n_tg > 0) ? (n_tg * 1000.0 / tg_ms) : 0.0;

    std::string sep(64, '-');
    std::cout << "\n" << sep << "\n"
              << std::fixed << std::setprecision(2)
              << std::setw(18) << std::right << "prompt eval time"
              << " = " << std::setw(9) << pp_ms  << " ms / "
              << std::setw(6) << n_pp << " tokens"
              << "  (" << std::setw(8) << pp_tps << " tokens/s)  [PP]\n";
    if (n_tg > 0)
        std::cout << std::setw(18) << std::right << "eval time"
                  << " = " << std::setw(9) << tg_ms  << " ms / "
                  << std::setw(6) << n_tg << " tokens"
                  << "  (" << std::setw(8) << tg_tps << " tokens/s)  [TG]\n";
    std::cout << std::setw(18) << std::right << "total time"
              << " = " << std::setw(9) << total << " ms\n"
              << sep << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark mode
// Directly feeds raw token ids via generator->AppendTokens() — no tokenizer,
// no chat template, exactly N tokens every time.
// ─────────────────────────────────────────────────────────────────────────────
static void run_bench(const Args& args, OgaModel& model) {
    std::string eq(64, '=');
    std::cout << "\n" << eq << "\n  BENCHMARK MODE\n  pp sizes    : [";
    for (size_t i = 0; i < args.pp.size(); ++i)
        std::cout << args.pp[i] << (i+1 < args.pp.size() ? ", " : "");
    std::cout << "]\n  tg steps    : " << args.tg
              << "\n  repetitions : " << args.reps << "\n" << eq << "\n\n";

    // Token id 1 is a safe content token on virtually every model.
    // (0 is often PAD/BOS, 2 is often EOS — avoid those.)
    const int32_t FILL_ID = 1;

    for (int pp_n : args.pp) {
        std::vector<int32_t> ids(pp_n, FILL_ID);
        std::vector<double>  pp_ms_v, tg_per_tok_ms_v;

        for (int rep = 0; rep < args.reps; ++rep) {
            auto params = OgaGeneratorParams::Create(model);
            params->SetSearchOption("max_length",
                static_cast<double>(pp_n + std::max(args.tg, 1) + 16));
            params->SetSearchOption("min_length", 0.0);

            auto gen = OgaGenerator::Create(model, *params);

            // ── PP ─────────────────────────────────────────────────
            // AppendTokens triggers prefill; GenerateNextToken produces token 1
            auto t0 = Clock::now();
            gen->AppendTokens(ids.data(), static_cast<size_t>(pp_n));
            gen->GenerateNextToken();
            auto t1 = Clock::now();
            double pp_ms = elapsed_ms(t0, t1);
            pp_ms_v.push_back(pp_ms);

            // ── TG ─────────────────────────────────────────────────
            int    tg_actual = 0;
            double tg_ms     = 0.0;
            if (args.tg > 0) {
                auto t2 = Clock::now();
                for (int s = 0; s < args.tg; ++s) {
                    if (gen->IsDone()) break;
                    gen->GenerateNextToken();
                    ++tg_actual;
                }
                tg_ms = elapsed_ms(t2, Clock::now());
                if (tg_actual > 0)
                    tg_per_tok_ms_v.push_back(tg_ms / tg_actual);
            }

            double pp_tps = (pp_ms > 0) ? (pp_n * 1000.0 / pp_ms) : 0.0;
            std::cout << "  rep " << std::setw(2) << rep+1 << "/" << args.reps
                      << "  pp=" << std::setw(5) << pp_n
                      << "  PP " << std::fixed << std::setprecision(1)
                      << std::setw(7) << pp_ms << " ms"
                      << "  (" << std::setw(7) << pp_tps << " tok/s)";
            if (tg_actual > 0) {
                double tg_tps = tg_actual * 1000.0 / tg_ms;
                std::cout << "  |  TG " << std::setw(7) << tg_ms << " ms"
                          << " / " << tg_actual << " tok"
                          << "  (" << std::setw(7) << tg_tps << " tok/s)";
            }
            std::cout << "\n";
        }

        // ── Summary ───────────────────────────────────────────────
        auto stat = [](const std::vector<double>& v)
            -> std::tuple<double,double,double>
        {
            double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double mn  = *std::min_element(v.begin(), v.end());
            double mx  = *std::max_element(v.begin(), v.end());
            return {avg, mn, mx};
        };

        auto [pp_avg, pp_min, pp_max] = stat(pp_ms_v);
        std::string sep(64, '-');
        std::cout << "\n" << sep
                  << "\n  ▶ pp" << pp_n << "  (n=" << args.reps << " runs)\n"
                  << std::fixed << std::setprecision(2)
                  << "    PP latency  : avg=" << std::setw(8) << pp_avg << " ms"
                  << "  min=" << std::setw(8) << pp_min << " ms"
                  << "  max=" << std::setw(8) << pp_max << " ms\n"
                  << "    PP throughput: avg=" << std::setw(8) << (pp_n*1000.0/pp_avg) << " tok/s"
                  << "  best=" << std::setw(8) << (pp_n*1000.0/pp_min) << " tok/s\n";
        if (!tg_per_tok_ms_v.empty()) {
            auto [tg_avg, tg_min, tg_max] = stat(tg_per_tok_ms_v);
            std::cout << "    TG throughput: avg=" << std::setw(8) << (1000.0/tg_avg) << " tok/s"
                      << "  best=" << std::setw(8) << (1000.0/tg_min) << " tok/s\n";
        }
        std::cout << sep << "\n\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal inference mode
// Uses Tokenizer::Encode -> generator->AppendTokens() pattern (no SetInputs).
// For vision models with real images, switch to the processor path shown below.
// ─────────────────────────────────────────────────────────────────────────────
static void run_inference(const Args& args, OgaModel& model) {
    auto tokenizer  = OgaTokenizer::Create(model);
    auto tok_stream = OgaTokenizerStream::Create(*tokenizer);

    bool interactive = args.prompt.empty();

    while (true) {
        // ── Get prompt ────────────────────────────────────────────
        std::string text;
        if (interactive) {
            std::cout << "Prompt: " << std::flush;
            if (!std::getline(std::cin, text) || text.empty()) break;
        } else {
            text = args.prompt;
        }

        // ── Encode prompt to token ids ────────────────────────────
        auto seqs = OgaSequences::Create();
        tokenizer->Encode(text.c_str(), *seqs);
        size_t   n_prompt_tokens = seqs->SequenceCount(0);
        const int32_t* prompt_ids = seqs->SequenceData(0);

        // ── Setup ─────────────────────────────────────────────────
        auto params = OgaGeneratorParams::Create(model);
        params->SetSearchOption("max_length", static_cast<double>(args.max_length));
        params->SetSearchOption("min_length", 0.0);

        auto gen = OgaGenerator::Create(model, *params);

        // ── PP: AppendTokens triggers prefill ─────────────────────
        double pp_ms        = 0.0;
        double tg_ms        = 0.0;
        int    n_gen        = 0;
        TimePoint t_start, t_first, t_end;

        t_start = Clock::now();
        gen->AppendTokens(prompt_ids, n_prompt_tokens);
        gen->GenerateNextToken();
        t_first = Clock::now();
        pp_ms   = elapsed_ms(t_start, t_first);

        // Print first generated token
        {
            size_t len = gen->GetSequenceCount(0);
            if (len > 0) {
                int32_t tok = gen->GetSequenceData(0)[len - 1];
                std::cout << tok_stream->Decode(tok) << std::flush;
                ++n_gen;
            }
        }

        // ── TG: remaining tokens ──────────────────────────────────
        while (!gen->IsDone()) {
            gen->GenerateNextToken();
            size_t len = gen->GetSequenceCount(0);
            if (len > 0) {
                int32_t tok = gen->GetSequenceData(0)[len - 1];
                std::cout << tok_stream->Decode(tok) << std::flush;
                ++n_gen;
            }
        }
        t_end = Clock::now();
        tg_ms = elapsed_ms(t_first, t_end);

        std::cout << "\n";
        print_perf(static_cast<int>(n_prompt_tokens), pp_ms, n_gen, tg_ms);

        if (!interactive) break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    try {
        std::cout << "Loading model...\n";

        std::unique_ptr<OgaModel> model;
        if (args.execution_provider.empty() ||
            args.execution_provider == "follow_config") {
            model = OgaModel::Create(args.model_path.c_str());
        } else {
            auto config = OgaConfig::Create(args.model_path.c_str());
            config->ClearProviders();
            if (args.execution_provider != "cpu")
                config->AppendProvider(args.execution_provider.c_str());
            model = OgaModel::Create(*config);
        }
        std::cout << "Model loaded\n";

        if (args.bench)
            run_bench(args, *model);
        else
            run_inference(args, *model);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
