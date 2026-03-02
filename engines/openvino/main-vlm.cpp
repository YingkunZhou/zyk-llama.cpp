bench_vlm.py#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <random>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static std::mt19937 rng(std::random_device{}());

// ── 词汇表，用于构造随机 prompt ───────────────────────────────────────────────
static const std::vector<std::string> VOCAB = {
    "hello", "world", "the", "a", "is", "are", "was", "were",
    "this", "that", "what", "how", "why", "when", "where", "which",
    "cat", "dog", "car", "tree", "book", "food", "water", "light",
    "time", "day", "year", "work", "life", "man", "woman", "child"
};

// ── 构造 encode 后恰好 n 个 token 的字符串 ───────────────────────────────────
static std::string make_prompt_of_n_tokens(ov::genai::Tokenizer& tok, int n, bool randomize) {
    std::string prompt;
    int count = 0;

    std::vector<std::string> words;
    for (int i = 0; i < n + 20; ++i)
        words.push_back(VOCAB[i % VOCAB.size()]);
    if (randomize)
        std::shuffle(words.begin(), words.end(), rng);

    // 逐词添加直到达到 n 个 token
    for (auto& w : words) {
        std::string candidate = prompt.empty() ? w : prompt + " " + w;
        int cur = (int)tok.encode(candidate).input_ids.get_shape()[1];
        if (cur > n) break;
        prompt = candidate;
        count  = cur;
        if (count == n) break;
    }

    // 用随机数字补足
    if (count < n) {
        std::uniform_int_distribution<int> dist(0, 99999);
        while (count < n) {
            std::string candidate = prompt + " " + std::to_string(randomize ? dist(rng) : count);
            int cur = (int)tok.encode(candidate).input_ids.get_shape()[1];
            if (cur > n) break;
            prompt = candidate;
            count  = cur;
        }
    }

    return prompt;
}

static int count_tokens(ov::genai::Tokenizer& tok, const std::string& prompt) {
    return (int)tok.encode(prompt).input_ids.get_shape()[1];
}

// ── 结果结构体 ────────────────────────────────────────────────────────────────
struct BenchResult {
    double pp_latency_ms;
    double pp_throughput;
    double tg_throughput;
    int    pp_actual;
    int    tg_actual;
};

// ── 空图片列表（纯文本 benchmark）────────────────────────────────────────────
static const std::vector<ov::Tensor> NO_IMAGES{};

// ── combined 模式 ─────────────────────────────────────────────────────────────
BenchResult run_combined(
    ov::genai::VLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int pp_n, int tg_n,
    int warmup, int iters)
{
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = tg_n;
    cfg.do_sample      = false;

    std::vector<double> pp_lats, tg_thrs;
    int actual_pp = pp_n;

    for (int i = 0; i < warmup + iters; ++i) {
        std::string prompt = make_prompt_of_n_tokens(tok, pp_n, /*randomize=*/true);
        if (i == 0) actual_pp = count_tokens(tok, prompt);

        // ★ 纯文本调用：传空 images，streamer 用 std::monostate
        auto result = pipe.generate(
            prompt, NO_IMAGES, cfg, std::monostate{}
        );
        auto& pm = result.perf_metrics;

        if (i >= warmup) {
            double ttft_ms = pm.get_ttft().mean;
            double tpot_ms = pm.get_tpot().mean;
            pp_lats.push_back(ttft_ms);
            tg_thrs.push_back(tpot_ms > 0 ? 1000.0 / tpot_ms : 0.0);
        }
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    BenchResult r;
    r.pp_latency_ms = mean(pp_lats);
    r.pp_throughput = actual_pp / (r.pp_latency_ms / 1000.0);
    r.tg_throughput = mean(tg_thrs);
    r.pp_actual     = actual_pp;
    r.tg_actual     = tg_n;
    return r;
}

// ── separate 模式：PP only ────────────────────────────────────────────────────
BenchResult run_pp_only(
    ov::genai::VLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int pp_n, int warmup, int iters)
{
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = 1;
    cfg.do_sample      = false;

    std::vector<double> times;
    int actual_pp = pp_n;

    for (int i = 0; i < warmup + iters; ++i) {
        std::string prompt = make_prompt_of_n_tokens(tok, pp_n, /*randomize=*/true);
        if (i == 0) actual_pp = count_tokens(tok, prompt);

        auto result = pipe.generate(prompt, NO_IMAGES, cfg, std::monostate{});
        if (i >= warmup)
            times.push_back(result.perf_metrics.get_ttft().mean);
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    BenchResult r{};
    r.pp_latency_ms = avg;
    r.pp_throughput = actual_pp / (avg / 1000.0);
    r.pp_actual     = actual_pp;
    return r;
}

// ── separate 模式：TG only ────────────────────────────────────────────────────
BenchResult run_tg_only(
    ov::genai::VLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int tg_n, int warmup, int iters)
{
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = tg_n;
    cfg.do_sample      = false;

    std::string prompt = make_prompt_of_n_tokens(tok, 1, false);

    std::vector<double> thrs;
    for (int i = 0; i < warmup + iters; ++i) {
        auto result = pipe.generate(prompt, NO_IMAGES, cfg, std::monostate{});
        if (i >= warmup) {
            double tpot_ms = result.perf_metrics.get_tpot().mean;
            thrs.push_back(tpot_ms > 0 ? 1000.0 / tpot_ms : 0.0);
        }
    }

    double avg_thr = std::accumulate(thrs.begin(), thrs.end(), 0.0) / thrs.size();
    BenchResult r{};
    r.pp_latency_ms = (avg_thr > 0) ? (1000.0 / avg_thr * tg_n) : 0.0;
    r.tg_throughput = avg_thr;
    r.tg_actual     = tg_n;
    return r;
}

// ── 打印辅助 ──────────────────────────────────────────────────────────────────
static void print_header_combined() {
    std::cout << std::right
              << std::setw(22) << "测试"
              << std::setw(8)  << "实际PP"
              << std::setw(14) << "PP延迟(ms)"
              << std::setw(16) << "PP吞吐(tok/s)"
              << std::setw(16) << "TG吞吐(tok/s)"
              << "\n" << std::string(76, '-') << "\n";
}

static void print_row(const std::string& tag, const BenchResult& r) {
    std::cout << std::right << std::fixed << std::setprecision(1)
              << std::setw(22) << tag
              << std::setw(8)  << r.pp_actual
              << std::setw(14) << r.pp_latency_ms
              << std::setw(16) << r.pp_throughput
              << std::setw(16) << r.tg_throughput
              << "\n";
}

static void verify_token_count(ov::genai::Tokenizer& tok,
                                const std::vector<int>& pp_list)
{
    std::cout << "[ tokenizer 对齐验证 ]\n";
    for (int n : pp_list) {
        std::string prompt = make_prompt_of_n_tokens(tok, n, false);
        int got = count_tokens(tok, prompt);
        std::cout << "  目标=" << std::setw(4) << n
                  << "  实际=" << std::setw(4) << got
                  << (got == n ? "  ✓" : "  △ 最近似=" + std::to_string(got))
                  << "\n";
    }
    std::cout << "\n";
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string      model_path = "./model";
    std::string      device     = "CPU";
    std::vector<int> pp_list    = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<int> tg_list    = {32, 64, 128, 256};
    int         warmup   = 1;
    int         iters    = 3;
    int         threads  = 0;
    std::string mode     = "combined";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc)
            model_path = argv[++i];
        else if ((a == "-d" || a == "--device") && i + 1 < argc)
            device = argv[++i];
        else if (a == "--warmup" && i + 1 < argc)
            warmup = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)
            threads = std::stoi(argv[++i]);
        else if (a == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (a == "--pp") {
            pp_list.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                pp_list.push_back(std::stoi(argv[++i]));
        }
        else if (a == "--tg") {
            tg_list.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                tg_list.push_back(std::stoi(argv[++i]));
        }
    }

    std::cout << "模型: " << model_path << "  设备: " << device
              << "  模式: " << mode;
    if (threads > 0)
        std::cout << "  线程数: " << threads;
    std::cout << "\n\n";

    std::cout << "加载 tokenizer..." << std::flush;
    ov::genai::Tokenizer tokenizer(model_path);
    std::cout << " 完成\n";

    verify_token_count(tokenizer, pp_list);

    std::cout << "加载模型..." << std::flush;
    ov::AnyMap properties;
    properties["PERFORMANCE_HINT"] = std::string("LATENCY");
    if (threads > 0)
        properties["INFERENCE_NUM_THREADS"] = threads;
    ov::genai::VLMPipeline pipe(model_path, device, properties);
    std::cout << " 完成\n\n";

    if (mode == "combined") {
        print_header_combined();
        for (int pp : pp_list)
            for (int tg : tg_list) {
                auto r = run_combined(pipe, tokenizer, pp, tg, warmup, iters);
                std::ostringstream tag;
                tag << "pp" << pp << "+tg" << tg;
                print_row(tag.str(), r);
            }
    } else {
        std::cout << std::right
                  << std::setw(12) << "测试"
                  << std::setw(8)  << "实际PP"
                  << std::setw(14) << "PP延迟(ms)"
                  << std::setw(14) << "PP吞吐(tok/s)" << "\n"
                  << std::string(48, '-') << "\n";
        for (int pp : pp_list) {
            auto r = run_pp_only(pipe, tokenizer, pp, warmup, iters);
            std::ostringstream tag; tag << "pp" << pp;
            std::cout << std::setw(12) << tag.str()
                      << std::setw(8)  << r.pp_actual
                      << std::setw(14) << std::fixed << std::setprecision(1)
                      << r.pp_latency_ms
                      << std::setw(14) << r.pp_throughput << "\n";
        }

        std::cout << "\n"
                  << std::setw(12) << "测试"
                  << std::setw(14) << "总时间(ms)"
                  << std::setw(14) << "TG吞吐(tok/s)" << "\n"
                  << std::string(40, '-') << "\n";
        for (int tg : tg_list) {
            auto r = run_tg_only(pipe, tokenizer, tg, warmup, iters);
            std::ostringstream tag; tag << "tg" << tg;
            std::cout << std::setw(12) << tag.str()
                      << std::setw(14) << std::fixed << std::setprecision(1)
                      << r.pp_latency_ms
                      << std::setw(14) << r.tg_throughput << "\n";
        }
    }

    return 0;
}
