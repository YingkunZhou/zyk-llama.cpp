#include "openvino/genai/llm_pipeline.hpp"
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

// ── 随机数生成器，防止 prefix cache 命中 ─────────────────────────────────────
static std::mt19937 rng(std::random_device{}());

// ── 构造恰好 n 个 token 的 EncodedInputs，每次随机打乱顺序 ───────────────────
// 用随机 token id 填充，完全避免 prefix cache 命中
static ov::genai::EncodedInputs make_exact_tokens(ov::genai::Tokenizer& tok, int n, bool randomize = false) {
    std::string long_text;
    long_text.reserve((n + 10) * 7);
    for (int i = 0; i < n + 10; ++i)
        long_text += "hello ";

    auto encoded   = tok.encode(long_text);
    size_t seq_len = encoded.input_ids.get_shape()[1];

    if ((int)seq_len < n) {
        throw std::runtime_error(
            "编码后 token 数 (" + std::to_string(seq_len) +
            ") 不足目标长度 " + std::to_string(n));
    }

    ov::Tensor input_ids(ov::element::i64, {1, (size_t)n});
    ov::Tensor attn_mask(ov::element::i64, {1, (size_t)n});

    int64_t* dst_ids  = input_ids.data<int64_t>();
    int64_t* dst_mask = attn_mask.data<int64_t>();

    std::copy(encoded.input_ids.data<int64_t>(),
              encoded.input_ids.data<int64_t>() + n,
              dst_ids);
    std::copy(encoded.attention_mask.data<int64_t>(),
              encoded.attention_mask.data<int64_t>() + n,
              dst_mask);

    if (randomize && n > 1) {
        // 打乱 token 顺序，确保每次调用的 prefix 不同，防止 KV cache 命中
        // 保留第一个 token 不动（通常是 BOS），打乱其余
        std::shuffle(dst_ids + 1, dst_ids + n, rng);
    }

    ov::genai::TokenizedInputs ti{input_ids, attn_mask};
    return ov::genai::EncodedInputs{ti};
}

// ── 结果结构体 ────────────────────────────────────────────────────────────────
struct BenchResult {
    double pp_latency_ms;   // prefill 延迟 (TTFT)
    double pp_throughput;   // prefill 吞吐 tok/s
    double tg_throughput;   // decode 吞吐 tok/s (TPOT 倒数)
    int    pp_actual;
    int    tg_actual;
};

// ── combined 模式 ─────────────────────────────────────────────────────────────
BenchResult run_combined(
    ov::genai::LLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int pp_n, int tg_n,
    int warmup, int iters)
{
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = tg_n;
    cfg.do_sample      = false;

    std::vector<double> pp_lats, tg_thrs;

    for (int i = 0; i < warmup + iters; ++i) {
        // 每次迭代都随机化输入，防止 prefix cache
        auto inputs = make_exact_tokens(tok, pp_n, /*randomize=*/true);

        // 使用 perf_metrics 获取精确的 TTFT 和 TPOT
        auto result = pipe.generate(inputs, cfg);
        auto& pm    = result.perf_metrics;

        if (i >= warmup) {
            double ttft_ms = pm.get_ttft().mean;          // ms
            double tpot_ms = pm.get_tpot().mean;          // ms/token
            pp_lats.push_back(ttft_ms);
            tg_thrs.push_back(tpot_ms > 0 ? 1000.0 / tpot_ms : 0.0);
        }
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    BenchResult r;
    r.pp_latency_ms = mean(pp_lats);
    r.pp_throughput = pp_n / (r.pp_latency_ms / 1000.0);
    r.tg_throughput = mean(tg_thrs);
    r.pp_actual     = pp_n;
    r.tg_actual     = tg_n;
    return r;
}

// ── separate 模式：PP only ────────────────────────────────────────────────────
BenchResult run_pp_only(
    ov::genai::LLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int pp_n, int warmup, int iters)
{
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = 1;
    cfg.do_sample      = false;

    std::vector<double> times;
    for (int i = 0; i < warmup + iters; ++i) {
        auto inputs = make_exact_tokens(tok, pp_n, /*randomize=*/true);
        auto result = pipe.generate(inputs, cfg);

        if (i >= warmup)
            times.push_back(result.perf_metrics.get_ttft().mean);
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    BenchResult r{};
    r.pp_latency_ms = avg;
    r.pp_throughput = pp_n / (avg / 1000.0);
    r.pp_actual     = pp_n;
    return r;
}

// ── separate 模式：TG only ────────────────────────────────────────────────────
BenchResult run_tg_only(
    ov::genai::LLMPipeline&  pipe,
    ov::genai::Tokenizer&    tok,
    int tg_n, int warmup, int iters)
{
    // prefill 只用 1 个 token
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = tg_n;
    cfg.do_sample      = false;

    std::vector<double> thrs;
    for (int i = 0; i < warmup + iters; ++i) {
        auto inputs = make_exact_tokens(tok, 1, /*randomize=*/false);
        auto result = pipe.generate(inputs, cfg);

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
              << std::setw(14) << "PP延迟(ms)"
              << std::setw(16) << "PP吞吐(tok/s)"
              << std::setw(16) << "TG吞吐(tok/s)"
              << "\n" << std::string(68, '-') << "\n";
}

static void print_row(const std::string& tag, const BenchResult& r) {
    std::cout << std::right << std::fixed << std::setprecision(1)
              << std::setw(22) << tag
              << std::setw(14) << r.pp_latency_ms
              << std::setw(16) << r.pp_throughput
              << std::setw(16) << r.tg_throughput
              << "\n";
}

// ── 验证 tokenizer 对齐 ───────────────────────────────────────────────────────
static void verify_token_count(ov::genai::Tokenizer& tok,
                                const std::vector<int>& pp_list)
{
    std::cout << "[ tokenizer 对齐验证 ]\n";
    for (int n : pp_list) {
        auto enc = make_exact_tokens(tok, n, false);
        auto& ti  = std::get<ov::genai::TokenizedInputs>(enc);
        int   got = (int)ti.input_ids.get_shape()[1];
        std::cout << "  目标=" << std::setw(4) << n
                  << "  实际=" << std::setw(4) << got
                  << (got == n ? "  ✓" : "  ✗ 不匹配!") << "\n";
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
    ov::genai::LLMPipeline pipe(model_path, device, properties);
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
                  << std::setw(14) << "PP延迟(ms)"
                  << std::setw(14) << "PP吞吐(tok/s)" << "\n"
                  << std::string(40, '-') << "\n";
        for (int pp : pp_list) {
            auto r = run_pp_only(pipe, tokenizer, pp, warmup, iters);
            std::ostringstream tag; tag << "pp" << pp;
            std::cout << std::setw(12) << tag.str()
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