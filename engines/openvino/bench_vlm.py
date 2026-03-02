import time
import argparse
import numpy as np
import openvino_genai as ov_genai

def parse_args():
    parser = argparse.ArgumentParser(description="VLMPipeline NPU benchmark")
    parser.add_argument("-m", "--model",   default="gemma-3-4b-it-int4-cw-ov")
    parser.add_argument("-d", "--device",  default="NPU")
    parser.add_argument("--pp",    nargs="+", type=int, default=[2,3,4,5,6,7,8])
    parser.add_argument("--tg",    nargs="+", type=int, default=[32])
    parser.add_argument("--warmup",type=int,  default=1)
    parser.add_argument("--iters", type=int,  default=3)
    parser.add_argument("--mode",  choices=["separate","combined"], default="separate")
    return parser.parse_args()

# ── tokenizer 工具 ────────────────────────────────────────────────────────────
def to_ids(tensor) -> np.ndarray:
    """ov.Tensor → 1D numpy array"""
    return np.array(tensor.data).flatten()

def count_tokens(tok, prompt: str) -> int:
    return int(to_ids(tok.encode(prompt).input_ids).shape[0])

def make_prompt_exact(tok, n: int) -> str:
    """构造 re-encode 后恰好 n 个 token 的字符串"""
    long_text = "hello " * (n + 20)
    all_ids   = to_ids(tok.encode(long_text).input_ids)

    if len(all_ids) < n:
        raise RuntimeError(f"源文本 token 数 {len(all_ids)} 不足目标 {n}")

    # 截取前 n 个 id → decode → re-encode 验证
    prompt = tok.decode(all_ids[:n].tolist())
    actual = count_tokens(tok, prompt)

    if actual != n:
        # roundtrip 有偏差时微调
        for end in range(n, min(n + 10, len(all_ids)) + 1):
            candidate = tok.decode(all_ids[:end].tolist())
            if count_tokens(tok, candidate) == n:
                return candidate
        print(f"  ⚠ pp={n} 无法精确对齐，最近似={actual}")

    return prompt

# ── benchmark 函数 ────────────────────────────────────────────────────────────
def bench_pp(pipe, prompt, warmup, iters):
    """prefill benchmark：生成 1 个 token，测 TTFT"""
    times = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=1, do_sample=False)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
    return sum(times) / len(times)

def bench_tg(pipe, prompt, tg_n, warmup, iters):
    """decode benchmark：固定 1 token prefill，生成 tg_n 个 token，测 TPOT"""
    times = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=tg_n, do_sample=False)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(ms)
    avg_total = sum(times) / len(times)
    # 减去 prefill 时间（1 token），剩下是 decode 时间
    pp1_ms    = bench_pp(pipe, prompt, warmup=1, iters=1)
    decode_ms = avg_total - pp1_ms
    tg_thr    = (tg_n - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    return avg_total, tg_thr

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"模型: {args.model}  设备: {args.device}  模式: {args.mode}\n")

    print("加载模型...", end="", flush=True)
    pipe = ov_genai.VLMPipeline(args.model, args.device)
    tok  = pipe.get_tokenizer()
    print(" 完成\n")

    # 预先构造所有 prompt，验证对齐
    all_pp = sorted(set(args.pp))
    print("[ tokenizer 对齐验证 ]")
    prompts = {}
    for n in all_pp:
        p   = make_prompt_exact(tok, n)
        got = count_tokens(tok, p)
        mark = "✓" if got == n else f"✗ 实际={got}"
        print(f"  目标={n:4d}  实际={got:4d}  {mark}")
        prompts[n] = p
    print()

    # 1-token prompt 用于 tg 测试
    prompt_1tok = make_prompt_exact(tok, 1)

    if args.mode == "separate":
        # ── PP ──────────────────────────────────────────────────────────────
        print(f"{'测试':>8}  {'实际PP':>6}  {'PP延迟(ms)':>11}  {'PP吞吐(tok/s)':>14}")
        print("-" * 46)
        for pp_n in args.pp:
            prompt    = prompts[pp_n]
            actual_pp = count_tokens(tok, prompt)
            avg_ms    = bench_pp(pipe, prompt, args.warmup, args.iters)
            thr       = actual_pp / (avg_ms / 1000)
            print(f"{'pp'+str(pp_n):>8}  {actual_pp:>6}  {avg_ms:>11.1f}  {thr:>14.1f}")

        print()

        # ── TG ──────────────────────────────────────────────────────────────
        print(f"{'测试':>8}  {'总时间(ms)':>11}  {'TG吞吐(tok/s)':>14}")
        print("-" * 38)
        for tg_n in args.tg:
            total_ms, tg_thr = bench_tg(pipe, prompt_1tok, tg_n, args.warmup, args.iters)
            print(f"{'tg'+str(tg_n):>8}  {total_ms:>11.1f}  {tg_thr:>14.1f}")

    else:  # combined
        print(f"{'测试':>20}  {'实际PP':>6}  {'PP延迟(ms)':>11}  {'PP吞吐(tok/s)':>14}  {'TG吞吐(tok/s)':>14}")
        print("-" * 74)
        for pp_n in args.pp:
            prompt    = prompts[pp_n]
            actual_pp = count_tokens(tok, prompt)
            for tg_n in args.tg:
                pp_ms = bench_pp(pipe, prompt, args.warmup, args.iters)

                # decode：用 pp prompt 生成 tg_n token
                times = []
                for i in range(args.warmup + args.iters):
                    t0 = time.perf_counter()
                    pipe.generate(prompt, max_new_tokens=tg_n, do_sample=False)
                    ms = (time.perf_counter() - t0) * 1000
                    if i >= args.warmup:
                        times.append(ms)
                total_ms  = sum(times) / len(times)
                decode_ms = total_ms - pp_ms
                tg_thr    = (tg_n - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
                pp_thr    = actual_pp / (pp_ms / 1000)
                tag       = f"pp{pp_n}+tg{tg_n}"
                print(f"{tag:>20}  {actual_pp:>6}  {pp_ms:>11.1f}  {pp_thr:>14.1f}  {tg_thr:>14.1f}")

if __name__ == "__main__":
    main()
