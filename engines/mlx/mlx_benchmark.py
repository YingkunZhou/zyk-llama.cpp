"""
MLX Benchmark Script – 对标 llama.cpp 的 pp/tg 性能测试 (v4)
新增：--device cpu/gpu 参数，Header 打印当前设备。

用法:
    # the default is GPU
    python mlx_benchmark.py --model mlx-community/gemma-3-4b-it-qat-4bit
    python mlx_benchmark.py --model mlx-community/gemma-3-4b-it-qat-4bit \
        --pp 2 4 8 --tg 32 --runs 3 --warmup 1 --device gpu
    python mlx_benchmark.py --model mlx-community/gemma-3-4b-it-qat-4bit \
        --pp 2 4 8 --tg 32 --device cpu
"""

import argparse
import time
import statistics
import platform

import mlx.core as mx
from mlx_lm import load


# ── 设备设置 ────────────────────────────────────────────────────
def setup_device(device_str: str) -> str:
    """
    设置 MLX 默认设备，返回实际生效的设备名称字符串。
    必须在任何 mx.array 创建之前调用。
    """
    if device_str == "cpu":
        mx.set_default_device(mx.cpu)
        return "cpu"
    else:
        # MLX 在 Apple Silicon 上默认即为 GPU（Metal）
        # 显式设置以确保不被环境变量覆盖
        mx.set_default_device(mx.gpu)
        return "gpu (Metal)"


def get_device_info(device_str: str) -> str:
    """拼一段可读的设备描述，含芯片型号（macOS 上可查）。"""
    chip = "unknown chip"
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            chip = out
    except Exception:
        chip = platform.processor() or "unknown chip"

    label = "CPU" if device_str == "cpu" else "GPU (Metal)"
    return f"{label}  [{chip}]"


# ── KV cache 兼容不同版本 mlx_lm ───────────────────────────────
def _make_cache(model):
    try:
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(model)
    except ImportError:
        pass
    try:
        from mlx_lm.utils import make_kv_caches
        return make_kv_caches(model)
    except ImportError:
        pass
    if hasattr(model, "make_cache"):
        return model.make_cache()
    raise RuntimeError("无法创建 KV cache，请 pip install -U mlx-lm")


# ── 严格构造 prompt ─────────────────────────────────────────────
def make_prompt(n_tokens: int, fill_id: int, bos_id: int | None) -> mx.array:
    """
    构造精确 n_tokens 个 token 的 prompt，BOS 计入总数。
    与 llama.cpp bench 行为对齐：
      有 BOS → [BOS] + [fill_id] * (n_tokens - 1)
      无 BOS → [fill_id] * n_tokens
    """
    if bos_id is not None and n_tokens >= 1:
        ids = [bos_id] + [fill_id] * (n_tokens - 1)
    else:
        ids = [fill_id] * n_tokens
    assert len(ids) == n_tokens, f"prompt 长度不对: {len(ids)} != {n_tokens}"
    return mx.array([ids])  # [1, n_tokens]


# ── 计时辅助 ────────────────────────────────────────────────────
def sync_eval(x: mx.array):
    """强制 MLX lazy graph 完成执行，保证计时精准。"""
    mx.eval(x)

def now_ms() -> float:
    return time.perf_counter() * 1000.0


# ── PP 基准（Prefill 速度）──────────────────────────────────────
def bench_pp(model, n_tokens: int, fill_id: int, bos_id: int | None,
             n_warmup: int, n_runs: int) -> dict:
    prompt = make_prompt(n_tokens, fill_id, bos_id)
    times_ms = []

    for i in range(n_warmup + n_runs):
        cache = _make_cache(model)

        t0 = now_ms()
        logits = model(prompt, cache=cache)
        sync_eval(logits)
        t1 = now_ms()

        if i >= n_warmup:
            times_ms.append(t1 - t0)

    avg_ms   = statistics.mean(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    tps      = n_tokens / (avg_ms / 1000.0)

    return {
        "label":    f"pp{n_tokens}",
        "n_tokens": n_tokens,
        "avg_ms":   avg_ms,
        "stdev_ms": stdev_ms,
        "tps":      tps,
    }


# ── TG 基准（Token Decode 速度）────────────────────────────────
def bench_tg(model, n_generate: int, prompt_len: int,
             fill_id: int, bos_id: int | None,
             n_warmup: int, n_runs: int) -> dict:
    prompt = make_prompt(prompt_len, fill_id, bos_id)
    all_tps = []

    for i in range(n_warmup + n_runs):
        cache = _make_cache(model)

        # ── prefill（不计时）──
        logits = model(prompt, cache=cache)
        sync_eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        sync_eval(token)

        # ── decode（计时）──
        intervals_ms = []
        t_prev = now_ms()

        for _ in range(n_generate):
            logits = model(token, cache=cache)
            token  = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            sync_eval(token)
            t_now = now_ms()
            intervals_ms.append(t_now - t_prev)
            t_prev = t_now

        if i >= n_warmup and intervals_ms:
            avg_interval = statistics.mean(intervals_ms)
            all_tps.append(1000.0 / avg_interval)

    if not all_tps:
        return {}

    avg_tps   = statistics.mean(all_tps)
    stdev_tps = statistics.stdev(all_tps) if len(all_tps) > 1 else 0.0
    avg_ms    = 1000.0 / avg_tps if avg_tps > 0 else float("inf")

    return {
        "label":     f"tg{n_generate}",
        "n_tokens":  n_generate,
        "avg_ms":    avg_ms,
        "stdev_tps": stdev_tps,
        "tps":       avg_tps,
    }


# ── 输出格式 ────────────────────────────────────────────────────
def print_header(model_path: str, device_info: str,
                 bos_id, fill_id: int, n_warmup: int, n_runs: int):
    print("=" * 64)
    print(f"  MLX Benchmark  (v4 – strict token count)")
    print(f"  model    : {model_path}")
    print(f"  device   : {device_info}")
    print(f"  bos_id   : {bos_id}  |  fill_id : {fill_id}")
    print(f"  warmup   : {n_warmup}  |  runs    : {n_runs}")
    print("=" * 64)


def print_results(pp_results: list, tg_results: list):
    if pp_results:
        print(f"\n  {'─'*60}")
        print(f"  Prompt Processing (PP) – prefill speed")
        print(f"  {'─'*60}")
        print(f"  {'label':<10} {'tokens/sec':>12} {'ms/run':>10} {'±ms':>8}")
        print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
        for r in pp_results:
            print(f"  {r['label']:<10} {r['tps']:>12.2f} "
                  f"{r['avg_ms']:>10.1f} {r['stdev_ms']:>8.1f}")

    if tg_results:
        print(f"\n  {'─'*60}")
        print(f"  Token Generation (TG) – decode speed")
        print(f"  {'─'*60}")
        print(f"  {'label':<10} {'tokens/sec':>12} {'ms/tok':>10} {'±tps':>8}")
        print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
        for r in tg_results:
            print(f"  {r['label']:<10} {r['tps']:>12.2f} "
                  f"{r['avg_ms']:>10.3f} {r['stdev_tps']:>8.2f}")

    print(f"\n{'='*64}\n")


# ── 主程序 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MLX LLM benchmark – pp & tg（对标 llama.cpp bench）"
    )
    parser.add_argument("--model", "-m",
        default="mlx-community/gemma-3-4b-it-qat-4bit",
        help="HuggingFace model ID 或本地路径")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
        help="运行设备：gpu（默认，使用 Metal）或 cpu")
    parser.add_argument("--pp", type=int, nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="PP 测试的 prompt token 数（严格计入 BOS）")
    parser.add_argument("--tg", type=int, nargs="+",
        default=[32, 64, 128, 256],
        help="TG 测试的生成 token 数")
    parser.add_argument("--prompt-len", type=int, default=4,
        help="TG 测试的 prefill prompt 长度（默认 4，含 BOS）")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs",   type=int, default=3)
    parser.add_argument("--no-pp",  action="store_true", help="跳过 PP 测试")
    parser.add_argument("--no-tg",  action="store_true", help="跳过 TG 测试")
    parser.add_argument("--no-bos", action="store_true",
        help="强制不加 BOS（覆盖 tokenizer 设置）")
    args = parser.parse_args()

    # ── 必须在加载模型之前设置设备 ──
    setup_device(args.device)
    device_info = get_device_info(args.device)

    print(f"\nDevice   : {device_info}")
    print(f"Loading  : {args.model} …")
    model, tokenizer = load(args.model)
    model.eval()

    bos_id  = None if args.no_bos else getattr(tokenizer, "bos_token_id", None)
    ids     = tokenizer.encode(" hello", add_special_tokens=False)
    fill_id = ids[0] if ids else 1

    print_header(args.model, device_info, bos_id, fill_id,
                 args.warmup, args.runs)

    pp_results, tg_results = [], []

    if not args.no_pp:
        print("\nRunning PP benchmarks …")
        for n in sorted(args.pp):
            print(f"  pp{n:>5} … ", end="", flush=True)
            r = bench_pp(model, n, fill_id, bos_id, args.warmup, args.runs)
            pp_results.append(r)
            print(f"{r['tps']:>9.1f} tok/s  "
                  f"(avg {r['avg_ms']:.1f} ms ± {r['stdev_ms']:.1f})")

    if not args.no_tg:
        print("\nRunning TG benchmarks …")
        for n in sorted(args.tg):
            print(f"  tg{n:>5} … ", end="", flush=True)
            r = bench_tg(model, n, args.prompt_len,
                         fill_id, bos_id, args.warmup, args.runs)
            if r:
                tg_results.append(r)
                print(f"{r['tps']:>9.1f} tok/s  "
                      f"({r['avg_ms']:.3f} ms/tok ± {r['stdev_tps']:.2f})")

    print_results(pp_results, tg_results)


if __name__ == "__main__":
    main()
