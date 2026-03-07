#!/usr/bin/env python3
"""
MLC-LLM Benchmark -- raw prefill (pp) & decode (tg) speed test.

* NO chat template -- uses engine.completions.create() directly.
* Token counts are exact: verified by the tokenizer before each run.
* Thread control: two-layer (env var + tvm.runtime.set_num_threads API).

Usage:
  python mlc_bench.py
  python mlc_bench.py --model gemma-3-4b-it-q4f32_1-MLC --device cpu
  python mlc_bench.py --pp 2 4 8 16 64 512 --tg 32 128 --runs 3
  python mlc_bench.py --num-threads 4 --pp 128 --tg 32 --verbose
  python mlc_bench.py --hf-tokenizer google/gemma-3-4b-it --pp 64 128 512
"""

import argparse
import os
import sys
import time
import statistics
from dataclasses import dataclass, field

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"

DEFAULT_MODEL  = "gemma-3-4b-it-q4f32_1-MLC"
DEFAULT_DEVICE = "auto"
DEFAULT_PP     = [2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_TG     = [32, 64, 128, 256]
DEFAULT_RUNS   = 3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label:    str
    tokens:   int          # requested token count
    actual:   int          # verified by tokenizer
    times_ms: list = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def tps(self) -> float:
        return (self.actual / self.mean_ms * 1000) if self.mean_ms > 0 else 0.0


# ---------------------------------------------------------------------------
# Thread control
#
# TVM's thread count has TWO independent controls:
#
#   Layer 1 -- TVM_NUM_THREADS env var
#     Must be set BEFORE tvm is first imported. TVM reads it once when
#     creating its internal thread pool at module-load time.
#
#   Layer 2 -- tvm.runtime.set_num_threads(n)
#     A direct runtime API call. This is the ONLY reliable way to change
#     threads after tvm has already been imported (e.g. by a dependency).
#     We call this BOTH before and after MLCEngine.__init__ because the
#     engine init may reset TVM's thread pool internally.
# ---------------------------------------------------------------------------
def apply_thread_env(n: int, verbose: bool) -> None:
    """Layer 1: set env vars -- must run before any tvm/mlc_llm import."""
    for var in ("TVM_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    if verbose:
        print(f"  {DIM}[threads] env vars -> {n}{RESET}")


def apply_thread_api(n: int, verbose: bool) -> int | None:
    """
    Layer 2: reconfigure TVM thread pool via runtime.config_threadpool.
    tvm.runtime.set_num_threads() does NOT exist in most TVM builds.
    The correct API is:
        tvm.get_global_func("runtime.config_threadpool")(-1, n)
    where mode=-1 means reconfigure ALL thread pools.
    """
    try:
        import tvm

        # Primary: runtime.config_threadpool (TVM >= 0.9)
        config_fn = tvm.get_global_func(
            "runtime.config_threadpool", allow_missing=True)
        if config_fn is not None:
            config_fn(-1, n)
            actual = tvm.runtime.num_threads()
            if verbose:
                print(f"  {DIM}[threads] config_threadpool(-1, {n}) "
                      f"-> num_threads()={actual}{RESET}")
            if actual != n:
                print(f"  {YELLOW}[threads] Warning: requested {n} "
                      f"but TVM reports {actual}{RESET}")
            return actual

        # Fallback: some TVM forks expose set_num_threads directly
        set_fn = getattr(tvm.runtime, "set_num_threads", None)
        if set_fn is not None:
            set_fn(n)
            actual = tvm.runtime.num_threads()
            if verbose:
                print(f"  {DIM}[threads] set_num_threads({n}) "
                      f"-> num_threads()={actual}{RESET}")
            return actual

        # Nothing worked — tell the user to set env var in shell
        print(f"  {YELLOW}[threads] No TVM thread-control API found.{RESET}")
        print(f"  {YELLOW}  Run instead:  TVM_NUM_THREADS={n} python mlc_bench.py ...{RESET}")
        return None

    except Exception as e:
        print(f"  {YELLOW}[threads] API call failed: {e}{RESET}")
        return None


def get_tvm_threads() -> int | None:
    try:
        import tvm
        return tvm.runtime.num_threads()
    except Exception:
        return None


def make_engine_config(n: int):
    """Try EngineConfig(num_threads=n) for newer MLC builds (best-effort)."""
    try:
        from mlc_llm.serve import EngineConfig
        import inspect
        if "num_threads" in inspect.signature(EngineConfig.__init__).parameters:
            return EngineConfig(num_threads=n)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------
def get_tokenizer(engine):
    tok = getattr(engine, "tokenizer", None)
    if tok is None:
        raise RuntimeError(
            "engine.tokenizer not found -- use --hf-tokenizer <repo>.")
    return tok


def tok_encode(tokenizer, text: str) -> list:
    """Encode raw text to token ids WITHOUT special tokens."""
    try:
        result = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        result = tokenizer.encode(text)
    return list(result.ids) if hasattr(result, "ids") else list(result)


def find_unit_token(tokenizer) -> str:
    """Find a short string that encodes to exactly 1 token."""
    for candidate in ["a", "x", "1", "A", "hello", "the"]:
        try:
            if len(tok_encode(tokenizer, candidate)) == 1:
                return candidate
        except Exception:
            continue
    raise RuntimeError("Cannot find a single-token unit. Check your tokenizer.")


# ---------------------------------------------------------------------------
# Exact prompt builder (no chat template)
# ---------------------------------------------------------------------------
def build_exact_prompt(tokenizer, n_tokens: int,
                       unit: str, verbose: bool = False) -> tuple[str, int]:
    """
    Build a raw string whose token count is EXACTLY n_tokens (no chat template).

    Steps:
      1. Seed with (n_tokens + 4) repetitions of `unit` (safe overestimate).
      2. Trim one unit at a time while count > n_tokens.
      3. Pad one unit at a time while count < n_tokens.
         If adding one unit overshoots, fall back to single-character padding.
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")

    content = (unit + " ") * (n_tokens + 4)
    content = content.rstrip()

    for _ in range(500):
        actual = len(tok_encode(tokenizer, content))
        if actual == n_tokens:
            break
        elif actual > n_tokens:
            tail = " " + unit
            if content.endswith(tail):
                content = content[: -len(tail)]
            elif content.endswith(unit):
                content = content[: -len(unit)]
            else:
                content = content[:-1]
        else:
            candidate = content + " " + unit
            if len(tok_encode(tokenizer, candidate)) <= n_tokens:
                content = candidate
            else:
                for ch in "abcdefghijklmnopqrstuvwxyz":
                    c2 = content + ch
                    if len(tok_encode(tokenizer, c2)) <= n_tokens:
                        content = c2
                        break
                else:
                    break  # cannot reach target without overshooting

    actual = len(tok_encode(tokenizer, content))
    if verbose:
        print(f"  {DIM}prompt built: target={n_tokens}  actual={actual}  "
              f"chars={len(content)}{RESET}")
    return content, actual


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def sep(c: str = "-", w: int = 80) -> str:
    return DIM + c * w + RESET


def print_header(model: str, device: str, threads, actual_threads) -> None:
    print()
    print(sep("="))
    print(f"{BOLD}  MLC-LLM Benchmark  (raw completion -- no chat template){RESET}")
    print(f"  model   : {CYAN}{model}{RESET}")
    print(f"  device  : {YELLOW}{device}{RESET}")
    if threads is not None:
        mismatch = actual_threads is not None and actual_threads != threads
        extra = (f"  {YELLOW}(TVM reports {actual_threads}){RESET}" if mismatch
                 else f"  {DIM}(confirmed: {actual_threads}){RESET}"
                 if actual_threads else "")
        print(f"  threads : {GREEN}{threads}{RESET}{extra}")
    else:
        extra = (f"  {DIM}(TVM default: {actual_threads}){RESET}"
                 if actual_threads else "")
        print(f"  threads : {DIM}system default{RESET}{extra}")
    print(sep("="))


def print_table_header() -> None:
    print(f"  {DIM}{'label':<10}  {'req':>5}  {'actual':>7}  "
          f"{'mean ms':>10}  {'stdev':>10}  {'t/s':>9}   bar{RESET}")
    print(sep())


def print_row(res: BenchResult, runs: int) -> None:
    bar_w  = 26
    filled = min(int(res.tps / 3000.0 * bar_w), bar_w)
    bar    = GREEN + "#" * filled + DIM + "." * (bar_w - filled) + RESET
    stdev  = f"+-{res.stdev_ms:6.1f} ms" if runs > 1 else "          "
    mark   = (GREEN + "ok" + RESET) if res.actual == res.tokens \
             else (YELLOW + "~1" + RESET)
    print(f"  {BOLD}{res.label:<10}{RESET}"
          f"  {res.tokens:>5}"
          f"  {res.actual:>5} {mark}"
          f"  {res.mean_ms:>10.1f} ms"
          f"  {stdev}"
          f"  {GREEN}{res.tps:>9.1f} t/s{RESET}"
          f"  {bar}")


# ---------------------------------------------------------------------------
# Benchmark: prefill
# ---------------------------------------------------------------------------
def bench_pp(engine, model: str, prompt: str, actual_tokens: int,
             n_req: int, runs: int, verbose: bool) -> BenchResult:
    """
    Prefill benchmark via raw engine.completions (no chat template).
    Timing: API call start -> arrival of the FIRST streamed chunk.
    max_tokens=1 so the engine exits immediately after prefill.
    """
    res = BenchResult(label=f"pp{n_req}", tokens=n_req, actual=actual_tokens)

    for run in range(runs):
        t0      = time.perf_counter()
        t_first = None
        for _resp in engine.completions.create(
            prompt      = prompt,
            model       = model,
            max_tokens  = 1,
            stream      = True,
            temperature = 0.0,
        ):
            if t_first is None:
                t_first = time.perf_counter()

        elapsed = ((t_first or time.perf_counter()) - t0) * 1000
        res.times_ms.append(elapsed)
        if verbose:
            print(f"    {DIM}run {run+1}/{runs}: {elapsed:.1f} ms{RESET}")

    return res


# ---------------------------------------------------------------------------
# Benchmark: decode
# ---------------------------------------------------------------------------
def bench_tg(engine, model: str, n_tokens: int,
             runs: int, verbose: bool) -> BenchResult:
    """
    Decode benchmark via raw engine.completions (no chat template).

    - Short fixed prompt ("hello") keeps prefill cost negligible.
    - Timing starts after the SECOND chunk (skips first-token latency).
    - Timing ends at the last chunk.
    - t/s = (generated - 1) / decode_duration
    """
    res = BenchResult(label=f"tg{n_tokens}", tokens=n_tokens, actual=n_tokens)

    for run in range(runs):
        chunk_n = 0
        t_start = None
        t_last  = None
        gen     = 0

        for resp in engine.completions.create(
            prompt      = "hello",
            model       = model,
            max_tokens  = n_tokens,
            stream      = True,
            temperature = 1.0,
        ):
            now = time.perf_counter()
            for choice in resp.choices:
                if choice.text:
                    chunk_n += 1
                    gen     += 1
                    if chunk_n == 2:
                        t_start = now
                    if chunk_n >= 2:
                        t_last  = now

        res.actual = gen

        if t_start is None or t_last is None or t_last <= t_start:
            if verbose:
                print(f"    {YELLOW}run {run+1}/{runs}: "
                      f"only {gen} tok generated -- skip{RESET}")
            continue

        decode_ms = (t_last - t_start) * 1000
        res.times_ms.append(decode_ms)

        if verbose:
            measured = max(gen - 1, 1)
            print(f"    {DIM}run {run+1}/{runs}: gen={gen}  "
                  f"decode={decode_ms:.1f} ms  "
                  f"{measured / decode_ms * 1000:.1f} t/s{RESET}")

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLC-LLM raw PP/TG benchmark (no chat template)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--device",         default=DEFAULT_DEVICE,
                        help="auto | cpu | cuda | metal | vulkan")
    parser.add_argument("--num-threads", "--threads", "-t",
                        type=int, default=None, metavar="N", dest="threads",
                        help="CPU thread count. Applies BOTH TVM_NUM_THREADS "
                             "env var (pre-import) and tvm.runtime.set_num_threads() "
                             "(post-import) for maximum reliability.")
    parser.add_argument("--pp",             nargs="+", type=int, default=DEFAULT_PP)
    parser.add_argument("--tg",             nargs="+", type=int, default=DEFAULT_TG)
    parser.add_argument("--runs",           type=int, default=DEFAULT_RUNS)
    parser.add_argument("--hf-tokenizer",   default=None, metavar="REPO",
                        help="HuggingFace tokenizer repo "
                             "(e.g. google/gemma-3-4b-it)")
    parser.add_argument("--skip-pp",        action="store_true")
    parser.add_argument("--skip-tg",        action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # STEP 1: env vars -- MUST be before any tvm/mlc_llm import
    # ------------------------------------------------------------------
    if args.threads is not None:
        apply_thread_env(args.threads, verbose=args.verbose)

    # ------------------------------------------------------------------
    # STEP 2: import MLC (TVM initialises its thread pool here)
    # ------------------------------------------------------------------
    try:
        from mlc_llm import MLCEngine
    except ImportError:
        print(f"{RED}mlc_llm not found.  pip install mlc-llm{RESET}",
              file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3: runtime API thread override (most reliable)
    # ------------------------------------------------------------------
    if args.threads is not None:
        actual_threads = apply_thread_api(args.threads, verbose=args.verbose)
    else:
        actual_threads = get_tvm_threads()

    print_header(args.model, args.device, args.threads, actual_threads)

    # ------------------------------------------------------------------
    # STEP 4: optional EngineConfig for newer MLC builds
    # ------------------------------------------------------------------
    engine_config = None
    if args.threads is not None:
        engine_config = make_engine_config(args.threads)
        if args.verbose:
            status = "applied" if engine_config else "not supported (env+API used)"
            print(f"  {DIM}EngineConfig.num_threads: {status}{RESET}")

    # ------------------------------------------------------------------
    # STEP 5: load engine
    # ------------------------------------------------------------------
    print(f"\n  {DIM}Loading model ...{RESET}", flush=True)
    t0 = time.perf_counter()
    try:
        kw = dict(device=args.device)
        if engine_config is not None:
            kw["engine_config"] = engine_config
        engine = MLCEngine(args.model, **kw)
    except Exception as e:
        print(f"{RED}Failed to load model: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    print(f"  {GREEN}Loaded in {time.perf_counter()-t0:.2f} s{RESET}")

    # ------------------------------------------------------------------
    # STEP 6: re-apply runtime API after engine load
    # MLCEngine.__init__ may reset TVM's thread pool internally,
    # so we call set_num_threads again here to be safe.
    # ------------------------------------------------------------------
    if args.threads is not None:
        apply_thread_api(args.threads, verbose=True)

    # ------------------------------------------------------------------
    # STEP 7: resolve tokenizer
    # ------------------------------------------------------------------
    if args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"\n  {DIM}Loading HF tokenizer: {args.hf_tokenizer} ...{RESET}",
                  flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
            print(f"  {GREEN}HF tokenizer loaded.{RESET}")
        except Exception as e:
            print(f"{RED}HF tokenizer failed: {e}{RESET}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            tokenizer = get_tokenizer(engine)
            print(f"  {GREEN}Using MLC built-in tokenizer.{RESET}")
        except RuntimeError as e:
            print(f"  {YELLOW}{e}{RESET}", file=sys.stderr)
            sys.exit(1)

    unit = find_unit_token(tokenizer)
    if args.verbose:
        print(f"  {DIM}pad unit token: {repr(unit)}{RESET}")
    print()

    pp_results: list[BenchResult] = []
    tg_results: list[BenchResult] = []

    # ------------------------------------------------------------------
    # PP benchmarks
    # ------------------------------------------------------------------
    if not args.skip_pp:
        print(sep())
        print(f"  {BOLD}PREFILL  (pp){RESET}  time-to-first-token  "
              f"{DIM}({args.runs} run(s) each){RESET}")
        print(sep())
        print_table_header()

        for n in args.pp:
            sys.stdout.write(f"  {DIM}building pp{n} prompt ...{RESET}\r")
            sys.stdout.flush()
            try:
                prompt, actual = build_exact_prompt(
                    tokenizer, n, unit, verbose=args.verbose)
                res = bench_pp(engine, args.model, prompt, actual,
                               n, args.runs, args.verbose)
                pp_results.append(res)
                print_row(res, args.runs)
            except Exception as e:
                print(f"  {RED}pp{n} failed: {e}{RESET}")

    # ------------------------------------------------------------------
    # TG benchmarks
    # ------------------------------------------------------------------
    if not args.skip_tg:
        print()
        print(sep())
        print(f"  {BOLD}DECODE   (tg){RESET}  tokens per second     "
              f"{DIM}({args.runs} run(s) each){RESET}")
        print(sep())
        print_table_header()

        for n in args.tg:
            sys.stdout.write(f"  {DIM}testing tg{n} ...{RESET}\r")
            sys.stdout.flush()
            try:
                res = bench_tg(engine, args.model, n, args.runs, args.verbose)
                tg_results.append(res)
                print_row(res, args.runs)
            except Exception as e:
                print(f"  {RED}tg{n} failed: {e}{RESET}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(sep("="))
    thr = f"threads={args.threads}" if args.threads else "threads=default"
    print(f"  {BOLD}SUMMARY{RESET}  {DIM}{thr}{RESET}")
    print(sep())

    if pp_results:
        best = max(pp_results, key=lambda r: r.tps)
        print(f"  PP best  : {GREEN}{best.label:<8}{RESET}"
              f"  {best.tps:>8.1f} t/s"
              f"  ({best.mean_ms:.1f} ms, actual={best.actual} tok)")

    if tg_results:
        best = max(tg_results, key=lambda r: r.tps)
        avg  = statistics.mean(r.tps for r in tg_results)
        print(f"  TG best  : {GREEN}{best.label:<8}{RESET}"
              f"  {best.tps:>8.1f} t/s"
              f"  ({best.mean_ms:.1f} ms)")
        print(f"  TG avg   : {avg:>8.1f} t/s  across {len(tg_results)} sizes")

    print()
    print(f"  {DIM}ok = exact token count   ~1 = off by 1 (tokenizer boundary){RESET}")
    print(sep("="))
    print()

    engine.terminate()


if __name__ == "__main__":
    main()
