from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cache_sim import segment_reuse_statistics, simulate_capacities
from .io_utils import read_jsonl
from .segmentation import SEGMENT_NAMES, canonical_prompt_text, canonical_segment_text, split_prompt
from .tokenization import UnicodeCodepointTokenizer, load_tokenizer, longest_common_prefix


def analyze_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    previous_by_run: dict[str, list[int]] = {}
    for row in sorted(rows, key=lambda item: (item["run_id"], int(item["call_index"]))):
        sections = split_prompt(row.get("messages", []), row.get("functions", []))
        segment_counts = {
            name: len(tokenizer.encode(canonical_segment_text(name, sections[name]), add_special_tokens=False))
            for name in SEGMENT_NAMES
        }
        prompt_tokens = tokenizer.encode(
            canonical_prompt_text(row.get("messages", []), row.get("functions", [])),
            add_special_tokens=False,
        )
        total = len(prompt_tokens)
        previous = previous_by_run.get(row["run_id"])
        shared = longest_common_prefix(previous, prompt_tokens) if previous is not None else 0
        denominator = sum(segment_counts.values())
        result = {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "call_index": int(row["call_index"]),
            "model": row.get("model"),
            "latency_seconds": row.get("latency_seconds"),
            "total_prompt_tokens": total,
            "shared_prefix_tokens": shared,
            "reuse_ratio": shared / total if total else 0.0,
            "theoretical_saved_prefill_tokens": shared,
        }
        for name, count in segment_counts.items():
            result[f"{name}_tokens"] = count
            result[f"{name}_ratio"] = count / denominator if denominator else 0.0
        results.append(result)
        previous_by_run[row["run_id"]] = prompt_tokens
    return results


def save_outputs(
    results: list[dict[str, Any]],
    cache_results: list[dict[str, Any]],
    reuse_results: list[dict[str, Any]],
    output_root: Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(tables / "prompt_call_metrics.csv", index=False, encoding="utf-8-sig")
    cache_frame = pd.DataFrame(cache_results)
    cache_frame.to_csv(tables / "lru_capacity_comparison.csv", index=False, encoding="utf-8-sig")
    reuse_frame = pd.DataFrame(reuse_results)
    reuse_frame.to_csv(tables / "segment_reuse_statistics.csv", index=False, encoding="utf-8-sig")

    summary = frame.groupby("call_index", as_index=False).agg(
        total_prompt_tokens=("total_prompt_tokens", "mean"),
        shared_prefix_tokens=("shared_prefix_tokens", "mean"),
        reuse_ratio=("reuse_ratio", "mean"),
    )
    summary.to_csv(tables / "summary_by_call.csv", index=False, encoding="utf-8-sig")

    segment_columns = [f"{name}_tokens" for name in SEGMENT_NAMES]
    by_call = frame.groupby("call_index")[segment_columns].mean()
    ax = by_call.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_xlabel("LLM call index")
    ax.set_ylabel("Mean normalized tokens")
    ax.set_title("Prompt composition by LLM call")
    plt.tight_layout()
    plt.savefig(figures / "prompt_composition.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["call_index"], summary["reuse_ratio"], marker="o")
    ax.set_xlabel("LLM call index")
    ax.set_ylabel("Mean reuse ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_title("Adjacent-call prefix reuse")
    plt.tight_layout()
    plt.savefig(figures / "prefix_reuse.png", dpi=180)
    plt.close()

    total_prefill = int(frame["total_prompt_tokens"].sum())
    saved = int(frame["theoretical_saved_prefill_tokens"].sum())
    pd.DataFrame([{
        "total_prefill_tokens": total_prefill,
        "theoretical_saved_prefill_tokens": saved,
        "theoretical_saving_ratio": saved / total_prefill if total_prefill else 0.0,
    }]).to_csv(tables / "cache_benefit_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(cache_frame["capacity_tokens"], cache_frame["block_hit_rate"], marker="o", label="Block hit rate")
    ax1.plot(cache_frame["capacity_tokens"], cache_frame["token_hit_rate"], marker="s", label="Token hit rate")
    ax1.set_xlabel("LRU capacity (tokens)")
    ax1.set_ylabel("Hit rate")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(cache_frame["capacity_tokens"], cache_frame["reusable_tokens"], alpha=0.18, width=180)
    ax2.set_ylabel("Reusable tokens")
    ax1.legend(loc="upper left")
    ax1.set_title("LRU capacity sensitivity")
    plt.tight_layout()
    plt.savefig(figures / "lru_capacity_comparison.png", dpi=180)
    plt.close()

    labels = reuse_frame["block_type"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, reuse_frame["reuse_count"])
    axes[0].set_title("Reuse count")
    axes[1].bar(labels, reuse_frame["mean_reuse_interval_calls"])
    axes[1].set_title("Mean reuse interval (calls)")
    axes[2].bar(labels, reuse_frame["mean_lifetime_calls"])
    axes[2].set_title("Mean lifetime (calls)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figures / "segment_reuse_comparison.png", dpi=180)
    plt.close()


def demo_rows() -> list[dict[str, Any]]:
    base = [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Find x."}]
    functions = [{"name": "lookup", "description": "Find a value."}]
    return [
        {"run_id": "demo", "task_id": "demo", "call_index": 1, "messages": base, "functions": functions},
        {"run_id": "demo", "task_id": "demo", "call_index": 2, "messages": base + [
            {"role": "assistant", "function_call": {"name": "lookup", "arguments": "{}"}},
            {"role": "function", "name": "lookup", "content": "42"},
        ], "functions": functions},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path("outputs/traces/calls.jsonl"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        rows = demo_rows()
        tokenizer = UnicodeCodepointTokenizer()
        output = args.output / "demo"
        print("注意：demo 使用字符级 tokenizer，仅用于检查流程，不可作为实验数据。")
    else:
        rows = read_jsonl(args.traces)
        tokenizer = load_tokenizer(args.tokenizer)
        output = args.output
    results = analyze_rows(rows, tokenizer)
    capacities = [0, 256, 512, 1024, 2048, 4096, 8192]
    cache_results = simulate_capacities(rows, tokenizer, capacities)
    reuse_results = segment_reuse_statistics(rows, tokenizer)
    save_outputs(results, cache_results, reuse_results, output)
    print(f"已分析 {len(results)} 次 LLM 调用，结果保存到 {output}")


if __name__ == "__main__":
    main()
