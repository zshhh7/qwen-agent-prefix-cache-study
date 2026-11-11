"""Replay recorded Qwen-Agent calls against llama-server with and without prefix reuse."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import format_as_text_message

from .io_utils import append_jsonl, read_jsonl
from .tokenization import longest_common_prefix


class LocalServer:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.http = build_opener(ProxyHandler({}))

    def open(self, path, payload=None, timeout=60):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base + path, data, {"Content-Type": "application/json"})
        return self.http.open(request, timeout=timeout)

    def json(self, path, payload=None, timeout=60):
        with self.open(path, payload, timeout) as response:
            return json.load(response)

    def wait_ready(self, process):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and process.poll() is None:
            try:
                if self.json("/health", timeout=2)["status"] == "ok":
                    return
            except (URLError, TimeoutError):
                pass
            time.sleep(0.4)
        raise RuntimeError("llama-server did not become ready")

    def prompt_tokens(self, messages):
        prompt = self.json("/apply-template", {"messages": messages})["prompt"]
        return self.json("/tokenize", {
            "content": prompt, "add_special": True, "parse_special": True
        })["tokens"]

    def prime(self, tokens):
        # Evaluate only the previous input, leaving its KV state in slot 0.
        result = self.json("/completion", {
            "prompt": tokens, "id_slot": 0, "cache_prompt": False, "n_predict": 0
        })
        if result["timings"]["prompt_n"] != len(tokens):
            raise RuntimeError("Priming did not process the full previous prompt")

    def measure(self, tokens, reuse):
        payload = {
            "prompt": tokens, "id_slot": 0, "cache_prompt": reuse,
            "stream": True, "return_tokens": True, "n_predict": 1,
            "temperature": 0, "seed": 1234, "ignore_eos": True,
        }
        started = time.perf_counter()
        first_token_ms = None
        final = None
        with self.open("/completion", payload) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if first_token_ms is None and event.get("tokens") and not event.get("stop"):
                    first_token_ms = (time.perf_counter() - started) * 1000
                if event.get("stop"):
                    final = event
        if first_token_ms is None or final is None or final.get("truncated"):
            raise RuntimeError("Incomplete streaming measurement")
        timings = final["timings"]
        return {
            "first_token_ms": first_token_ms,
            "prefill_ms": timings["prompt_ms"],
            "processed_tokens": timings["prompt_n"],
            "cached_tokens": timings["cache_n"],
        }


def load_transitions(server, traces):
    model = get_chat_model({"model": "qwen3-local:4b", "model_type": "oai", "api_key": "EMPTY"})
    runs = defaultdict(list)
    for row in read_jsonl(traces):
        messages = model._preprocess_messages(
            [Message.model_validate(item) for item in row["messages"]],
            lang="zh", generate_cfg={}, functions=row["functions"],
        )
        messages = [format_as_text_message(item, add_upload_info=False) for item in messages]
        tokens = server.prompt_tokens(model.convert_messages_to_dicts(messages))
        runs[row["run_id"]].append((row, tokens))

    transitions = []
    for run in runs.values():
        run.sort(key=lambda item: item[0]["call_index"])
        for (_, previous), (row, current) in zip(run, run[1:]):
            transitions.append((row, previous, current))
    if not transitions:
        raise ValueError("Trace file contains no adjacent calls")
    return transitions


def run_pair(server, transition, warm_first):
    row, previous, current = transition
    shared = longest_common_prefix(previous, current)
    if not 0 < shared < len(current):
        raise ValueError("Expected a nonempty prefix and a new suffix")
    results = {}
    for reuse in (warm_first, not warm_first):
        if reuse:
            server.prime(previous)
        measurement = server.measure(current, reuse)
        expected_cache = shared if reuse else 0
        if (measurement["cached_tokens"] != expected_cache
                or measurement["processed_tokens"] != len(current) - expected_cache):
            raise RuntimeError(f"Unexpected cache state for {row['call_id']}: {measurement}, "
                               f"expected cache={expected_cache}, prompt={len(current)}")
        results["warm" if reuse else "cold"] = measurement
    return {
        "run_id": row["run_id"], "task_id": row["task_id"],
        "target_call": row["call_index"], "prompt_tokens": len(current),
        "shared_tokens": shared, "warm_first": warm_first, **results,
    }


def summarize(pairs):
    summary = {}
    for key in ("first_token_ms", "prefill_ms", "processed_tokens"):
        cold = statistics.median(pair["cold"][key] for pair in pairs)
        warm = statistics.median(pair["warm"][key] for pair in pairs)
        summary[key] = {"cold_median": cold, "warm_median": warm,
                        "median_paired_reduction": statistics.median(
                            1 - pair["warm"][key] / pair["cold"][key] for pair in pairs)}
    return summary


def benchmark(args, server):
    transitions = load_transitions(server, args.traces)
    run_pair(server, transitions[0], warm_first=False)
    rng = random.Random(20261111)
    order = [(repeat, i) for repeat in range(args.repeats) for i in range(len(transitions))]
    rng.shuffle(order)
    pairs = []
    for repeat, i in order:
        pair = run_pair(server, transitions[i], warm_first=(i + repeat) % 2 == 0)
        pairs.append({"repeat": repeat + 1, **pair})
        if len(pairs) % len(transitions) == 0:
            print(f"Measured {len(pairs)} / {len(order)} pairs", flush=True)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend-library", type=Path)
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use a positive repeat count and a new output file")

    server_exe = args.server_exe.resolve(strict=True)
    model_file = args.model.resolve(strict=True)
    server_env = os.environ.copy()
    if args.backend_library:
        library = args.backend_library.resolve(strict=True)
        server_env["GGML_BACKEND_PATH"] = str(library)
        server_env["PATH"] = str(library.parent) + os.pathsep + server_env.get("PATH", "")
    flags = ["--host", "127.0.0.1", "--port", str(args.port),
             "--parallel", "1", "--cache-ram", "0", "--device", args.device, "--gpu-layers", "99",
             "--ctx-size", "2048", "--reasoning", "off", "--no-webui"]
    version = subprocess.run([str(server_exe), "--version"], capture_output=True,
                             text=True, check=True, timeout=15)
    print("Starting llama-server", flush=True)
    with tempfile.TemporaryFile() as server_log:
        process = subprocess.Popen(
            [str(server_exe), "--model", str(model_file), *flags],
            stdout=server_log, stderr=subprocess.STDOUT,
            env=server_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            server = LocalServer(args.port)
            server.wait_ready(process)
            print("Model ready; warming up and checking cache state", flush=True)
            pairs = benchmark(args, server)
        except Exception:
            server_log.seek(0, 2)
            server_log.seek(max(0, server_log.tell() - 7000))
            print(server_log.read().decode(errors="replace"), flush=True)
            raise
        finally:
            process.terminate()
            process.wait(timeout=15)
        server_log.seek(0)
        gpu_lines = [
            line.decode(errors="replace").strip() for line in server_log
            if b"offloaded " in line or b"CUDA0 (" in line
        ]
    metadata = {
        "event": "setup", "server_version": (version.stdout or version.stderr).strip(),
        "model_file": model_file.name, "server_flags": flags, "gpu_log": gpu_lines,
        "pair_count": len(pairs), "repeats": args.repeats,
    }
    append_jsonl(args.output, [metadata, *({"event": "pair", **pair} for pair in pairs)])
    print(json.dumps({"setup": metadata, "summary": summarize(pairs)}, indent=2))


if __name__ == "__main__":
    main()
