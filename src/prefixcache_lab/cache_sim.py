from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from .segmentation import canonical_segment_text


@dataclass(frozen=True)
class LogicalBlock:
    block_type: str
    text: str
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _role(message: dict[str, Any]) -> str:
    return str(message.get("role", "")).lower()


def build_logical_blocks(row: dict[str, Any], tokenizer: Any) -> list[LogicalBlock]:
    """Build cacheable blocks while preserving normalized prompt order.

    Order: system messages, tool definitions, then remaining messages in their
    original order. Each conversational message is an independent logical block.
    """
    messages = list(row.get("messages", []))
    functions = list(row.get("functions", []))
    last_user = max(
        (index for index, message in enumerate(messages) if _role(message) == "user"),
        default=-1,
    )
    typed_values: list[tuple[str, list[Any]]] = []
    system_messages = [message for message in messages if _role(message) == "system"]
    if system_messages:
        typed_values.append(("system_prompt", system_messages))
    if functions:
        typed_values.append(("tool_description", functions))

    for index, message in enumerate(messages):
        role = _role(message)
        if role == "system":
            continue
        if role in {"function", "tool"}:
            block_type = "tool_result"
        elif role == "user" and index == last_user:
            block_type = "current_query"
        else:
            block_type = "conversation_history"
        typed_values.append((block_type, [message]))

    blocks: list[LogicalBlock] = []
    for block_type, value in typed_values:
        text = canonical_segment_text(block_type, value)
        token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        blocks.append(LogicalBlock(block_type, text, token_ids))
    return blocks


def _node_key(parent_key: str, block: LogicalBlock) -> str:
    payload = parent_key.encode("ascii") + b"\0" + bytes.fromhex(block.content_hash)
    return hashlib.sha256(payload).hexdigest()


class PrefixLRUCache:
    """Token-capacity LRU over logical prefix-tree nodes.

    A block hits only when every preceding block in the same prompt has hit.
    Each node stores only its incremental block tokens, avoiding double-counting
    a complete prefix at every depth.
    """

    def __init__(self, capacity_tokens: int):
        if capacity_tokens < 0:
            raise ValueError("capacity_tokens must be non-negative")
        self.capacity_tokens = capacity_tokens
        self.used_tokens = 0
        self.entries: OrderedDict[str, int] = OrderedDict()

    def access(self, blocks: Iterable[LogicalBlock]) -> dict[str, int | float]:
        block_list = list(blocks)
        hit_blocks = 0
        hit_tokens = 0
        total_tokens = sum(block.token_count for block in block_list)
        parent_key = "root"
        prefix_intact = True
        keys: list[tuple[str, LogicalBlock]] = []

        for block in block_list:
            key = _node_key(parent_key, block)
            keys.append((key, block))
            if prefix_intact and key in self.entries:
                hit_blocks += 1
                hit_tokens += block.token_count
                self.entries.move_to_end(key)
            else:
                prefix_intact = False
            parent_key = key

        for key, block in keys:
            self._insert(key, block.token_count)

        return {
            "accessed_blocks": len(block_list),
            "hit_blocks": hit_blocks,
            "accessed_tokens": total_tokens,
            "hit_tokens": hit_tokens,
        }

    def _insert(self, key: str, size: int) -> None:
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        if size > self.capacity_tokens or self.capacity_tokens == 0:
            return
        while self.entries and self.used_tokens + size > self.capacity_tokens:
            _, evicted_size = self.entries.popitem(last=False)
            self.used_tokens -= evicted_size
        self.entries[key] = size
        self.used_tokens += size


def simulate_capacities(
    rows: list[dict[str, Any]], tokenizer: Any, capacities: Iterable[int]
) -> list[dict[str, int | float]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("started_at_unix", 0.0)),
            str(row.get("run_id", "")),
            int(row.get("call_index", 0)),
        ),
    )
    prompts = [build_logical_blocks(row, tokenizer) for row in ordered_rows]
    output: list[dict[str, int | float]] = []
    for capacity in capacities:
        cache = PrefixLRUCache(int(capacity))
        totals = defaultdict(int)
        for blocks in prompts:
            metrics = cache.access(blocks)
            for name, value in metrics.items():
                totals[name] += int(value)
        output.append({
            "capacity_tokens": int(capacity),
            **totals,
            "block_hit_rate": totals["hit_blocks"] / totals["accessed_blocks"] if totals["accessed_blocks"] else 0.0,
            "token_hit_rate": totals["hit_tokens"] / totals["accessed_tokens"] if totals["accessed_tokens"] else 0.0,
            "reusable_tokens": totals["hit_tokens"],
        })
    return output


def segment_reuse_statistics(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Measure exact-block reuse count, access interval, and lifetime by type."""
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("started_at_unix", 0.0)),
            str(row.get("run_id", "")),
            int(row.get("call_index", 0)),
        ),
    )
    accesses: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for call_position, row in enumerate(ordered_rows, start=1):
        for block in build_logical_blocks(row, tokenizer):
            accesses[(block.block_type, block.content_hash)].append((call_position, block.token_count))

    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"occurrences": 0, "unique_blocks": 0, "reuse_count": 0, "intervals": [], "lifetimes": [], "token_counts": []}
    )
    for (block_type, _), positions_and_sizes in accesses.items():
        stats = by_type[block_type]
        positions = [item[0] for item in positions_and_sizes]
        sizes = [item[1] for item in positions_and_sizes]
        stats["occurrences"] += len(positions)
        stats["unique_blocks"] += 1
        stats["reuse_count"] += max(0, len(positions) - 1)
        stats["intervals"].extend(b - a for a, b in zip(positions, positions[1:]))
        stats["lifetimes"].append(positions[-1] - positions[0] + 1)
        stats["token_counts"].extend(sizes)

    output = []
    for block_type, stats in sorted(by_type.items()):
        output.append({
            "block_type": block_type,
            "occurrences": stats["occurrences"],
            "unique_blocks": stats["unique_blocks"],
            "reuse_count": stats["reuse_count"],
            "reuse_rate": stats["reuse_count"] / stats["occurrences"] if stats["occurrences"] else 0.0,
            "mean_reuse_interval_calls": mean(stats["intervals"]) if stats["intervals"] else 0.0,
            "max_reuse_interval_calls": max(stats["intervals"], default=0),
            "mean_lifetime_calls": mean(stats["lifetimes"]) if stats["lifetimes"] else 0.0,
            "max_lifetime_calls": max(stats["lifetimes"], default=0),
            "mean_block_tokens": mean(stats["token_counts"]) if stats["token_counts"] else 0.0,
        })
    return output

