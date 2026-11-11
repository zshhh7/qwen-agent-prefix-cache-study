from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any


SEGMENT_NAMES = (
    "system_prompt",
    "tool_description",
    "conversation_history",
    "tool_result",
    "current_query",
)


def _role(message: dict[str, Any]) -> str:
    return str(message.get("role", "")).lower()


def split_prompt(
    messages: list[dict[str, Any]], functions: list[dict[str, Any]] | None
) -> OrderedDict[str, list[Any]]:
    """Split one LLM request into five mutually exclusive semantic sections."""
    sections: OrderedDict[str, list[Any]] = OrderedDict(
        (name, []) for name in SEGMENT_NAMES
    )
    sections["tool_description"] = list(functions or [])

    last_user = -1
    for index, message in enumerate(messages):
        if _role(message) == "user":
            last_user = index

    for index, message in enumerate(messages):
        role = _role(message)
        if role == "system":
            sections["system_prompt"].append(message)
        elif role in {"function", "tool"}:
            sections["tool_result"].append(message)
        elif role == "user" and index == last_user:
            sections["current_query"].append(message)
        else:
            sections["conversation_history"].append(message)
    return sections


def canonical_segment_text(name: str, value: list[Any]) -> str:
    """Stable representation used for cross-backend comparison."""
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<|segment:{name}|>\n{body}\n"


def canonical_prompt_text(
    messages: list[dict[str, Any]], functions: list[dict[str, Any]] | None
) -> str:
    """Serialize in request order so LCP reflects a real append-only chat prefix.

    Sections are useful for composition statistics, but concatenating all section
    buckets would reorder interleaved assistant and tool messages and would make
    the prefix-cache calculation invalid.
    """
    system_messages = [message for message in messages if _role(message) == "system"]
    parts = []
    if system_messages:
        parts.append(canonical_segment_text("system_prompt", system_messages))
    if functions:
        parts.append(canonical_segment_text("tool_description", list(functions)))
    for message in messages:
        if _role(message) == "system":
            continue
        body = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        parts.append(f"<|message:{_role(message)}|>\n{body}\n")
    return "".join(parts)
