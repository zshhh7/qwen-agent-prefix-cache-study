from __future__ import annotations

from typing import Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class UnicodeCodepointTokenizer:
    """Offline test fallback; not valid for final model-token experiments."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class QwenAgentTokenizer:
    """Use the tokenizer bundled with Qwen-Agent without another model download."""

    def __init__(self) -> None:
        from qwen_agent.utils.tokenization_qwen import tokenizer

        self._tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text)


def load_tokenizer(model_name: str) -> TokenizerLike:
    if model_name == "qwen-agent-builtin":
        return QwenAgentTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index

