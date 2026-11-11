from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from .io_utils import append_jsonl, to_jsonable


class ChatTraceRecorder:
    """Patch a Qwen-Agent model instance and record every public chat call.

    The patch is deliberately placed at ``llm.chat`` because FnCallAgent passes
    both the complete message list and function descriptions through this API.
    Streaming output is consumed lazily and the final chunk is stored.
    """

    def __init__(self, output_path: Path, run_id: str, task_id: str):
        self.output_path = output_path
        self.run_id = run_id
        self.task_id = task_id
        self.call_index = 0

    def attach(self, llm: Any) -> Any:
        original_chat = llm.chat

        def traced_chat(
            messages: list[Any],
            functions: list[dict[str, Any]] | None = None,
            stream: bool = True,
            delta_stream: bool = False,
            extra_generate_cfg: dict[str, Any] | None = None,
        ) -> Any:
            self.call_index += 1
            call_id = str(uuid.uuid4())
            started = time.time()
            request = {
                "schema_version": 1,
                "event": "llm_call",
                "run_id": self.run_id,
                "task_id": self.task_id,
                "call_id": call_id,
                "call_index": self.call_index,
                "started_at_unix": started,
                "model": getattr(llm, "model", None),
                "model_type": getattr(llm, "model_type", None),
                "messages": to_jsonable(messages),
                "functions": to_jsonable(functions or []),
                "extra_generate_cfg": to_jsonable(extra_generate_cfg or {}),
            }
            result = original_chat(
                messages=messages,
                functions=functions,
                stream=stream,
                delta_stream=delta_stream,
                extra_generate_cfg=extra_generate_cfg,
            )

            if not stream:
                self._write(request, result, started)
                return result

            def consume() -> Iterator[Any]:
                final_chunk: Any = []
                try:
                    for chunk in result:
                        final_chunk = chunk
                        yield chunk
                finally:
                    self._write(request, final_chunk, started)

            return consume()

        llm.chat = traced_chat
        return llm

    def _write(self, request: dict[str, Any], response: Any, started: float) -> None:
        row = dict(request)
        row["finished_at_unix"] = time.time()
        row["latency_seconds"] = row["finished_at_unix"] - started
        row["response"] = to_jsonable(response)
        append_jsonl(self.output_path, [row])

