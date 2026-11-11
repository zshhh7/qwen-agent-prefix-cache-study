from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model

from .io_utils import append_jsonl, read_jsonl, to_jsonable
from .tools import Calculator, UnitConverter  # noqa: F401
from .tracing import ChatTraceRecorder


SYSTEM_MESSAGE = """你是可复现实验中的工具调用助手。严格依次完成以下步骤：
1. 调用 unit_converter 完成题目要求的单位换算；
2. 使用换算结果调用 calculator 计算多件物品的总长度；
3. 给出包含换算结果和总长度的简短最终答案。
每轮只能调用一个工具；必须等待上一个工具返回结果后，才能调用下一个工具。
不得凭空编造工具结果，也不得跳过步骤。"""


def build_llm_cfg(config: dict) -> dict:
    cfg = {
        "model": config["model"],
        "model_type": config["model_type"],
        "generate_cfg": config.get("generate_cfg", {}),
    }
    if config.get("model_server"):
        cfg["model_server"] = config["model_server"]
    if config.get("api_key"):
        cfg["api_key"] = config["api_key"]
    return cfg


def run(config_path: Path, tasks_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["backend"] == "dashscope" and not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 环境变量；不要把 Key 写入配置文件。")

    tasks = read_jsonl(tasks_path)
    calls_path = output_dir / "calls.jsonl"
    runs_path = output_dir / "runs.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(1, int(config.get("repetitions", 1)) + 1):
        for task in tasks:
            run_id = str(uuid.uuid4())
            llm = get_chat_model(build_llm_cfg(config))
            recorder = ChatTraceRecorder(calls_path, run_id, task["task_id"])
            recorder.attach(llm)
            bot = Assistant(
                llm=llm,
                system_message=SYSTEM_MESSAGE,
                function_list=["unit_converter", "calculator"],
            )
            messages = [{"role": "user", "content": task["query"]}]
            final_response = []
            try:
                for final_response in bot.run(messages=messages):
                    pass
                status, error = "ok", None
            except Exception as exc:
                status, error = "error", repr(exc)
            append_jsonl(runs_path, [{
                "schema_version": 1,
                "run_id": run_id,
                "task_id": task["task_id"],
                "repetition": repetition,
                "query": task["query"],
                "status": status,
                "error": error,
                "response": to_jsonable(final_response),
                "llm_call_count": recorder.call_index,
                "backend": config["backend"],
                "model": config["model"],
            }])
            print(f"[{status}] repetition={repetition} task={task['task_id']} calls={recorder.call_index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dashscope.json"))
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/traces"))
    args = parser.parse_args()
    run(args.config, args.tasks, args.output)


if __name__ == "__main__":
    main()

