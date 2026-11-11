"""Qwen-Agent official Assistant connected to the local Qwen3 model."""

from qwen_agent.agents import Assistant


def main() -> None:
    bot = Assistant(
        llm={
            "model": "qwen3-local:4b",
            "model_server": "http://127.0.0.1:11434/v1",
            "api_key": "EMPTY",
            "generate_cfg": {"thought_in_content": True},
        },
        system_message="你是一个用于Qwen-Agent课程实验的助手，请使用中文简洁回答。",
        name="本地Qwen-Agent Demo",
        description="使用本地Qwen3 4B模型运行的Qwen-Agent基础示例。",
    )

    messages = [{"role": "user", "content": "请用一句话说明Qwen-Agent是什么。 /no_think"}]
    for response in bot.run(messages=messages):
        print(response)


if __name__ == "__main__":
    main()
