# 面向智能体的Prompt结构与KV Cache复用特征分析

给本地 Qwen 模型一道长度题，让它先换算单位，再用计算器求总长。记录模型每一步收到的内容，看看后续调用有多少输入和前一次的开头一样。

## 结题成果

- [结题报告](结题报告.md)
- [正式实验轨迹](outputs/final_experiment_v2/traces)、[统计表](outputs/final_experiment_v2/tables)、[实验图](outputs/final_experiment_v2/figures)

4 道题各跑 2 次，一共记录了 24 次模型调用。只比较同一次做题过程中的相邻调用，整批输入约有 **59.85%** 可以理论复用。另做的缓存模拟允许不同题目共用缓存，容量设为 2048 Token 时，模拟命中率为 **87.90%**。

## 目录

```text
configs/                 实验配置
data/tasks.jsonl         统一任务集
outputs/final_experiment_v2/traces/   原始运行轨迹
outputs/final_experiment_v2/tables/   CSV 统计表
outputs/final_experiment_v2/figures/  实验图
src/prefixcache_lab/     采集与分析代码
tests/                   单元测试
```

## 安装与复现

以下命令在 Windows PowerShell 中执行，原实验使用 Python 3.12。离线分析只需要仓库内的轨迹与 Python 依赖。

```powershell
py -3.12 -m venv .venv
$Py = '.\.venv\Scripts\python.exe'
& $Py -m pip install -r requirements.txt
& $Py -m pip install -e .
& $Py -m pytest -q
& $Py -m prefixcache_lab.analyze --traces outputs/final_experiment_v2/traces/calls.jsonl --tokenizer qwen-agent-builtin --output outputs/reproduced
```

### 重新采集真实模型轨迹

安装并启动 [Ollama](https://ollama.com/download)，将 [Qwen 官方 Q4_K_M 模型](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf) 放到 `models` 目录，再在同一终端执行：

```powershell
ollama create qwen3-local:4b -f models/Modelfile.qwen3-4b
$RunDir = "outputs/run_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $Py -m prefixcache_lab.run_experiment --config configs/local_ollama.json --tasks data/tasks.jsonl --output "$RunDir/traces"
& $Py -m prefixcache_lab.analyze --traces "$RunDir/traces/calls.jsonl" --tokenizer qwen-agent-builtin --output $RunDir
```

轨迹文件按次追加，重新采集时请使用新的输出目录。正式批次使用 Ollama 0.40.4，实际模型生成受运行环境影响，重新采集可能出现不同的工具调用或回答。

## 怎么看实验文件

| 想看什么 | 对应文件 |
| --- | --- |
| 模型每次收到什么、返回什么 | [调用轨迹](outputs/final_experiment_v2/traces/calls.jsonl) |
| 每道题有没有用工具、最后答了什么 | [任务结果](outputs/final_experiment_v2/traces/runs.jsonl) |
| 系统提示、工具说明等各占多少 | [每次调用的分段明细](outputs/final_experiment_v2/tables/prompt_call_metrics.csv) |
| 同一段内容重复了几次、相隔多久 | [片段复用统计](outputs/final_experiment_v2/tables/segment_reuse_statistics.csv) |
| 缓存容量变化后，能多复用多少 | [缓存容量对比](outputs/final_experiment_v2/tables/lru_capacity_comparison.csv) |

代码里的 `run_experiment.py` 负责出题和调用模型，`tracing.py` 负责记录，`analyze.py` 和 `cache_sim.py` 用记录算出表格和图。算法的计算口径和参考资料见[结题报告](结题报告.md)。
