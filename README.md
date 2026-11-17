# 面向智能体的Prompt结构与KV Cache复用特征分析

本项目基于 Qwen-Agent 采集多轮工具调用轨迹，分析 Prompt 的组成及相邻请求的公共前缀。实验覆盖轨迹统计、不同容量下的 LRU 缓存模拟，以及本地推理的前缀复用对照测试。

## 技术报告与实验数据

- [结题报告](结题报告.md)
- [正式实验轨迹](outputs/final_experiment_v2/traces)、[统计表](outputs/final_experiment_v2/tables)、[实验图](outputs/final_experiment_v2/figures)
- [缓存计时记录](outputs/final_experiment_v2/latency/pairs.jsonl)

4 个同类任务各执行 2 次，共采集 24 条模型调用轨迹。同次运行内的相邻请求共有 6,904 个公共前缀 Token，占全部规范化输入的 **59.85%**。在允许跨任务保留缓存的 LRU 模拟中，2,048 Token 容量下的 Token 命中率为 **87.90%**。

在模型已加载、请求串行执行的条件下，64 对本地测量中，启用前缀复用后首 Token 延迟的中位数由 **461.0 毫秒**降至 **91.9 毫秒**。

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

## 复现前缀缓存对照实验

以下命令使用 Ollama 随附的 llama-server 和 CUDA 12 后端。按实际安装路径设置可执行文件及后端库位置，并在运行前释放其他本地模型占用的显存。脚本负责启动和关闭测试服务。

```powershell
$Py = '.\.venv\Scripts\python.exe'
$Bench = @(
    '--server-exe', 'Ollama/lib/ollama/llama-server.exe',
    '--backend-library', 'Ollama/lib/ollama/cuda_v12/ggml-cuda.dll',
    '--model', 'models/Qwen3-4B-Q4_K_M.gguf',
    '--traces', 'outputs/final_experiment_v2/traces/calls.jsonl',
    '--output', "outputs/latency_$(Get-Date -Format 'yyyyMMdd_HHmmss').jsonl",
    '--repeats', '4',
    '--port', '18084'
)
& $Py -m prefixcache_lab.benchmark_latency @Bench
```

## 数据文件说明

| 数据内容 | 对应文件 |
| --- | --- |
| 各次调用的输入消息、工具说明与响应 | [调用轨迹](outputs/final_experiment_v2/traces/calls.jsonl) |
| 工具调用过程与任务结果 | [任务运行记录](outputs/final_experiment_v2/traces/runs.jsonl) |
| 各类 Prompt 的 Token 数、占比与公共前缀长度 | [分段统计明细](outputs/final_experiment_v2/tables/prompt_call_metrics.csv) |
| 片段复用次数、间隔与生命周期 | [片段复用统计](outputs/final_experiment_v2/tables/segment_reuse_statistics.csv) |
| 不同缓存容量下的命中率与复用量 | [缓存容量对比](outputs/final_experiment_v2/tables/lru_capacity_comparison.csv) |

`run_experiment.py` 执行任务，`tracing.py` 采集模型调用轨迹，`analyze.py` 和 `cache_sim.py` 负责统计、缓存模拟和结果输出。指标定义与实验条件见[技术报告](结题报告.md)。
