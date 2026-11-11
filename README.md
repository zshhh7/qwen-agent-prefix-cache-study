# Qwen-Agent Prompt 复用与 Prefix Cache 实验

本项目按照“面向智能体的 Prompt 结构与 KV Cache 复用特征分析”课题要求，实现 Qwen-Agent 多轮工具调用轨迹采集、Prompt 类型标注、复用间隔与生命周期统计、逻辑缓存块划分、连续前缀命中判断和简化 LRU 模拟。

## 结题成果

- [结题报告](结题报告.md)
- [正式实验轨迹](outputs/final_experiment_v2/traces)、[统计表](outputs/final_experiment_v2/tables)、[实验图](outputs/final_experiment_v2/figures)

4 个单位换算任务各执行 2 次，共采集 24 次真实模型调用。按同次任务内的相邻请求计算，规范化输入的理论可复用比例为 **59.85%**。跨任务 LRU 重放在 2048 Token 容量下达到 **87.90%** 的模拟 Token 命中率。两项指标使用各自的统计范围和分母。

## 目录

```text
configs/                 实验配置
data/tasks.jsonl         统一任务集
outputs/traces/          原始运行轨迹（运行后生成）
outputs/tables/          CSV 统计表（分析后生成）
outputs/figures/         实验图（分析后生成）
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

## 指标定义

设第 `i` 次调用的规范化 Prompt Token 序列为 `P_i`：

- 公共前缀：`L_i = LCP(P_(i-1), P_i)`
- 当前调用重复比例：`R_i = L_i / |P_i|`
- 理论可减少 Prefill Token：`Saved_i = L_i`
- 理论节省率：`sum(Saved_i) / sum(|P_i|)`（首个调用的 `Saved=0`）

这里的 `Saved` 是不考虑缓存块大小、淘汰、路由和服务端实现的理论上界，不等同于账单优惠或真实延迟降幅。SGLang 验证实验应另行记录运行时缓存命中和 TTFT。

除相邻调用 LCP 外，程序还统计：

- `reuse_count`：同类型、同内容逻辑块在首次出现之后的重复出现次数；
- `reuse_interval`：同一逻辑块相邻两次访问之间相隔的 LLM 调用数；
- `lifetime`：逻辑块从首次出现到最后一次出现覆盖的调用数；
- `block_hit_rate`：满足连续前缀完整性的命中块数占访问块数比例；
- `token_hit_rate`：命中逻辑块的 Token 数占全部访问 Token 数比例。

## Prompt 分段规则

- `System Prompt`：system 角色消息。
- `Tool Description`：传给模型的 functions/tools JSON 描述。
- `Conversation History`：当前查询之前的 user/assistant 历史，以及此前的工具调用请求。
- `Tool Result`：function/tool 角色消息。
- `Current Query`：最后一条 user 消息。

分析器保存每段的 Token 数与占比。为保证跨后端可比，使用固定的规范化 JSON 序列化。服务端可能再应用自己的 chat template，因此报告中应将其表述为“规范化 Prompt Token 统计”；若需要与服务端逐 Token 完全一致，应在 SGLang 侧额外导出最终渲染 Prompt。

## 逻辑缓存块与 LRU 模拟

规范化 Prompt 按以下顺序划分为缓存块：

1. System Prompt 整体块；
2. Tool Description 整体块；
3. 其余每条 user、assistant、function/tool 消息各一个块，并保留原始调用顺序。

命中采用连续前缀规则：只有当前块及其之前的所有块均已存在于缓存中，当前块才计为命中。缓存容量按增量块的 Token 数计算，默认比较 `0、256、512、1024、2048、4096、8192` Token。满容量时淘汰最久未访问块。该实现是用于教学分析的简化 LRU，而不是 SGLang RadixAttention 的逐行复刻。

分析完成后主要生成：

- `segment_reuse_statistics.csv`：各类型复用次数、间隔和生命周期；
- `lru_capacity_comparison.csv`：不同容量的块命中率、Token 命中率及可复用 Token；
- `segment_reuse_comparison.png`：片段类型对比图；
- `lru_capacity_comparison.png`：容量敏感性图；
- `prompt_composition.png`、`prefix_reuse.png`：Prompt 组成与相邻调用复用图。

## 与正式任务要求的对应关系

| 正式要求 | 项目实现 |
|---|---|
| 简化的多轮工具调用 Agent | 单位换算 → 计算器两步场景 |
| 记录每轮 Prompt 和调用顺序 | `calls.jsonl` 中的 `run_id`、`call_index`、messages、functions、response |
| 按类型标注片段 | 五类 Prompt 分段及每消息逻辑块 |
| 统计复用次数、间隔和生命周期 | `segment_reuse_statistics.csv` |
| 连续前缀匹配 | LCP 指标和逻辑块前缀完整性判断 |
| 简化 LRU 缓存 | `PrefixLRUCache` |
| 比较不同缓存容量 | 7 档容量的 CSV 和对比图 |
| 结题成果 | 代码、真实轨迹、统计表、实验图与结题报告 |

## 建议先读的资料

1. Qwen-Agent Quickstart 与 LLM configuration：理解 `Assistant`、`llm.chat`、工具定义及后端切换。
2. *Qwen3 Technical Report*：重点阅读模型概览、thinking/non-thinking 模式和 Agent 能力，不必先通读训练细节。
3. *SGLang* 论文的 RadixAttention 章节：重点理解公共前缀、radix tree、KV Cache 的插入/匹配/淘汰。

## 主要参考文献

[1] Qwen Team. Qwen-Agent: LLM Application Development Framework. GitHub, 2026.

[2] Yang, A., et al. Qwen3 Technical Report. arXiv:2505.09388, 2025.

[3] Zheng, L., et al. SGLang: Efficient Execution of Structured Language Model Programs. NeurIPS, 2024.
