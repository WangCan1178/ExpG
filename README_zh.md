## ExpG：为智能体 Harness 构建经验驱动的自适应指导，实现持续改进的工具使用

**语言**：中文 / [English](./README.md)

<p align="center">
  <img src="paper/latex/figures/gitcha.png" alt="ExpG 概览" width="85%">
</p>

本仓库是论文 **“Constructing Experience-Driven Adaptive Guidance for Continually Improving Tool Use in Agent Harnesses”** 的实现代码与实验脚本。

智能体的性能瓶颈正逐渐从模型能力转移到控制执行过程的 harness 上。在 harness 中，工具是智能体与外部环境交互的主要接口，但现有的工具使用指导很少能随运行时条件自适应调整。**ExpG**（Experience-driven Guidance）从历史工具调用中构建经验驱动的自适应指导，刻画每个工具的 **能力边界** 与 **最佳实践**，使工具使用能够随着经验积累而持续改进。

我们的实现基于 [ReMe](https://github.com/agentscope-ai/ReMe) 的记忆框架。

**如何使用 ReMe：** 启动 ReMe 的 Tool Memory 服务后，历史工具调用经 `add_tool_call_result` 写入并评估，经 `summary_tool_memory` 蒸馏成工具级指导，再经 `retrieve_tool_memory` 取回并注入后续推理。向量存储与服务接口由 ReMe 提供，经验获取 / 蒸馏 / 复用策略由 ExpG 实现。本实现基于 **带 Tool Memory 的 ReMe 0.1.x–0.3.x**（即本仓库中的 `reme_ai/`）；当前 ReMe `main`（0.4+，Memory-as-File）已移除上述接口，需另行适配。

---

### 方法

ExpG 包含三个阶段（完整细节见 `paper/example_paper.tex`）：

1. **经验获取（Experience Acquisition）**  
   由 $\mathrm{LLM}_{\mathrm{evaluator}}$ 对历史轨迹中的每次工具调用进行多维度归因分析，从 10 个方面打分，并归为三个视角：**工具使用环境**（工具是否被正确选择、调用时机是否恰当）、**智能体行为**（输入在语法和语义上是否正确）、**工具可靠性**（工具是否稳定执行并返回预期结果）。每条经验按工具名索引并存入向量库。

2. **经验蒸馏（Experience Distillation）**  
   随着时间推移剔除无益经验；按打分向量将剩余经验划分为 **等价类**，在上下文配额 $Q$ 内选出既符合原始分布、又覆盖稀有调用模式的代表性子集；再由 $\mathrm{LLM}_{\mathrm{summarizer}}$ 总结为工具级指导（guidance）。

3. **经验复用（Experience Reuse）**  
   智能体每次调用同一工具时复用对应指导。**动态指导** 以轻量的上下文提示注入；当经验数量达到 $Q/2$ 且大多数落在同一个非完美等价类时，指导升级为 **稳定指导**，并作为约束合并进工具 schema。新的调用记录会持续修订这两类指导。

主要结论：

- 在 MetaTool、API-Bank、BFCL-V3 上，对 **工具选择、工具调用和响应生成** 均带来一致提升；
- 配备 ExpG 的小模型可以超过不使用 ExpG 的更大模型；
- 在困难场景（相似工具、不可靠工具、缺失参数 / 缺失函数、无关函数等）中提升尤为明显。

---

### 主实验结果

MetaTool、API-Bank、BFCL-V3 上的性能对比（%）。**加粗** 为各模型组内最优。Total 按各数据集任务数加权（MetaTool 1,669，API-Bank 399，BFCL-V3 461）。

| Model | Method | MetaTool Pass@1 | MetaTool Avg@3 | MetaTool Pass@3 | API-Bank Pass@1 | API-Bank Avg@3 | API-Bank Pass@3 | BFCL-V3 Pass@1 | BFCL-V3 Avg@3 | BFCL-V3 Pass@3 | Total Pass@1 | Total Avg@3 | Total Pass@3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 nano | No Method | 72.62 | 72.76 | 78.49 | 82.96 | 83.46 | 86.97 | 53.80 | 53.00 | 60.95 | 70.82 | 70.62 | 76.63 |
| GPT-5 nano | Few-shot | 74.12 | 75.11 | 82.32 | 83.71 | 83.96 | **87.22** | 56.18 | 55.24 | 61.39 | 72.36 | 72.65 | 79.28 |
| GPT-5 nano | DRAFT | 73.94 | 73.04 | 78.97 | 84.21 | 83.46 | **87.22** | 57.27 | 57.27 | 62.26 | 72.52 | 71.58 | 77.23 |
| GPT-5 nano | Mem0 | 74.96 | 76.13 | 82.92 | 84.96 | 85.21 | **87.22** | 60.95 | 61.61 | 65.08 | 73.98 | 74.67 | 80.35 |
| GPT-5 nano | **ExpG** | **81.67** | **82.07** | **84.60** | **86.72** | **86.55** | **87.22** | **64.43** | **63.99** | **66.38** | **79.32** | **79.22** | **81.69** |
| DeepSeek-V3 | No Method | 83.10 | 82.94 | 84.66 | 84.71 | 84.38 | 85.46 | 58.79 | 59.65 | 65.94 | 78.92 | 78.66 | 81.37 |
| DeepSeek-V3 | Few-shot | 82.74 | 83.90 | 86.28 | 85.21 | 84.63 | 86.22 | 60.52 | 60.30 | 67.90 | 79.08 | 79.45 | 82.92 |
| DeepSeek-V3 | DRAFT | 80.23 | 80.79 | 82.44 | 84.96 | 85.63 | 86.47 | 62.26 | 61.61 | 68.55 | 77.70 | 77.80 | 80.54 |
| DeepSeek-V3 | Mem0 | 83.88 | 84.56 | 86.40 | 85.46 | 85.55 | 86.47 | 65.08 | 65.15 | 68.33 | 80.70 | 80.91 | 83.12 |
| DeepSeek-V3 | **ExpG** | **85.26** | **85.38** | **86.52** | **87.72** | **87.39** | **87.97** | **69.41** | **69.92** | **72.02** | **82.76** | **82.61** | **84.11** |
| Qwen3-8B | No Method | 76.51 | 76.97 | 77.71 | 83.96 | 83.88 | 84.21 | 58.79 | 58.28 | 60.30 | 74.46 | 74.41 | 75.56 |
| Qwen3-8B | Few-shot | 79.93 | 79.83 | 82.92 | 83.71 | 82.62 | 84.96 | 60.09 | 59.29 | 61.39 | 76.91 | 76.27 | 79.32 |
| Qwen3-8B | DRAFT | 78.19 | 77.33 | 77.89 | 85.71 | 84.96 | 85.46 | 60.74 | 60.30 | 62.91 | 76.20 | 75.18 | 76.35 |
| Qwen3-8B | Mem0 | 75.07 | 75.47 | 82.38 | 86.22 | 86.05 | 86.47 | 63.34 | 64.93 | 66.16 | 74.69 | 74.98 | 80.07 |
| Qwen3-8B | **ExpG** | **83.52** | **84.88** | **85.08** | **86.47** | **87.89** | **87.97** | **67.46** | **66.96** | **68.33** | **81.06** | **81.82** | **82.48** |
| Qwen3-32B | No Method | 80.05 | 79.43 | 80.17 | 84.71 | 84.88 | 85.21 | 65.15 | 65.08 | 66.16 | 78.05 | 77.55 | 78.41 |
| Qwen3-32B | **ExpG** | **84.68** | **85.02** | **86.28** | **86.97** | **87.30** | **87.72** | **70.72** | **71.01** | **73.32** | **82.48** | **82.56** | **84.14** |
| Qwen3-235B | No Method | 78.25 | 79.23 | 80.29 | 85.46 | 85.46 | 85.71 | 71.37 | 71.15 | 73.54 | 78.13 | 78.49 | 79.91 |
| Qwen3-235B | **ExpG** | **86.34** | **86.70** | **86.94** | **87.47** | **86.97** | **88.22** | **79.61** | **78.52** | **80.04** | **85.29** | **84.98** | **85.69** |

---

### 仓库结构

- **`reme_ai/`**：ExpG 使用的工具记忆与经验池实现
  - `summary/tool/`：经验获取与蒸馏（多维评估、过滤、基于等价类的选择、指导总结）；
  - `retrieve/tool/`：经验复用（根据当前工具调用上下文检索指导）；
  - `vector_store/`：经验的存储与索引后端；
  - `agent/`：将指导注入智能体的工具调用流程；
  - `service/` 与 `config/`：实验中使用的服务封装与默认配置。
- **`experiment/MetaTool/`**：MetaTool 实验（工具选择）
  - `src/`：任务定义、数据处理、经验生成与使用；
  - `quickstart.sh`：端到端运行示例脚本。
- **`experiment/api-bank/`**：API-Bank 实验（工具调用）
  - `apis/` 与 `API-Bank/`：工具定义与官方数据；
  - `tool_memory.py` / `tool_manager.py`：工具调用记录与经验池管理；
  - `evaluator.py`：评测逻辑与指标计算。
- **`experiment/gorilla/`**：BFCL-V3 实验（响应生成与多轮任务）
  - `berkeley-function-call-leaderboard/bfcl_eval/tool_memory.py`：BFCL 任务下的工具经验建模；
  - `berkeley-function-call-leaderboard/bfcl_eval/model_handler/`：不同基座模型的推理封装。
- **`experiment/0_draw_and_analysis/`**：结果汇总、分数分布分析与绘图脚本。
- **`paper/`**：论文源码（`example_paper.tex`）与图片（`latex/figures/`）。

---

### 环境与安装

- **Python**：3.12+（见 `pyproject.toml`）

```bash
conda create -n expg python=3.12
conda activate expg
pip install --upgrade pip
pip install -e .
```

核心依赖：`flowllm[reme]>=0.1.11.6`（LLM、Embedding 与向量库后端）。

各实验子项目（如 MetaTool）还需安装各自的依赖：

```bash
cd experiment/MetaTool
pip install -r requirements.txt
```

#### LLM 与向量库配置

配置方式与 ReMe 保持一致：

```bash
export FLOW_LLM_API_KEY=sk-xxxx
export FLOW_LLM_BASE_URL=https://xxxx/v1
export FLOW_EMBEDDING_API_KEY=sk-xxxx
export FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

更多选项见 `reme_ai/config/default.yaml`。论文中 $\mathrm{LLM}_{\mathrm{evaluator}}$ 与 $\mathrm{LLM}_{\mathrm{summarizer}}$ 均使用 Qwen3-max，轨迹采样温度为 0.5。

---

### 外部基准与数据集

本仓库复用了已有工作发布的公开基准。如使用本代码运行这些基准，请同时引用原始论文：

- **MetaTool**（`experiment/MetaTool/`）：选取最常用的 100 个工具，每个工具在每种子任务设置下最多采样 10 个任务，共 1,669 个任务，采用留一交叉验证。
- **API-Bank**（`experiment/api-bank/`）：使用完整的 Level-1 数据，共 399 个任务，采用留一交叉验证。
- **BFCL-V3**（`experiment/gorilla/berkeley-function-call-leaderboard/`）：过滤掉工具不会被复用的任务，保留 461 个测试任务和 1,542 个训练任务。

```bibtex
@inproceedings{metatool,
  title     = {MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use},
  author    = {Yue Huang and Jiawen Shi and Yuan Li and Chenrui Fan and Siyuan Wu and Qihui Zhang and Yixin Liu and Pan Zhou and Yao Wan and Neil Zhenqiang Gong and Lichao Sun},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)},
  year      = {2024},
}

@inproceedings{apibank,
  title     = {API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs},
  author    = {Minghao Li and Feifan Song and Yu Bowen and Haiyang Yu and Zhoujun Li and Fei Huang and Yongbin Li},
  booktitle = {Conference on Empirical Methods in Natural Language Processing},
  year      = {2023},
}

@inproceedings{patilberkeley,
  title     = {The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models},
  author    = {Patil, Shishir G and Mao, Huanzhi and Yan, Fanjia and Ji, Charlie Cheng-Jie and Suresh, Vishnu and Stoica, Ion and Gonzalez, Joseph E},
  booktitle = {Forty-second International Conference on Machine Learning (ICML)},
  year      = {2025},
}
```

---

### 引用

```bibtex
@article{ExpG2026,
  title   = {Constructing Experience-Driven Adaptive Guidance for Continually Improving Tool Use in Agent Harnesses},
  author  = {Anonymous Authors},
  journal = {Under review},
  year    = {2026},
}
```

由于本项目复用了 ReMe 的记忆系统，也建议引用：

```bibtex
@software{ReMe2026,
  title = {Remember me, Refine me: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2026}
}
```

---

### 许可证

本项目基于 **Apache License 2.0** 发布，完整条款见 `LICENSE`。
