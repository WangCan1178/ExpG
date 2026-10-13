## ExpG: Constructing Experience-Driven Adaptive Guidance for Continually Improving Tool Use in Agent Harnesses

**Language**: English (default) / [中文](./README_zh.md)

<p align="center">
  <img src="paper/latex/figures/gitcha.png" alt="ExpG overview" width="85%">
</p>

This repository contains the implementation and experiment code for the paper  
**“Constructing Experience-Driven Adaptive Guidance for Continually Improving Tool Use in Agent Harnesses”**.

The performance bottleneck of agents is increasingly shifting from model capability to the agent harness that governs execution. Within this harness, tools are the primary interface to external environments, yet available guidance rarely adapts across runtime conditions. **ExpG** (Experience-driven Guidance) constructs experience-driven adaptive guidance that captures each tool's **capability boundaries** and **best practices**, so that tool use can improve as experience accumulates.

Our implementation is built on top of the memory framework of [ReMe](https://github.com/agentscope-ai/ReMe).

**How ReMe is used:** We run ReMe's Tool Memory service; historical tool calls are written and evaluated via `add_tool_call_result`, distilled into tool-level guidance via `summary_tool_memory`, then retrieved and injected into later reasoning via `retrieve_tool_memory`. ReMe provides the vector store and service APIs; the acquisition / distillation / reuse strategy is implemented by ExpG. This implementation targets **ReMe 0.1.x–0.3.x with Tool Memory** (the `reme_ai/` package in this repo). Current ReMe `main` (0.4+, Memory-as-File) has removed these APIs and would need additional adaptation.

---

## Method

ExpG consists of three phases (see `paper/example_paper.tex` for full details):

1. **Experience Acquisition**  
   An $\mathrm{LLM}_{\mathrm{evaluator}}$ analyzes the quality of each tool invocation in historical trajectories through multi-aspect attribution. It scores 10 aspects grouped into three perspectives: **tool-use environment** (is the tool selected and invoked at the right time), **agent behavior** (is the input syntactically and semantically correct), and **tool reliability** (does the tool execute stably and respond as expected). Each experience is indexed by tool name in a vector store.

2. **Experience Distillation**  
   Unhelpful experiences are filtered out over time. The remaining ones are partitioned into **equivalence classes** by their score vectors, and a representative subset is selected under a context quota $Q$ so that it both follows the original distribution and covers rare invocation patterns. An $\mathrm{LLM}_{\mathrm{summarizer}}$ then summarizes the selected experiences into tool-level guidance.

3. **Experience Reuse**  
   Guidance is reused whenever the agent invokes the same tool. **Dynamic guidance** is injected as lightweight contextual prompts. When enough experiences accumulate (at least $Q/2$) and most of them fall into a single non-perfect class, the guidance becomes **stable guidance** and is merged into the tool schema as a constraint. As new invocations are recorded, both forms of guidance are revised.

Key findings:

- Consistent improvements across **tool selection, tool calling, and response generation** on MetaTool, API-Bank, and BFCL-V3;
- Smaller agents with ExpG can outperform larger agents that do not use it;
- Particularly strong gains in challenging settings (similar tools, unreliable tools, missing parameters / functions, irrelevant functions).

---

## Main Results

Performance (%) across MetaTool, API-Bank, and BFCL-V3. **Bold** indicates the best within each model. Total is weighted by the number of tasks in each dataset (MetaTool 1,669, API-Bank 399, BFCL-V3 461).

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

## Repository Structure

- **`reme_ai/`** – Tool memory and experience-pool implementation used by ExpG
  - `summary/tool/`: experience acquisition and distillation (multi-aspect evaluation, filtering, equivalence-class-based selection, guidance summarization);
  - `retrieve/tool/`: experience reuse (retrieving guidance for the current tool-use context);
  - `vector_store/`: shared storage and indexing backend for experiences;
  - `agent/`: injects guidance into the agent's tool-calling pipeline;
  - `service/` and `config/`: service wrappers and default configurations used by the experiments.
- **`experiment/MetaTool/`** – Experiments on MetaTool (tool selection)
  - `src/`: task definitions, data processing, experience generation and usage;
  - `quickstart.sh`: example end-to-end script.
- **`experiment/api-bank/`** – Experiments on API-Bank (tool calling)
  - `apis/` and `API-Bank/`: tool definitions and official data;
  - `tool_memory.py` / `tool_manager.py`: recording tool calls and managing experience pools;
  - `evaluator.py`: evaluation logic and metrics.
- **`experiment/gorilla/`** – Experiments on BFCL-V3 (response generation and multi-turn)
  - `berkeley-function-call-leaderboard/bfcl_eval/tool_memory.py`: tool experience modeling for BFCL tasks;
  - `berkeley-function-call-leaderboard/bfcl_eval/model_handler/`: inference wrappers for different backbone models.
- **`experiment/0_draw_and_analysis/`** – Scripts for result aggregation, score distribution analysis, and figures.
- **`paper/`** – Paper source (`example_paper.tex`) and figures (`latex/figures/`).

---

## Environment & Installation

- **Python**: 3.12+ (as specified in `pyproject.toml`)

```bash
conda create -n expg python=3.12
conda activate expg
pip install --upgrade pip
pip install -e .
```

Key dependency: `flowllm[reme]>=0.1.11.6` (backend for LLMs, embeddings, and vector stores).

For individual experiment suites (e.g., MetaTool), also install their local requirements:

```bash
cd experiment/MetaTool
pip install -r requirements.txt
```

### LLM & Vector Store Configuration

The project follows ReMe's configuration style:

```bash
export FLOW_LLM_API_KEY=sk-xxxx
export FLOW_LLM_BASE_URL=https://xxxx/v1
export FLOW_EMBEDDING_API_KEY=sk-xxxx
export FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

See `reme_ai/config/default.yaml` for more options. In the paper, Qwen3-max is used as both $\mathrm{LLM}_{\mathrm{evaluator}}$ and $\mathrm{LLM}_{\mathrm{summarizer}}$, and trajectories are sampled with temperature 0.5.

---

## External Benchmarks & Datasets

This repository reuses public benchmarks released by prior work. If you use our code to run these benchmarks, please also cite the original papers:

- **MetaTool** (`experiment/MetaTool/`): the 100 most frequently used tools, up to 10 tasks per tool per sub-task setting, 1,669 tasks in total, leave-one-out cross-validation.
- **API-Bank** (`experiment/api-bank/`): the full Level-1 set, 399 tasks, leave-one-out cross-validation.
- **BFCL-V3** (`experiment/gorilla/berkeley-function-call-leaderboard/`): tasks whose tools are reused elsewhere, 461 test tasks and 1,542 training tasks.

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

## Citation

```bibtex
@article{ExpG2026,
  title   = {Constructing Experience-Driven Adaptive Guidance for Continually Improving Tool Use in Agent Harnesses},
  author  = {Anonymous Authors},
  journal = {Under review},
  year    = {2026},
}
```

Since this project reuses the ReMe memory system, we also recommend citing:

```bibtex
@software{ReMe2026,
  title = {Remember me, Refine me: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2026}
}
```

---

## License

This project is released under the **Apache License 2.0**. See `LICENSE` for the full text.
