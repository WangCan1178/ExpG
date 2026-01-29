## ExpG: Towards Robust Tool Use in LLM Agents via Experience-Driven Adaptive Guidance

**Language**: English (default) / [中文](./README_zh.md)


<p align="center">
  <img src="[paper/latex/figures/gitcha.png](paper/latex/figures/gitcha.png)" alt="ExpG Background" width="60%">
</p>


This repository contains the official implementation and experiment code for the paper  
**“ExpG: Towards Robust Tool Use in LLM Agents via Experience-Driven Adaptive Guidance”**.  
Our method is **built on top of the underlying memory framework provided by [Agentscope ReMe](https://github.com/agentscope-ai/ReMe)**, and introduces a new experience-driven guidance mechanism and benchmark pipelines specifically for robust tool use.

`ExpG` (Experience-driven Guidance) turns raw tool-use trajectories into reusable experience that provides:

- **Capability boundaries** and **best-practice guidance** for each tool;  
- More **robust tool selection and invocation** under dynamic or noisy environments;  
- Consistent improvements across **tool selection, tool invocation, and response generation** stages.

---

## Paper

- **Title**: *ExpG: Towards Robust Tool Use in LLM Agents via Experience-Driven Adaptive Guidance*  
- **Abbreviation**: ExpG  
- **Keywords**: Tool Use, Agent Memory, Experience Pool, Robustness, LLM Agents  

Core ideas (see `paper/example_paper.tex` for full details):

- View historical tool invocations as analyzable, learnable experiences instead of black-box events;  
- Structure the lifecycle into **Experience Acquisition → Experience Distillation → Experience Reuse**;  
- Demonstrate consistent gains on MetaTool, API-Bank, and BFCL-V3 benchmarks in both tool behavior and final task metrics;  
- Observe a clear **non-parametric scaling effect**: small models with ExpG can outperform larger, memoryless baselines.


<p align="center">
  <img src="[paper/latex/figures/gitpic.png](paper/latex/figures/gitpic.png)" alt="ExpG Overview" width="85%">
</p>


All LaTeX sources and figures (including the main overview figure above) are in:

```text
paper/
  latex/
    figures/
      overview.pdf
      ablation.png
      overlap.png
      ...
```

---

## Repository Structure

This repo includes the core **tool-memory implementation used by ExpG (within `reme_ai/`)** and multiple experiment suites (`experiment/`):

- **`reme_ai/`** – Tool memory and experience-pool implementation for ExpG
  - The implementation follows the three-stage pipeline defined in the paper (acquisition / distillation / reuse) and is reflected in the following submodules:
    - `summary/`: focuses on **experience acquisition and distillation**, extracting, filtering, clustering, and summarizing past tool calls into reusable experience units and guidance snippets;  
    - `retrieve/`: focuses on **experience reuse**, retrieving appropriate guidance conditioned on the current tool-use context for downstream agents;  
    - `vector_store/`: shared backend for storing and indexing experience vectors across all three stages;  
    - `agent/`: injects tool-memory guidance into the agent’s tool-calling pipeline;  
    - `service/` and `config/`: service wrappers and default configurations used by the experiments.  

- **`experiment/MetaTool/`** – ExpG experiments on the MetaTool benchmark  
  - `src/`: task definitions, data processing, experience generation and usage  
  - `quickstart.sh`: example end-to-end script (model download, vector DB, etc.)  

- **`experiment/api-bank/`** – Experiments on API-Bank with tool memory & ExpG  
  - `apis/`: tool / API definitions  
  - `API-Bank/`: official API-Bank data and configs  
  - `tool_memory.py` / `tool_manager.py`: recording tool calls and managing experience pools  
  - `evaluator.py`: evaluation logic and metrics (e.g., pass@k)  

- **`experiment/gorilla/`** – Experiments in Gorilla / BFCL framework  
  - `berkeley-function-call-leaderboard/`: BFCL-V3 related code and data  
  - `.../bfcl_eval/tool_memory.py`: tool experience modeling for BFCL tasks  
  - `.../bfcl_eval/model_handler/`: inference wrappers for different base models  

- **`paper/`** – Paper source and figures  
  - `example_paper.tex`: main paper source  
  - `latex/figures/`: all figures used in the paper  

- **Root**
  - `pyproject.toml`: Python package configuration (`reme_ai`)  
  - `LICENSE`: Apache 2.0 license  

---

## Environment & Installation

- **Python**: 3.12+ (as specified in `pyproject.toml`)  
- Recommended to use an isolated environment:

```bash
conda create -n expg python=3.12
conda activate expg
```

Install the core library from the project root:

```bash
pip install --upgrade pip
pip install -e .
```

Key dependency (see `pyproject.toml`):

- `flowllm[reme]>=0.1.11.6` – backend for LLMs, embeddings, and vector stores

For individual experiment suites (e.g., MetaTool), also install their local requirements:

```bash
cd experiment/MetaTool
pip install -r requirements.txt
```

### LLM & Vector Store Configuration

The project follows ReMe’s configuration style. Typical environment variables:

```bash
export FLOW_LLM_API_KEY=sk-xxxx
export FLOW_LLM_BASE_URL=https://xxxx/v1
export FLOW_EMBEDDING_API_KEY=sk-xxxx
export FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

For more details, please refer to `reme_ai/config/default.yaml` and the internal docs.

---

## ExpG Mechanism (High-level)

Following `paper/example_paper.tex`, ExpG treats tool usage as learnable experience and runs a three-stage pipeline:

1. **Experience Acquisition**  
   Analyze historical tool trajectories (success / failure, token cost, latency, etc.) and build structured experience units with context, parameter patterns, and outcomes.

2. **Experience Distillation**  
   Filter noisy or unhelpful experiences, cluster them into equivalence classes of invocation patterns, and summarize them with LLMs into generalizable textual guidance.

3. **Experience Reuse**  
   Retrieve relevant guidance for future tasks and inject it into tool selection, argument generation, and response synthesis, improving robustness under dynamic environments and imperfect feedback.

ExpG is implemented on top of ReMe’s task / personal / tool memory abstractions, vector-store-based retrieval, and multi-stage summarization / filtering operators.

---

## Citation

If you find this repository or ExpG useful in your research or applications, please consider citing (placeholder, to be updated with the final venue):

```bibtex
@article{ExpG2026,
  title   = {{ExpG}: Towards Robust Tool Use in LLM Agents via Experience-Driven Adaptive Guidance},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026},
}
```

Since this project is forked from ReMe and reuses its memory system, we also recommend citing:

```bibtex
@software{AgentscopeReMe2025,
  title  = {AgentscopeReMe: Memory Management Kit for Agents},
  author = {Li Yu and Jiaji Deng and Zouying Cao},
  url    = {https://github.com/agentscope-ai/ReMe},
  year   = {2025}
}
```

---

## License

This project is released under the **Apache License 2.0**. See `LICENSE` for the full text.  
You are welcome to use this codebase for research, extensions, and downstream systems, and we welcome issues and pull requests.
