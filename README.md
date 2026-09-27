# Signal-SQL: Signal Detection & MCTS-Driven Example Selection for Text-to-SQL

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: Spider](https://img.shields.io/badge/Benchmark-Spider-green.svg)](https://yale-lily.github.io/spider)
[![Benchmark: BIRD](https://img.shields.io/badge/Benchmark-BIRD-orange.svg)](https://bird-bench.github.io/)
[![Agent: Signal-SQL-Agent](https://img.shields.io/badge/Agent%20System-Signal--SQL--Agent-blueviolet.svg)](https://github.com/huohuo0212/Signal-SQL-Agent)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/huohuo0212/Signal-SQL/pulls)

**An advanced In-Context Learning (ICL) and prompt engineering framework for Text-to-SQL, powered by Signal Detection Theory, CFAR clutter calibration, analytic SINR reward, and Monte Carlo Tree Search (MCTS) combinatorial optimization.**

[Overview](#-overview--项目简介) | [Core Architecture](#-core-architecture--核心架构) | [Quick Start](#-quick-start--快速上手) | [Benchmark](#-experimental-results--实验表现) | [Agent Version](#-ecosystem--关联项目)

</div>

---

> [!TIP]
> 🤖 **Looking for the Agentic Version? / 寻找智能体版本？**
> Check out **[Signal-SQL-Agent](https://github.com/huohuo0212/Signal-SQL-Agent)** — our full-fledged Multi-Agent system featuring database sandbox execution, dynamic schema exploration, and automated self-correction loops powered by Signal-SQL!

---

## 🌟 Overview / 项目简介

In Few-Shot In-Context Learning for Text-to-SQL, the **quality and diversity of demonstration examples** play a decisive role in model accuracy. Traditional selection methods (such as BM25, Euclidean Distance, or naive Cosine Similarity) suffer from three critical bottlenecks:
1. **Entity Bias & Semantic Drift**: Embeddings overfit to superficial domain nouns (e.g. "student", "stadium") rather than underlying SQL relational operations (`JOIN`, `GROUP BY`, `HAVING`).
2. **High Intra-Subset Redundancy**: Greedily selecting Top-$k$ nearest neighbors often yields near-duplicate examples that waste context window tokens without offering complementary guidance.
3. **Multi-Clause Blindspot**: Complex natural language queries require multiple relational steps. A single retrieved example rarely covers all needed clauses, while greedy selection fails to form an orthogonal portfolio.

**Signal-SQL** resolves these bottlenecks by formulating Few-Shot selection as a **Signal Detection and Combinatorial Search** problem:
- **Pre-whitening Filter**: Decorrelates dense embedding dimensions to eliminate colored background noise.
- **CFAR Detection**: Calibrates dynamic clutter thresholds via cross-domain hard negative mining.
- **Hybrid Atomic Decomposition**: Breaks complex queries into atomic sub-intents and builds a dual-channel (BM25 + Dense) pool.
- **Analytic SINR Reward**: Evaluates Signal Coverage ($E_{\text{match}}$), Redundant Noise ($E_{\text{noise}}$), and Intra-Subset Interference ($E_{\text{inter}}$) with zero LLM rollout cost.
- **MCTS Combinatorial Optimization**: Discovers the globally optimal, mutually complementary example portfolio.

---

## 🚀 Core Architecture / 核心架构

```mermaid
flowchart TD
    subgraph Input ["1. Input Target"]
        Q["User Question + Schema Linking"]
    end

    subgraph SignalProcessing ["2. Signal Detection & Calibration"]
        W["Covariance Pre-whitening W = C^(-1/2)"]
        CFAR["CFAR Clutter Threshold Calibration"]
        Q --> W
        W --> CFAR
    end

    subgraph CandidatePool ["3. Hybrid Pooling & Atomic Decomposition"]
        BM25["BM25 Lexical Retrieval"]
        Dense["Pre-whitened Dense Retrieval"]
        Decomp["LLM Atomic Sub-query Decomposition"]
        CFAR --> BM25
        CFAR --> Dense
        CFAR --> Decomp
        BM25 & Dense & Decomp --> Pool["Candidate Pool C (Zipper Merge)"]
    end

    subgraph MCTS_SINR ["4. Combinatorial Optimization (MCTS + SINR)"]
        MCTS_Node["MCTS State Search (UCT Selection)"]
        SINR["Analytic SINR Reward: E_match / (1 + λ1·E_noise + λ2·E_inter)"]
        Pool --> MCTS_Node
        MCTS_Node <--> SINR
        MCTS_Node --> BestSub["Optimal Few-Shot Portfolio S*"]
    end

    subgraph PromptFactory ["5. Disentangled Prompt Assembly"]
        Repr["SQL DDL Representation"]
        Format["QA / ONLYSQL Formatting"]
        BestSub --> PromptFactoryEngine["prompt_factory"]
        Repr --> PromptFactoryEngine
        Format --> PromptFactoryEngine
        PromptFactoryEngine --> FinalPrompt["Final Few-Shot Prompt"]
    end

    FinalPrompt --> LLM["Target LLM (GPT-4 / Claude / DeepSeek)"]
```

---

## 🔬 Analytic SINR Reward Function / 信干噪比公式

Instead of relying on slow, expensive LLM rollouts during candidate selection, Signal-SQL evaluates candidate subsets $S$ in microseconds via an analytic SINR formulation:

$$R(S) = \frac{E_{\text{match}}(S, x)}{1 + \lambda_1 E_{\text{noise}}(S, x) + \lambda_2 E_{\text{inter}}(S)}$$

- **Signal Energy ($E_{\text{match}}$)**: Logic keyword and structural feature coverage rate:
  $$E_{\text{match}} = \frac{|\text{Feat}(x) \cap (\bigcup_{e \in S} \text{Feat}(e))|}{|\text{Feat}(x)|}$$
- **Noise Energy ($E_{\text{noise}}$)**: Redundant SQL logic present in candidates but unneeded by target $x$:
  $$E_{\text{noise}} = \frac{\sum_{e \in S} |\text{Feat}(e) - \text{Feat}(x)|}{\text{Total Features}}$$
- **Interference Energy ($E_{\text{inter}}$)**: Intra-subset redundancy (cosine similarity $> 0.9$) plus SQL coding style inconsistency:
  $$E_{\text{inter}} = \frac{1}{\binom{n}{2}} \sum_{i<j} \left( \text{Sim}_{\cos}(e_i, e_j) + \mathbb{I}[\text{Style}(e_i) \neq \text{Style}(e_j)] \right)$$

---

## 🛠️ Project Structure / 目录结构

```text
├── prompt/
│   ├── PromptReprTemplate.py      # 18 schema representation styles (SQL DDL, Text, CoT, CBR, etc.)
│   ├── ExampleFormatTemplate.py   # Example format styles (QA, ONLYSQL, COMPLETE, QAWRULE, etc.)
│   ├── ExampleSelectorTemplate.py # Baseline & DAIL-SQL selectors (BM25, XiYan, QuestionMask, etc.)
│   ├── SignalSelector.py          # SIGNAL_DETECTION selector implementation
│   ├── PromptICLTemplate.py       # ICL Prompt template with token budget management
│   └── prompt_builder.py          # Decoupled prompt factory (prompt_factory)
├── utils/
│   ├── signal_lib.py              # Pre-whitening & CFAR clutter threshold calibration
│   ├── sinr_reward.py             # Feature extraction & analytic SINR reward calculation
│   ├── mcts_optimizer.py          # Monte Carlo Tree Search with UCT combinatorial optimization
│   ├── data_builder.py            # Dataset loaders for Spider, Spider-Realistic, and BIRD
│   ├── post_process.py            # SQL execution accuracy & multi-set equivalence checker
│   ├── enums.py                   # Enums (REPR_TYPE, SELECTOR_TYPE, EXAMPLE_TYPE, LLM)
│   └── linking_utils/             # 5-gram & CoreNLP schema linking utilities
├── llm/
│   └── chatgpt.py                 # OpenAI / OpenRouter API client with exponential backoff
├── results/                       # Experimental logs and prediction outputs
├── run_demo.py                    # Standalone Prompt & MCTS demo (Runs out of the box!)
├── main.py                        # Unified CLI entrypoint for Spider/BIRD benchmark evaluation
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Environment configuration template
└── LICENSE                        # MIT License
```

---

## ⚡ Quick Start / 快速上手

### 1. Installation

```bash
git clone https://github.com/huohuo0212/Signal-SQL.git
cd Signal-SQL
pip install -r requirements.txt
```

### 2. Run Instant Demo (10 Seconds, No Dataset Download Required)

Experience feature extraction, SINR evaluation, MCTS search, and prompt assembly instantly:

```bash
python run_demo.py
```

### 3. Python API Integration

```python
from prompt.prompt_builder import prompt_factory
from utils.enums import REPR_TYPE, SELECTOR_TYPE, EXAMPLE_TYPE
from utils.data_builder import load_data

# 1. Load dataset (Spider / BIRD)
data = load_data("spider", "./dataset")

# 2. Build customized ICL Prompt class
PromptClass = prompt_factory(
    repr_type=REPR_TYPE.CODE_REPRESENTATION,         # "SQL" (DDL schema)
    k_shot=7,                                        # 7-shot demonstration
    example_format=EXAMPLE_TYPE.QA,                  # "QA" pair style
    selector_type=SELECTOR_TYPE.SIGNAL_DETECTION     # Signal-SQL Selector
)

# 3. Instantiate and format
builder = PromptClass(data=data)
target_question = data.get_test_json()[0]
prompt_info = builder.format(target=target_question, max_seq_len=4096, max_ans_len=512, scope_factor=1)

print(prompt_info["prompt"])
```

---

## 📊 Experimental Results / 实验表现

Evaluated on standard cross-domain benchmarks **Spider** and **BIRD**:

| Method | Selector | Representation | Format | Shots | Spider Dev (EX %) | BIRD Dev (EX %) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| Baseline | Random | TEXT | QA | 5 | 69.4% | 46.2% |
| Baseline | BM25 | SQL | QA | 5 | 74.2% | 49.5% |
| DAIL-SQL | EUCDISQUESTIONMASK | SQL | QA | 7 | 78.9% | 53.1% |
| DAIL-SQL | EUCDISMASKPRESKLSIMTHR | SQL | QA | 9 | 82.4% | 55.4% |
| **Signal-SQL (Ours)** | **SIGNAL_DETECTION (MCTS)** | **SQL** | **QA** | **7** | **84.8%** | **58.7%** |

---

## 🔗 Ecosystem / 关联项目

- **[Signal-SQL-Agent](https://github.com/huohuo0212/Signal-SQL-Agent)**: The full-featured Multi-Agent system powered by Signal-SQL, adding dynamic schema exploration, safe sandbox execution, and automated self-correction loops.

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/huohuo0212/Signal-SQL/issues).

This project is licensed under the [MIT License](LICENSE).
