# Scraping & Summarization Agentic AI — Capstone

## Use Case
Give the agent a topic or URL → it fetches relevant content → chunks and stores it → retrieves the most relevant chunks → summarizes into a structured report → evaluates its own output quality.

## Architecture
User → Fetch (MCP/LLM) → Chunker → RAG Retrieval → Guardrails → LLM Summarizer → LLM-as-Judge → Structured Report

## Technical Components
| Component | Implementation |
|-----------|---------------|
| MCP Tool | Web fetch via OpenAI GPT-4o + urllib scraper |
| Agentic RAG | BM25-lite keyword chunking + top-k retrieval |
| Flow Engineering | Sequential pipeline with retry logic (3 attempts) |
| Guardrails | Keyword-based topic filter applied pre-summarization |
| Observability | Python logging → `agent_trace.log` with JSON traces per step |
| LLM-as-Judge | Secondary GPT-4o call scores on 4 dimensions (1–5 each) |
| Caching | MD5 chunk deduplication |
| Retry | Exponential backoff on fetch failures |

## Tech Stack
- Python 3.12
- OpenAI GPT-4o (`gpt-4o`)
- Jupyter Notebook (via Anaconda)
- Standard library: `urllib`, `logging`, `hashlib`, `json`, `dataclasses`

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/scrape-agent.git
cd scrape-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY=sk-...your key here...
```

## Run via Jupyter Notebook
1. Open Anaconda Navigator → Launch Jupyter Notebook
2. Open `agent.ipynb`
3. Add your API key in Cell 1
4. Run all cells with `Shift + Enter`

## Sample Output
```
============================================================
📋 AGENT REPORT
============================================================
Query    : latest developments in agentic AI 2026
Chunks   : 10 | Retrieved: 5
Guardrail: ✅ PASS
------------------------------------------------------------
## Key Findings
- Designing AI systems that align with human values is a key challenge
- Human-in-the-loop approach advocated for alignment
- Transparency in AI decision-making crucial for user trust
- AI industry growing rapidly with investment across sectors

## Summary
Agentic AI systems capable of making autonomous decisions are gaining
traction across automotive, healthcare, and finance sectors...

## Data Points
- Total latency: 15.45s end-to-end
- Chunks retrieved: 5 of 10
- Eval score: 3/5
------------------------------------------------------------
🧑‍⚖️  Eval Score : 3/5
   Feedback  : Summary provides solid overview but lacks 2026-specific details
============================================================
📁 Full trace saved to: agent_trace.log
```

## Observability — agent_trace.log
Every step emits a structured JSON trace:
```
2026-04-15 22:34:55 [INFO] AGENT START | query='latest developments in agentic AI 2026'
2026-04-15 22:35:05 [INFO] TRACE | fetch | {"method": "llm_fallback", "chars": 4283, "latency_s": 10.16}
2026-04-15 22:35:05 [INFO] TRACE | chunker | {"n_chunks": 10}
2026-04-15 22:35:05 [INFO] TRACE | rag_retrieve | {"top_k": 5, "scores": [26, 24, 18, 17, 17]}
2026-04-15 22:35:05 [INFO] TRACE | guardrails | {"blocked": false}
2026-04-15 22:35:09 [INFO] TRACE | summarizer | {"tokens_out": 307, "latency_s": 3.62}
2026-04-15 22:35:10 [INFO] TRACE | llm_judge | {"score": 3, "feedback": "..."}
2026-04-15 22:35:10 [INFO] AGENT DONE | total_latency=15.45s | eval_score=3/5
```

## Deliverables Checklist
- [x] Codebase — `agent.ipynb`
- [x] Architecture diagram — included in presentation
- [x] Use case illustration — web research summarization agent
- [x] Deployment & telemetry — `agent_trace.log` with per-step JSON traces
- [x] MCP integration — web fetch tool
- [x] Agentic RAG — chunking + keyword retrieval pipeline
- [x] Flow engineering — sequential pipeline with retry + exponential backoff
- [x] Guardrails — pre-summarization safety keyword filter
- [x] LLM-as-Judge — eval with 4-dimension score + written feedback
- [x] Observability — structured JSON traces per step with latency tracking
- [x] Presentation — PDF slide deck

## Project Structure
```
scrape-agent/
├── agent.ipynb           # Jupyter notebook (codebase + executed output)
├── requirements.txt      # Dependencies
└── README.md             # This file
```
