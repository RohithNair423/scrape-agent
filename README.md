# Scraping & Summarization Agentic AI — Capstone

## Architecture
User → MCP Fetch (web_search tool) → Chunker → RAG Retrieval → Guardrails → LLM Summarizer → LLM-as-Judge → Structured Report

## Technical Components
| Component | Implementation |
|-----------|---------------|
| MCP Tool | Anthropic `web_search_20250305` tool |
| Agentic RAG | BM25-lite keyword chunking + top-k retrieval |
| Flow Engineering | Sequential pipeline with retry logic |
| Guardrails | Keyword-based topic filter pre-LLM |
| Observability | Python logging → `agent_trace.log` with JSON traces |
| LLM-as-Judge | Secondary Claude call scores on 4 dimensions |
| Caching | MD5 chunk deduplication |
| Retry | Exponential backoff on fetch (3 attempts) |

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run
```bash
python agent.py "what is the impact of AI on software engineering"
python agent.py "latest LLM benchmarks 2025"
```

## Output
- Printed structured report (Key Findings, Summary, Data Points)
- `agent_trace.log` — full observability trace with latencies
- Eval score (1–5) from LLM-as-Judge

## Deliverables Covered
- [x] Codebase (this repo)
- [x] Architecture diagram (see diagram image)
- [x] Use case: web research summarization agent
- [x] Deployment & telemetry: `agent_trace.log`
- [x] MCP integration: `web_search` tool
- [x] Agentic RAG: chunking + retrieval pipeline
- [x] Guardrails: safety filter
- [x] LLM-as-Judge: eval with score + feedback
- [x] Observability: structured JSON traces per step
