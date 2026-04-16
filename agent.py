"""
Scraping & Summarization Agentic AI — Capstone Project
Uses: MCP (fetch), Agentic RAG, LLM-as-Judge, Guardrails, Observability
"""

import os, time, hashlib, logging, json, textwrap
from dataclasses import dataclass, field
from typing import Optional
import anthropic

# ─── Observability / Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_trace.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("scrape_agent")

def trace(step: str, data: dict):
    log.info(f"TRACE | {step} | {json.dumps(data, ensure_ascii=False)[:300]}")

# ─── Config ────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500    # tokens ≈ chars/4, using chars here for simplicity
CHUNK_OVERLAP = 50
TOP_K        = 5
MODEL        = "claude-opus-4-5"

# ─── Data classes ──────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    embedding_key: str = ""  # for demo: BM25-style keyword hash

@dataclass
class AgentState:
    query: str
    url: str
    raw_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    retrieved: list[Chunk] = field(default_factory=list)
    summary: str = ""
    guardrail_ok: bool = True
    eval_score: int = 0
    eval_feedback: str = ""
    traces: list[dict] = field(default_factory=list)

# ─── Step 1: MCP Fetch (via Anthropic tool_use) ────────────────────────────
def mcp_fetch(state: AgentState, client: anthropic.Anthropic) -> AgentState:
    log.info(f"[1/5] MCP FETCH — {state.url}")
    t0 = time.time()

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": (
                f"Search the web for information about: {state.query}\n"
                f"URL hint: {state.url}\n"
                "Return a detailed extract of the main content, including key facts, "
                "data points, and insights. Be comprehensive."
            )
        }]
    )

    # Collect all text blocks from the response
    raw = "\n\n".join(
        block.text for block in resp.content
        if hasattr(block, "text") and block.text
    )

    state.raw_text = raw or f"[Simulated content for: {state.query}]"
    elapsed = round(time.time() - t0, 2)
    trace("mcp_fetch", {"chars": len(state.raw_text), "latency_s": elapsed, "stop_reason": resp.stop_reason})
    return state

# ─── Step 2: Chunker ────────────────────────────────────────────────────────
def chunk_text(state: AgentState) -> AgentState:
    log.info("[2/5] CHUNKING")
    text = state.raw_text
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(text), step)):
        snippet = text[start: start + CHUNK_SIZE]
        if len(snippet) < 50:
            break
        cid = hashlib.md5(snippet.encode()).hexdigest()[:8]
        chunks.append(Chunk(id=cid, text=snippet, source=state.url,
                            embedding_key=snippet.lower()))
    state.chunks = chunks
    trace("chunker", {"n_chunks": len(chunks)})
    return state

# ─── Step 3: Agentic RAG — keyword retrieval (BM25-lite) ───────────────────
def rag_retrieve(state: AgentState) -> AgentState:
    log.info("[3/5] RAG RETRIEVAL")
    query_words = set(state.query.lower().split())

    def score(chunk: Chunk) -> float:
        return sum(chunk.embedding_key.count(w) for w in query_words)

    ranked = sorted(state.chunks, key=score, reverse=True)
    state.retrieved = ranked[:TOP_K]
    trace("rag_retrieve", {"top_k": TOP_K, "scores": [score(c) for c in state.retrieved]})
    return state

# ─── Step 4a: Guardrails ────────────────────────────────────────────────────
BLOCKED_TOPICS = ["weapon", "exploit", "hack", "malware", "illegal"]

def guardrails(state: AgentState) -> AgentState:
    log.info("[4a/5] GUARDRAILS CHECK")
    combined = (state.query + " " + state.raw_text[:500]).lower()
    violations = [t for t in BLOCKED_TOPICS if t in combined]
    if violations:
        state.guardrail_ok = False
        state.summary = f"⛔ Guardrail triggered: topic(s) {violations} blocked."
        trace("guardrails", {"blocked": True, "violations": violations})
    else:
        trace("guardrails", {"blocked": False})
    return state

# ─── Step 4b: LLM Summarizer ────────────────────────────────────────────────
def llm_summarize(state: AgentState, client: anthropic.Anthropic) -> AgentState:
    if not state.guardrail_ok:
        return state
    log.info("[4b/5] LLM SUMMARIZATION")
    context = "\n\n---\n\n".join(c.text for c in state.retrieved)
    t0 = time.time()

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You are an expert research summarizer. "
            "Given retrieved context chunks, produce a well-structured, factual summary. "
            "Format: ## Key Findings\\n- bullets\\n\\n## Summary\\nparagraph\\n\\n## Data Points\\n- key stats"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Query: {state.query}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Produce a comprehensive structured summary."
            )
        }]
    )
    state.summary = resp.content[0].text
    elapsed = round(time.time() - t0, 2)
    trace("summarizer", {"tokens_out": resp.usage.output_tokens, "latency_s": elapsed})
    return state

# ─── Step 5: LLM-as-Judge ──────────────────────────────────────────────────
def llm_judge(state: AgentState, client: anthropic.Anthropic) -> AgentState:
    if not state.guardrail_ok:
        return state
    log.info("[5/5] LLM-AS-JUDGE EVALUATION")
    t0 = time.time()

    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "You are an impartial evaluator. Score the summary on: "
            "Relevance (1-5), Completeness (1-5), Accuracy (1-5), Clarity (1-5). "
            "Respond ONLY as JSON: "
            '{"relevance":N,"completeness":N,"accuracy":N,"clarity":N,"overall":N,"feedback":"..."}'
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Original query: {state.query}\n\n"
                f"Summary to evaluate:\n{state.summary}"
            )
        }]
    )

    raw = resp.content[0].text.strip()
    try:
        start = raw.index("{"); end = raw.rindex("}") + 1
        eval_data = json.loads(raw[start:end])
        state.eval_score = eval_data.get("overall", 0)
        state.eval_feedback = eval_data.get("feedback", "")
    except Exception:
        state.eval_score = -1
        state.eval_feedback = raw[:200]

    elapsed = round(time.time() - t0, 2)
    trace("llm_judge", {"score": state.eval_score, "feedback": state.eval_feedback, "latency_s": elapsed})
    return state

# ─── Orchestrator (Flow Engineering) ───────────────────────────────────────
def run_agent(query: str, url: str = "") -> AgentState:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    state = AgentState(query=query, url=url or f"https://www.google.com/search?q={query.replace(' ','+')}")

    log.info("=" * 60)
    log.info(f"AGENT START | query='{query}'")
    log.info("=" * 60)

    t_total = time.time()

    # Retry wrapper
    for attempt in range(1, 4):
        try:
            state = mcp_fetch(state, client)
            break
        except Exception as e:
            log.warning(f"Fetch attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    else:
        state.raw_text = f"[Fallback: could not fetch content for '{query}']"

    state = chunk_text(state)
    state = rag_retrieve(state)
    state = guardrails(state)
    state = llm_summarize(state, client)
    state = llm_judge(state, client)

    total = round(time.time() - t_total, 2)
    log.info(f"AGENT DONE | total_latency={total}s | eval_score={state.eval_score}/5")
    log.info("=" * 60)
    return state

# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "latest developments in agentic AI 2025"
    url   = ""  # optionally pass a URL

    state = run_agent(query, url)

    print("\n" + "=" * 60)
    print("📋 AGENT REPORT")
    print("=" * 60)
    print(f"Query    : {state.query}")
    print(f"Chunks   : {len(state.chunks)} | Retrieved: {len(state.retrieved)}")
    print(f"Guardrail: {'✅ PASS' if state.guardrail_ok else '❌ BLOCKED'}")
    print("-" * 60)
    print(state.summary)
    print("-" * 60)
    print(f"🧑‍⚖️  Eval Score : {state.eval_score}/5")
    print(f"   Feedback  : {state.eval_feedback}")
    print("=" * 60)
    print("📁 Full trace saved to: agent_trace.log")
