"""
Phase 6a: FastAPI Backend
----------------------------
Wraps the Phase 5 LangGraph pipeline in a single POST /query endpoint.

Setup:
  export GROQ_API_KEY="your-key-here"
  pip install fastapi uvicorn

Run locally:
  uvicorn api:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"query": "I want to cancel my order"}'
"""

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Reuse everything from Phase 5 -- artifacts loading + graph building
from orch import load_all_artifacts, build_graph, SupportState

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Support Triage Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for a demo; restrict in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across all requests -- this is the whole
# point of a persistent API server vs. a script that reloads everything
# per query.
print("Starting up: loading all artifacts and building the agent graph...")
_artifacts = load_all_artifacts()
_graph = build_graph(_artifacts)
print("Startup complete. Ready to serve requests.")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    intent: Optional[str]
    confidence: Optional[float]
    answer: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]
    sources: List[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Support Triage Agent API is running"}


@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    start = time.perf_counter()

    initial_state: SupportState = {
        "query": request.query,
        "intent": None,
        "classifier_confidence": None,
        "retrieved_docs": None,
        "answer": None,
        "escalated": False,
        "escalation_reason": None,
    }

    final_state = _graph.invoke(initial_state)

    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    if final_state.get("retrieved_docs"):
        sources = [doc["text"][:120] for doc in final_state["retrieved_docs"]]

    return QueryResponse(
        query=request.query,
        intent=final_state.get("intent"),
        confidence=final_state.get("classifier_confidence"),
        answer=final_state.get("answer"),
        escalated=final_state.get("escalated", False),
        escalation_reason=final_state.get("escalation_reason"),
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
