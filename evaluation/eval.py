"""
End-to-end evaluation of the running LangGraph agent.

Hits POST /chat, so it tests the whole pipeline: agent, query formulation, MCP
server, web fallback. Retrieval config is whatever the container is running,
so compare runs by relabelling and restarting with a different .env.

    docker compose --profile chat up -d
    uv run --no-project --with-requirements requirements.txt eval.py <testname>

Appends to runs.csv so labels can be compared across restarts.
"""

import csv
import os
import sys
import time
from pathlib import Path
import json
import httpx

# Ragas deprecation for 1.0 version
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness, ResponseRelevancy, SemanticSimilarity,
    FactualCorrectness, LLMContextPrecisionWithReference, LLMContextRecall,
)

API = os.getenv("API", "http://localhost:8080/chat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.6-luna")
NO_LOCAL = "NO_LOCAL_ANSWER"
OUT = Path("runs.csv")

# num:      stringa che DEVE comparire nella risposta (None = non verificato)
# sentinel: True se la domanda è fuori SRD e la sentinella deve scattare
# web:      valore del toggle per questo caso

CASES_FILE = Path(os.getenv("CASES", Path(__file__).parent / "golden_set.json"))
REQUIRED = {"q", "ref", "num", "sentinel", "web"}
 
 
def load_cases(path: Path) -> list[dict]:
    """Carica i casi e valida lo schema: un typo qui costerebbe un run intero."""
    if not path.exists():
        raise SystemExit(f"File dei casi non trovato: {path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(cases):
        if missing := REQUIRED - c.keys():
            raise SystemExit(f"caso {i} ({c.get('q', '?')[:40]}): mancano {missing}")
    return cases
 
 
CASES = load_cases(CASES_FILE)


def ask(case: dict) -> dict:
    """Un turno, sessione nuova: niente contaminazione fra casi."""
    t0 = time.perf_counter()
    r = httpx.post(API, json={"query": case["q"], "web": case["web"]}, timeout=180)
    r.raise_for_status()
    d = r.json()
    return {
        "answer": d["answer"],
        "contexts": d.get("contexts") or [],
        "calls": sum(1 for t in d["trace"] if t["kind"] == "call"),
        "secs": time.perf_counter() - t0,
    }


label = sys.argv[1] if len(sys.argv) > 1 else "run"
samples, ok_num, ok_sent, calls, secs, no_call = [], 0, 0, 0, 0.0, 0

for c in CASES:
    out = ask(c)
    hit_num = c["num"] is None or c["num"] in out["answer"]
    hit_sent = (NO_LOCAL in out["answer"]) == c["sentinel"]

    ok_num += int(hit_num)
    ok_sent += int(hit_sent)
    calls += out["calls"]
    secs += out["secs"]
    no_call += int(out["calls"] == 0)

    flag = "" if (hit_num and hit_sent) else "  <-- FAIL"
    print(f"  {out['calls']} calls  {out['secs']:5.1f}s  {c['q'][:52]}{flag}")

    # Remove sentinel cases from RAGAS
    if not c["sentinel"]:
        samples.append({
            "user_input": c["q"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"] or ["(no context returned)"],
            "reference": c["ref"],
        })

judge = LangchainLLMWrapper(ChatOpenRouter(model=JUDGE_MODEL, temperature=0))
judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
result = evaluate(EvaluationDataset.from_list(samples),
                  metrics=[Faithfulness(llm=judge),
                           ResponseRelevancy(llm=judge, embeddings=judge_emb),
                           SemanticSimilarity(embeddings=judge_emb), 
                           FactualCorrectness(llm=judge, mode="f1"),
                           LLMContextPrecisionWithReference(llm=judge), 
                           LLMContextRecall(llm=judge)])


scores = result.to_pandas().mean(numeric_only=True).round(3).to_dict()

n = len(CASES)
row = {
    "label": label,
    **scores,
    # "faithfulness": avg(result["faithfulness"], 3),
    # "relevancy": round(result["answer_relevancy"], 3),
    # "similarity": round(result["semantic_similarity"], 3),
    # "factual": round(result["factual_correctness(mode=f1)"], 3),
    # "ctx_precision": round(result["llm_context_precision_with_reference"], 3),
    # "ctx_recall": round(result["context_recall"], 3),
    "numeric": round(ok_num / n, 3),
    "sentinel": round(ok_sent / n, 3),
    "avg_calls": round(calls / n, 2),
    "no_call": no_call,
    "avg_secs": round(secs / n, 1),
}

write_header = not OUT.exists()
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        w.writeheader()
    w.writerow(row)

print("\n" + " ".join(f"{k}={v}" for k, v in row.items()))
print(f"appeso a {OUT}")