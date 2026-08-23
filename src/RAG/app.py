# app.py

import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Optional, Required, TypedDict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openrouter import ChatOpenRouter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

MODEL = os.environ.get("MODEL", "google/gemma-4-31b-it")
MODEL_WEB = os.environ.get("MODEL_WEB", "perplexity/sonar")
MCP_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")
MAX_MESSAGES = 20
MAX_TOOL_ROUNDS = 3
RECURSION_LIMIT = 20

NO_LOCAL = "NO_LOCAL_ANSWER"

SYSTEM_PROMPT = f"""
You are a D&D rules retrieval agent.

You MUST follow this process:

For every question:
1. Decide if the answer requires locating a specific document.

2. If info is needed, call retrieve_rules with:
   - `query`: the specific term you are looking for (e.g. "owlbear", "fireball")

Some questions need MORE THAN ONE retrieval. Rule CONCEPTS are retrievable
too, not just named entities. Valid queries include: "hit points per level",
"level advancement", "saving throw", "advantage", "armor class".
   
Reuse previously retrieved text ONLY for terms already retrieved in this
conversation. Any NEW named entity (class, monster, spell, item) requires a
fresh retrieve_sources call, even if the conversation already contains
related rules text.

NEVER state a numeric value (hit points, hit dice, damage, AC, DC, range)
unless it appears in retrieved text visible above. If you need a number you
have not retrieved, call the tools. Do not rely on your own knowledge of D&D.

If numerical estimates are needed to answer the user request, use average values
for dice throw, but explicitly say so. When performing math, state the formulas explicitly.

Every answer ends with a "Sources:" line listing the SOURCE paths you used.

Then, if the retrieved text does not actually contain something the user asked
about and you are not able to infer it from the local sources — the retrieval 
returned related material but not that specific entity — add ONE MORE LINE, alone,
after "Sources:", nothing following it:

{NO_LOCAL}: <the exact term that is missing>
Before emitting NO_LOCAL_ANSWER, do it only if the answer is incomplete and 
you cannot find the information in the tool.
"""

WEB_PROMPT = """You are completing a D&D 5e rules answer. Part of it was already
answered from the local rules text below; one specific thing was missing from
it, so search the web for that missing part.

Write the answer as ONE coherent piece that blends the local part and what you
found on the web — do not present them as two separate answers, and do not
repeat what the local part already said. Be brief.

End with a single "Sources:" section listing every source actually used: keep
local paths exactly as given, and prefix each web URL with "web: ".
"""

class AgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    tool_rounds: int
    web_enabled: bool
    web_tried: bool

def build_graph(tools):
    llm = ChatOpenRouter(model=MODEL, temperature=0.2, max_tokens=1024, max_retries=2)
    llm_with_tools = llm.bind_tools(tools)

    llm_web = ChatOpenRouter(
        model=MODEL_WEB, temperature=0.2, max_tokens=1024,
        timeout=60000,
        model_kwargs={"retries": None},
    )

    async def assistant(state: AgentState):
        rounds = state.get("tool_rounds", 0)
        model = llm if rounds >= MAX_TOOL_ROUNDS else llm_with_tools

        trimmed = trim_messages(
            state["messages"],
            max_tokens=MAX_MESSAGES,
            token_counter=len,
            strategy="last",
            start_on="human",
            end_on=("human", "ai", "tool"),
            include_system=False,
            allow_partial=False,
        )
        response = await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT)] + trimmed)
        return {"messages": [response], "tool_rounds": rounds + 1}

    # Returns partial_response, missing term
    def find_no_local(text: str) -> tuple[str, str | None]:
        """Ritorna (risposta ripulita, termine mancante | None)."""
        lines = text.rstrip().splitlines()
        if lines and lines[-1].lstrip().startswith(NO_LOCAL):
            term = lines[-1].split(":", 1)[-1].strip()
            return "\n".join(lines[:-1]).rstrip(), term
        return text, None

    async def web_fallback(state: AgentState):
        question = next((str(m.content) for m in reversed(state["messages"])
                        if isinstance(m, HumanMessage)), "")
        partial, missing = find_no_local(str(state["messages"][-1].content))

        ctx = f"Question: {question}\nMissing from the local rules text: {missing}"
        if partial:
            ctx += f"\n\nAlready answered from the local rules text:\n{partial}"

        resp = await llm_web.ainvoke([SystemMessage(content=WEB_PROMPT),
                                    HumanMessage(content=ctx)])
        return {"messages": [AIMessage(content=resp.content)], "web_tried": True}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        _, missing = find_no_local(str(last.content))
        if missing and state.get("web_enabled") and not state.get("web_tried"):
            return "web"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("web", web_fallback)
    builder.set_entry_point("assistant")
    builder.add_conditional_edges("assistant", route,
                                  {"tools": "tools", "web": "web", END: END})
    builder.add_edge("tools", "assistant")
    builder.add_edge("web", END)

    return builder.compile(checkpointer=InMemorySaver())


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MultiServerMCPClient({
        "dnd_rules": {"url": MCP_URL, "transport": "streamable_http"}
    })
    tools = await client.get_tools()
    app.state.graph = build_graph(tools)
    app.state.tool_names = [t.name for t in tools]
    print(f"[startup] MCP tools: {app.state.tool_names}")
    yield


app = FastAPI(title="D&D RAG agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    web: bool = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    trace: list[dict]


def _trace_of_last_turn(messages: list[AnyMessage]) -> list[dict]:
    """Tool calls and results produced after the most recent HumanMessage."""
    idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
              default=-1)
    trace = []
    for m in messages[idx + 1:]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                trace.append({"kind": "call", "name": tc["name"], "args": tc["args"]})
        elif isinstance(m, ToolMessage):
            body = str(m.content)
            trace.append({
                "kind": "result",
                "name": m.name,
                "chars": len(body),
                "preview": body[:400] + ("..." if len(body) > 400 else ""),
            })
        for a in (m.additional_kwargs or {}).get("annotations", []):
            if a.get("type") == "url_citation":
                trace.append({"kind": "web", "name": a["url_citation"].get("title", ""),
                            "url": a["url_citation"]["url"]})
    return trace


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
        "tags": ["dnd", "langgraph-rag"],
        "metadata": {"model": MODEL, "embedding": "bge-small-en-v1.5"},
    }
    # tool_rounds azzerato a ogni turno; add_messages appende, il default
    # reducer di tool_rounds sovrascrive
    payload = {"messages": [HumanMessage(content=req.query)], "tool_rounds": 0, "web_enabled": req.web, "web_tried": False}

    try:
        result = await app.state.graph.ainvoke(payload, config=config)
    except GraphRecursionError:
        return ChatResponse(
            answer="Tool-call budget esaurito senza una risposta finale.",
            session_id=session_id,
            trace=[],
        )

    last = result["messages"][-1]
    answer = last.content or "(risposta vuota dal modello)"
    return ChatResponse(
        answer=answer if isinstance(answer, str) else str(answer),
        session_id=session_id,
        trace=_trace_of_last_turn(result["messages"]),
    )


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL, "tools": app.state.tool_names}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")