# mcp_server.py
import os

from fastmcp import FastMCP

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENT_K = int(os.getenv("DOCUMENT_K", 4))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.5))

mcp = FastMCP("dnd-rules")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="DungeonsandDragons",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",
)
fs = LocalFileStore("./parent_store")
docstore = create_kv_docstore(fs)

parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(),
    parent_splitter=RecursiveCharacterTextSplitter(),
    search_kwargs={"k": DOCUMENT_K},
)

import re
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for",
        "with", "what", "how", "many", "much", "does", "do", "it", "its",
        "hit", "points", "point", "damage", "rules"}

def _tokenize(text: str) -> list[str]:
    """lowercase + remove stopword. Help BM25 matching names"""
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


def _build_retriever():

    if BM25_WEIGHT <= 0:
        print("[startup] chroma", flush=True)
        return parent_retriever

    parents = [d for d in docstore.mget(list(docstore.yield_keys())) if d]
    if not parents:
        print("[startup] docstore vuoto: solo ricerca densa. "
              "Hai eseguito il profilo populate?", flush=True)
        return parent_retriever

    bm25 = BM25Retriever.from_documents(parents, preprocess_func=_tokenize)
    bm25.k = DOCUMENT_K

    print(f"[startup] hybrid: {len(parents)} parent, "
          f"BM25 {BM25_WEIGHT:.2f} / dense {1 - BM25_WEIGHT:.2f}, "
          f"k={DOCUMENT_K}", flush=True)

    return EnsembleRetriever(
        retrievers=[bm25, parent_retriever],
        weights=[BM25_WEIGHT, 1 - BM25_WEIGHT],
    )


retriever = _build_retriever()


@mcp.tool
def retrieve_rules(query: str) -> str:
    """
    Retrieve D&D rules.

    Use this tool when you need official rules text, class features,
    spells, mechanics, or other D&D rule references.
    """

    docs = retriever.invoke(query)

    # :DOCUMENT_K, to avoid receiving 2k documents after the merge
    docs = docs[:DOCUMENT_K]

    if not docs:
        return "No matching rules found."

    return "\n\n".join(
        f"SOURCE: {d.metadata.get('source')}\nCONTENT:\n{d.page_content}"
        for d in docs
    )

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
    )