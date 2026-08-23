# mcp_server.py
from fastmcp import FastMCP

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

DOCUMENT_K = int(os.getenv("DOCUMENT_K", 4))

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


@mcp.tool
def retrieve_rules(
    query: str,
) -> str:
    """
    Retrieve D&D rules.

    Use this tool when you need official rules text, class features,
    spells, mechanics, or other D&D rule references.
    """

    print("[Retriever]", query)

    docs = parent_retriever.invoke(query)

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