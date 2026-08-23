from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from pathlib import Path
import os


class BatchedChroma(Chroma):
    """Chroma che spezza automaticamente le upsert oltre il limite del client.

    Serve perche' ParentDocumentRetriever espande ogni parent in N child chunk
    prima di scrivere: il batch a monte non controlla quanti record arrivano
    davvero a Chroma.
    """

    _FALLBACK_MAX_BATCH = 5000  # sotto il limite tipico di 5461

    @property
    def max_batch_size(self) -> int:
        cached = getattr(self, "_max_batch_size", None)
        if cached is not None:
            return cached
        try:
            size = self._client.get_max_batch_size()
        except Exception:
            size = getattr(self._client, "max_batch_size", self._FALLBACK_MAX_BATCH)
        # margine di sicurezza
        self._max_batch_size = max(1, int(size) - 100)
        return self._max_batch_size

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else None
        ids = list(ids) if ids is not None else None

        step = self.max_batch_size
        out = []
        for i in range(0, len(texts), step):
            sl = slice(i, i + step)
            out.extend(
                super().add_texts(
                    texts[sl],
                    metadatas[sl] if metadatas is not None else None,
                    ids[sl] if ids is not None else None,
                    **kwargs,
                )
            )
        return out


md_loader = DirectoryLoader(
    "./markdown",
    glob="**/*.md",
    loader_cls=TextLoader,
)

docs = md_loader.load()

ROOT = Path("./markdown")
for doc in docs:
    path = Path(doc.metadata["source"])
    relative = path.relative_to(ROOT)
    doc.metadata["source"] = str(relative)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=[
        "\n# ",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        " ",
    ],
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        " ",
    ],
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = BatchedChroma(
    collection_name="DungeonsandDragons",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",
)

# Parent document storage
fs = LocalFileStore("./parent_store")
store = create_kv_docstore(fs)

# Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)

# Index documents
batch_size = int(os.getenv("BATCH_SIZE", 200))

for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    print(f"Adding documents {i} -> {i + len(batch)}", flush=True)
    retriever.add_documents(batch)

print("Done.", flush=True)