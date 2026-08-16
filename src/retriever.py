# src/retriever.py
import os
import json
import hashlib
from pathlib import Path
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Path setup — anchored to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "data" / "chroma_store"
INGESTED_LOG = PERSIST_DIR / "ingested_files.json"

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY"),
)


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def _load_ingested_log() -> dict:
    if INGESTED_LOG.exists():
        with open(INGESTED_LOG, "r") as f:
            return json.load(f)
    return {}


def _save_ingested_log(log: dict):
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(INGESTED_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _assign_stable_chunk_ids(chunks: List[Document]) -> List[Document]:
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:8]
        chunk.metadata["chunk_id"] = content_hash
    return chunks


def _load_single_file(path: Path) -> List[Document]:
    if path.suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif path.suffix == ".txt":
        return TextLoader(str(path)).load()
    return []


def ingest(data_dir: Path = DATA_DIR, persist_dir: Path = PERSIST_DIR):
    ingested_log = _load_ingested_log()
    new_or_changed_docs = []
    files_processed = []

    for path in data_dir.glob("*"):
        if path.suffix not in (".pdf", ".txt"):
            continue

        current_hash = _file_hash(path)
        filename = path.name

        if ingested_log.get(filename) == current_hash:
            print(f"Skipping '{filename}' — already ingested, unchanged.")
            continue

        print(f"Ingesting '{filename}' (new or changed)...")
        docs = _load_single_file(path)
        new_or_changed_docs.extend(docs)
        files_processed.append((filename, current_hash))

    if not new_or_changed_docs:
        print("Nothing new to ingest. Vector store is up to date.")
        return get_vector_store(persist_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(new_or_changed_docs)
    chunks = _assign_stable_chunk_ids(chunks)

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )
    vector_store.add_documents(chunks)

    for filename, file_hash in files_processed:
        ingested_log[filename] = file_hash
    _save_ingested_log(ingested_log)

    print(f"Ingested {len(chunks)} new chunks from {len(files_processed)} file(s).")
    return vector_store


def get_vector_store(persist_dir: Path = PERSIST_DIR) -> Chroma:
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )


# Alias load_store to get_vector_store for compatibility
load_store = get_vector_store


def retrieve(question: str, k: int = 4) -> List[Document]:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(question)


def get_all_chunks() -> List[Document]:
    vector_store = get_vector_store()
    data = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data["documents"], data["metadatas"])
    ]


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    ingest()

    # retrieved_docs = retrieve("What are the key contacts?", k=3)
    # for i, doc in enumerate(retrieved_docs, start=1):
    #     print(f"\n--- Retrieved Document {i} ---")
    #     print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
    #     print(f"Content: {doc.page_content[:200]}...")  # Print first 200 chars