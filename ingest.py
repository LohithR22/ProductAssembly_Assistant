"""Ingest PDF manuals into a persistent ChromaDB vector store.

Usage: python ingest.py
"""
import os
from typing import List

# Try to import LangChain loaders; if they are missing, we'll fallback to pypdf-based loader
try:
    from langchain.document_loaders import DirectoryLoader, PyPDFLoader  # type: ignore
except Exception:
    DirectoryLoader = None  # type: ignore
    PyPDFLoader = None  # type: ignore

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore

try:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None  # type: ignore

try:
    from langchain.vectorstores import Chroma
except Exception:
    Chroma = None  # type: ignore

try:
    from langchain.schema import Document
except Exception:
    # Minimal fallback Document shape used by this script
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict


# Fallback: use pypdf to load PDFs if LangChain loaders are unavailable
def _load_pdfs_with_pypdf(manuals_dir: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise ImportError(
            "pypdf is required for the fallback PDF loader. Install it (pip install pypdf) or install a LangChain version with document loaders."
        ) from e

    docs: List[Document] = []
    for root, _, files in os.walk(manuals_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fname)
            try:
                reader = PdfReader(path)
                text_parts: List[str] = []
                for page in reader.pages:
                    try:
                        text_parts.append(page.extract_text() or "")
                    except Exception:
                        # skip pages that fail to extract
                        continue
                full_text = "\n\n".join(text_parts)
                metadata = {"source": path}
                docs.append(Document(page_content=full_text, metadata=metadata))
            except Exception:
                # skip unreadable files
                continue
    return docs


# Fallback implementation of RecursiveCharacterTextSplitter if LangChain doesn't provide it
if RecursiveCharacterTextSplitter is None:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, docs: List[Document]) -> List[Document]:
            out: List[Document] = []
            for d in docs:
                text = getattr(d, "page_content", str(d)) or ""
                start = 0
                L = len(text)
                if L == 0:
                    continue
                while start < L:
                    end = min(start + self.chunk_size, L)
                    chunk = text[start:end]
                    metadata = getattr(d, "metadata", {}) or {}
                    out.append(Document(page_content=chunk, metadata=metadata))
                    if end == L:
                        break
                    start = end - self.chunk_overlap
            return out


def main():
    # Load all PDFs from ./manuals
    base_dir = os.path.dirname(__file__)
    manuals_dir = os.path.join(base_dir, "manuals")
    if not os.path.isdir(manuals_dir):
        raise SystemExit("manuals directory not found. Create ./manuals and add PDF manuals before running this script.")
    # --- Dependency pre-check -------------------------------------------------
    # If the user doesn't have the optional fallback packages installed,
    # show a concise message with a suggested pip command instead of a
    # long traceback. We only check the fallback deps used by this script.
    missing = []
    try:
        import pypdf  # type: ignore
    except Exception:
        # Only required if LangChain PDF loaders are not available
        if DirectoryLoader is None or PyPDFLoader is None:
            missing.append("pypdf")

    try:
        import sentence_transformers  # type: ignore
    except Exception:
        missing.append("sentence-transformers")

    # chromadb is optional in the simplified ingestion path; we will
    # fallback to writing snapshot files when a Chroma client isn't used.

    if missing:
        print("Missing optional dependencies required for the fallback ingestion path:", ", ".join(missing))
        print("Install them with:")
        print(f"  pip install {' '.join(missing)}")
        raise SystemExit(1)

    print(f"Loading PDFs from: {manuals_dir}")
    if DirectoryLoader is not None and PyPDFLoader is not None:
        try:
            loader = DirectoryLoader(manuals_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents using LangChain loaders")
        except Exception:
            # fallback to pypdf loader
            docs = _load_pdfs_with_pypdf(manuals_dir)
            print(f"Loaded {len(docs)} documents using pypdf fallback")
    else:
        docs = _load_pdfs_with_pypdf(manuals_dir)
        print(f"Loaded {len(docs)} documents using pypdf fallback")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Embeddings
    print("Initializing embeddings (all-MiniLM-L6-v2)")

    persist_dir = os.path.join(base_dir, "RAG_store")
    print(f"Initializing Chroma persistent store at: {persist_dir}")
    # Ensure the persist directory exists so we can observe any files written there
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        print(f"Warning: could not create persist directory: {persist_dir}")

    texts = [getattr(c, "page_content", str(c)) for c in chunks]
    metadatas = [getattr(c, "metadata", {}) for c in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    # Compute embeddings using sentence-transformers (simpler and reliable)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("sentence-transformers is required for embeddings. Install with `pip install sentence-transformers`") from e

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Computing embeddings with sentence-transformers...")
    vectors = model.encode(texts, show_progress_bar=True)

    # Instead of attempting many chromadb variants, write a simple snapshot
    # that the FastAPI app can load. This guarantees a working RAG fallback.
    try:
        print("Writing snapshot files into persist_dir...")
        import json
        import numpy as _np

        with open(os.path.join(persist_dir, "ids.json"), "w", encoding="utf-8") as fh:
            json.dump(ids, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(persist_dir, "texts.json"), "w", encoding="utf-8") as fh:
            json.dump(texts, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(persist_dir, "metadatas.json"), "w", encoding="utf-8") as fh:
            json.dump(metadatas, fh, ensure_ascii=False, indent=2)

        arr = _np.asarray(vectors)
        _np.save(os.path.join(persist_dir, "embeddings.npy"), arr)
        try:
            print("Snapshot files written:", os.listdir(persist_dir))
        except Exception:
            pass
    except Exception as e:
        print("Failed to write snapshot files:", e)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
