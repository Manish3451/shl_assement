# backend/services/embedder.py
from pathlib import Path
import json
import os
import re
import logging
from dotenv import load_dotenv

# Import these inside functions to avoid heavy imports at module-import time
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
INPUT_FILE = DATA_DIR / "shl_all_assessments.json"
OUTPUT_DIR = DATA_DIR / "vector_store" / "faiss_langchain_store"

# Load .env only for local dev; on deployment prefer actual env vars/secrets
load_dotenv(REPO_ROOT / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_api_key(provided_key: str | None = None):
    key = provided_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Provide as argument or env var.")
    return key


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        # import inside constructor to avoid heavy top-level import
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def normalize(self, record):
        return {k: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v for k, v in record.items()}

    def to_documents(self, data):
        from langchain_core.documents import Document

        docs = []
        for rec in data:
            rec = self.normalize(rec)
            name = rec.get("name", "Unknown Assessment")
            content = (
                f"Assessment Name: {name}\n"
                f"Category: {rec.get('category','')}\n"
                f"Type: {rec.get('test_type','')}\n"
                f"Duration: {rec.get('duration','Not specified')}\n"
                f"Format: {rec.get('format','')}\n"
                f"Remote Testing: {rec.get('remote_testing','No')}\n"
                f"Adaptive: {rec.get('adaptive_irt','No')}\n"
                f"URL: {rec.get('url','')}\n"
                f"Description: {rec.get('description','')}\n"
                f"Suitable for: {rec.get('suitable_for','')}\n"
            )
            docs.append(Document(page_content=content, metadata={"assessment_name": name, "url": rec.get("url", "")}))
        return docs

    def chunk_documents(self, docs):
        final = []
        for d in docs:
            name = d.metadata.get("assessment_name", "Unknown Assessment")
            subs = self.splitter.split_documents([d])
            for s in subs:
                s.page_content = f"Assessment Name: {name}\n{s.page_content}"
                final.append(s)
        return final


def build_langchain_store(openai_api_key: str | None = None, chunk_size=500, chunk_overlap=100):
    """
    Builds a FAISS store from the INPUT_FILE. Returns a dict with info for the caller.
    """
    key = ensure_api_key(openai_api_key)

    # Lazy imports that may be heavy / rely on native libs
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input JSON not found at {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = processor.to_documents(data)
    chunks = processor.chunk_documents(docs)

    logger.info("Creating embeddings (this may take time for many chunks)...")

    # Some versions accept an explicit key parameter; pass it explicitly to be safe.
    # If your OpenAIEmbeddings ctor expects a different kwarg name, adjust accordingly.
    embeddings = OpenAIEmbeddings(openai_api_key=key)

    vs = FAISS.from_documents(chunks, embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(OUTPUT_DIR))
    logger.info("Saved FAISS store to %s", OUTPUT_DIR)

    return {"output_dir": str(OUTPUT_DIR), "num_chunks": len(chunks)}


# Export a main symbol so `from backend.services.embedder import main` works
def main(*, openai_api_key: str | None = None, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    wrapper for CLI or other modules. Returns the same dict that build_langchain_store returns.
    """
    return build_langchain_store(openai_api_key=openai_api_key, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    # When run directly allow optional environment override
    print(main())
