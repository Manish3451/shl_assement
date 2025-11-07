from pathlib import Path
import json, os, re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
INPUT_FILE = DATA_DIR / "shl_all_assessments.json"
OUTPUT_DIR = DATA_DIR / "vector_store" / "faiss_langchain_store"

load_dotenv(REPO_ROOT / ".env")

def ensure_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in .env")

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def normalize(self, record):
        return {k: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v for k, v in record.items()}

    def to_documents(self, data):
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
            docs.append(Document(page_content=content, metadata={"assessment_name": name, "url": rec.get("url","")}))
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

def build_langchain_store():
    ensure_api_key()
    data = json.loads(open(INPUT_FILE, encoding="utf-8").read())
    if isinstance(data, dict):
        data = [data]
    processor = DocumentProcessor()
    docs = processor.to_documents(data)
    chunks = processor.chunk_documents(docs)
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(OUTPUT_DIR))

if __name__ == "__main__":
    build_langchain_store()
