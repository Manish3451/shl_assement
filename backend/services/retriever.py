from pathlib import Path
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = REPO_ROOT / "data" / "vector_store" / "faiss_langchain_store"

load_dotenv(REPO_ROOT / ".env")

DEFAULT_ALPHA = 0.7
DEFAULT_K = 10  # Number of results to return
DEFAULT_MODE = "hybrid"

def _name_from_doc(doc):
    """Extract assessment name from document."""
    meta_name = None
    if isinstance(doc.metadata, dict):
        meta_name = doc.metadata.get("assessment_name") or doc.metadata.get("name")
    if meta_name:
        return meta_name
    text = doc.page_content or ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("assessment name:"):
            return stripped.split(":", 1)[1].strip()
        if stripped and len(stripped) < 120:
            return stripped
    return "Unknown"

def _unique_by_name(docs):
    """Remove duplicate documents based on assessment name."""
    seen = set()
    unique = []
    for d in docs:
        name = _name_from_doc(d)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, d))
    return unique

def load_retrievers(alpha=DEFAULT_ALPHA, k=DEFAULT_K):
    """Load and return hybrid retriever combining dense and sparse search."""
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(str(STORE_DIR), embeddings, allow_dangerous_deserialization=True)
    
    # Dense retriever (vector similarity)
    dense = vs.as_retriever(search_kwargs={"k": k})
    
    # Sparse retriever (BM25 keyword search)
    texts = [doc.page_content for doc in vs.docstore._dict.values()]
    sparse = BM25Retriever.from_texts(texts)
    sparse.k = k
    
    # Hybrid retriever
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[alpha, 1 - alpha])
    
    return hybrid, vs

def retrieve_documents(query_text, alpha=DEFAULT_ALPHA, k=DEFAULT_K, mode="hybrid"):
    """
    Retrieve documents without re-ranking.
    
    Args:
        query_text: Search query
        alpha: Weight for dense retriever (0-1), sparse gets (1-alpha)
        k: Number of documents to return
        mode: "dense", "sparse", or "hybrid"
    
    Returns:
        List of dicts with name, text, and score
    """
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(str(STORE_DIR), embeddings, allow_dangerous_deserialization=True)

    if mode == "dense":
        dense = vs.as_retriever(search_kwargs={"k": k})
        docs = dense.invoke(query_text)
    elif mode == "sparse":
        texts = [doc.page_content for doc in vs.docstore._dict.values()]
        sparse = BM25Retriever.from_texts(texts)
        sparse.k = k
        docs = sparse.get_relevant_documents(query_text)
    elif mode == "hybrid":
        hybrid, _ = load_retrievers(alpha=alpha, k=k)
        docs = hybrid.invoke(query_text)
    else:
        raise ValueError("mode must be dense, sparse, or hybrid")

    # Remove duplicates
    unique = _unique_by_name(docs)
    
    # Convert to list of dicts
    results = []
    for name, doc in unique[:k]:  # Limit to k results
        results.append({
            "name": name,
            "text": doc.page_content,
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        })
    
    return results

def print_results(results):
    """Print retrieval results in a readable format."""
    for i, item in enumerate(results, start=1):
        name = item.get("name", "Unknown")
        text = item.get("text", "")[:400].replace("\n", " ")
        print(f"[{i}] {name}")
        print(text + "...\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--query", type=str, default="account manager assessment")
    args = parser.parse_args()

    print(f"Mode={args.mode} alpha={args.alpha} k={args.k}")
    print(f"Query: {args.query}\n")

    results = retrieve_documents(args.query, alpha=args.alpha, k=args.k, mode=args.mode)
    print_results(results)