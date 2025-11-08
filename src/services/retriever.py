from pathlib import Path
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from src.services.chat import get_test_type_probs_via_llm
from langchain_community.vectorstores import FAISS
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = REPO_ROOT / "data" / "vector_store" / "faiss_langchain_store"

load_dotenv(REPO_ROOT / ".env")

DEFAULT_ALPHA = 0.7
DEFAULT_K = 10 
DEFAULT_MODE = "hybrid"

TYPES = ["A", "B", "C", "D", "E", "K", "P", "S"]

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

def _compute_type_score(test_type_probs, candidate_types):
    """Compute average probability alignment between query types and test types."""
    if not candidate_types:
        return 0.0
    values = []
    for t in candidate_types:
        if t in test_type_probs:
            values.append(test_type_probs[t])
    if not values:
        return 0.0
    return float(np.mean(values))

def _normalize_scores(candidates, key):
    """Normalize a score field between 0 and 1."""
    scores = [c.get(key, 0.0) for c in candidates]
    if not scores:
        return
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        for c in candidates:
            c[key] = 1.0
    else:
        for c in candidates:
            c[key] = (c.get(key, 0.0) - min_s) / (max_s - min_s)

def re_rank_candidates(candidates, query_text,test_type_probs, w_sim=0.5, w_desc=0.3, w_type=0.2):
    """
    Re-rank using: similarity + description match + test-type alignment
    w_sim  = weight for semantic similarity (from retrieval)
    w_desc = weight for description relevance (query-description similarity)
    w_type = weight for test-type alignment
    """
    if not candidates:
        return []

    _normalize_scores(candidates, key="score")
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query_text)  # Need to pass query_text
    
    for c in candidates:
        # Get description from metadata or text
        desc = c.get("metadata", {}).get("description", "") or c.get("text", "")[:500]
        if desc.strip():
            desc_embedding = embeddings.embed_query(desc)
            # Cosine similarity
            similarity = np.dot(query_embedding, desc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(desc_embedding)
            )
            c["desc_score"] = max(0.0, similarity)  # Ensure non-negative
        else:
            c["desc_score"] = 0.0
    
    # Normalize description scores
    _normalize_scores(candidates, key="desc_score")

    # Compute test-type scores
    for c in candidates:
        metadata = c.get("metadata", {})
        raw_types = metadata.get("test_type") or metadata.get("test_types") or ""
        if isinstance(raw_types, str):
            candidate_types = [t.strip() for t in raw_types.split() if t.strip() in TYPES]
        elif isinstance(raw_types, list):
            candidate_types = [t for t in raw_types if t in TYPES]
        else:
            candidate_types = []
        
        s_type = _compute_type_score(test_type_probs, candidate_types)
        c["s_type"] = s_type
        
        # Final score with 3 components
        c["final_score"] = (
            w_sim * c.get("score", 0.0) + 
            w_desc * c.get("desc_score", 0.0) + 
            w_type * s_type
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates

def load_retrievers(alpha=DEFAULT_ALPHA, k=DEFAULT_K):
    """Load and return hybrid retriever combining dense and sparse search."""
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(str(STORE_DIR), embeddings, allow_dangerous_deserialization=True)
    
    dense = vs.as_retriever(search_kwargs={"k": k})
    texts = [doc.page_content for doc in vs.docstore._dict.values()]
    sparse = BM25Retriever.from_texts(texts)
    sparse.k = k
    
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[alpha, 1 - alpha])
    return hybrid, vs

def retrieve_documents(query_text, alpha=DEFAULT_ALPHA, k=DEFAULT_K, mode="hybrid", test_type_probs=None):
    """
    Retrieve and optionally re-rank documents based on query and test-type alignment.
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

    unique = _unique_by_name(docs)

    results = []
    for name, doc in unique[:k]:
        result = {
            "name": name,
            "text": doc.page_content,
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
            "score": getattr(doc, "score", 0.0)
        }
        results.append(result)

    if test_type_probs:
        results = re_rank_candidates(results,query_text, test_type_probs)

    return results

def print_results(results):
    """Print retrieval results in a readable format."""
    for i, item in enumerate(results, start=1):
        name = item.get("name", "Unknown")
        text = item.get("text", "")[:400].replace("\n", " ")
        print(f"[{i}] {name}")
        print(f"score={item.get('score', 0.0):.2f}, type_score={item.get('s_type', 0.0):.2f}, final={item.get('final_score', 0.0):.2f}")
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

    test_type_probs = get_test_type_probs_via_llm(args.query)
    print("Test type probabilities:", test_type_probs)

    results = retrieve_documents(args.query, alpha=args.alpha, k=args.k, mode=args.mode, test_type_probs=test_type_probs)
    print_results(results)
