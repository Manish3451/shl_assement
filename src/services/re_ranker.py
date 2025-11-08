# from pathlib import Path
# import json
# import os
# from typing import List, Tuple, Dict, Any

# import numpy as np
# from dotenv import load_dotenv

# # optional imports
# try:
#     from sentence_transformers import CrossEncoder
# except Exception:
#     CrossEncoder = None

# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# REPO_ROOT = Path(__file__).resolve().parents[2]
# load_dotenv(REPO_ROOT / ".env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DEFAULT_OPENAI_MODEL = os.getenv("RERANK_MODEL", "gpt-3.5-turbo")
# DEFAULT_CROSSENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# def _candidate_texts(docs: List[Any], max_chars: int = 1200) -> List[str]:
#     out = []
#     for d in docs:
#         txt = (d.page_content or "")
#         txt = " ".join(txt.split())
#         out.append(txt[:max_chars])
#     return out


# class CrossEncoderReranker:
#     def __init__(self, model_name: str = DEFAULT_CROSSENCODER_MODEL):
#         if CrossEncoder is None:
#             raise RuntimeError("sentence-transformers not installed")
#         self.model = CrossEncoder(model_name)

#     def rerank(self, query: str, docs: List[Any], top_n: int = 20) -> List[Tuple[float, Any]]:
#         texts = _candidate_texts(docs)
#         pairs = [[query, t] for t in texts]
#         scores = self.model.predict(pairs, convert_to_numpy=True)
#         idxs = np.argsort(scores)[::-1]
#         return [(float(scores[i]), docs[i]) for i in idxs[:top_n]]


# class OpenAIReranker:
#     def __init__(self, model: str = DEFAULT_OPENAI_MODEL):
#         if OpenAI is None:
#             raise RuntimeError("openai package not installed")
#         if not OPENAI_API_KEY:
#             raise RuntimeError("OPENAI_API_KEY not set in environment")
#         self.model = model
#         self.client = OpenAI(api_key=OPENAI_API_KEY)

#     def _build_prompt(self, query: str, texts: List[str]) -> str:
#         blocks = []
#         for i, t in enumerate(texts, start=1):
#             blocks.append(f"{i}. {t}")
#         candidates = "\n\n".join(blocks)
#         prompt = (
#             "You are given a user query and a list of candidate documents.\n"
#             "For each candidate, return a JSON array of objects with fields: index (1-based) and score (0-100).\n"
#             "Return only the JSON array and nothing else.\n\n"
#             f"Query: {query}\n\nCandidates:\n{candidates}\n\nJSON:"
#         )
#         return prompt

#     def rerank(self, query: str, docs: List[Any], top_n: int = 20) -> List[Tuple[float, Any]]:
#         texts = _candidate_texts(docs)[:top_n]
#         prompt = self._build_prompt(query, texts)
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             max_tokens=512
#         )
#         text = resp.choices[0].message.content.strip()
#         try:
#             parsed = json.loads(text)
#             index_to_score = {int(item["index"]) - 1: float(item["score"]) for item in parsed if "index" in item}
#             scored = []
#             for idx in range(len(texts)):
#                 s = float(index_to_score.get(idx, 0.0))
#                 scored.append((s, docs[idx]))
#             scored = sorted(scored, key=lambda x: x[0], reverse=True)
#             return scored[:top_n]
#         except Exception:
#             tokens = [t for t in text.replace(",", " ").split() if t.replace(".", "", 1).isdigit()]
#             scores = [float(t) for t in tokens][:len(texts)]
#             scored = [(scores[i] if i < len(scores) else 0.0, docs[i]) for i in range(len(texts))]
#             scored = sorted(scored, key=lambda x: x[0], reverse=True)
#             return scored[:top_n]


# def rerank_pipeline(query: str,
#                    candidates: List[Any],
#                    reranker_type: str = "openai",
#                    re_rank_n: int = 8,
#                    display_k: int = 10,
#                    crossencoder_model: str = DEFAULT_CROSSENCODER_MODEL,
#                    openai_model: str = DEFAULT_OPENAI_MODEL) -> List[Dict[str, Any]]:
#     # dedupe by assessment_name
#     seen = set()
#     uniq = []
#     for d in candidates:
#         name = None
#         if isinstance(d.metadata, dict):
#             name = d.metadata.get("assessment_name") or d.metadata.get("name")
#         if not name:
#             text = (d.page_content or "").splitlines()
#             name = text[0].strip() if text else "Unknown"
#         key = name.strip().lower()
#         if key in seen:
#             continue
#         seen.add(key)
#         uniq.append(d)
#     candidates = uniq
#     if not candidates:
#         return []
#     top_candidates = candidates[:max(len(candidates), re_rank_n)]
#     if reranker_type == "crossencoder":
#         reranker = CrossEncoderReranker(model_name=crossencoder_model)
#         scored = reranker.rerank(query, top_candidates, top_n=re_rank_n)
#     elif reranker_type == "openai":
#         reranker = OpenAIReranker(model=openai_model)
#         scored = reranker.rerank(query, top_candidates, top_n=re_rank_n)
#     else:
#         raise ValueError("reranker_type must be 'crossencoder' or 'openai'")

#     final = []
#     for score, doc in scored[:display_k]:
#         final.append({
#             "score": float(score),
#             "name": doc.metadata.get("assessment_name") if isinstance(doc.metadata, dict) else None,
#             "text": (doc.page_content or "")[:2000],
#             "metadata": doc.metadata
#         })
#     return final
