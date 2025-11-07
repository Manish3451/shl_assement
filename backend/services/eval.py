# backend/services/eval_rag.py
import time
import json
import csv
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.retriever import retrieve_documents
from backend.services.chat import call_chat

# Directory to store eval results
OUT_DIR = REPO_ROOT / "data" / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# Test pairs: (query, expected_assessment_name)
# ------------------------------------------------------
TEST_PAIRS: List[Tuple[str, str]] = [
    # C++ and Programming
    ("c++ programming test", "C++ Programming (New)"),
    ("c++ skills assessment", "C++ Programming (New)"),
    ("c++ coding test new", "C++ Programming (New)"),
    ("assess c++ developers", "C++ Programming (New)"),

    # Cardiology and Diabetes
    ("cardiology and diabetes management test", "Cardiology and Diabetes Management (New)"),
    ("diabetes care assessment", "Cardiology and Diabetes Management (New)"),
    ("cardiology knowledge test", "Cardiology and Diabetes Management (New)"),
    ("medical specialty diabetes exam", "Cardiology and Diabetes Management (New)"),

    # Engineering Fields
    ("ceramic engineering assessment", "Ceramic Engineering (New)"),
    ("chemical engineering test", "Chemical Engineering (New)"),
    ("civil engineering knowledge test", "Civil Engineering (New)"),
    ("electrical engineering assessment", "Electrical Engineering (New)"),
    ("electronics and telecommunications test", "Electronics & Telecommunications Engineering (New)"),
    ("electronics embedded systems exam", "Electronics and Embedded Systems Engineering (New)"),
    ("semiconductor engineering test", "Electronics and Semiconductor Engineering (New)"),

    # Cloud & DevOps
    ("cloud computing skills test", "Cloud Computing (New)"),
    ("docker assessment", "Docker (New)"),
    ("docker container skills", "Docker (New)"),
    ("devops docker test", "Docker (New)"),

    # Programming & Web Tech
    ("cobol programming test", "COBOL Programming (New)"),
    ("core java advanced test", "Core Java (Advanced Level) (New)"),
    ("core java entry level exam", "Core Java (Entry Level) (New)"),
    ("java programming assessment", "Core Java (Entry Level) (New)"),
    ("css3 skills test", "CSS3 (New)"),
    ("css3 web development", "CSS3 (New)"),
    ("dojo framework test", "Dojo (New)"),
    ("drupal cms assessment", "Drupal (New)"),

    # IT & Support
    ("desktop support skills test", "Desktop Support (New)"),
    ("it desktop support assessment", "Desktop Support (New)"),

    # Data & Science
    ("data science assessment", "Data Science (New)"),
    ("data scientist skills test", "Data Science (New)"),
    ("data warehousing concepts test", "Data Warehousing Concepts"),
    ("data entry assessment", "Data Entry (New)"),
    ("alphanumeric data entry test", "Data Entry Alphanumeric Split Screen - US"),
    ("numeric data entry test", "Data Entry Numeric Split Screen - US"),
    ("ten key data entry test", "Data Entry Ten Key Split Screen"),
    ("econometrics test", "Econometrics (New)"),
    ("economics knowledge assessment", "Economics (New)"),

    # Medical Specialties
    ("dermatology assessment", "Dermatology (New)"),

    # Cisco & Monitoring
    ("cisco appdynamics test", "Cisco AppDynamics (New)"),

    # Computer Science & General
    ("computer science fundamentals test", "Computer Science (New)"),

    # Cybersecurity
    ("cyber risk assessment", "Cyber Risk (New)"),
    ("cybersecurity risk knowledge test", "Cyber Risk (New)"),

    # Culinary
    ("culinary skills assessment", "Culinary Skills (New)"),
    ("chef cooking skills test", "Culinary Skills (New)"),

    # Digital Advertising
    ("digital advertising skills test", "Digital Advertising (New)"),
    ("online advertising assessment", "Digital Advertising (New)"),

    # Simulation & Behavioral Assessments
    ("contact center call simulation", "Contact Center Call Simulation (New)"),
    ("call center phone simulation", "Contact Center Call Simulation (New)"),
    ("multichat simulation test", "Conversational Multichat Simulation"),
    ("live chat customer service simulation", "Conversational Multichat Simulation"),
    ("customer service phone simulation", "Customer Service Phone Simulation"),
    ("phone support simulation test", "Customer Service Phone Simulation"),
    ("customer service phone solution", "Customer Service Phone Solution"),
    ("multi-skill customer service test", "Customer Service Phone Solution"),

    # Money & Counting
    ("count out the money test", "Count Out The Money"),
    ("cash handling skills assessment", "Count Out The Money"),

    # Personality & Behavior (P)
    ("dependability and safety instrument", "Dependability and Safety Instrument (DSI)"),
    ("safety behavior assessment", "Dependability and Safety Instrument (DSI)"),
    ("dsi v1.1 report", "DSI v1.1 Interpretation Report"),
    ("dsi interpretation guide", "DSI v1.1 Interpretation Report"),

    # Digital Readiness Reports
    ("digital readiness development report individual", "Digital Readiness Development Report - IC"),
    ("digital readiness for employees", "Digital Readiness Development Report - IC"),
    ("digital readiness manager report", "Digital Readiness Development Report - Manager"),
    ("manager digital skills assessment", "Digital Readiness Development Report - Manager"),

    # Reservation Agent (from original example)
    ("reservation agent assessment details", "Reservation Agent Solution"),
    ("airline reservation skills test", "Reservation Agent Solution"),
    ("travel agent assessment", "Reservation Agent Solution"),
    ("flight booking simulation", "Reservation Agent Solution"),

    # Restaurant Manager (from original example)
    ("restaurant manager solution description", "Restaurant Manager Solution"),
    ("restaurant leadership assessment", "Restaurant Manager Solution"),
    ("food service management test", "Restaurant Manager Solution"),
    ("hospitality manager evaluation", "Restaurant Manager Solution"),

    # Telenurse (from original example)
    ("telenurse solution assessment", "Telenurse Solution"),
    ("nurse telehealth skills test", "Telenurse Solution"),
    ("remote nursing assessment", "Telenurse Solution"),
    ("telemedicine nurse evaluation", "Telenurse Solution"),

    # Teller 7.0 (from original example)
    ("teller 7.0 test", "Teller 7.0"),
    ("bank teller skills assessment", "Teller 7.0"),
    ("cash handling bank test", "Teller 7.0"),
    ("teller performance evaluation", "Teller 7.0"),

    # .NET MVC (from original example)
    (".net mvc new test", ".NET MVC (New)"),
    ("asp.net mvc developer assessment", ".NET MVC (New)"),
    ("dot net mvc skills test", ".NET MVC (New)"),
    ("mvc framework coding test", ".NET MVC (New)"),

    # Additional variations using synonyms and phrasing
    ("technical programming assessment c++", "C++ Programming (New)"),
    ("evaluate civil engineers", "Civil Engineering (New)"),
    ("cloud infrastructure skills test", "Cloud Computing (New)"),
    ("advanced java programming test", "Core Java (Advanced Level) (New)"),
    ("entry-level java coder assessment", "Core Java (Entry Level) (New)"),
    ("web styling with css3 exam", "CSS3 (New)"),
    ("front-end dojo framework test", "Dojo (New)"),
    ("cms drupal skills evaluation", "Drupal (New)"),
    ("data analyst science test", "Data Science (New)"),
    ("statistical modeling econometrics", "Econometrics (New)"),
    ("microelectronics engineering test", "Electronics and Semiconductor Engineering (New)"),
    ("embedded systems developer assessment", "Electronics and Embedded Systems Engineering (New)"),
    ("telecom engineering knowledge test", "Electronics & Telecommunications Engineering (New)"),
    ("network security cyber risk test", "Cyber Risk (New)"),
    ("ad tech digital advertising test", "Digital Advertising (New)"),
    ("cooking chef assessment", "Culinary Skills (New)"),
    ("kitchen skills evaluation", "Culinary Skills (New)"),
    ("customer interaction multichannel simulation", "Conversational Multichat Simulation"),
    ("live support chat assessment", "Conversational Multichat Simulation"),
    ("call center performance test", "Contact Center Call Simulation (New)"),
    ("phone-based customer service test", "Customer Service Phone Simulation"),
    ("integrated customer service solution", "Customer Service Phone Solution"),
    ("behavioral and skill assessment customer service", "Customer Service Phone Solution"),
    ("money counting accuracy test", "Count Out The Money"),
    ("cash reconciliation assessment", "Count Out The Money"),
    ("reliability and safety at work test", "Dependability and Safety Instrument (DSI)"),
    ("workplace safety behavior assessment", "Dependability and Safety Instrument (DSI)"),
    ("interpret dsi results", "DSI v1.1 Interpretation Report"),
    ("dsi scoring report", "DSI v1.1 Interpretation Report"),
    ("digital competency employee report", "Digital Readiness Development Report - IC"),
    ("employee tech readiness assessment", "Digital Readiness Development Report - IC"),
    ("leadership digital transformation report", "Digital Readiness Development Report - Manager"),
    ("managerial digital fluency test", "Digital Readiness Development Report - Manager"),
    ("software developer cobol test", "COBOL Programming (New)"),
    ("legacy systems programming assessment", "COBOL Programming (New)"),
    ("general computer science knowledge test", "Computer Science (New)"),
    ("cs fundamentals exam", "Computer Science (New)"),
    ("appdynamics monitoring tool test", "Cisco AppDynamics (New)"),
    ("application performance management cisco", "Cisco AppDynamics (New)"),
    ("dermatology medical knowledge test", "Dermatology (New)"),
    ("skin care specialist assessment", "Dermatology (New)"),
    ("helpdesk technical support test", "Desktop Support (New)"),
    ("it support technician evaluation", "Desktop Support (New)"),
    ("big data warehousing concepts", "Data Warehousing Concepts"),
    ("etl data warehouse fundamentals", "Data Warehousing Concepts"),
    ("data entry speed and accuracy", "Data Entry (New)"),
    ("split screen alphanumeric entry", "Data Entry Alphanumeric Split Screen - US"),
    ("numeric keypad data input", "Data Entry Numeric Split Screen - US"),
    ("ten-key calculator test", "Data Entry Ten Key Split Screen"),
    ("docker containerization skills", "Docker (New)"),
    ("container deployment assessment", "Docker (New)"),
    ("javascript dojo library test", "Dojo (New)"),
    ("content management drupal test", "Drupal (New)"),
    ("economic theory assessment", "Economics (New)"),
    ("financial economics test", "Economics (New)"),
    ("power systems electrical engineering", "Electrical Engineering (New)"),
    ("power electronics test", "Electrical and Electronics Engineering (New)"),
    ("electronics engineering combined test", "Electrical and Electronics Engineering (New)"),
    ("medical cardiology diabetes test", "Cardiology and Diabetes Management (New)"),
    ("patient care diabetes assessment", "Cardiology and Diabetes Management (New)"),
    ("ceramics materials engineering", "Ceramic Engineering (New)"),
    ("industrial ceramics test", "Ceramic Engineering (New)"),
    ("chemical process engineering test", "Chemical Engineering (New)"),
    ("process design chemical assessment", "Chemical Engineering (New)"),
    ("cloud platform skills evaluation", "Cloud Computing (New)"),
    ("aws azure cloud fundamentals", "Cloud Computing (New)"),
]


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def parse_expected_list(expected_raw: str) -> List[str]:
    if not expected_raw:
        return []
    return [e.strip() for e in re.split(r"[,;]\s*", expected_raw) if e.strip()]

def extract_names_from_llm(answer: str) -> List[str]:
    """Extract assessment names from LLM output (handles JSON or text)."""
    try:
        obj = json.loads(answer)
        if isinstance(obj, dict) and "recommendations" in obj:
            return [r["name"].strip() for r in obj["recommendations"] if "name" in r]
        if isinstance(obj, list):
            return [str(r).strip() if not isinstance(r, dict) else r.get("name", "").strip() for r in obj]
    except Exception:
        pass

    # Regex fallback
    names = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\.\s*(.+?)(?:\s+[-—–:]\s*.*)?$", line)
        if match:
            names.append(match.group(1))
            continue
        quoted = re.findall(r'["“](.+?)["”]', line)
        if quoted:
            names.extend(quoted)

    # Deduplicate
    seen, out = set(), []
    for n in names:
        if not n:
            continue
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def compute_rank(preds: List[str], expected_list: List[str]) -> Optional[int]:
    expected_lc = [e.lower() for e in expected_list]
    for i, p in enumerate(preds, start=1):
        if p and p.lower() in expected_lc:
            return i
    return None

def hit_at_k(preds: List[str], expected_list: List[str], k: int) -> int:
    r = compute_rank(preds, expected_list)
    return 1 if (r and r <= k) else 0

def precision_at_k(preds: List[str], expected_list: List[str], k: int) -> float:
    expected_set = set([e.lower() for e in expected_list])
    preds_k = [p.lower() for p in preds[:k]]
    if not preds_k or not expected_set:
        return 0.0
    hits = sum(1 for p in preds_k if p in expected_set)
    return hits / k

def recall_at_k(preds: List[str], expected_list: List[str], k: int) -> float:
    expected_set = set([e.lower() for e in expected_list])
    preds_k = [p.lower() for p in preds[:k]]
    if not expected_set:
        return 0.0
    hits = sum(1 for e in expected_set if e in preds_k)
    return hits / len(expected_set)

def ndcg_at_k(preds: List[str], expected_list: List[str], k: int) -> float:
    expected_set = set([e.lower() for e in expected_list])
    dcg = 0.0
    for i, p in enumerate(preds[:k], start=1):
        rel = 1.0 if p and p.lower() in expected_set else 0.0
        dcg += (2**rel - 1) / math.log2(i + 1)
    R = min(len(expected_set), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, R + 1)) if R > 0 else 1.0
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rag(
    test_pairs: List[Tuple[str, str]],
    alpha: float = 0.7,
    k: int = 10,
    context_n: int = 5,
    mode: str = "hybrid",
):
    rows = []
    retrieval_times, llm_times = [], []

    for q, expected_raw in test_pairs:
        expected = parse_expected_list(expected_raw)
        print(f"\nQuery: {q}")
        t0 = time.time()
        retrieved = retrieve_documents(q, alpha=alpha, k=k, mode=mode)
        t1 = time.time()
        retrieval_times.append(t1 - t0)

        # Build context
        context_docs = retrieved[:context_n]
        context_for_chat = [
            {"name": d["name"], "text": d["text"], "metadata": d["metadata"]}
            for d in context_docs
        ]

        # Call LLM
        t2 = time.time()
        chat_out = call_chat(q, context_for_chat)
        t3 = time.time()
        llm_times.append(t3 - t2)

        answer = chat_out.get("answer", "")
        preds = extract_names_from_llm(answer)
        if not preds:
            preds = [d["name"] for d in retrieved[:context_n]]

        rank = compute_rank(preds, expected)
        hit1 = hit_at_k(preds, expected, 1)
        hit3 = hit_at_k(preds, expected, 3)
        hit5 = hit_at_k(preds, expected, 5)
        precision_k = precision_at_k(preds, expected, k)
        recall_k = recall_at_k(preds, expected, k)
        ndcg = ndcg_at_k(preds, expected, k)

        print(f"Expected: {expected}")
        print(f"Predicted top: {preds[:5]}")
        print(f"Rank: {rank}")

        rows.append({
            "query": q,
            "expected": "; ".join(expected),
            "preds": json.dumps(preds, ensure_ascii=False),
            "rank": rank if rank else "",
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "precision@k": round(precision_k, 3),
            "recall@k": round(recall_k, 3),
            "ndcg": round(ndcg, 3),
            "retrieval_time": round(t1 - t0, 3),
            "llm_time": round(t3 - t2, 3),
        })

    total = len(rows)
    avg = lambda key: sum(r[key] for r in rows) / total if total else 0.0

    mrr = sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / total if total else 0.0
    avg_ret = sum(retrieval_times) / total if total else 0.0
    avg_llm = sum(llm_times) / total if total else 0.0

    summary = {
        "top1": avg("hit@1"),
        "top3": avg("hit@3"),
        "top5": avg("hit@5"),
        "mrr": round(mrr, 3),
        "precision@k": avg("precision@k"),
        "recall@k": avg("recall@k"),
        "ndcg": avg("ndcg"),
        "avg_retrieval_time": avg_ret,
        "avg_llm_time": avg_llm,
    }

    # Save CSV
    ts = int(time.time())
    fname = OUT_DIR / f"eval_rag_{ts}.csv"
    with open(fname, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Results saved to: {fname}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--context_n", type=int, default=5)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["dense", "sparse", "hybrid"])
    args = parser.parse_args()

    evaluate_rag(TEST_PAIRS, alpha=args.alpha, k=args.k, context_n=args.context_n, mode=args.mode)
