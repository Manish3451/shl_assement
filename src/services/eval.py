import json
import requests
from typing import List, Dict, Tuple
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/api/recommend"  
GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"

def load_ground_truth() -> Dict:
    """Load ground truth labels"""
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_recommendations(query: str, k: int = 10) -> List[str]:
    """Get top K recommendations from API"""
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "k": k},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract assessment names from response
        recommendations = data.get("recommended_assessments", [])
        return [rec["name"] for rec in recommendations]
    
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

def calculate_recall_at_k(
    retrieved: List[str], 
    relevant: List[str], 
    k: int
) -> float:
    """
    Calculate Recall@K for a single query
    
    Recall@K = (Number of relevant items in top K) / (Total relevant items)
    """
    if not relevant:
        return 0.0
    
    # Get top K retrieved items
    top_k = retrieved[:k]
    
    # Count how many relevant items are in top K
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    # Calculate recall
    recall = relevant_in_top_k / len(relevant)
    
    return recall

def fuzzy_match(retrieved_name: str, relevant_names: List[str]) -> bool:
    """
    Check if retrieved assessment matches any relevant assessment
    (handles slight name variations)
    """
    retrieved_lower = retrieved_name.lower().strip()
    
    for relevant_name in relevant_names:
        relevant_lower = relevant_name.lower().strip()
        
        # Exact match
        if retrieved_lower == relevant_lower:
            return True
        
        # Partial match (one contains the other)
        if retrieved_lower in relevant_lower or relevant_lower in retrieved_lower:
            return True
    
    return False

def calculate_recall_at_k_fuzzy(
    retrieved: List[str], 
    relevant: List[str], 
    k: int
) -> Tuple[float, List[str]]:
    """
    Calculate Recall@K with fuzzy matching
    Returns (recall_score, list_of_matched_items)
    """
    if not relevant:
        return 0.0, []
    
    top_k = retrieved[:k]
    matched_items = []
    
    for retrieved_item in top_k:
        if fuzzy_match(retrieved_item, relevant):
            matched_items.append(retrieved_item)
    
    recall = len(matched_items) / len(relevant)
    
    return recall, matched_items

def evaluate_system(k_values: List[int] = [5, 10]) -> Dict:
    """
    Evaluate the recommendation system across all queries
    """
    ground_truth = load_ground_truth()
    queries = ground_truth["queries"]
    
    results = {
        "total_queries": len(queries),
        "k_values": k_values,
        "per_query_results": [],
        "mean_recall": {}
    }
    
    # Evaluate each query
    for idx, query_data in enumerate(queries, 1):
        query = query_data["query"]
        relevant = query_data["relevant_assessments"]
        
        print(f"\n{'='*80}")
        print(f"Query {idx}/{len(queries)}")
        print(f"{'='*80}")
        print(f"Query: {query[:100]}...")
        print(f"Relevant assessments: {len(relevant)}")
        
        # Get recommendations
        retrieved = get_recommendations(query, k=max(k_values))
        
        print(f"Retrieved assessments: {len(retrieved)}")
        
        query_result = {
            "query": query,
            "relevant_count": len(relevant),
            "retrieved_count": len(retrieved),
            "relevant_assessments": relevant,
            "retrieved_assessments": retrieved,
            "recall_scores": {}
        }
        
        # Calculate Recall@K for each K
        for k in k_values:
            recall, matched = calculate_recall_at_k_fuzzy(retrieved, relevant, k)
            query_result["recall_scores"][f"recall@{k}"] = recall
            
            print(f"\nRecall@{k}: {recall:.4f}")
            print(f"Matched assessments ({len(matched)}):")
            for match in matched:
                print(f"  ✓ {match}")
            
            if len(matched) < len(relevant):
                print(f"Missing assessments ({len(relevant) - len(matched)}):")
                for rel in relevant:
                    if not fuzzy_match(rel, retrieved[:k]):
                        print(f"  ✗ {rel}")
        
        results["per_query_results"].append(query_result)
    
    # Calculate Mean Recall@K
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    
    for k in k_values:
        recalls = [
            qr["recall_scores"][f"recall@{k}"] 
            for qr in results["per_query_results"]
        ]
        mean_recall = sum(recalls) / len(recalls)
        results["mean_recall"][f"mean_recall@{k}"] = mean_recall
        
        print(f"\nMean Recall@{k}: {mean_recall:.4f}")
        print(f"  Min: {min(recalls):.4f}")
        print(f"  Max: {max(recalls):.4f}")
        print(f"  Individual scores: {[f'{r:.4f}' for r in recalls]}")
    
    return results

def save_results(results: Dict):
    """Save evaluation results to JSON"""
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    print("="*80)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM - EVALUATION")
    print("="*80)
    
    # Run evaluation for K=5 and K=10
    results = evaluate_system(k_values=[5, 10])
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)