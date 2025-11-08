# backend/services/chat.py
from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY, CHAT_MODEL, MAX_CONTEXT_CHARS
# NOTE: removed top-level import of retrieve_documents to avoid circular imports
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY missing in environment")

# Use modern OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are an SHL Admin with 20 years of experience in talent assessment and psychometrics. 
You specialize in analyzing role requirements and selecting the most suitable SHL assessments across technical, cognitive, and behavioral dimensions.

Your task:
1. Analyze the user's hiring request carefully to understand the role, skill level, and core competencies needed.
2. Review the provided SHL assessment data or catalog content thoroughly.
3. Recommend the most relevant SHL assessments that match the role’s technical and behavioral needs.
4. Ensure that your recommendations are realistic — e.g., if the request is for a Software Developer, suggest appropriate coding or problem-solving tests (basic, intermediate, or advanced depending on the role).
5. Support every recommendation with a brief but strong justification.
6. Include official SHL URLs when available.

Guidelines:
- Recommend up to 10 specific SHL assessments whenever possible (minimum 1, maximum 10).
- Tailor the recommendations to the role type and seniority (e.g., entry-level, mid, or senior).
- For each test, provide: the name, URL, and justification (1–2 sentences).
- If no matching assessments are found, explain why and ask for clarifying details (e.g., job level, focus area, skills needed).
- Keep the tone professional, concise, and confident.

Response format (JSON):
{
  "recommendations": [
    {
      "name": "Assessment Name",
      "url": "https://...",
      "justification": "Why this assessment fits the role and level"
    }
  ],
  "explanation": "Brief summary of why these assessments were chosen and any additional insights or next steps"
}
"""

CONTEXT_SEPARATOR = "\n" + "=" * 80 + "\n"


def build_context_blocks(candidates: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not candidates:
        return "No relevant assessments found in the database."
    
    pieces = []
    total = 0

    for idx, c in enumerate(candidates, 1):
        name = (
            c.get("name")
            or c.get("metadata", {}).get("assessment_name")
            or f"Assessment {idx}"
        )
        url = c.get("metadata", {}).get("url", "URL not available")
        text = c.get("text", "").strip()

        if len(text) > 500:
            text = text[:500] + "..."

        block = f"[{idx}] {name}\nURL: {url}\nDescription: {text}"

        block_len = len(block) + len(CONTEXT_SEPARATOR)
        if total + block_len > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                pieces.append(block[:remaining] + "...")
            break

        pieces.append(block)
        total += block_len

    context = CONTEXT_SEPARATOR.join(pieces)
    context += f"\n\nTotal assessments provided: {len(pieces)}"
    return context


def get_test_type_probs_via_llm(query: str) -> Dict[str, float]:
    classification_prompt = f"""
You are an SHL test classification expert.
Given the following hiring query, classify it into SHL test types.
Return a JSON object with probability values (0.0–1.0) for each type.
The probabilities MUST sum to exactly 1.0.

Test Types:
A: Ability & Aptitude
B: Biodata & Situational Judgement
C: Competencies
D: Development & 360
E: Assessment Exercises
K: Knowledge & Skills
P: Personality & Behavior
S: Simulations

Use probabilities that reflect the relevance of each test type.

Examples:

Query: "Need a Java developer test"
Classification: {{"A":0.15,"B":0.0,"C":0.05,"D":0.0,"E":0.0,"K":0.75,"P":0.05,"S":0.0}}

Query: "Assess leadership and communication skills"
Classification: {{"A":0.0,"B":0.0,"C":0.25,"D":0.05,"E":0.0,"K":0.05,"P":0.60,"S":0.05}}

Query: "Evaluate analytical and reasoning ability"
Classification: {{"A":0.70,"B":0.0,"C":0.15,"D":0.0,"E":0.0,"K":0.10,"P":0.0,"S":0.05}}

Query: "Test interpersonal style and workplace behavior"
Classification: {{"A":0.0,"B":0.05,"C":0.15,"D":0.0,"E":0.0,"K":0.0,"P":0.75,"S":0.05}}

Query: "Assess situational judgement and decision making in customer service roles"
Classification: {{"A":0.05,"B":0.55,"C":0.20,"D":0.0,"E":0.0,"K":0.10,"P":0.10,"S":0.0}}

Query: "Find a test to evaluate managerial potential and development needs"
Classification: {{"A":0.10,"B":0.05,"C":0.30,"D":0.40,"E":0.05,"K":0.05,"P":0.05,"S":0.0}}

Query: "Simulate a real-life sales conversation to assess persuasion skills"
Classification: {{"A":0.05,"B":0.10,"C":0.15,"D":0.0,"E":0.05,"K":0.10,"P":0.30,"S":0.25}}

Query: "Evaluate spreadsheet and Excel knowledge for finance roles"
Classification: {{"A":0.20,"B":0.0,"C":0.05,"D":0.0,"E":0.0,"K":0.70,"P":0.05,"S":0.0}}

Query: "Use an in-basket exercise to test how managers prioritize tasks"
Classification: {{"A":0.05,"B":0.0,"C":0.20,"D":0.15,"E":0.45,"K":0.10,"P":0.05,"S":0.0}}

Query: "Assess coding and algorithmic problem-solving for software engineers"
Classification: {{"A":0.40,"B":0.0,"C":0.05,"D":0.0,"E":0.0,"K":0.50,"P":0.05,"S":0.0}}

Now classify this query:
"{query}"

Output JSON strictly in this format (probabilities MUST sum to 1.0):
{{
  "A": 0.0,
  "B": 0.0,
  "C": 0.0,
  "D": 0.0,
  "E": 0.0,
  "K": 0.0,
  "P": 0.0,
  "S": 0.0
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an SHL test-type classifier."},
                {"role": "user", "content": classification_prompt},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        probs = json.loads(response.choices[0].message.content)
        probs_dict = {k: float(v) for k, v in probs.items()}
        
        # Normalize so they sum to 1.0
        total = sum(probs_dict.values())
        if total > 0:
            probs_dict = {k: v / total for k, v in probs_dict.items()}
        
        return probs_dict
    except Exception as e:
        print("⚠️ Type classification failed, defaulting to zeros:", e)
        return {t: 0.0 for t in ["A", "B", "C", "D", "E", "K", "P", "S"]}


def make_messages(
    query: str,
    candidates: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    test_type_probs: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    context = build_context_blocks(candidates)
    user_message = (
        f"User Question: {query}\n\n"
        f"Available Assessments:\n{context}\n\n"
        "Please analyze these assessments and provide recommendations in JSON format."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})
    return messages


def parse_json_response(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {
            "recommendations": [],
            "explanation": content,
            "error": f"Failed to parse JSON: {str(e)}",
        }


def call_chat(
    query: str,
    candidates: Optional[List[Dict[str, Any]]] = None,
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    test_type_probs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Call OpenAI Chat API to generate assessment recommendations.

    If `candidates` is provided, the function uses them directly and does not
    run classification or retrieval. If `candidates` is None, the function
    will run classification and retrieval internally (backward-compatible).
    """
    try:
        # If candidates not provided, run classification and retrieval as fallback
        if candidates is None:
            # produce test-type probabilities and retrieve documents internally
            local_probs = get_test_type_probs_via_llm(query)
            # Local (deferred) import to avoid circular import at module import time
            from src.services.retriever import retrieve_documents
            candidates = retrieve_documents(query_text=query, test_type_probs=local_probs, mode="hybrid", k=10)
            # if caller didn't pass test_type_probs, keep the local one
            if test_type_probs is None:
                test_type_probs = local_probs
        else:
            # candidates were provided by caller; if no test_type_probs present, try to compute them
            if test_type_probs is None:
                test_type_probs = get_test_type_probs_via_llm(query)

        # Build messages including test_type_probs for transparency to the LLM
        # Modify make_messages to accept test_type_probs if not already supporting it.
        messages = make_messages(query, candidates, conversation_history, test_type_probs=test_type_probs)

        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content.strip()
        parsed = parse_json_response(content)

        result = {
            "answer": content,
            "parsed_response": parsed,
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None)
            },
            "elapsed": round(elapsed, 2),
            "model": model,
            "success": True,
            "test_type_probs": test_type_probs,
            "candidates": candidates
        }

        return result

    except Exception as e:
        return {
            "answer": "",
            "parsed_response": {
                "recommendations": [],
                "explanation": f"An error occurred: {str(e)}"
            },
            "usage": None,
            "elapsed": 0,
            "model": model,
            "success": False,
            "error": str(e)
        }


def format_recommendations(parsed_response: Dict[str, Any]) -> str:
    output = []
    if "explanation" in parsed_response:
        output.append(parsed_response["explanation"])
        output.append("")

    recommendations = parsed_response.get("recommendations", [])
    if recommendations:
        output.append("Recommended Assessments:")
        output.append("")
        for idx, rec in enumerate(recommendations, 1):
            name = rec.get("name", "Unknown")
            url = rec.get("url", "")
            justification = rec.get("justification", "")
            output.append(f"{idx}. {name}")
            if justification:
                output.append(f"   {justification}")
            if url:
                output.append(f"   🔗 {url}")
            output.append("")
    else:
        output.append("No specific assessments could be recommended based on your query.")
    return "\n".join(output)


if __name__ == "__main__":
    mock_candidates = [
        {
            "name": "Verify G+ Test",
            "text": "Measures general cognitive ability and reasoning skills.",
            "metadata": {"url": "https://example.com/verify-g-plus"},
        }
    ]
    query = "I need to assess problem-solving skills for software engineers"
    result = call_chat(query, mock_candidates)
    print(f"Response time: {result['elapsed']}s")
    print(f"Tokens used: {result['usage']['total_tokens']}")
    print("\n" + "=" * 80)
    print(format_recommendations(result["parsed_response"]))