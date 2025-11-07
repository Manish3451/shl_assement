# backend/services/chat.py
from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
from openai import OpenAI
from backend.config import OPENAI_API_KEY, CHAT_MODEL, MAX_CONTEXT_CHARS

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
- Recommend 2–4 specific SHL assessments whenever possible.
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

CONTEXT_SEPARATOR = "\n" + "="*80 + "\n"

def build_context_blocks(
    candidates: List[Dict[str, Any]], 
    max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """
    Build context string from candidate documents with smart truncation.
    
    Args:
        candidates: List of dicts with keys: name, text, metadata
        max_chars: Maximum characters for context
    
    Returns:
        Formatted context string
    """
    if not candidates:
        return "No relevant assessments found in the database."
    
    pieces = []
    total = 0
    
    for idx, c in enumerate(candidates, 1):
        # Extract metadata
        name = (
            c.get("name") or 
            c.get("metadata", {}).get("assessment_name") or 
            f"Assessment {idx}"
        )
        url = c.get("metadata", {}).get("url", "URL not available")
        text = c.get("text", "").strip()
        
        # Truncate text if too long
        if len(text) > 500:
            text = text[:500] + "..."
        
        # Build block
        block = (
            f"[{idx}] {name}\n"
            f"URL: {url}\n"
            f"Description: {text}"
        )
        
        # Check if we can fit this block
        block_len = len(block) + len(CONTEXT_SEPARATOR)
        if total + block_len > max_chars:
            remaining = max_chars - total
            if remaining > 100:  # Only add if we have meaningful space
                pieces.append(block[:remaining] + "...")
            break
        
        pieces.append(block)
        total += block_len
    
    context = CONTEXT_SEPARATOR.join(pieces)
    context += f"\n\nTotal assessments provided: {len(pieces)}"
    return context

def make_messages(
    query: str, 
    candidates: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Create messages for ChatCompletion API.
    
    Args:
        query: User's question
        candidates: Retrieved assessment documents
        conversation_history: Optional previous conversation turns
    
    Returns:
        List of message dicts for OpenAI API
    """
    context = build_context_blocks(candidates)
    
    user_message = (
        f"User Question: {query}\n\n"
        f"Available Assessments:\n{context}\n\n"
        "Please analyze these assessments and provide recommendations in JSON format."
    )
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": user_message})
    
    return messages

def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        content: Raw response content
    
    Returns:
        Parsed JSON dict or structured error response
    """
    # Remove markdown code blocks if present
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
        # Return structured error if JSON parsing fails
        return {
            "recommendations": [],
            "explanation": content,  # Use raw content as explanation
            "error": f"Failed to parse JSON: {str(e)}"
        }

def call_chat(
    query: str,
    candidates: List[Dict[str, Any]],
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Call OpenAI Chat API to generate assessment recommendations.
    
    Args:
        query: User's question
        candidates: Retrieved assessment documents
        model: OpenAI model to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        conversation_history: Optional previous conversation
    
    Returns:
        Dict with answer, parsed_response, usage stats, and timing
    """
    try:
        messages = make_messages(query, candidates, conversation_history)
        start = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        elapsed = time.time() - start
        
        # Extract response
        content = response.choices[0].message.content.strip()
        parsed = parse_json_response(content)
        
        # Build result
        result = {
            "answer": content,
            "parsed_response": parsed,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "elapsed": round(elapsed, 2),
            "model": model,
            "success": True
        }
        
        return result
        
    except Exception as e:
        # Handle errors gracefully
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
    """
    Format parsed recommendations into readable text.
    
    Args:
        parsed_response: Parsed JSON response
    
    Returns:
        Formatted string for display
    """
    output = []
    
    # Add explanation
    if "explanation" in parsed_response:
        output.append(parsed_response["explanation"])
        output.append("")
    
    # Add recommendations
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


# Example usage
if __name__ == "__main__":
    # Mock candidates for testing
    mock_candidates = [
        {
            "name": "Verify G+ Test",
            "text": "Measures general cognitive ability and reasoning skills.",
            "metadata": {"url": "https://example.com/verify-g-plus"}
        }
    ]
    
    query = "I need to assess problem-solving skills for software engineers"
    result = call_chat(query, mock_candidates)
    
    print(f"Response time: {result['elapsed']}s")
    print(f"Tokens used: {result['usage']['total_tokens']}")
    print("\n" + "="*80)
    print(format_recommendations(result['parsed_response']))