# backend/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

# Retrieval defaults
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", "0.7"))
DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))