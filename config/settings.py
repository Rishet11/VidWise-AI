"""Runtime configuration. Environment variables are the only secret source."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("VIDWISE_DATA_DIR", ROOT / "data"))
TRANSCRIPT_CACHE_DIR = DATA_DIR / "transcripts"
LOG_DIR = Path(os.getenv("VIDWISE_LOG_DIR", ROOT / "logs"))
MODEL_ID = os.getenv("VIDWISE_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL_ID = os.getenv(
    "VIDWISE_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

MIN_VIDEOS = 1  # upload/demo modes remain useful with one; URL research prompts for 3+
MAX_VIDEOS = 6
SESSION_DAILY_QUESTION_LIMIT = int(os.getenv("VIDWISE_SESSION_DAILY_LIMIT", "15"))
GLOBAL_DAILY_QUESTION_LIMIT = int(os.getenv("VIDWISE_GLOBAL_DAILY_LIMIT", "500"))
DISCOVERY_DAILY_LIMIT = int(os.getenv("VIDWISE_DISCOVERY_DAILY_LIMIT", "100"))
MAX_LLM_CALLS_PER_QUESTION = 3
RETRIEVAL_K = 12
RERANK_THRESHOLD = 8
FINAL_K = 6

SUPADATA_URL = "https://api.supadata.ai/v1/transcript"
SUPADATA_MONTHLY_FREE_CREDITS = 100

