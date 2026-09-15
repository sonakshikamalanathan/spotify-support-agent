"""Central configuration: paths, brand, split and model choices."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LABELS_DIR = ROOT / "labels"
EVAL_DIR = ROOT / "eval"
REPORTS_DIR = ROOT / "reports"
# LLM responses are cached on disk and committed, so results reproduce without API keys.
CACHE_DIR = ROOT / "cache"

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
BRAND = "SpotifyCares"

SEED = 42
# Time-based split: the most recent 20% of conversations are held out for evaluation.
TEST_FRACTION = 0.20

PAIRS_CSV = DATA_PROCESSED / "spotify_pairs.csv"

# The agent's two LLM steps use different Groq models: each model has its own free-tier
# token-per-minute budget, and classification benefits from a reasoning model while drafting
# short replies does not.
ANALYSER = {"provider": "groq", "model": "openai/gpt-oss-120b"}
DRAFTER = {"provider": "groq", "model": "qwen/qwen3.8-27b"}
# The judge grades replies, so it must not be the model that wrote them (Qwen). Gemini 2.5 Flash
# was the first choice but its free tier allows only 20 requests/day; gpt-oss-120b on Groq allows 1,000.
JUDGE = {"provider": "groq", "model": "openai/gpt-oss-120b"}

# Minimum seconds between calls per provider (free-tier rate limits).
MIN_INTERVAL_S = {"groq": 2.5, "gemini": 4.5}
# A batch of judge cases is ~2,500 tokens and Groq's free tier allows 8,000 tokens/minute.
JUDGE_MIN_INTERVAL_S = 20
