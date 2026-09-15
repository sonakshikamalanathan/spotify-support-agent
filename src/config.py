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
# short replies does not. The judge is a different vendor's model to limit self-preference bias.
ANALYSER = {"provider": "groq", "model": "openai/gpt-oss-120b"}
DRAFTER = {"provider": "groq", "model": "qwen/qwen3.8-27b"}
JUDGE = {"provider": "gemini", "model": "gemini-2.5-flash"}

# Minimum seconds between calls per provider (free-tier rate limits).
MIN_INTERVAL_S = {"groq": 2.5, "gemini": 4.5}
