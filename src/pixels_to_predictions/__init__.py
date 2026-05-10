"""Pixels-to-Predictions: SmolVLM-500M fine-tune for NYU DL Spring 2026 Kaggle final."""

__version__ = "0.1.0"

# Hard competition constraints, imported where they're needed.
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TRAINABLE_PARAM_BUDGET = 5_000_000
