"""Shared constants and small helpers for TinyToolRL.

Everything that must stay IDENTICAL across SFT, GRPO and evaluation
(system prompt, model id, jsonl IO) lives here, so the three stages never
drift apart in their prompt format.
"""

import json
import os
import random

# Base model used everywhere (SFT -> GRPO -> eval). Override via CLI/--model.
MODEL_ID = os.environ.get("TINYTOOLRL_MODEL", "Qwen/Qwen3-0.6B")

# One neutral system prompt for ALL stages.
# Note: it deliberately does NOT tell the model when to use the calculator.
# *When to call the tool* is exactly what SFT data + reward shaping must
# teach -- that is the research question of this project.
SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to a calculator tool. "
    "End your final reply with exactly one line of the form:\n"
    "Answer: <value>"
)

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

SEED = 42


def get_tokenizer(model_id: str = MODEL_ID):
    """Return tokenizer with TRL training chat template pre-configured.

    Guarantees prefix-preservation and byte-level consistency across SFT,
    GRPO, and evaluation.
    """
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, qwen3_template

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.chat_template = get_training_chat_template(tok)
    tok.response_template = qwen3_template
    return tok



def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
