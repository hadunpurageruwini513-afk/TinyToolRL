"""Parsing utilities shared by reward functions and evaluation.

Two jobs:
  * extract the final answer from a model completion ("Answer: <value>")
  * detect / count / extract calculator tool calls from raw text

Both rewards and eval must use EXACTLY the same logic, otherwise the
training signal and the reported metrics would measure different things.
"""

import json
import re

# --------------------------------------------------------------------------
# Final-answer extraction
# --------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"[Aa]nswer\s*[:：]\s*([^\n]+)")

# Chat-template special tokens (<|im_end|>, <|endoftext|>, ...) can end up on
# the same line as the answer when text is decoded with skip_special_tokens
# =False (eval multi-turn loop) or rendered via apply_chat_template. Strip
# them before matching so reward and eval never see "3901</tool_response>".
_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")


def extract_answer(text: str):
    """Return the last 'Answer: ...' value in text, or None."""
    text = _SPECIAL_TOKEN_RE.sub(" ", text or "")
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    ans = ans.strip().strip(".*_`").strip()
    ans = re.sub(r"[。.]+$", "", ans).strip()
    ans = re.sub(r"^[$\\(\[]+|[$\\)\]]+$", "", ans).strip()
    return ans or None


def normalize_number(value):
    """Parse '3,901', '25%', '12.50' etc. into a float; None if impossible."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().replace(",", "").replace(" ", "")
    s = s.rstrip("%")
    s = re.sub(r"^[$]+", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_string(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"^(a|an|the)\s+", "", s)
    return s


def answers_match(pred, gold, answer_type: str = "numeric", alternatives=()) -> bool:
    """Compare a predicted answer with the gold one.

    numeric : float equality with a small tolerance (accepts '3,901' == 3901,
              '25%' == 25, '12.50' == 12.5)
    string  : case/punctuation-insensitive; pred is correct if it equals or
              contains the gold string or any alternative (and vice versa).
    """
    if pred is None:
        return False
    if answer_type == "numeric":
        p, g = normalize_number(pred), normalize_number(gold)
        if p is None or g is None:
            return False
        if abs(g - round(g)) < 1e-6:
            return abs(p - round(g)) < 1e-4
        return abs(p - g) <= 1e-4
    p = _normalize_string(str(pred))
    candidates = [str(gold)] + [str(a) for a in (alternatives or [])]
    for c in candidates:
        c = _normalize_string(c)
        if not c:
            continue
        if p == c or c in p or p in c:
            return True
    return False


# --------------------------------------------------------------------------
# Tool-call detection
# --------------------------------------------------------------------------

# Qwen3 renders assistant tool calls inside
# <tool_response>{"name": "calculator", "arguments": {...}}</tool_response>
# blocks (exact markup can vary slightly by template
# version). We detect the tagged form and also fall back to any JSON object
# that names our tool, so counting keeps working if the template changes.

_TAGGED_CALL_RE = re.compile(r"<tool_call\b", re.IGNORECASE)
_JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"([a-zA-Z_][\w-]*)"')

# Full block extraction for the eval loop: grab JSON inside tool_call tags.
_CALL_BLOCK_RE = re.compile(
    r"<tool_call[^>]*>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response\b", re.IGNORECASE)


def count_tool_calls(text: str) -> int:
    """Number of assistant tool calls in a raw generation string."""
    text = text or ""
    n_tagged = len(_TAGGED_CALL_RE.findall(text))
    if n_tagged:
        return n_tagged
    # fallback: JSON objects naming a tool, but exclude tool-response echoes
    stripped = _TOOL_RESPONSE_RE.sub("", text)
    return len(_JSON_CALL_RE.findall(stripped))


def has_tool_call(text: str) -> bool:
    return count_tool_calls(text) > 0


def extract_tool_calls(text: str):
    """Return a list of {'name': str, 'arguments': dict} from raw text.

    Used by the evaluation loop to actually execute what the model asked for.
    Tolerant to slightly malformed JSON (single quotes, python-style args).
    """
    calls = []
    for m in _CALL_BLOCK_RE.finditer(text or ""):
        raw = m.group(1)
        call = _parse_call_json(raw)
        if call is not None:
            calls.append(call)
    if calls:
        return calls
    # fallback: any JSON-looking object that names our calculator
    for m in re.finditer(r"\{[^{}]*\"name\"\s*:[^{}]*\}", text or "", re.DOTALL):
        call = _parse_call_json(m.group(0))
        if call is not None:
            calls.append(call)
    return calls


def _parse_call_json(raw: str):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(raw.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"expression": args}
    if not isinstance(args, dict):
        args = {"expression": str(args)}
    if not name:
        return None
    return {"name": str(name), "arguments": args}


def parse_tool_response(text: str) -> str:
    """Extract content of the last <tool_response>...</tool_response> block (eval loop)."""
    m = re.findall(r"<tool_response>(.*?)</tool_response>", text or "", re.DOTALL)
    return m[-1].strip() if m else ""


def extract_completion_text(completion) -> str:
    """Extract completion string from either a raw string or conversation messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return str(msg["content"])
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
        return str(completion)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def check_has_tool_call(completion) -> bool:
    """Check if tool was called from either raw string or conversation messages."""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict):
                if msg.get("tool_calls"):
                    return True
                if msg.get("role") == "tool":
                    return True
                if has_tool_call(str(msg.get("content", ""))):
                    return True
        return False
    if isinstance(completion, dict):
        if completion.get("tool_calls"):
            return True
        return has_tool_call(str(completion.get("content", "")))
    return has_tool_call(str(completion))

