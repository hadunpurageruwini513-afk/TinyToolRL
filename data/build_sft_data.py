"""Construct dual-trajectory SFT training and evaluation datasets.

Design convention (keep in sync with CLAUDE.md):
  * Dual-trajectory balance:
    - hard_arith (needs_tool=True, 300 items in train): tool trajectory
      Turn 1 Assistant: calls calculator(expression)
      Turn 2 Tool:      calculator result
      Turn 3 Assistant: Answer: <gold>
    - easy_arith & common_sense (needs_tool=False, 300 items in train): direct answer trajectory
      Turn 1 Assistant: Answer: <gold>

Output format: JSONL with 'messages' list compatible with TRL SFTTrainer.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SYSTEM_PROMPT, load_jsonl, save_jsonl
from tools.calculator import TOOL_SCHEMA, run_calculator


def build_trajectory(item: dict) -> list:
    """Build conversational message list for a single item."""
    q = item["question"]
    gold = item["answer"]
    needs_tool = bool(item.get("needs_tool", False))
    expr = item.get("expression")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]

    if needs_tool and expr:
        # Multi-turn tool trajectory
        tool_res = run_calculator(expr)
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": {"expression": expr},
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "name": "calculator",
                "content": tool_res,
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Answer: {gold}",
            }
        )
    else:
        # Direct answer trajectory
        msgs.append(
            {
                "role": "assistant",
                "content": f"Answer: {gold}",
            }
        )

    return msgs


def convert_split(src_path: str, dst_path: str):
    raw = load_jsonl(src_path)
    sft_rows = []
    n_tool = 0
    n_direct = 0

    for item in raw:
        msgs = build_trajectory(item)
        if len(msgs) > 3:
            n_tool += 1
        else:
            n_direct += 1

        sft_rows.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "needs_tool": item.get("needs_tool"),
                "messages": msgs,
                "tools": [TOOL_SCHEMA],
            }
        )

    save_jsonl(dst_path, sft_rows)
    print(
        f"Converted {src_path} -> {dst_path}: total={len(sft_rows)}, "
        f"tool_trajectories={n_tool} ({n_tool/len(sft_rows)*100:.1f}%), "
        f"direct_trajectories={n_direct} ({n_direct/len(sft_rows)*100:.1f}%)"
    )


def main():
    os.makedirs("data", exist_ok=True)
    convert_split("data/train.jsonl", "data/sft_train.jsonl")
    convert_split("data/eval.jsonl", "data/sft_eval.jsonl")


if __name__ == "__main__":
    main()
