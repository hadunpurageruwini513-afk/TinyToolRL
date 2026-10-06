"""Gate check: Qwen3 chat template + TOOL_SCHEMA + parsing.py must all agree.

Run this BEFORE building SFT data / GRPO / eval (todo item 4). Asserts the
"three places byte-identical" design convention is actually achievable:

  1. tools=[TOOL_SCHEMA] (dict) and tools=[calculator] (function, what
     GRPOTrainer receives) render byte-identical prompts.
  2. enable_thinking=False is respected (no thinking markers anywhere).
  3. An assistant tool_calls message renders into markup that
     rewards/parsing.py can detect/extract (count_tool_calls,
     extract_tool_calls, parse_tool_response) -- training reward and eval
     metrics measure the same thing.
  4. The tokenizer's chat template is prefix-preserving (trl requirement
     for the GRPOTrainer multi-turn tool loop).
  5. Prompt ends with a clean generation prompt for the assistant turn.

Usage:
    python scripts/verify_chat_template.py [--model Qwen/Qwen3-0.6B]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CHAT_TEMPLATE_KWARGS, MODEL_ID, SYSTEM_PROMPT, get_tokenizer
from rewards.parsing import count_tool_calls, extract_tool_calls, parse_tool_response
from tools.calculator import TOOL_SCHEMA, calculator, run_calculator

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    from trl.chat_template_utils import is_chat_template_prefix_preserving

    tok = get_tokenizer(args.model)
    ctk = CHAT_TEMPLATE_KWARGS

    user_q = "Calculate 47 × 83."
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_q},
    ]

    # --- 1. dict schema vs function render byte-identical -------------------
    r_schema = tok.apply_chat_template(
        prompt_msgs, tools=[TOOL_SCHEMA], add_generation_prompt=True,
        tokenize=False, **ctk,
    )
    r_func = tok.apply_chat_template(
        prompt_msgs, tools=[calculator], add_generation_prompt=True,
        tokenize=False, **ctk,
    )
    check("tools=[TOOL_SCHEMA] == tools=[calculator] byte-identical", r_schema == r_func)

    # --- 2. empty think block in generation prompt (prefix-preserving) ------
    check("enable_thinking=False: empty think block at prompt end",
          r_schema.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
          f"tail={r_schema[-50:]!r}")

    # --- 5. generation prompt check ---------------------------------------
    check("prompt ends with assistant generation prompt",
          "<|im_start|>assistant\n<think>\n\n</think>\n\n" in r_schema)

    # --- tool section present & names our calculator ------------------------
    check("rendered prompt contains calculator tool definition",
          "calculator" in r_schema and "expression" in r_schema)

    # --- 3. assistant tool_calls message <-> parsing.py agreement -----------
    expr = "47 * 83"
    result = run_calculator(expr)  # '3901'
    call_msgs = prompt_msgs + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": "calculator",
                                         "arguments": {"expression": expr}}}],
        },
    ]
    r_call = tok.apply_chat_template(
        call_msgs, tools=[TOOL_SCHEMA], tokenize=False, **ctk,
    )
    check("multi-turn tool trajectory prefix-preserving (r_call.startswith(r_schema))",
          r_call.startswith(r_schema))

    assistant_part = r_call[len(r_schema):]
    n_calls = count_tool_calls(assistant_part)
    check("parsing.count_tool_calls detects exactly 1 call in rendered assistant turn",
          n_calls == 1, f"got {n_calls}")

    extracted = extract_tool_calls(assistant_part)
    ok_extract = (
        len(extracted) == 1
        and extracted[0]["name"] == "calculator"
        and extracted[0]["arguments"].get("expression") == expr
    )
    check("parsing.extract_tool_calls recovers name+arguments", ok_extract, f"got {extracted}")

    tool_msgs = call_msgs + [{"role": "tool", "content": result, "name": "calculator"}]
    try:
        r_tool = tok.apply_chat_template(
            tool_msgs, tools=[TOOL_SCHEMA], tokenize=False, **ctk,
        )
        tool_render_ok = True
    except Exception as e:  # template may want a different tool-message shape
        r_tool, tool_render_ok = "", False
        check("tool-role message renders (shape {'role':'tool','content',...})", False, str(e))
    if tool_render_ok:
        got = parse_tool_response(r_tool)
        check("parsing.parse_tool_response extracts tool result from rendered convo",
          got == result, f"got {got!r}")

        r_tool_prompt = tok.apply_chat_template(
            tool_msgs, tools=[TOOL_SCHEMA], add_generation_prompt=True,
            tokenize=False, **ctk,
        )

        # final answer turn renders and extract_answer-compatible convention
        final_msgs = tool_msgs + [{"role": "assistant", "content": f"Answer: {result}"}]
        r_final = tok.apply_chat_template(
            final_msgs, tools=[TOOL_SCHEMA], tokenize=False, **ctk,
        )
        check("turn-2 prompt prefix-preserving (r_final.startswith(r_tool_prompt))",
              r_final.startswith(r_tool_prompt))

        final_part = r_final[len(r_tool_prompt):]
        from rewards.parsing import extract_answer
        check("extract_answer works on rendered final assistant turn",
              extract_answer(final_part) == result)

    # --- 4. prefix-preserving (trl GRPO multi-turn tool loop requirement) ----
    try:
        pp = is_chat_template_prefix_preserving(tok)
    except Exception as e:
        pp, detail = False, str(e)
        check("trl prefix-preserving check runs", False, detail)
    else:
        check("chat template is prefix-preserving (trl requirement)", bool(pp))

    # --- print rendered prompt for eyeballing --------------------------------
    print("\n" + "=" * 70)
    print("RENDERED PROMPT (dict schema, enable_thinking=False):")
    print("=" * 70)
    print(r_schema)
    print("=" * 70)
    print("RENDERED ASSISTANT TOOL-CALL TURN (appended part):")
    print(repr(r_call[len(r_schema):]))

    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL CHECKS PASSED -- template/parsing/trl are consistent.")


if __name__ == "__main__":
    main()
