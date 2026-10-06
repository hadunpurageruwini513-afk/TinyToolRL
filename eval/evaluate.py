"""Evaluation pipeline for TinyToolRL.

Evaluates base models, SFT checkpoints, or GRPO policies on dataset splits.
Supports two modes:
  - with_tools: agent can call calculator (max_turns=1) -> reports Acc + TCR
  - no_tools: agent cannot call calculator -> reports pure direct Acc (e.g. for c(x) estimation)

Strictly uses common.get_tokenizer(), common.SYSTEM_PROMPT, tools.TOOL_SCHEMA,
and rewards.parsing for byte-level consistency with training.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CHAT_TEMPLATE_KWARGS, MODEL_ID, SYSTEM_PROMPT, get_tokenizer, load_jsonl, save_jsonl, set_seed
from rewards.parsing import answers_match, count_tool_calls, extract_answer, extract_tool_calls
from tools.calculator import TOOL_SCHEMA, run_calculator


def load_model_and_tokenizer(model_path_or_id: str, adapter_path: str = None, device: str = "cuda"):
    """Load model (base or LoRA adapter) and tokenizer with training template."""
    is_adapter = adapter_path or os.path.exists(os.path.join(model_path_or_id, "adapter_config.json"))
    actual_adapter = adapter_path or (model_path_or_id if is_adapter else None)

    base_id = MODEL_ID
    if is_adapter and os.path.exists(os.path.join(actual_adapter, "adapter_config.json")):
        with open(os.path.join(actual_adapter, "adapter_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path", MODEL_ID)

    tok = get_tokenizer(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {model_path_or_id}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if is_adapter:
        from peft import PeftModel
        print(f"Loading base model {base_id} with adapter {actual_adapter}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            dtype=dtype,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, actual_adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            dtype=dtype,
            device_map=device,
        )

    model.eval()
    return model, tok


def generate_batch(
    model,
    tok,
    list_of_messages: List[List[Dict[str, Any]]],
    tools: List[Dict[str, Any]] | None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> List[str]:
    """Generate completions for a batch of conversation messages."""
    if not list_of_messages:
        return []

    orig_padding_side = tok.padding_side
    tok.padding_side = "left"

    prompts = [
        tok.apply_chat_template(
            m,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **CHAT_TEMPLATE_KWARGS,
        )
        for m in list_of_messages
    ]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for o in out:
        gen_ids = o[prompt_len:]
        completions.append(tok.decode(gen_ids, skip_special_tokens=False))

    tok.padding_side = orig_padding_side
    return completions


def evaluate_dataset(
    model,
    tok,
    data: List[Dict[str, Any]],
    allow_tools: bool = True,
    max_turns: int = 1,
    batch_size: int = 8,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run batched evaluation on a list of dataset records."""
    records = []
    tools_list = [TOOL_SCHEMA] if allow_tools else None

    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating (tools={allow_tools})"):
        batch = data[i : i + batch_size]
        batch_msgs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            for item in batch
        ]

        # Turn 1 (all items in batch)
        completions_t1 = generate_batch(
            model=model,
            tok=tok,
            list_of_messages=batch_msgs,
            tools=tools_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Check tool calls
        turn2_indices = []
        turn2_msgs = []
        turn2_tool_infos = {}

        final_completions = list(completions_t1)

        for b_idx, (item, comp1) in enumerate(zip(batch, completions_t1)):
            tool_calls = extract_tool_calls(comp1) if allow_tools else []
            if allow_tools and tool_calls and max_turns >= 1:
                tc = tool_calls[0]
                expr = tc["arguments"].get("expression", "")
                res = run_calculator(expr)
                turn2_tool_infos[b_idx] = {"call": tc, "result": res}

                m2 = list(batch_msgs[b_idx])
                m2.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"type": "function", "function": tc}],
                    }
                )
                m2.append(
                    {
                        "role": "tool",
                        "name": "calculator",
                        "content": res,
                    }
                )
                turn2_indices.append(b_idx)
                turn2_msgs.append(m2)

        # Turn 2 for items that called tools
        if turn2_msgs:
            completions_t2 = generate_batch(
                model=model,
                tok=tok,
                list_of_messages=turn2_msgs,
                tools=tools_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for orig_idx, comp2 in zip(turn2_indices, completions_t2):
                final_completions[orig_idx] = comp2

        # Record outcomes
        for b_idx, (item, comp1, final_comp) in enumerate(
            zip(batch, completions_t1, final_completions)
        ):
            q = item["question"]
            gold = item["answer"]
            ans_type = item.get("answer_type", "numeric")
            alts = item.get("alternatives", [])
            category = item.get("category", "unknown")

            pred = extract_answer(final_comp)
            correct = answers_match(pred, gold, ans_type, alts)
            called_tool = b_idx in turn2_tool_infos or (allow_tools and bool(extract_tool_calls(comp1)))

            records.append(
                {
                    "id": item.get("id"),
                    "category": category,
                    "needs_tool": item.get("needs_tool"),
                    "capability": item.get("capability"),
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": bool(correct),
                    "called_tool": bool(called_tool),
                    "has_format": bool(pred is not None),
                    "turn1_completion": comp1,
                    "tool_info": turn2_tool_infos.get(b_idx),
                    "final_completion": final_comp,
                }
            )

    # Aggregate metrics
    total = len(records)
    acc = sum(r["correct"] for r in records) / total if total > 0 else 0.0
    tcr = sum(r["called_tool"] for r in records) / total if total > 0 else 0.0
    format_rate = sum(r["has_format"] for r in records) / total if total > 0 else 0.0

    by_cat = {}
    categories = sorted(set(r["category"] for r in records))
    for cat in categories:
        sub = [r for r in records if r["category"] == cat]
        n_sub = len(sub)
        by_cat[cat] = {
            "count": n_sub,
            "accuracy": sum(r["correct"] for r in sub) / n_sub if n_sub > 0 else 0.0,
            "tool_call_rate": sum(r["called_tool"] for r in sub) / n_sub if n_sub > 0 else 0.0,
            "format_rate": sum(r["has_format"] for r in sub) / n_sub if n_sub > 0 else 0.0,
        }

    # Aggregate by capability if present
    by_cap = {}
    cap_groups = {}
    has_caps = any(r["capability"] is not None for r in records)
    if has_caps:
        unique_caps = sorted(set(r["capability"] for r in records if r["capability"] is not None))
        for c_val in unique_caps:
            sub = [r for r in records if r["capability"] == c_val]
            n_sub = len(sub)
            by_cap[str(round(c_val, 3))] = {
                "count": n_sub,
                "accuracy": sum(r["correct"] for r in sub) / n_sub if n_sub > 0 else 0.0,
                "tool_call_rate": sum(r["called_tool"] for r in sub) / n_sub if n_sub > 0 else 0.0,
                "format_rate": sum(r["has_format"] for r in sub) / n_sub if n_sub > 0 else 0.0,
            }

        # Groups: Incapable (c <= 1/3) vs Capable (c >= 2/3)
        incapable_sub = [r for r in records if r["capability"] is not None and r["capability"] <= 0.334]
        capable_sub = [r for r in records if r["capability"] is not None and r["capability"] >= 0.666]
        for grp_name, grp_sub in [("Incapable (c<=1/3)", incapable_sub), ("Capable (c>=2/3)", capable_sub)]:
            n_sub = len(grp_sub)
            cap_groups[grp_name] = {
                "count": n_sub,
                "accuracy": sum(r["correct"] for r in grp_sub) / n_sub if n_sub > 0 else 0.0,
                "tool_call_rate": sum(r["called_tool"] for r in grp_sub) / n_sub if n_sub > 0 else 0.0,
                "format_rate": sum(r["has_format"] for r in grp_sub) / n_sub if n_sub > 0 else 0.0,
            }

    return {
        "summary": {
            "total": total,
            "accuracy": acc,
            "tool_call_rate": tcr,
            "format_rate": format_rate,
            "by_category": by_cat,
            "by_capability": by_cap,
            "capability_groups": cap_groups,
        },
        "records": records,
    }


def print_summary(summary: Dict[str, Any], title: str = "Evaluation Results"):
    """Pretty-print summary metrics table."""
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)
    s = summary
    print(f"Total samples:     {s['total']}")
    print(f"Overall Accuracy:  {s['accuracy'] * 100:.2f}%")
    print(f"Tool Call Rate:    {s['tool_call_rate'] * 100:.2f}%")
    print(f"Format Valid Rate: {s['format_rate'] * 100:.2f}%")
    print("-" * 65)
    print(f"{'Category':<16} | {'Count':<6} | {'Acc (%)':<8} | {'TCR (%)':<8} | {'Fmt (%)':<8}")
    print("-" * 65)
    for cat, m in s["by_category"].items():
        print(
            f"{cat:<16} | {m['count']:<6} | {m['accuracy']*100:<8.2f} | {m['tool_call_rate']*100:<8.2f} | {m['format_rate']*100:<8.2f}"
        )

    if s.get("capability_groups"):
        print("-" * 65)
        print(f"{'Cap Group':<20} | {'Count':<6} | {'Acc (%)':<8} | {'TCR (%)':<8} | {'Fmt (%)':<8}")
        print("-" * 65)
        for grp, m in s["capability_groups"].items():
            print(
                f"{grp:<20} | {m['count']:<6} | {m['accuracy']*100:<8.2f} | {m['tool_call_rate']*100:<8.2f} | {m['format_rate']*100:<8.2f}"
            )

    if s.get("by_capability"):
        print("-" * 65)
        print(f"{'c(x) Bucket':<16} | {'Count':<6} | {'Acc (%)':<8} | {'TCR (%)':<8} | {'Fmt (%)':<8}")
        print("-" * 65)
        for c_val, m in s["by_capability"].items():
            print(
                f"c={c_val:<14} | {m['count']:<6} | {m['accuracy']*100:<8.2f} | {m['tool_call_rate']*100:<8.2f} | {m['format_rate']*100:<8.2f}"
            )
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyToolRL model")
    parser.add_argument("--model", default=MODEL_ID, help="Model name or checkpoint directory")
    parser.add_argument("--adapter", default=None, help="LoRA adapter checkpoint directory (optional)")
    parser.add_argument("--data", default="data/eval.jsonl", help="Dataset jsonl file")
    parser.add_argument("--tools", action="store_true", default=False, help="Allow tool calling")
    parser.add_argument("--no-tools", dest="tools", action="store_false", help="Forbid tool calling")
    parser.add_argument("--max-turns", type=int, default=1, help="Max tool turns (default 1)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples evaluated")
    parser.add_argument("--output", default=None, help="Output file path for full predictions JSONL")
    parser.add_argument("--summary-output", default=None, help="Output file path for summary JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    print(f"Loaded {len(data)} items from {args.data}")

    model, tok = load_model_and_tokenizer(args.model, adapter_path=args.adapter, device=args.device)

    res = evaluate_dataset(
        model=model,
        tok=tok,
        data=data,
        allow_tools=args.tools,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )

    mode_str = "With Tools" if args.tools else "No Tools (Direct)"
    print_summary(res["summary"], title=f"Results: {args.model} [{mode_str}] on {args.data}")

    if args.output:
        save_jsonl(args.output, res["records"])
        print(f"Saved {len(res['records'])} detailed records to {args.output}")

    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(res["summary"], f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
