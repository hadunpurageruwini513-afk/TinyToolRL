"""Estimate problem capability c(x) using frozen SFT model pi_0 (Step E).

Design specification (CLAUDE.md):
  * Model: frozen SFT checkpoint checkpoints/sft (= pi_0)
  * Condition: forbidden tool (no calculator provided in prompt)
  * Sampling: K=3 samples per item at temperature T=0.7
  * Metric: c(x) = (number of correct direct samples) / K in {0.0, 1/3, 2/3, 1.0}
  * Write back: new column 'capability' in train.jsonl and all 3 eval splits.
"""

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CHAT_TEMPLATE_KWARGS, SYSTEM_PROMPT, load_jsonl, save_jsonl, set_seed
from eval.evaluate import load_model_and_tokenizer
from rewards.parsing import answers_match, extract_answer


def estimate_split_capability(
    model,
    tok,
    split_path: str,
    k: int = 3,
    temperature: float = 0.7,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Estimate capability c(x) for each item in a jsonl split."""
    data = load_jsonl(split_path)
    print(f"\nProcessing {split_path} ({len(data)} items, K={k}, T={temperature})...")

    orig_padding_side = tok.padding_side
    tok.padding_side = "left"

    updated_data = []

    for i in tqdm(range(0, len(data), batch_size), desc=f"Estimating c(x) [{os.path.basename(split_path)}]"):
        batch = data[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                ],
                tools=None,  # Forbidden tool
                add_generation_prompt=True,
                tokenize=False,
                **CHAT_TEMPLATE_KWARGS,
            )
            for item in batch
        ]

        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 1.0,
            "num_return_sequences": k,
        }

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]

        for b_idx, item in enumerate(batch):
            gold = item["answer"]
            ans_type = item.get("answer_type", "numeric")
            alts = item.get("alternatives", [])

            # Extract the K sequences for this batch element
            sub_out = out[b_idx * k : (b_idx + 1) * k]
            correct_count = 0
            sample_answers = []

            for seq in sub_out:
                gen_ids = seq[prompt_len:]
                text = tok.decode(gen_ids, skip_special_tokens=False)
                pred = extract_answer(text)
                sample_answers.append(pred)
                if answers_match(pred, gold, ans_type, alts):
                    correct_count += 1

            cap = round(correct_count / k, 4)
            row = dict(item)
            row["capability"] = cap
            updated_data.append(row)

    tok.padding_side = orig_padding_side

    # Summarize distribution
    caps = [r["capability"] for r in updated_data]
    counter = Counter(caps)
    mean_cap = sum(caps) / len(caps) if caps else 0.0

    print(f"\n--- Capability Summary for {split_path} ---")
    print(f"Total: {len(updated_data)}, Mean c(x): {mean_cap:.3f}")
    for val in sorted(counter.keys()):
        print(f"  c={val:.3f}: {counter[val]} items ({counter[val]/len(caps)*100:.1f}%)")

    # Category breakdown
    cats = sorted(set(r.get("category", "unknown") for r in updated_data))
    print("By category mean c(x):")
    for cat in cats:
        sub = [r["capability"] for r in updated_data if r.get("category") == cat]
        print(f"  {cat:<14}: count={len(sub)}, mean_c={sum(sub)/len(sub):.3f}")

    # Overwrite in place
    save_jsonl(split_path, updated_data)
    print(f"Updated {split_path} with 'capability' column.\n")

    return updated_data


def main():
    parser = argparse.ArgumentParser(description="Estimate capability c(x) using frozen pi_0")
    parser.add_argument("--model", default="checkpoints/sft", help="Frozen SFT model checkpoint")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "data/train.jsonl",
            "data/eval.jsonl",
            "data/eval_ood.jsonl",
            "data/eval_ratio20.jsonl",
        ],
        help="JSONL splits to process",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of samples per item")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading frozen SFT model from {args.model}...")
    model, tok = load_model_and_tokenizer(args.model, device=args.device)

    for split in args.splits:
        if os.path.exists(split):
            estimate_split_capability(
                model=model,
                tok=tok,
                split_path=split,
                k=args.k,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )
        else:
            print(f"Warning: split file {split} not found, skipping.")


if __name__ == "__main__":
    main()
