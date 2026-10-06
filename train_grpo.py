"""GRPO policy training script for TinyToolRL (Step G).

Trains the three RL policies from identical frozen SFT starting weights (pi_0):
  1. GRPO-Outcome: purely accuracy-driven (no tool cost)
  2. GRPO-FixedCost: OTC-style uniform penalty (-lambda * I_tool)
  3. GRPO-CapabilityAware (Ours): capability-aware penalty (-lambda * c(x) * I_tool)

Usage:
  python train_grpo.py --reward outcome --tag run1
  python train_grpo.py --reward fixed --lam 0.1 --tag run1
  python train_grpo.py --reward cap --lam 0.1 --tag run1
"""

import argparse
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import CHAT_TEMPLATE_KWARGS, SYSTEM_PROMPT, get_tokenizer, load_jsonl, set_seed
from rewards.cost_aware_reward import get_reward_functions
from tools.calculator import calculator


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO policy for TinyToolRL")
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        choices=["outcome", "fixed", "cap", "capability"],
        help="Reward mode: outcome, fixed, or cap",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="Penalty multiplier lambda for fixed / cap rewards (default 0.1)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="run1",
        help="Run identifier tag (e.g. run1, seed42, etc.)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="checkpoints/sft_merged",
        help="Starting model checkpoint (merged SFT = frozen pi_0)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train.jsonl",
        help="Path to training dataset jsonl with capability column",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (overrides epochs if > 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per device train batch size (default 4)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default 4)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of generations G per prompt in GRPO (default 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for LoRA adapter (default 2e-5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL divergence penalty coefficient beta (default 0.04)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Rollout generation temperature (default 1.0)",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=384,
        help="Max generated completion length in tokens (default 384)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1,
        help="Max tool calling iterations (default 1)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank r (default 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output checkpoint directory (default checkpoints/grpo_{reward}_{tag})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    mode = "cap" if args.reward in ("cap", "capability") else args.reward
    if args.output_dir is None:
        args.output_dir = f"checkpoints/grpo_{mode}_{args.tag}"

    print("=" * 65)
    print("TinyToolRL GRPO Training")
    print(f"  Reward Mode:       {mode}")
    print(f"  Penalty Lambda:    {args.lam}")
    print(f"  Base Model (pi_0): {args.base_model}")
    print(f"  Training Data:     {args.train_data}")
    print(f"  Output Dir:        {args.output_dir}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch / GradAccum: {args.batch_size} / {args.grad_accum}")
    print(f"  Num Generations:   {args.num_generations}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Beta (KL):         {args.beta}")
    print(f"  Max Tool Turns:    {args.max_turns}")
    print("=" * 65)

    tok = get_tokenizer()

    raw_data = load_jsonl(args.train_data)
    print(f"Loaded {len(raw_data)} training items.")

    # Format conversational prompt
    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        for item in raw_data
    ]

    train_ds = Dataset.from_dict(
        {
            "prompt": prompts,
            "answer": [item["answer"] for item in raw_data],
            "answer_type": [item.get("answer_type", "numeric") for item in raw_data],
            "alternatives": [item.get("alternatives") for item in raw_data],
            "capability": [item.get("capability") for item in raw_data],
            "needs_tool": [item.get("needs_tool") for item in raw_data],
        }
    )

    reward_funcs = get_reward_functions(mode=mode, penalty_lambda=args.lam)
    print(f"Configured {len(reward_funcs)} reward function(s): {[f.__name__ for f in reward_funcs]}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        max_tool_calling_iterations=args.max_turns,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=peft_config,
        tools=[calculator],
    )

    print("\nStarting GRPO training...")
    trainer.train()

    print(f"\nSaving trained GRPO adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Training and checkpoint save complete!\n")


if __name__ == "__main__":
    main()
