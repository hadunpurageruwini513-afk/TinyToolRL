"""SFT Training script for TinyToolRL (Step D).

Trains Qwen3-0.6B on dual-trajectory dataset (data/sft_train.jsonl)
using LoRA (r=16, alpha=32, all-linear).

Features:
  - Preserves exact chat template prefix-preservation (via common.get_tokenizer)
  - Trains only assistant response tokens (assistant_only_loss=True)
  - Balanced tool vs direct trajectories
  - Freezes checkpoint into checkpoints/sft (= pi_0 baseline and capability reference)
"""

import argparse
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import MODEL_ID, get_tokenizer, load_jsonl, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for TinyToolRL")
    parser.add_argument("--model", default=MODEL_ID, help="Base model identifier")
    parser.add_argument("--train-data", default="data/sft_train.jsonl", help="Train dataset")
    parser.add_argument("--eval-data", default="data/sft_eval.jsonl", help="Eval dataset")
    parser.add_argument("--output-dir", default="checkpoints/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading tokenizer with training chat template: {args.model}...")
    tok = get_tokenizer(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base model: {args.model}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Loading datasets: {args.train_data}, {args.eval_data}...")
    train_rows = load_jsonl(args.train_data)
    eval_rows = load_jsonl(args.eval_data)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        max_length=args.max_length,
        assistant_only_loss=True,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    print(f"Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tok,
        peft_config=lora_config,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"SFT model successfully saved to {args.output_dir} (= pi_0).")


if __name__ == "__main__":
    main()
