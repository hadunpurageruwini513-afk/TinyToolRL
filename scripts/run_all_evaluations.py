"""Run comprehensive 4-model evaluation across all test splits (Step H).

Evaluates:
  1. SFT (pi_0 baseline)
  2. GRPO-Outcome (accuracy only)
  3. GRPO-FixedCost (OTC uniform penalty)
  4. GRPO-CapabilityAware (Ours)

Across:
  - eval (in-distribution 33% tool need)
  - eval_ood (out-of-domain 3-digit large numbers)
  - eval_ratio20 (distribution shift: 20% tool need)
"""

import argparse
import json
import os
import subprocess
import sys

SPLITS = {
    "eval": "data/eval.jsonl",
    "eval_ood": "data/eval_ood.jsonl",
    "eval_ratio20": "data/eval_ratio20.jsonl",
}

MODELS = {
    "SFT": "checkpoints/sft_merged",
    "GRPO-Outcome": "checkpoints/grpo_outcome_run1",
    "GRPO-Fixed": "checkpoints/grpo_fixed_run1",
    "GRPO-Cap": "checkpoints/grpo_cap_run1",
    "GRPO-Fixed-lam03": "checkpoints/grpo_fixed_lam03",
    "GRPO-Cap-lam03": "checkpoints/grpo_cap_lam03",
    "GRPO-Fixed-lam05": "checkpoints/grpo_fixed_lam05",
    "GRPO-Cap-lam05": "checkpoints/grpo_cap_lam05",
}


def run_eval(model_name: str, model_path: str, split_name: str, split_path: str, output_dir: str = "eval_results"):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{model_name}_{split_name}_summary.json")
    records_path = os.path.join(output_dir, f"{model_name}_{split_name}_records.jsonl")

    cmd = [
        sys.executable,
        "-X", "utf8",
        "eval/evaluate.py",
        "--model", model_path,
        "--data", split_path,
        "--tools",
        "--output", records_path,
        "--summary-output", summary_path,
    ]
    print(f"\n>>> Running evaluation: {model_name} on {split_name} ({split_path})...")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"Error evaluating {model_name} on {split_name}:")
        print(res.stderr)
        return None
    print(res.stdout)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_comparison_report(output_dir: str = "eval_results"):
    all_summaries = {}
    for m in MODELS:
        all_summaries[m] = {}
        for s in SPLITS:
            path = os.path.join(output_dir, f"{m}_{s}_summary.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    all_summaries[m][s] = json.load(f)

    report_lines = []
    report_lines.append("# TinyToolRL 4-Model Comparative Evaluation\n")

    for s in SPLITS:
        report_lines.append(f"### Split: {s} ({SPLITS[s]})\n")
        header = "| Model | Overall Acc (%) | Tool Call Rate (%) | Incapable (c<=1/3) TCR (%) | Capable (c>=2/3) TCR (%) | Format Rate (%) |"
        sep = "|---|---|---|---|---|---|"
        report_lines.append(header)
        report_lines.append(sep)

        for m in MODELS:
            data = all_summaries.get(m, {}).get(s)
            if not data:
                continue
            acc = data.get("accuracy", 0.0) * 100
            tcr = data.get("tool_call_rate", 0.0) * 100
            fmt = data.get("format_rate", 0.0) * 100
            groups = data.get("capability_groups", {})
            incap_tcr = groups.get("Incapable (c<=1/3)", {}).get("tool_call_rate", 0.0) * 100
            cap_tcr = groups.get("Capable (c>=2/3)", {}).get("tool_call_rate", 0.0) * 100

            row = f"| {m} | {acc:.2f} | {tcr:.2f} | {incap_tcr:.2f} | {cap_tcr:.2f} | {fmt:.2f} |"
            report_lines.append(row)
        report_lines.append("\n")

    report_md = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "comparative_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nComparative report written to {report_path}")
    print(report_md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyToolRL models")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()), help="Splits to evaluate")
    parser.add_argument("--report-only", action="store_true", help="Only compile report from existing JSON summaries")
    args = parser.parse_args()

    if not args.report_only:
        for m in args.models:
            if m not in MODELS:
                continue
            m_path = MODELS[m]
            if not os.path.exists(m_path):
                print(f"Skipping {m} (checkpoint {m_path} does not exist yet).")
                continue
            for s in args.splits:
                if s not in SPLITS:
                    continue
                s_path = SPLITS[s]
                run_eval(m, m_path, s, s_path)

    generate_comparison_report()


if __name__ == "__main__":
    main()
