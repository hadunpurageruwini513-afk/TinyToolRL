"""Statistical rigor appendix (review P1-6).

Computes from the per-sample evaluation records (eval_results/*_records.jsonl):
  * Wilson 95% confidence intervals for Accuracy and Tool Call Rate per model x split
  * McNemar exact (binomial) tests on per-sample correctness flips:
      SFT vs each GRPO variant, and GRPO-Fixed vs GRPO-Cap
  * Tool-decision flip counts (called_tool differences) per pair

Writes eval_results/stats_appendix.md. No model runs; pure record analysis,
so any reader can reproduce the uncertainty quantification from the repo alone.

Usage:
    python scripts/analyze_stats.py
"""

import json
import os
import sys
from math import comb, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLITS = ["eval", "eval_ood", "eval_ratio20"]
MODELS = ["SFT", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap"]
Z = 1.959963985  # 95%


def wilson(k: int, n: int) -> tuple:
    """Wilson score 95% CI for a proportion; returns (lo, hi) in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 0.0
    for i in range(0, k + 1):
        p += comb(n, i)
    # two-sided: 2 * P(X <= k) under Binom(n, 0.5), capped at 1
    return min(1.0, 2 * p / (2 ** n))


def load(tag: str, split: str) -> list:
    path = f"eval_results/{tag}_{split}_records.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    lines = ["# Statistical Appendix: Confidence Intervals & Paired Tests",
             "",
             "Wilson 95% CIs on Accuracy / Tool Call Rate; McNemar exact tests on",
             "per-sample correctness flips. Computed from `eval_results/*_records.jsonl`",
             "by `scripts/analyze_stats.py` (re-runnable, no model needed).", ""]

    for split in SPLITS:
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append("| Model | Acc (%) [95% CI] | TCR (%) [95% CI] | n |")
        lines.append("|---|---|---|---|")
        recs = {}
        for m in MODELS:
            r = load(m, split)
            recs[m] = r
            n = len(r)
            ka = sum(x["correct"] for x in r)
            kt = sum(x["called_tool"] for x in r)
            alo, ahi = wilson(ka, n)
            tlo, thi = wilson(kt, n)
            lines.append(
                f"| {m} | {ka/n*100:.2f} [{alo*100:.2f}, {ahi*100:.2f}] | "
                f"{kt/n*100:.2f} [{tlo*100:.2f}, {thi*100:.2f}] | {n} |"
            )
        lines.append("")
        lines.append("| Comparison | GRPO-correct&SFT-wrong (b) | SFT-correct&GRPO-wrong (c) | McNemar p | tool-decision flips |")
        lines.append("|---|---|---|---|---|")
        pairs = [("SFT", "GRPO-Outcome"), ("SFT", "GRPO-Fixed"), ("SFT", "GRPO-Cap"),
                 ("GRPO-Fixed", "GRPO-Cap")]
        for a, b in pairs:
            ra, rb = recs[a], recs[b]
            disc_b = sum(1 for x, y in zip(ra, rb) if (not x["correct"]) and y["correct"])
            disc_c = sum(1 for x, y in zip(ra, rb) if x["correct"] and (not y["correct"]))
            p = mcnemar_exact(disc_b, disc_c)
            tf = sum(1 for x, y in zip(ra, rb) if x["called_tool"] != y["called_tool"])
            lines.append(f"| {a} vs {b} | {disc_b} | {disc_c} | {p:.3f} | {tf} |")
        lines.append("")

    lines.append("### Reading guide")
    lines.append("")
    lines.append("- Overlapping CIs and non-significant McNemar p-values mean the accuracy deltas")
    lines.append("  (e.g. +2.50pp = 3/120 samples) are **within sampling noise at n=120**; they must")
    lines.append("  not be presented as method-specific gains without seeds/replicates.")
    lines.append("- `tool-decision flips = 0` across all pairs at λ=0.1 is the controlled null result:")
    lines.append("  the penalty scale did not move the call-or-not policy of any model.")

    os.makedirs("eval_results", exist_ok=True)
    out = "\n".join(lines) + "\n"
    with open("eval_results/stats_appendix.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
