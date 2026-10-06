# Statistical Appendix: Confidence Intervals & Paired Tests

Wilson 95% CIs on Accuracy / Tool Call Rate; McNemar exact tests on
per-sample correctness flips. Computed from `eval_results/*_records.jsonl`
by `scripts/analyze_stats.py` (re-runnable, no model needed).

## Split: eval

| Model | Acc (%) [95% CI] | TCR (%) [95% CI] | n |
|---|---|---|---|
| SFT | 94.00 [88.99, 96.81] | 33.33 [26.29, 41.21] | 150 |
| GRPO-Outcome | 94.67 [89.83, 97.27] | 33.33 [26.29, 41.21] | 150 |
| GRPO-Fixed | 94.67 [89.83, 97.27] | 33.33 [26.29, 41.21] | 150 |
| GRPO-Cap | 94.67 [89.83, 97.27] | 33.33 [26.29, 41.21] | 150 |

| Comparison | GRPO-correct&SFT-wrong (b) | SFT-correct&GRPO-wrong (c) | McNemar p | tool-decision flips |
|---|---|---|---|---|
| SFT vs GRPO-Outcome | 1 | 0 | 1.000 | 0 |
| SFT vs GRPO-Fixed | 1 | 0 | 1.000 | 0 |
| SFT vs GRPO-Cap | 1 | 0 | 1.000 | 0 |
| GRPO-Fixed vs GRPO-Cap | 0 | 0 | 1.000 | 0 |

## Split: eval_ood

| Model | Acc (%) [95% CI] | TCR (%) [95% CI] | n |
|---|---|---|---|
| SFT | 96.67 [91.74, 98.70] | 50.00 [41.19, 58.81] | 120 |
| GRPO-Outcome | 99.17 [95.43, 99.85] | 50.00 [41.19, 58.81] | 120 |
| GRPO-Fixed | 99.17 [95.43, 99.85] | 50.00 [41.19, 58.81] | 120 |
| GRPO-Cap | 99.17 [95.43, 99.85] | 50.00 [41.19, 58.81] | 120 |

| Comparison | GRPO-correct&SFT-wrong (b) | SFT-correct&GRPO-wrong (c) | McNemar p | tool-decision flips |
|---|---|---|---|---|
| SFT vs GRPO-Outcome | 3 | 0 | 0.250 | 0 |
| SFT vs GRPO-Fixed | 3 | 0 | 0.250 | 0 |
| SFT vs GRPO-Cap | 3 | 0 | 0.250 | 0 |
| GRPO-Fixed vs GRPO-Cap | 0 | 0 | 1.000 | 0 |

## Split: eval_ratio20

| Model | Acc (%) [95% CI] | TCR (%) [95% CI] | n |
|---|---|---|---|
| SFT | 93.33 [87.39, 96.58] | 20.00 [13.82, 28.04] | 120 |
| GRPO-Outcome | 95.00 [89.52, 97.69] | 20.00 [13.82, 28.04] | 120 |
| GRPO-Fixed | 95.83 [90.62, 98.21] | 20.00 [13.82, 28.04] | 120 |
| GRPO-Cap | 95.83 [90.62, 98.21] | 20.00 [13.82, 28.04] | 120 |

| Comparison | GRPO-correct&SFT-wrong (b) | SFT-correct&GRPO-wrong (c) | McNemar p | tool-decision flips |
|---|---|---|---|---|
| SFT vs GRPO-Outcome | 3 | 1 | 0.625 | 0 |
| SFT vs GRPO-Fixed | 4 | 1 | 0.375 | 0 |
| SFT vs GRPO-Cap | 4 | 1 | 0.375 | 0 |
| GRPO-Fixed vs GRPO-Cap | 0 | 0 | 1.000 | 0 |

### Reading guide

- Overlapping CIs and non-significant McNemar p-values mean the accuracy deltas
  (e.g. +2.50pp = 3/120 samples) are **within sampling noise at n=120**; they must
  not be presented as method-specific gains without seeds/replicates.
- `tool-decision flips = 0` across all pairs at λ=0.1 is the controlled null result:
  the penalty scale did not move the call-or-not policy of any model.
