# TinyToolRL 4-Model Comparative Evaluation

### Split: eval (data/eval.jsonl)

| Model | Overall Acc (%) | Tool Call Rate (%) | Incapable (c<=1/3) TCR (%) | Capable (c>=2/3) TCR (%) | Format Rate (%) |
|---|---|---|---|---|---|
| SFT | 94.00 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Outcome | 94.67 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Fixed | 94.67 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Cap | 94.67 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Fixed-lam03 | 95.33 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Cap-lam03 | 95.33 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Fixed-lam05 | 94.67 | 33.33 | 77.78 | 14.29 | 100.00 |
| GRPO-Cap-lam05 | 94.67 | 33.33 | 77.78 | 14.29 | 100.00 |


### Split: eval_ood (data/eval_ood.jsonl)

| Model | Overall Acc (%) | Tool Call Rate (%) | Incapable (c<=1/3) TCR (%) | Capable (c>=2/3) TCR (%) | Format Rate (%) |
|---|---|---|---|---|---|
| SFT | 96.67 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Outcome | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Fixed | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Cap | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Fixed-lam03 | 98.33 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Cap-lam03 | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Fixed-lam05 | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |
| GRPO-Cap-lam05 | 99.17 | 50.00 | 89.58 | 23.61 | 100.00 |


### Split: eval_ratio20 (data/eval_ratio20.jsonl)

| Model | Overall Acc (%) | Tool Call Rate (%) | Incapable (c<=1/3) TCR (%) | Capable (c>=2/3) TCR (%) | Format Rate (%) |
|---|---|---|---|---|---|
| SFT | 93.33 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Outcome | 95.00 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Fixed | 95.83 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Cap | 95.83 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Fixed-lam03 | 94.17 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Cap-lam03 | 95.00 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Fixed-lam05 | 95.00 | 20.00 | 60.87 | 10.31 | 100.00 |
| GRPO-Cap-lam05 | 95.00 | 20.00 | 60.87 | 10.31 | 100.00 |

