"""Generate research figures for TinyToolRL.

Figure 1: Observation Phenomenon (Step F)
- c(x) vs Tool Call Rate (TCR) on pi_0 (frozen SFT); shows unnecessary use on
  capable items and under-use on incapable items.

Figure 2: Tool policy by capability group at lambda=0.1 (controlled null result)
- Grouped incapable/capable TCR bars for the 4 policies, WITH Wilson 95% CIs.
  Honest subtitle: at this penalty scale all policies coincide.

Figure 3: Accuracy vs Tool-Usage trade-off (eval set)
- Coincident points are merged into a single marker with a combined label
  (fixes the old overlapping-label defect); Wilson CI error bars added.

Figure 4: Lambda sweep (Fixed vs Capability-aware), if the sweep summaries exist
- Left: capable / incapable TCR vs lambda. Right: overall accuracy vs lambda.
  This is the figure that tests the §6.1 divergence hypothesis with real runs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from analyze_stats import wilson

os.makedirs("figures", exist_ok=True)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
})

MODELS = ["SFT", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap"]
LABELS = [r"SFT ($\pi_0$)", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap (Ours)"]


def load_summary(eval_dir: str, tag: str, split: str):
    path = os.path.join(eval_dir, f"{tag}_{split}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_observation_figure(summary_path: str = "eval_results/sft_summary.json", save_path: str = "figures/fig1_observation.png"):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_cap = summary["by_capability"]
    c_vals = [0.0, 0.333, 0.667, 1.0]
    c_keys = [str(round(c, 3)) for c in c_vals]

    tcrs = [by_cap[k]["tool_call_rate"] * 100 for k in c_keys]
    counts = [by_cap[k]["count"] for k in c_keys]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color_tcr = "#D9534F"  # Coral red
    bar_color = "#E9ECEF"  # Soft gray

    x = np.arange(len(c_vals))
    width = 0.45

    # Sample counts as background bars
    ax2 = ax1.twinx()
    ax2.bar(x, counts, width=width, color=bar_color, alpha=0.6, label="Sample Count")
    ax2.set_ylabel("Sample Count", color="#6C757D")
    ax2.tick_params(axis="y", labelcolor="#6C757D")
    ax2.grid(False)

    # TCR curve with Wilson 95% CI error bars
    errs_lo, errs_hi = [], []
    for k, c in zip([by_cap[key]["count"] for key in c_keys],
                    [by_cap[key]["tool_call_rate"] for key in c_keys]):
        lo, hi = wilson(int(round(c * k)), k)
        errs_lo.append(c * 100 - lo * 100)
        errs_hi.append(hi * 100 - c * 100)
    ax1.errorbar(x, tcrs, yerr=[errs_lo, errs_hi], marker="o", markersize=8,
                 linewidth=2.5, capsize=4, color=color_tcr,
                 label="Tool Call Rate (TCR %) [Wilson 95% CI]")
    for i, txt in enumerate(tcrs):
        ax1.annotate(f"{txt:.1f}%", (x[i], tcrs[i] + 6), ha="center", fontsize=10,
                     fontweight="bold", color=color_tcr)

    # Annotate unnecessary tool use region
    ax1.axvspan(1.5, 3.3, color="#FFECEC", alpha=0.5, label="Unnecessary Tool Use Zone (c >= 2/3)")
    ax1.annotate("Unnecessary Tool Use\n(13.4% on c=1.0, n=97)", xy=(3, tcrs[3]), xytext=(2.05, 45),
                 arrowprops=dict(arrowstyle="->", color=color_tcr, lw=1.5),
                 fontsize=10, color=color_tcr, fontweight="semibold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3F3", ec=color_tcr, lw=1))

    ax1.set_xlabel(r"Model Capability without Tools $c(x)$", labelpad=8)
    ax1.set_ylabel("Rate (%)", color="#333333")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c:.2f}\n(n={counts[i]})" for i, c in enumerate(c_vals)])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.9)

    plt.title(r"Observation: $\pi_0$ Tool Usage vs. Inherent Capability $c(x)$", pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Observation figure saved to {save_path}")


def plot_selective_tool_use_figure(eval_dir: str = "eval_results", split: str = "eval",
                                   save_path: str = "figures/fig2_selective_tool_use.png"):
    """Figure 2: capability-grouped TCR for the 4 policies at lambda=0.1, with CIs."""
    incap_tcrs, cap_tcrs, incap_errs, cap_errs = [], [], [], []
    valid_labels = []

    for m, lbl in zip(MODELS, LABELS):
        data = load_summary(eval_dir, m, split)
        if data is None:
            continue
        groups = data.get("capability_groups", {})
        inc = groups.get("Incapable (c<=1/3)", {})
        cap = groups.get("Capable (c>=2/3)", {})
        ni, nc = inc.get("count", 0), cap.get("count", 0)
        ri, rc = inc.get("tool_call_rate", 0.0), cap.get("tool_call_rate", 0.0)
        lo_i, hi_i = wilson(int(round(ri * ni)), ni)
        lo_c, hi_c = wilson(int(round(rc * nc)), nc)
        incap_tcrs.append(ri * 100)
        cap_tcrs.append(rc * 100)
        incap_errs.append((ri * 100 - lo_i * 100, hi_i * 100 - ri * 100))
        cap_errs.append((rc * 100 - lo_c * 100, hi_c * 100 - rc * 100))
        valid_labels.append(lbl)

    if not valid_labels:
        print("No evaluation summaries found for Figure 2.")
        return

    x = np.arange(len(valid_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)

    bi = ax.bar(x - width / 2, incap_tcrs, width, yerr=[[e[0] for e in incap_errs], [e[1] for e in incap_errs]],
                capsize=4, label="Incapable (c <= 1/3) TCR % [necessary calls]", color="#2980B9", alpha=0.9)
    bc = ax.bar(x + width / 2, cap_tcrs, width, yerr=[[e[0] for e in cap_errs], [e[1] for e in cap_errs]],
                capsize=4, label="Capable (c >= 2/3) TCR % [redundant calls]", color="#E74C3C", alpha=0.9)

    for bar in list(bi) + list(bc):
        yval = bar.get_height()
        ax.annotate(f"{yval:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, yval + 6),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Tool Call Rate (%)", labelpad=8)
    ax.set_title(f"Tool Policy by Capability Group ({split.upper()} set, $\\lambda$=0.1)\n"
                 r"Controlled null result: all four policies coincide (0 tool-decision flips, per-sample audit)",
                 pad=12, fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Selective tool use figure saved to {save_path}")


def plot_pareto_frontier(eval_dir: str = "eval_results", split: str = "eval",
                         save_path: str = "figures/fig3_pareto_frontier.png"):
    """Figure 3: Accuracy vs Efficiency; coincident points merged, CIs shown."""
    colors = {"SFT": "#7F8C8D", "GRPO-Outcome": "#E67E22", "GRPO-Fixed": "#D9534F", "GRPO-Cap": "#27AE60"}
    markers = {"SFT": "s", "GRPO-Outcome": "^", "GRPO-Fixed": "D", "GRPO-Cap": "o"}

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)

    merged = {}
    for m, lbl in zip(MODELS, LABELS):
        data = load_summary(eval_dir, m, split)
        if data is None:
            continue
        n = data.get("total", 0)
        acc = data.get("accuracy", 0.0) * 100
        tcr = data.get("tool_call_rate", 0.0) * 100
        key = (round(tcr, 2), round(acc, 2))
        merged.setdefault(key, {"labels": [], "models": [], "n": n})
        merged[key]["labels"].append(lbl)
        merged[key]["models"].append(m)

    if not merged:
        print("No evaluation summaries found for Figure 3.")
        return

    for (tcr, acc), info in merged.items():
        n = info["n"]
        ka, kt = int(round(acc / 100 * n)), int(round(tcr / 100 * n))
        alo, ahi = wilson(ka, n)
        tlo, thi = wilson(kt, n)
        m0 = info["models"][0]
        ax.errorbar([tcr], [acc],
                    xerr=[[tcr - tlo * 100], [thi * 100 - tcr]],
                    yerr=[[acc - alo * 100], [ahi * 100 - acc]],
                    color=colors[m0], marker=markers[m0], markersize=12, capsize=4,
                    linewidth=1.5, zorder=5, label=None)
        label_txt = " = ".join(info["labels"]) if len(info["labels"]) > 1 else info["labels"][0]
        suffix = f" ({len(info['labels'])} policies coincide)" if len(info["labels"]) > 1 else ""
        # place labels outside the CI whiskers so they never cross error bars
        if acc >= 94.5:
            ytxt, va = ahi * 100 + 0.2, "bottom"
        else:
            ytxt, va = alo * 100 - 0.2, "top"
        ax.annotate(f"{label_txt}{suffix}\n({acc:.2f}% acc, {tcr:.2f}% TCR)",
                    xy=(tcr, acc), xytext=(tcr + 0.5, ytxt), ha="left", va=va,
                    fontsize=9, fontweight="semibold", color=colors[m0])

    ax.set_xlabel(r"Tool Call Rate (%) [Cost $\leftarrow$ lower is better]", labelpad=8)
    ax.set_ylabel(r"Overall Accuracy (%) [$\uparrow$ higher is better]", labelpad=8)
    ax.set_title(f"Accuracy vs. Tool Usage Trade-off ({split.upper()} set, Wilson 95% CIs)\n"
                 "the three GRPO variants coincide at $\\lambda$=0.1", pad=12,
                 fontweight="bold", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Pareto frontier figure saved to {save_path}")


def plot_lambda_sweep(eval_dir: str = "eval_results", split: str = "eval",
                      save_path: str = "figures/fig4_lambda_sweep.png"):
    """Figure 4: Fixed vs Capability-aware across lambda in {0.1, 0.3, 0.5}."""
    lam_tags = [(0.1, {"fixed": "GRPO-Fixed", "cap": "GRPO-Cap"}),
                (0.3, {"fixed": "GRPO-Fixed-lam03", "cap": "GRPO-Cap-lam03"}),
                (0.5, {"fixed": "GRPO-Fixed-lam05", "cap": "GRPO-Cap-lam05"})]

    series = {("fixed", "cap_tcr"): [], ("cap", "cap_tcr"): [],
              ("fixed", "inc_tcr"): [], ("cap", "inc_tcr"): [],
              ("fixed", "acc"): [], ("cap", "acc"): []}
    lams = {k: [] for k in series}

    for lam, tags in lam_tags:
        for mode in ("fixed", "cap"):
            data = load_summary(eval_dir, tags[mode], split)
            if data is None:
                continue
            groups = data.get("capability_groups", {})
            cap_t = groups.get("Capable (c>=2/3)", {}).get("tool_call_rate")
            inc_t = groups.get("Incapable (c<=1/3)", {}).get("tool_call_rate")
            acc = data.get("accuracy")
            if cap_t is None or inc_t is None or acc is None:
                continue
            for key, val in ((("cap_tcr"), cap_t * 100), (("inc_tcr"), inc_t * 100), (("acc"), acc * 100)):
                series[(mode, key)].append(val)
                lams[(mode, key)].append(lam)

    if not any(series.values()):
        print("No lambda-sweep summaries found yet for Figure 4 (skipping).")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)

    style = {("fixed", "cap_tcr"): ("#D9534F", "o", "Fixed: capable-group TCR"),
             ("cap", "cap_tcr"): ("#27AE60", "o", "Cap (Ours): capable-group TCR"),
             ("fixed", "inc_tcr"): ("#D9534F", "s", "Fixed: incapable-group TCR"),
             ("cap", "inc_tcr"): ("#27AE60", "s", "Cap (Ours): incapable-group TCR")}
    for key, (col, mark, lbl) in style.items():
        if series[key]:
            ax1.plot(lams[key], series[key], marker=mark, color=col, linewidth=2,
                     markersize=8, label=lbl,
                     linestyle="--" if "inc" in key[1] else "-")
    ax1.set_xlabel(r"Penalty scale $\lambda$")
    ax1.set_ylabel("Tool Call Rate (%)")
    ax1.set_title("Selective pressure: capable vs incapable", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    sft = load_summary(eval_dir, "SFT", split)
    if sft is not None:
        ax2.axhline(sft["accuracy"] * 100, color="#7F8C8D", linestyle=":",
                    label=r"SFT ($\pi_0$) baseline")
    for mode, col, lbl in (("fixed", "#D9534F", "GRPO-Fixed"), ("cap", "#27AE60", "GRPO-Cap (Ours)")):
        key = (mode, "acc")
        if series[key]:
            ax2.plot(lams[key], series[key], marker="o", color=col, linewidth=2,
                     markersize=8, label=lbl)
    ax2.set_xlabel(r"Penalty scale $\lambda$")
    ax2.set_ylabel("Overall Accuracy (%)")
    ax2.set_title("Accuracy under increasing penalty", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Lambda sweep on {split.upper()}: where (if anywhere) Fixed and Capability-aware diverge",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Lambda sweep figure saved to {save_path}")


if __name__ == "__main__":
    plot_observation_figure()
    plot_selective_tool_use_figure()
    plot_pareto_frontier()
    plot_lambda_sweep()
