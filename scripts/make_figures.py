"""Generate research figures for TinyToolRL.

Figure 1: Observation Phenomenon (Step F)
- Shows c(x) vs Tool Call Rate (TCR) on pi_0 (frozen SFT).
- Demonstrates unnecessary tool usage on capable items (c >= 2/3)
  and under-use on incapable items (c = 0).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

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


def plot_observation_figure(summary_path: str = "eval_results/sft_summary.json", save_path: str = "figures/fig1_observation.png"):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_cap = summary["by_capability"]
    c_vals = [0.0, 0.333, 0.667, 1.0]
    c_keys = [str(round(c, 3)) for c in c_vals]
    
    tcrs = [by_cap[k]["tool_call_rate"] * 100 for k in c_keys]
    accs = [by_cap[k]["accuracy"] * 100 for k in c_keys]
    counts = [by_cap[k]["count"] for k in c_keys]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color_tcr = "#D9534F"  # Coral red
    color_acc = "#0275D8"  # Slate blue
    bar_color = "#E9ECEF"  # Soft gray

    x = np.arange(len(c_vals))
    width = 0.45

    # Sample counts as background bars
    ax2 = ax1.twinx()
    bars = ax2.bar(x, counts, width=width, color=bar_color, alpha=0.6, label="Sample Count")
    ax2.set_ylabel("Sample Count", color="#6C757D")
    ax2.tick_params(axis="y", labelcolor="#6C757D")
    ax2.grid(False)

    # Plot TCR curve
    line1 = ax1.plot(x, tcrs, marker="o", markersize=8, linewidth=2.5, color=color_tcr, label="Tool Call Rate (TCR %)")
    for i, txt in enumerate(tcrs):
        ax1.annotate(f"{txt:.1f}%", (x[i], tcrs[i] + 3), ha="center", fontsize=10, fontweight="bold", color=color_tcr)

    # Annotate unnecessary tool use region
    ax1.axvspan(1.5, 3.3, color="#FFECEC", alpha=0.5, label="Unnecessary Tool Use Zone (c >= 2/3)")
    ax1.annotate("Unnecessary Tool Use\n(13.4% on c=1.0)", xy=(3, tcrs[3]), xytext=(2.2, 42),
                 arrowprops=dict(arrowstyle="->", color=color_tcr, lw=1.5),
                 fontsize=10, color=color_tcr, fontweight="semibold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3F3", ec=color_tcr, lw=1))

    ax1.set_xlabel(r"Model Capability without Tools $c(x)$", labelpad=8)
    ax1.set_ylabel("Rate (%)", color="#333333")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c:.2f}\n(n={counts[i]})" for i, c in enumerate(c_vals)])

    # Combined legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.9)

    plt.title(r"Observation: $\pi_0$ Tool Usage vs. Inherent Capability $c(x)$", pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Observation figure saved to {save_path}")


def plot_selective_tool_use_figure(eval_dir: str = "eval_results", split: str = "eval", save_path: str = "figures/fig2_selective_tool_use.png"):
    """Figure 2: Selective Tool Use across 4 models."""
    models = ["SFT", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap"]
    labels = [r"SFT ($\pi_0$)", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap (Ours)"]
    colors = ["#6C757D", "#E67E22", "#D9534F", "#27AE60"]

    incap_tcrs = []
    cap_tcrs = []
    accs = []

    valid_models = []
    valid_labels = []
    valid_colors = []

    for m, lbl, col in zip(models, labels, colors):
        path = os.path.join(eval_dir, f"{m}_{split}_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        groups = data.get("capability_groups", {})
        incap = groups.get("Incapable (c<=1/3)", {}).get("tool_call_rate", 0.0) * 100
        cap = groups.get("Capable (c>=2/3)", {}).get("tool_call_rate", 0.0) * 100
        acc = data.get("accuracy", 0.0) * 100

        incap_tcrs.append(incap)
        cap_tcrs.append(cap)
        accs.append(acc)
        valid_models.append(m)
        valid_labels.append(lbl)
        valid_colors.append(col)

    if not valid_models:
        print("No evaluation summaries found for Figure 2.")
        return

    x = np.arange(len(valid_models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    bars_incap = ax.bar(x - width/2, incap_tcrs, width, label="Incapable (c <= 1/3) TCR % [Necessary]", color="#2980B9", alpha=0.9)
    bars_cap = ax.bar(x + width/2, cap_tcrs, width, label="Capable (c >= 2/3) TCR % [Unnecessary]", color="#E74C3C", alpha=0.9)

    for bar in bars_incap:
        yval = bar.get_height()
        ax.annotate(f"{yval:.1f}%", xy=(bar.get_x() + bar.get_width()/2, yval + 1.5), ha="center", fontsize=10, fontweight="bold")

    for bar in bars_cap:
        yval = bar.get_height()
        ax.annotate(f"{yval:.1f}%", xy=(bar.get_x() + bar.get_width()/2, yval + 1.5), ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Tool Call Rate (%)", labelpad=8)
    ax.set_title(f"Selective Tool Use by Problem Capability ({split.upper()} set)", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Selective tool use figure saved to {save_path}")


def plot_pareto_frontier(eval_dir: str = "eval_results", split: str = "eval", save_path: str = "figures/fig3_pareto_frontier.png"):
    """Figure 3: Accuracy vs Efficiency Trade-off."""
    models = ["SFT", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap"]
    labels = [r"SFT ($\pi_0$)", "GRPO-Outcome", "GRPO-Fixed", "GRPO-Cap (Ours)"]
    colors = ["#7F8C8D", "#E67E22", "#D9534F", "#27AE60"]
    markers = ["s", "^", "D", "o"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    pts = []
    for m, lbl, col, mark in zip(models, labels, colors, markers):
        path = os.path.join(eval_dir, f"{m}_{split}_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        acc = data.get("accuracy", 0.0) * 100
        tcr = data.get("tool_call_rate", 0.0) * 100
        pts.append((tcr, acc, lbl, col, mark))
        ax.scatter([tcr], [acc], color=col, marker=mark, s=150, zorder=5, label=lbl)
        ax.annotate(f"{lbl}\n({acc:.1f}%, {tcr:.1f}% TCR)", xy=(tcr, acc), xytext=(tcr + 0.3, acc + 0.1),
                    fontsize=9, fontweight="semibold", color=col)

    if not pts:
        print("No evaluation summaries found for Figure 3.")
        return

    ax.set_xlabel(r"Tool Call Rate (%) [Cost $\leftarrow$ lower is better]", labelpad=8)
    ax.set_ylabel(r"Overall Accuracy (%) [$\uparrow$ higher is better]", labelpad=8)
    ax.set_title(f"Accuracy vs. Tool Usage Trade-off ({split.upper()} set)", pad=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Pareto frontier figure saved to {save_path}")


if __name__ == "__main__":
    plot_observation_figure()
    plot_selective_tool_use_figure()
    plot_pareto_frontier()

