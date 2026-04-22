# draw.py
# -*- coding: utf-8 -*-

"""
Laminar Artifact Evaluation: Figure Generation Script
Maps directly to the evaluation figures presented in the paper (Fig. 2 - Fig. 5).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/master_evaluation_stream.csv" 
OUT_DIR = "figures_paper"

os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# -------------------------
# Global Style Configurations
# -------------------------
SCHED_ORDER = ["laminar", "slurm", "ray", "flux"]
SCHED_LABEL = {
    "laminar": "Laminar",
    "slurm": "Slurm",
    "ray": "Ray",
    "flux": "Flux",
}
SCHED_COLOR = {
    "laminar": "#1f77b4",
    "slurm": "#d62728",
    "ray": "#2ca02c",
    "flux": "#9467bd",
}
TP_LABEL = {True: "Two-phase", False: "No two-phase"}
REGEN_LABEL = {True: "Regeneration", False: "No regeneration"}

# -------------------------
# Helper functions
# -------------------------
def to_bool(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes"}:
        return True
    if s in {"false", "f", "0", "no"}:
        return False
    return np.nan

def extract_numeric(tag, pattern):
    m = re.search(pattern, str(tag))
    return float(m.group(1)) if m else np.nan

def smart_zoom_ylim(s, floor=None, ceil=None, pad_ratio=0.12):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return None
    vmin, vmax = s.min(), s.max()
    span = vmax - vmin
    pad = max(span * pad_ratio, 1e-5)
    lo, hi = vmin - pad, vmax + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    return (lo, hi)

def clean_df(path):
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        return pd.DataFrame()
        
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "rho",
        "nodes",
        "control_work_per_success_us",
        "execution_starts_per_sec",
        "honest_start_goodput",
        "p99_arrival_to_start_ms",
        "p99_decision_us_measured",
        "start_success_ratio",
        "loss",
        "duplicate_da_count",
        "false_optimism_rate",
        "uniqueness_violation_count",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "scheduler" in df.columns:
        df["scheduler"] = df["scheduler"].astype(str).str.strip().str.lower()
    if "Experiment_Tag" in df.columns:
        df["Experiment_Tag"] = df["Experiment_Tag"].astype(str).str.strip()

    for c in ["enable_two_phase", "enable_regeneration", "enable_missingness_guard"]:
        if c in df.columns:
            df[c] = df[c].map(to_bool)

    return df

def save(fig, name):
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)
    print(f"Generated academic artifact: {pdf}")

def apply_axis_style(ax, *, ylim=None, yticks=None, logy=False):
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=0.28, linewidth=0.8)

def plot_multischedule_line(
    df, x, y, name, xlabel, ylabel, title=None,
    logy=False, ylim=None, yticks=None
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for sched in SCHED_ORDER:
        d = df[df["scheduler"] == sched].copy()
        d = d.dropna(subset=[x, y]).sort_values(x)
        if d.empty:
            continue
        ax.plot(
            d[x],
            d[y],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=SCHED_LABEL.get(sched, sched),
            color=SCHED_COLOR.get(sched, None),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    apply_axis_style(ax, ylim=ylim, yticks=yticks, logy=logy)
    ax.legend(frameon=True)
    save(fig, name)

def plot_single_line(
    df, x, y, name, xlabel, ylabel, title=None,
    color="#1f77b4", ylim=None, yticks=None, logy=False
):
    d = df.dropna(subset=[x, y]).sort_values(x).copy()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(d[x], d[y], marker="o", linewidth=2.4, markersize=6, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    apply_axis_style(ax, ylim=ylim, yticks=yticks, logy=logy)
    save(fig, name)

def plot_boolean_compare(
    df, x, y, flag_col, name, xlabel, ylabel, title=None,
    label_map=None, ylim=None, yticks=None, logy=False
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for flag in [False, True]:
        d = df[df[flag_col] == flag].copy()
        d = d.dropna(subset=[x, y]).sort_values(x)
        if d.empty:
            continue
        label = label_map.get(flag, str(flag)) if label_map else str(flag)
        ax.plot(d[x], d[y], marker="o", linewidth=2.2, markersize=6, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    apply_axis_style(ax, ylim=ylim, yticks=yticks, logy=logy)
    ax.legend(frameon=True)
    save(fig, name)

# -------------------------
# Data Loading
# -------------------------
df = clean_df(CSV_PATH)
if df.empty:
    exit(0)

# =====================================================================
# Figure 2: Mixed-Load Comparison (Exp1)
# =====================================================================
exp1 = df[df["Experiment_Tag"].str.startswith("Exp1_MixedLoad_", na=False)].copy()
exp1 = exp1[exp1["rho"] != 0.95].copy()

plot_multischedule_line(
    exp1,
    x="rho",
    y="start_success_ratio",
    name="Fig2_left_MixedLoad_SuccessRatio",
    xlabel=r"Offered Load $\rho$",
    ylabel="Start success ratio",
    title="Mixed load: success ratio vs. load",
    ylim=(0.0, 1.05),
    yticks=np.linspace(0, 1, 6),
)

plot_multischedule_line(
    exp1,
    x="rho",
    y="p99_arrival_to_start_ms",
    name="Fig2_right_MixedLoad_p99",
    xlabel=r"Offered Load $\rho$",
    ylabel="p99 arrival-to-start latency (ms)",
    title="Mixed load: p99 latency vs. load",
    logy=True,
    ylim=(1e-1, 2e4),
)

# =====================================================================
# Figure 3: Scale-Out Behavior (Exp2 - Laminar only)
# =====================================================================
exp2 = df[
    df["Experiment_Tag"].str.startswith("Exp2_ScaleO1_", na=False)
    & (df["scheduler"] == "laminar")
].copy()

plot_single_line(
    exp2,
    x="nodes",
    y="start_success_ratio",
    name="Fig3_left_ScaleOut_SuccessRatio",
    xlabel="Cluster Size (Nodes)",
    ylabel="Start success ratio",
    title="Scale-out: success ratio vs. nodes",
    ylim=(0.99, 1.001),
    yticks=[0.99, 0.992, 0.994, 0.996, 0.998, 1.00],
)

plot_single_line(
    exp2,
    x="nodes",
    y="p99_arrival_to_start_ms",
    name="Fig3_right_ScaleOut_p99",
    xlabel="Cluster Size (Nodes)",
    ylabel="p99 arrival-to-start latency (ms)",
    title="Scale-out: p99 latency vs. nodes",
    ylim=(8, 18),
    yticks=[8, 10, 12, 14, 16, 18],
)

# =====================================================================
# Figure 4: State Staleness Tolerance (Exp3 - Laminar only)
# =====================================================================
exp3 = df[
    df["Experiment_Tag"].str.startswith("Exp3_Staleness_", na=False)
    & (df["scheduler"] == "laminar")
].copy()

exp3["delay"] = exp3["Experiment_Tag"].apply(
    lambda s: extract_numeric(s, r"delay([0-9.]+)")
)

plot_single_line(
    exp3,
    x="delay",
    y="start_success_ratio",
    name="Fig4_left_Staleness_SuccessRatio",
    xlabel="Z-HAF Sync Delay (ms)",
    ylabel="Start success ratio",
    title="Staleness: Laminar success ratio",
    ylim=(0.9997, 1.0001),
    yticks=[0.9997, 0.9998, 0.9999, 1.0000],
)

plot_single_line(
    exp3,
    x="delay",
    y="p99_arrival_to_start_ms",
    name="Fig4_right_Staleness_p99",
    xlabel="Z-HAF Sync Delay (ms)",
    ylabel="p99 arrival-to-start latency (ms)",
    title="Staleness: Laminar p99 latency",
    ylim=(10.4, 10.6),
    yticks=[10.40, 10.45, 10.50, 10.55, 10.60],
)

# =====================================================================
# Figure 5: Mechanism Ablations (Exp4)
# =====================================================================

exp4b = df[df["Experiment_Tag"].str.startswith("Exp4b_Pending_", na=False)].copy()

exp4b["squatter_x"] = exp4b["Experiment_Tag"].apply(
    lambda s: extract_numeric(s, r"sq_?([0-9.]+)")
)

tp_from_tag = exp4b["Experiment_Tag"].str.extract(r"_TP(True|False)", expand=False)
exp4b["tp_flag"] = tp_from_tag.map(to_bool)

if "enable_two_phase" in exp4b.columns:
    exp4b["tp_flag"] = exp4b["enable_two_phase"].fillna(exp4b["tp_flag"])

exp4b_ylim = smart_zoom_ylim(
    exp4b["start_success_ratio"],
    floor=0.0,
    ceil=1.05,
    pad_ratio=0.12,
)

plot_boolean_compare(
    exp4b,
    x="squatter_x",
    y="start_success_ratio",
    flag_col="tp_flag",
    name="Fig5_left_TwoPhase_Ablation",
    xlabel="Malicious Squatter Ratio",
    ylabel="Start success ratio",
    title="Two-phase reservation effect",
    label_map=TP_LABEL,
    ylim=exp4b_ylim,
)

exp4c = df[df["Experiment_Tag"].str.startswith("Exp4c_Regen_", na=False)].copy()

loss_from_tag = exp4c["Experiment_Tag"].apply(
    lambda s: extract_numeric(s, r"loss([0-9.]+)")
)
regen_from_tag = exp4c["Experiment_Tag"].str.extract(r"_Regen(True|False)", expand=False).map(to_bool)

if "loss" in exp4c.columns:
    exp4c["loss_x"] = exp4c["loss"].fillna(loss_from_tag)
else:
    exp4c["loss_x"] = loss_from_tag

if "enable_regeneration" in exp4c.columns:
    exp4c["regen_flag"] = exp4c["enable_regeneration"].fillna(regen_from_tag)
else:
    exp4c["regen_flag"] = regen_from_tag

exp4c_success_ylim = smart_zoom_ylim(
    exp4c["start_success_ratio"],
    floor=0.0,
    ceil=1.05,
    pad_ratio=0.12,
)

plot_boolean_compare(
    exp4c,
    x="loss_x",
    y="start_success_ratio",
    flag_col="regen_flag",
    name="Fig5_right_Regen_Ablation",
    xlabel="UDP Packet Loss Rate",
    ylabel="Start success ratio",
    title="Regeneration benefit",
    label_map=REGEN_LABEL,
    ylim=exp4c_success_ylim,
)

print("\nArtifact generation complete: All exact figures required by the paper are generated.")