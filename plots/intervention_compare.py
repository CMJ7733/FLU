"""
plots/intervention_compare.py — 防控方案效果对比可视化
=======================================================
输出图表：
    1. 多方案流行曲线对比折线图
    2. 干预效果热力图（方案 × 指标）
    3. 成本-效益散点图（Pareto 前沿）
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ._style import setup_style, COLORS, PALETTE_INTERVENTIONS


def plot_scenario_comparison(
    sim_results: dict[str, pd.DataFrame],
    metric:      str = "I_rate",
    title:       str = "防控方案对比",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    多场景/方案流行曲线对比折线图。

    Args:
        sim_results: {方案名称: solve_seiqr 输出 DataFrame}
        metric:      绘制指标列名（I_rate / I_total / attack_rate）
        title:       图标题
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=14, fontweight="bold")

    colors = PALETTE_INTERVENTIONS + list(COLORS.values())

    for i, (name, df) in enumerate(sim_results.items()):
        c = colors[i % len(colors)]
        lw = 2.5 if "无干预" in name or "基准" in name else 2.0
        ls = "--" if "无干预" in name or "基准" in name else "-"
        ax.plot(df["t"], df[metric], color=c, linewidth=lw,
                linestyle=ls, label=name, alpha=0.9)

    # 坐标轴格式
    if "rate" in metric:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
        ax.set_ylabel("感染率")
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_ylabel("感染人数")

    ax.set_xlabel("模拟天数（天）")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_intervention_heatmap(
    results_df:  pd.DataFrame,
    row_labels:  list[str] | None = None,
    title:       str = "防控方案效果热力图",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    防控方案效果热力图（方案 × 评价指标）。

    Args:
        results_df: 含方案名和评价指标的 DataFrame
                    期望列：scheme_name, AR_reduction_pct, PIP_reduction_pct, cost_score, F_objective
        row_labels: 行标签（方案名），None 时取 scheme_name 列或行索引
    """
    setup_style()

    # 选取关键指标列
    metric_cols = {
        "AR_reduction_pct":  "发病率\n降低(%)",
        "PIP_reduction_pct": "峰值感染\n降低(%)",
        "cost_score":        "成本\n评分",
        "F_objective":       "综合\n目标值",
    }
    available = [c for c in metric_cols if c in results_df.columns]
    if not available:
        available = [c for c in results_df.columns
                     if results_df[c].dtype in [np.float64, np.int64]][:4]

    data = results_df[available].values.astype(float)

    # 每列归一化至 [0,1]
    data_norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        vmin, vmax = col.min(), col.max()
        if vmax > vmin:
            data_norm[:, j] = (col - vmin) / (vmax - vmin)
        else:
            data_norm[:, j] = 0.5

    if row_labels is None:
        if "scheme_name" in results_df.columns:
            row_labels = results_df["scheme_name"].tolist()
        else:
            row_labels = [f"方案 {i+1}" for i in range(len(results_df))]

    col_labels = [metric_cols.get(c, c) for c in available]

    n_rows = min(len(row_labels), 15)
    fig_h  = max(4, n_rows * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.8), fig_h))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # "成本"列应使用反转色图（越低越好，显示为绿色）
    # 统一用 RdYlGn（数值越大越绿/越好）
    # 对成本列：先反转归一化
    cost_idx = [i for i, c in enumerate(available) if "cost" in c or "F_obj" in c]
    for idx in cost_idx:
        data_norm[:, idx] = 1.0 - data_norm[:, idx]

    im = ax.imshow(data_norm[:n_rows], cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=1)

    # 数值标注
    for i in range(n_rows):
        for j in range(len(available)):
            raw_val = data[i, j]
            txt = f"{raw_val:.3f}" if abs(raw_val) < 10 else f"{raw_val:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < data_norm[i, j] < 0.7 else "white")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels[:n_rows], fontsize=9)

    plt.colorbar(im, ax=ax, label="归一化效果（绿色=更优）",
                 fraction=0.04, pad=0.02)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_pareto_frontier(
    results_df:  pd.DataFrame,
    x_col:       str = "cost_score",
    y_col:       str = "AR_reduction_pct",
    label_col:   str | None = None,
    title:       str = "成本-效益 Pareto 前沿",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    绘制成本-效益散点图，标出 Pareto 前沿。

    Args:
        x_col: 成本轴列名（越小越好）
        y_col: 效益轴列名（越大越好）
    """
    setup_style()

    df = results_df.dropna(subset=[x_col, y_col]).copy()
    x = df[x_col].values
    y = df[y_col].values

    # 识别 Pareto 前沿点（非被支配点）
    is_pareto = _pareto_mask(x, y, minimize_x=True, maximize_y=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 所有点
    ax.scatter(x[~is_pareto], y[~is_pareto],
               c=COLORS["baseline"], s=40, alpha=0.5, label="非 Pareto 方案")

    # Pareto 前沿
    par_x = x[is_pareto]
    par_y = y[is_pareto]
    sort_idx = np.argsort(par_x)
    ax.scatter(par_x, par_y,
               c=COLORS["danger"], s=80, zorder=5, label="Pareto 最优方案")
    ax.plot(par_x[sort_idx], par_y[sort_idx],
            color=COLORS["danger"], linewidth=1.5, linestyle="--", alpha=0.7)

    # 标注 Pareto 点
    if label_col and label_col in df.columns:
        labels = df[label_col].values
        for xi, yi, lbl in zip(par_x, par_y, labels[is_pareto]):
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=COLORS["danger"])

    x_label_map = {"cost_score": "综合成本评分", "F_objective": "目标函数值"}
    y_label_map = {"AR_reduction_pct": "累计发病率降低（%）", "PIP_reduction_pct": "峰值感染降低（%）"}

    ax.set_xlabel(x_label_map.get(x_col, x_col), fontsize=12)
    ax.set_ylabel(y_label_map.get(y_col, y_col), fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_before_after_intervention(
    comparison:  dict,
    metric:      str = "I_rate",
    title:       str = "最优防控方案干预前后对比",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    干预前后感染曲线对比图（每个场景一个子图）。

    Args:
        comparison: 结构 = {
            scenario_name: {
                "before":  DataFrame (solve_seiqr 输出，无干预),
                "after":   DataFrame (solve_seiqr 输出，施加最优方案),
                "bundle":  InterventionBundle,
                "ar_before": float,
                "ar_after":  float,
                "peak_before": float,
                "peak_after":  float,
            }
        }
        metric: 绘制指标列名（默认 I_rate）
    """
    setup_style()

    names = list(comparison.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    axes = axes[0]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for ax, name in zip(axes, names):
        d = comparison[name]
        df_b, df_a = d["before"], d["after"]

        ax.plot(df_b["t"], df_b[metric],
                color=COLORS["baseline"], linewidth=2.2, linestyle="--",
                label="无干预", alpha=0.9)
        ax.plot(df_a["t"], df_a[metric],
                color=COLORS["danger"], linewidth=2.5, linestyle="-",
                label="最优方案", alpha=0.95)

        # 峰值标注
        ib = int(df_b[metric].values.argmax())
        ia = int(df_a[metric].values.argmax())
        ax.scatter([df_b["t"].iloc[ib]], [df_b[metric].iloc[ib]],
                   color=COLORS["baseline"], s=40, zorder=5)
        ax.scatter([df_a["t"].iloc[ia]], [df_a[metric].iloc[ia]],
                   color=COLORS["danger"], s=50, zorder=5)

        # 指标文本框
        ar_b, ar_a = d["ar_before"], d["ar_after"]
        pk_b, pk_a = d["peak_before"], d["peak_after"]
        ar_red = (1 - ar_a / max(ar_b, 1e-9)) * 100
        pk_red = (1 - pk_a / max(pk_b, 1e-9)) * 100
        bundle = d["bundle"]
        u_text = (
            f"u = ({bundle.mask_level:.1f}, {bundle.ventilation:.1f}, "
            f"{bundle.vaccination:.1f}, {bundle.isolation_rate:.1f}, "
            f"{bundle.online_teaching:.1f}, {bundle.activity_limit:.1f}, "
            f"{bundle.disinfection:.1f})"
        )
        info = (
            f"AR:   {ar_b:.1%} → {ar_a:.1%}  (↓{ar_red:.1f}%)\n"
            f"峰值: {pk_b:.1%} → {pk_a:.1%}  (↓{pk_red:.1f}%)\n"
            f"{u_text}"
        )
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="white", edgecolor=COLORS["baseline"], alpha=0.9))

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("模拟天数（天）")
        if "rate" in metric:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
            ax.set_ylabel("感染率")
        else:
            ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _pareto_mask(
    x: np.ndarray,
    y: np.ndarray,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """返回 Pareto 前沿点的布尔掩码。"""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j 支配 i：x更小（或等）且 y 更大（或等），且至少有一个严格更好
            x_dom = (x[j] <= x[i]) if minimize_x else (x[j] >= x[i])
            y_dom = (y[j] >= y[i]) if maximize_y else (y[j] <= y[i])
            x_str = (x[j] < x[i]) if minimize_x else (x[j] > x[i])
            y_str = (y[j] > y[i]) if maximize_y else (y[j] < y[i])
            if x_dom and y_dom and (x_str or y_str):
                is_pareto[i] = False
                break
    return is_pareto
