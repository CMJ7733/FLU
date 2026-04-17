"""
plots/sensitivity_plot.py — 敏感性分析可视化
==============================================
输出图表：
    1. PRCC Tornado 龙卷风水平条形图（按 |PRCC| 排序）
    2. R₀ 参数热力图（β₀ × γ 二维扫描）
    3. 单参数 R₀ 敏感性折线图
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ._style import setup_style, COLORS


def plot_prcc_tornado(
    prcc_df:     pd.DataFrame,
    output_name: str = "peak_infection_rate",
    title:       str = "PRCC 敏感性分析（龙卷风图）",
    output_path: str | Path | None = None,
    show:        bool = False,
    threshold:   float = 0.05,
) -> plt.Figure:
    """
    绘制 PRCC Tornado 水平条形图。

    Args:
        prcc_df:     PRCC 分析结果 DataFrame（含 parameter, PRCC, p_value, label）
        output_name: 目标量名称（用于标题）
        threshold:   显著性阈值（|PRCC| > threshold 才显示）
        output_path: 保存路径
        show:        是否显示

    Returns:
        matplotlib Figure
    """
    setup_style()

    # 过滤不显著的参数
    df = prcc_df.copy()
    df = df[df["PRCC"].abs() >= threshold].copy()
    df = df.sort_values("PRCC", ascending=True).reset_index(drop=True)

    if len(df) == 0:
        df = prcc_df.sort_values("PRCC", ascending=True).reset_index(drop=True)

    # 标签优先使用中文描述
    label_col = "label" if "label" in df.columns else "parameter"
    labels = df[label_col].fillna(df["parameter"]).tolist()

    colors = [COLORS["danger"] if v > 0 else COLORS["accent1"]
              for v in df["PRCC"]]

    # 显著性标记
    sig_markers = []
    for _, row in df.iterrows():
        p = row.get("p_value", 1.0)
        if p < 0.001:
            sig_markers.append("***")
        elif p < 0.01:
            sig_markers.append("**")
        elif p < 0.05:
            sig_markers.append("*")
        else:
            sig_markers.append("")

    fig_h = max(5, len(df) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    output_label_map = {
        "peak_infection_rate": "峰值感染率",
        "total_attack_rate":   "累计发病率",
        "peak_timing_days":    "峰值时间",
    }
    out_zh = output_label_map.get(output_name, output_name)
    ax.set_title(f"{title}\n目标量：{out_zh}", fontsize=13, fontweight="bold")

    bars = ax.barh(range(len(df)), df["PRCC"], color=colors, alpha=0.8,
                   edgecolor="white", linewidth=0.5)

    # 数值标注
    for i, (bar, sig) in enumerate(zip(bars, sig_markers)):
        w = bar.get_width()
        x_text = w + 0.01 * np.sign(w) if abs(w) > 0.05 else 0.01
        ax.text(
            x_text, i, f"{w:+.3f}{sig}",
            va="center", ha="left" if w >= 0 else "right",
            fontsize=9, color="gray"
        )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PRCC 系数（偏秩相关系数）")
    ax.set_xlim(-1.05, 1.05)

    # 图例
    pos_patch = plt.Rectangle((0, 0), 1, 1, color=COLORS["danger"],  alpha=0.8, label="正向影响")
    neg_patch = plt.Rectangle((0, 0), 1, 1, color=COLORS["accent1"], alpha=0.8, label="负向影响")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    # 显著性注释
    ax.text(0.01, -0.06,
            "显著性: * p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_r0_heatmap(
    Z:           np.ndarray,
    x_vals:      np.ndarray,
    y_vals:      np.ndarray,
    param_x:     str = "beta0",
    param_y:     str = "gamma",
    title:       str = "R₀ 参数敏感性热力图",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    绘制 R₀ 二维参数热力图。

    Args:
        Z:       R₀ 矩阵，形状 (len(y_vals), len(x_vals))
        x_vals:  x 轴参数值数组
        y_vals:  y 轴参数值数组
        param_x: x 轴参数名
        param_y: y 轴参数名
    """
    setup_style()

    PARAM_LABELS = {
        "beta0":  "基础传播系数 β₀",
        "gamma":  "康复率 γ",
        "sigma":  "潜伏转感染率 σ",
        "alpha":  "病例隔离率 α",
        "delta1": "年季节振幅 δ₁",
        "c11":    "学生接触率 c₁₁",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 以 R₀=1 为分界，使用分散色图
    vmin = min(Z.min(), 0.5)
    vmax = max(Z.max(), 2.5)
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(
        Z, cmap="RdYlBu_r", norm=norm,
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        aspect="auto", origin="lower",
    )

    # R₀=1 等高线
    cs = ax.contour(
        x_vals, y_vals, Z,
        levels=[1.0], colors=["black"], linewidths=[2.0]
    )
    ax.clabel(cs, fmt="R₀=1", fontsize=10, inline=True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("基本再生数 R₀", fontsize=11)
    cbar.ax.axhline(y=1.0, color="black", linewidth=1.5)

    ax.set_xlabel(PARAM_LABELS.get(param_x, param_x), fontsize=12)
    ax.set_ylabel(PARAM_LABELS.get(param_y, param_y), fontsize=12)

    # 标注当前参数值位置
    ax.text(0.98, 0.02, "黑线: R₀=1 临界线",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="black")

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_r0_sensitivity_line(
    param_values: np.ndarray,
    r0_values:    list,
    hit_values:   list,
    param_label:  str = "参数",
    base_val:     float | None = None,
    output_path:  str | Path | None = None,
    show:         bool = False,
) -> plt.Figure:
    """绘制单参数 R₀ 敏感性折线图。"""
    setup_style()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title(f"R₀ 对 {param_label} 的敏感性", fontsize=13, fontweight="bold")

    ax1.plot(param_values, r0_values, color=COLORS["danger"],
             linewidth=2.2, label="R₀")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="R₀=1 临界线")

    if base_val is not None:
        ax1.axvline(base_val, color=COLORS["accent1"], linestyle=":",
                    linewidth=1.5, label=f"基准值={base_val:.3f}")

    ax1.set_xlabel(param_label, fontsize=12)
    ax1.set_ylabel("基本再生数 R₀", fontsize=12, color=COLORS["danger"])
    ax1.tick_params(axis="y", colors=COLORS["danger"])

    # 群体免疫阈值（右轴）
    ax2 = ax1.twinx()
    ax2.plot(param_values, [h * 100 for h in hit_values],
             color=COLORS["accent2"], linestyle="--", linewidth=1.8, label="HIT（群体免疫阈值）")
    ax2.set_ylabel("群体免疫阈值 HIT (%)", fontsize=12, color=COLORS["accent2"])
    ax2.tick_params(axis="y", colors=COLORS["accent2"])
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
