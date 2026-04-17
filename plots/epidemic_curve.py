"""
plots/epidemic_curve.py — 流行曲线可视化
==========================================
输出图表：
    1. 双群体 I(t) 流行曲线（学生/教职工/总计）
    2. 与 FluNet 观测数据叠加的验证图（含 Bootstrap 置信带）
    3. β(t) 季节函数曲线（附学期调制）
    4. S/E/I/Q/R 各仓室全时序图
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ._style import setup_style, COLORS


def plot_epidemic_curve(
    sim_df: pd.DataFrame,
    obs_df:  pd.DataFrame | None = None,
    ci_band: dict | None = None,
    title:   str = "H3N2 校园传播流行曲线",
    output_path: str | Path | None = None,
    show:    bool = False,
) -> plt.Figure:
    """
    绘制流行曲线（I_total / I_rate，双群体分解 + 总计）。

    Args:
        sim_df:      solve_seiqr 输出的 DataFrame
        obs_df:      FluNet 观测数据（含 t_sim, h3n2_pos_rate）
        ci_band:     Bootstrap 置信带 {t, ci_lo, ci_hi, median}
        title:       图表标题
        output_path: 保存路径（None 则不保存）
        show:        是否显示图窗

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    t = sim_df["t"].values
    N1 = sim_df["I1"].iloc[0] + sim_df["S1"].iloc[0] + sim_df["E1"].iloc[0] + \
         sim_df["Q1"].iloc[0] + sim_df["R1"].iloc[0]
    N  = N1 + sim_df["I2"].iloc[0] + sim_df["S2"].iloc[0] + \
         sim_df["E2"].iloc[0] + sim_df["Q2"].iloc[0] + sim_df["R2"].iloc[0]

    # ── 上图：感染人数（绝对值） ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(t, sim_df["I_total"], alpha=0.15, color=COLORS["total"])
    ax1.plot(t, sim_df["I1_total"], color=COLORS["student"], label="学生感染者 (I+Q)")
    ax1.plot(t, sim_df["I2_total"], color=COLORS["staff"],   label="教职工感染者 (I+Q)", linestyle="--")
    ax1.plot(t, sim_df["I_total"],  color=COLORS["total"],   label="合计", linewidth=2.5)

    # 标注峰值
    peak_idx = sim_df["I_total"].idxmax()
    peak_t   = sim_df["t"].iloc[peak_idx]
    peak_I   = sim_df["I_total"].iloc[peak_idx]
    ax1.annotate(
        f"峰值: {peak_I:.0f} 人\n第 {peak_t:.0f} 天",
        xy=(peak_t, peak_I),
        xytext=(peak_t + 5, peak_I * 0.85),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray",
    )

    ax1.set_ylabel("感染人数（I+Q 舱）")
    ax1.legend(loc="upper right")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── 下图：感染率（相对值）+ 观测数据 + CI 带 ─────────────────────────
    ax2 = axes[1]

    # Bootstrap 置信带
    if ci_band is not None and "ci_lo" in ci_band:
        t_ci  = ci_band["t"]
        ax2.fill_between(
            t_ci, ci_band["ci_lo"], ci_band["ci_hi"],
            alpha=0.25, color=COLORS["ci"], label=f"{int(ci_band.get('ci_level',0.95)*100)}% 置信带"
        )

    ax2.plot(t, sim_df["I1_rate"], color=COLORS["student"], label="学生感染率", linewidth=1.8)
    ax2.plot(t, sim_df["I2_rate"], color=COLORS["staff"],   label="教职工感染率", linestyle="--")
    ax2.plot(t, sim_df["I_rate"],  color=COLORS["total"],   label="总感染率", linewidth=2.5)

    # 观测数据散点
    if obs_df is not None and "h3n2_pos_rate" in obs_df.columns:
        t_col = "t_sim" if "t_sim" in obs_df.columns else "t"
        obs_clean = obs_df.dropna(subset=[t_col, "h3n2_pos_rate"])
        ax2.scatter(
            obs_clean[t_col], obs_clean["h3n2_pos_rate"],
            color=COLORS["observed"], s=25, zorder=5,
            label="WHO FluNet 观测值", alpha=0.8,
        )

    ax2.set_xlabel("模拟天数（天）")
    ax2.set_ylabel("感染率（人数/总人口）")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
    ax2.legend(loc="upper right")

    # 标注 β(t) 调制事件
    if "beta_eff" in sim_df.columns:
        _add_holiday_shading(ax2, sim_df["t"].values)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _add_holiday_shading(ax: plt.Axes, t_vals: np.ndarray) -> None:
    """在图上添加寒/暑假阴影区域（按模拟时间轴）。"""
    # 简单示意：第 71–91 天（约1月下旬–2月中旬，从11月1日算起）
    for lo, hi, label in [(71, 91, "寒假"), (236, 282, "暑假")]:
        if lo < t_vals.max():
            ax.axvspan(lo, min(hi, t_vals.max()), alpha=0.08,
                       color="gray", label=f"{label}期间" if lo == 71 else "_")


def plot_seiqr_compartments(
    sim_df:      pd.DataFrame,
    group:       str = "all",
    title:       str = "SEIQR 各仓室时序",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """
    绘制 S/E/I/Q/R 各仓室随时间的变化。

    Args:
        group: "all"/"student"/"staff"
    """
    setup_style()
    t = sim_df["t"].values

    if group == "student":
        cols = {"S": "S1", "E": "E1", "I": "I1", "Q": "Q1", "R": "R1"}
        g_label = "学生群体"
    elif group == "staff":
        cols = {"S": "S2", "E": "E2", "I": "I2", "Q": "Q2", "R": "R2"}
        g_label = "教职工群体"
    else:
        cols = {}
        g_label = "全体"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{title}（{g_label}）", fontsize=14, fontweight="bold")

    compartment_colors = {
        "S": "#2196F3", "E": "#FF9800",
        "I": "#F44336", "Q": "#9C27B0", "R": "#4CAF50"
    }

    if cols:
        for label, col in cols.items():
            if col in sim_df.columns:
                ax.plot(t, sim_df[col], label=label, color=compartment_colors[label])
    else:
        for label, c1, c2 in [
            ("S", "S1", "S2"), ("I", "I1", "I2"), ("R", "R1", "R2")
        ]:
            total = sim_df[c1] + sim_df[c2]
            ax.plot(t, total, label=f"{label}（合计）",
                    color=compartment_colors[label], linewidth=2)

    ax.set_xlabel("模拟天数（天）")
    ax.set_ylabel("人数")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_beta_seasonal(
    sim_df:      pd.DataFrame,
    title:       str = "β(t) 季节性传播系数",
    output_path: str | Path | None = None,
    show:        bool = False,
) -> plt.Figure:
    """绘制 β(t) 和 c(t) 以及有效传播系数 β_eff(t) 的时序图。"""
    setup_style()
    t = sim_df["t"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(title, fontsize=14, fontweight="bold")

    if "beta_t" in sim_df.columns:
        ax.plot(t, sim_df["beta_t"],   color=COLORS["accent1"], label="β(t) 季节系数", linewidth=2)
    if "beta_eff" in sim_df.columns:
        ax.plot(t, sim_df["beta_eff"], color=COLORS["danger"],  label="β_eff(t) 有效系数", linewidth=2, linestyle="--")
    if "contact_t" in sim_df.columns:
        ax2 = ax.twinx()
        ax2.fill_between(t, sim_df["contact_t"], alpha=0.12,
                          color=COLORS["accent2"], label="c(t) 学期调制")
        ax2.set_ylabel("学期调制系数 c(t)", color=COLORS["accent2"])
        ax2.set_ylim(0, 1.5)
        ax2.tick_params(axis="y", colors=COLORS["accent2"])

    ax.set_xlabel("模拟天数（天）")
    ax.set_ylabel("传播系数 β")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
