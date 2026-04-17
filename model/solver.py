"""
model/solver.py — SEIQR ODE 数值求解封装
==========================================
封装 scipy.integrate.odeint，返回包含时间戳和派生指标的 DataFrame。
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from .params     import ModelParams
from .ode_system import seiqr_rhs, STATE_LABELS


def solve_seiqr(
    p: ModelParams,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """
    求解两群体 SEIQR ODE。

    Args:
        p:      ModelParams 实例
        t_eval: 求解时间点数组（天），默认为 [0, 1, ..., t_days]
        rtol:   相对误差容限
        atol:   绝对误差容限

    Returns:
        DataFrame，列为:
            t, S1,E1,I1,Q1,R1, S2,E2,I2,Q2,R2,
            I_total, I_rate, Infected_cumulative,
            beta_t, contact_t
    """
    if t_eval is None:
        t_eval = np.arange(0, p.t_days + 1, dtype=float)

    y0 = p.initial_state()

    # odeint 求解（LSODA 自动刚性检测）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(
            seiqr_rhs,
            y0,
            t_eval,
            args=(p,),
            rtol=rtol,
            atol=atol,
            full_output=False,
        )

    # 保证数值非负（ODE 积分舍入误差）
    sol = np.maximum(sol, 0.0)

    df = pd.DataFrame(sol, columns=STATE_LABELS)
    df.insert(0, "t", t_eval)

    # ── 派生指标 ──────────────────────────────────────────────────────────
    # 总感染人数（I 舱 + Q 舱）
    df["I_total"]  = df["I1"] + df["Q1"] + df["I2"] + df["Q2"]
    df["I1_total"] = df["I1"] + df["Q1"]
    df["I2_total"] = df["I2"] + df["Q2"]

    # 活跃感染率（相对于总人口）
    N = p.N1 + p.N2
    df["I_rate"]   = df["I_total"] / N
    df["I1_rate"]  = df["I1_total"] / p.N1
    df["I2_rate"]  = df["I2_total"] / p.N2

    # 累计发病人数（通过 R 舱增量估算）
    # 总康复 = R1 + R2 - 初始免疫人数
    R0_total = y0[4] + y0[9]   # 初始 R（疫苗保护）
    df["R_total"] = df["R1"] + df["R2"] - R0_total
    df["R_total"] = df["R_total"].clip(lower=0.0)
    df["attack_rate"] = df["R_total"] / (N - R0_total)

    # β(t) 和 c(t) 序列
    from .seasonal import beta_t as _beta_t, contact_t as _contact_t
    df["beta_t"]    = [_beta_t(t, p) for t in t_eval]
    df["contact_t"] = [_contact_t(t, p) for t in t_eval]
    df["beta_eff"]  = df["beta_t"] * df["contact_t"]

    return df


def extract_summary(df: pd.DataFrame, p: ModelParams) -> dict:
    """
    从求解结果中提取论文关键指标。

    Returns:
        dict 包含:
            peak_I_rate      峰值感染率
            peak_day         峰值时间（天）
            total_attack_rate 累计发病率（最终 R/N）
            peak_I_total     峰值感染人数（I+Q）
            doubling_time    翻倍时间（指数增长期）
    """
    N = p.N1 + p.N2

    peak_idx     = df["I_total"].idxmax()
    peak_I_total = df["I_total"].iloc[peak_idx]
    peak_day     = float(df["t"].iloc[peak_idx])
    peak_I_rate  = peak_I_total / N

    total_AR = float(df["attack_rate"].iloc[-1])

    # 翻倍时间：从 I_total > 10 开始，用对数线性拟合前 14 天
    growth_mask = df["I_total"] > max(10.0, df["I_total"].iloc[0] * 2)
    doubling_time = np.nan
    if growth_mask.sum() >= 5:
        t_g   = df.loc[growth_mask, "t"].values[:14]
        I_g   = df.loc[growth_mask, "I_total"].values[:14]
        valid = I_g > 0
        if valid.sum() >= 3:
            coef = np.polyfit(t_g[valid], np.log(I_g[valid]), 1)
            if coef[0] > 0:
                doubling_time = np.log(2) / coef[0]

    return {
        "peak_I_rate":       float(peak_I_rate),
        "peak_I_total":      float(peak_I_total),
        "peak_day":          peak_day,
        "total_attack_rate": total_AR,
        "doubling_time_days": float(doubling_time),
    }
