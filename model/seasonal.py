"""
model/seasonal.py — β(t) 双谐波季节函数 & 学期调制 c(t)
=========================================================
β_eff(t) = beta0 × [1 + δ₁·cos(2πt/365 + φ₁) + δ₂·cos(4πt/365 + φ₂)] × c(t)

其中 c(t) 为学期调制系数：
    开学期间 → 1.0（正常高密度接触）
    假期     → 0.3（学生离校，接触大幅减少）
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import ModelParams

# ── 上海高校学期日历（day-of-year 近似） ────────────────────────────────────
# 寒假：第 15–42 天（1月15日–2月11日左右）
# 暑假：第 197–243 天（7月16日–8月31日左右）
# 国庆：第 274–281 天（10月1日–10月8日）
# 五一：第 121–125 天（5月1日–5月5日）
HOLIDAY_PERIODS: list[tuple[int, int, float]] = [
    (15,  42,  0.3),   # 寒假（主要）
    (197, 243, 0.3),   # 暑假（主要）
    (274, 281, 0.65),  # 国庆
    (121, 125, 0.75),  # 五一
]

# 寒假期间学生返家后的接触强度（传播主要在宿舍，但人少）
_DEFAULT_HOLIDAY_C = 0.3


def beta_t(t: float, p: "ModelParams") -> float:
    """
    计算第 t 天（模拟内相对时间）的季节传播系数 β(t)。

    使用双谐波模型，t 以自然天为单位（可为小数）：
        β(t) = β₀ × [1 + δ₁·cos(2πt_abs/365 + φ₁)
                        + δ₂·cos(4πt_abs/365 + φ₂)]

    其中 t_abs = (t_start_doy + t) mod 365 是绝对一年中的天数。
    """
    t_abs = (p.t_start_doy + t) % 365
    annual     = p.delta1 * np.cos(2 * np.pi * t_abs / 365 + p.phi1)
    semiannual = p.delta2 * np.cos(4 * np.pi * t_abs / 365 + p.phi2)
    return max(p.beta0 * (1.0 + annual + semiannual), 0.0)


def contact_t(t: float, p: "ModelParams") -> float:
    """
    学期调制系数 c(t)。
    返回值 ∈ [0.3, 1.0]，乘以接触矩阵对应时段的接触强度。
    """
    doy = int(p.t_start_doy + t) % 365
    for lo, hi, multiplier in HOLIDAY_PERIODS:
        if lo <= doy <= hi:
            return multiplier
    return 1.0


def beta_eff(t: float, p: "ModelParams") -> float:
    """有效传播系数 = β(t) × c(t)。"""
    return beta_t(t, p) * contact_t(t, p)


def beta_series(t_arr: np.ndarray, p: "ModelParams") -> np.ndarray:
    """向量化计算 β(t) 序列（用于绘图和拟合）。"""
    return np.array([beta_t(float(t), p) for t in t_arr])


def contact_series(t_arr: np.ndarray, p: "ModelParams") -> np.ndarray:
    """向量化计算 c(t) 序列。"""
    return np.array([contact_t(float(t), p) for t in t_arr])


def beta_eff_series(t_arr: np.ndarray, p: "ModelParams") -> np.ndarray:
    """向量化计算有效传播系数序列 β_eff(t) = β(t) × c(t)。"""
    return beta_series(t_arr, p) * contact_series(t_arr, p)


def peak_timing(p: "ModelParams") -> tuple[int, float]:
    """
    在模拟时间范围内找到 β_eff(t) 的最大值及对应天数。
    返回 (peak_day, peak_beta)
    """
    t_arr = np.arange(p.t_days, dtype=float)
    beff  = beta_eff_series(t_arr, p)
    idx   = int(np.argmax(beff))
    return idx, float(beff[idx])
