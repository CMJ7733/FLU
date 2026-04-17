"""
model/ode_system.py — 两群体 SEIQR ODE 方程组
==============================================
状态向量（10维）:
    [S1, E1, I1, Q1, R1,  S2, E2, I2, Q2, R2]

下标 1 = 学生群体，下标 2 = 教职工群体。

仓室含义:
    S  — 易感者（Susceptible）
    E  — 潜伏期暴露者（Exposed，不具传染性）
    I  — 感染者（Infectious，可传播）
    Q  — 隔离者（Quarantined，已隔离，不传播）
    R  — 康复/免疫者（Recovered）

传播机制:
    λᵢ(t) = β(t) · c(t) · Σⱼ [Cᵢⱼ · Iⱼ / Nⱼ]

    S → E: λᵢ · Sᵢ
    E → I: σ · Eᵢ
    I → Q: α · p_iso · Iᵢ  （被发现并隔离）
    I → R: γ · (1−p_iso) · Iᵢ（自然康复，未被隔离）
    Q → R: γQ · Qᵢ
    R → S: ω · Rᵢ           （免疫衰退，单季设 ω=0）
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import ModelParams

from .seasonal import beta_t, contact_t


def seiqr_rhs(y: np.ndarray, t: float, p: "ModelParams") -> np.ndarray:
    """
    SEIQR 方程组右端项，兼容 scipy.integrate.odeint 接口。

    Args:
        y: 状态向量 [S1,E1,I1,Q1,R1, S2,E2,I2,Q2,R2]
        t: 时间（天，相对于模拟起始）
        p: ModelParams 实例

    Returns:
        dY/dt，形状 (10,)
    """
    S1, E1, I1, Q1, R1, S2, E2, I2, Q2, R2 = y

    N1 = float(p.N1)
    N2 = float(p.N2)

    # ── 当前时刻的传播参数 ──────────────────────────────────────────────────
    beta  = beta_t(t, p)         # 季节传播系数 β(t)
    ct    = contact_t(t, p)      # 学期调制系数 c(t)
    C     = p.contact_matrix()   # 2×2 接触矩阵

    # ── 感染压力（力感染项）λᵢ ─────────────────────────────────────────────
    # λᵢ = β(t) · c(t) · [Cᵢ₁ · I₁/N₁ + Cᵢ₂ · I₂/N₂]
    # 防止数值上 I 出现极小负值（ODE 积分舍入误差）
    I1_safe = max(I1, 0.0)
    I2_safe = max(I2, 0.0)

    lam1 = beta * ct * (C[0, 0] * I1_safe / N1 + C[0, 1] * I2_safe / N2)
    lam2 = beta * ct * (C[1, 0] * I1_safe / N1 + C[1, 1] * I2_safe / N2)

    # ── 各仓室离开速率 ──────────────────────────────────────────────────────
    sigma   = p.sigma
    gamma   = p.gamma
    gamma_Q = p.gamma_Q
    alpha   = p.alpha
    p_iso   = p.p_iso
    omega   = p.omega

    # I → Q 速率（感染者被发现隔离）
    iso_rate   = alpha * p_iso
    # I → R 速率（感染者自然康复，未被隔离）
    rec_rate_I = gamma * (1.0 - p_iso)

    # ── 学生群体（群体1） ───────────────────────────────────────────────────
    S1_safe = max(S1, 0.0)
    dS1 = -lam1 * S1_safe + omega * max(R1, 0.0)
    dE1 =  lam1 * S1_safe - sigma * max(E1, 0.0)
    dI1 =  sigma * max(E1, 0.0) - (rec_rate_I + iso_rate) * max(I1, 0.0)
    dQ1 =  iso_rate * max(I1, 0.0) - gamma_Q * max(Q1, 0.0)
    dR1 =  rec_rate_I * max(I1, 0.0) + gamma_Q * max(Q1, 0.0) - omega * max(R1, 0.0)

    # ── 教职工群体（群体2） ─────────────────────────────────────────────────
    S2_safe = max(S2, 0.0)
    dS2 = -lam2 * S2_safe + omega * max(R2, 0.0)
    dE2 =  lam2 * S2_safe - sigma * max(E2, 0.0)
    dI2 =  sigma * max(E2, 0.0) - (rec_rate_I + iso_rate) * max(I2, 0.0)
    dQ2 =  iso_rate * max(I2, 0.0) - gamma_Q * max(Q2, 0.0)
    dR2 =  rec_rate_I * max(I2, 0.0) + gamma_Q * max(Q2, 0.0) - omega * max(R2, 0.0)

    return np.array([dS1, dE1, dI1, dQ1, dR1,
                     dS2, dE2, dI2, dQ2, dR2])


def seiqr_rhs_flat(y: np.ndarray, t: float, *args) -> np.ndarray:
    """
    seiqr_rhs 的平铺参数版本，用于需要将参数展开为元组的场景。
    args 应为 (ModelParams,) 的解包形式。
    实际使用中优先用 seiqr_rhs。
    """
    p = args[0]
    return seiqr_rhs(y, t, p)


# ── 状态向量索引常量 ────────────────────────────────────────────────────────
IDX = {
    "S1": 0, "E1": 1, "I1": 2, "Q1": 3, "R1": 4,
    "S2": 5, "E2": 6, "I2": 7, "Q2": 8, "R2": 9,
}

STATE_LABELS = ["S1", "E1", "I1", "Q1", "R1", "S2", "E2", "I2", "Q2", "R2"]
