"""
model/r0.py — 次代矩阵法（Next-Generation Matrix）计算 R₀
==========================================================
参考文献：
    van den Driessche P, Watmough J. Reproduction numbers and sub-threshold
    endemic equilibria for compartmental models of disease transmission.
    Math Biosci. 2002.

对于 SEIQR 模型，新感染仓室为 E，需要考虑经 E → I 的完整感染链。
两群体（学生/教职工）次代矩阵为 2×2 矩阵，R₀ = ρ(K) = 最大特征值。
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import ModelParams


def compute_R0(p: "ModelParams", t: float = 0.0) -> float:
    """
    用次代矩阵（NGM）法计算 R₀。

    推导过程（两群体 SEIQR）：
    - 新感染矩阵 F[i,j] = β₀ · C[i,j] · S₀ᵢ / Nⱼ
      （群体j中1个感染者，在单位时间内使群体i中S产生新暴露的速率）
    - 转移矩阵 V_E = diag(σ₁, σ₂)
    - 转移矩阵 V_I = diag(μ₁, μ₂)，μ = γ(1-p_iso) + α·p_iso
    - K = F · V_I⁻¹  （E 经过 V_E 消去后，SEIR/SEIQR 的 NGM 与 SIR 形式一致）
    - R₀ = ρ(K) = max(eigenvalues(K))

    Args:
        p: ModelParams（默认用 t=0 时刻的季节参数，即 β₀）
        t: 可选，用于计算时刻 t 的有效再生数 Rₑff（传入实际 t）

    Returns:
        R₀（标量）
    """
    from .seasonal import beta_t, contact_t

    beta  = beta_t(t, p)       # β(t)，t=0 时即 β₀
    ct    = contact_t(t, p)    # 学期调制
    C     = p.contact_matrix()
    N     = np.array([float(p.N1), float(p.N2)])

    # 初始易感比例（扣除疫苗保护人群）
    vacc_frac = p.vax_coverage * p.vax_efficacy
    S0 = N * (1.0 - vacc_frac)

    # I 舱的总离开速率：μ = γ·(1-p_iso) + α·p_iso
    mu = p.gamma * (1.0 - p.p_iso) + p.alpha * p.p_iso

    # 2×2 次代矩阵 K[i,j]：
    # 群体 j 中一个 I 进入群体 i 的易感人群，
    # 在其感染期（1/μ）内产生新暴露，再经 E→I（1/σ 约掉），最终新感染数
    K = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            # F[i,j] = β(t)·c(t)·C[i,j]·S0[i]/N[j]
            # K[i,j] = F[i,j] / μ  （V_I⁻¹ = 1/μ；σ 在 SEIQR 中约掉）
            K[i, j] = beta * ct * C[i, j] * S0[i] / (N[j] * mu)

    eigenvalues = np.linalg.eigvals(K)
    R0 = float(np.max(np.real(eigenvalues)))
    return max(R0, 0.0)


def compute_Reff(df_sol, p: "ModelParams") -> np.ndarray:
    """
    计算时序有效再生数 Reff(t) = R₀ × S(t)/N（均质混合近似）。

    Args:
        df_sol: solve_seiqr 输出的 DataFrame
        p:      ModelParams

    Returns:
        Reff 数组，长度与 df_sol 相同
    """
    R0_base = compute_R0(p, t=0.0)
    N = float(p.N1 + p.N2)
    S_total = (df_sol["S1"] + df_sol["S2"]).values
    return R0_base * S_total / N


def compute_herd_immunity_threshold(R0: float) -> float:
    """
    群体免疫阈值 HIT = 1 - 1/R₀。
    （即需要免疫的人口比例，使 Reff < 1）
    """
    if R0 <= 1.0:
        return 0.0
    return 1.0 - 1.0 / R0


def r0_sensitivity_table(
    p: "ModelParams",
    param_name: str,
    values: np.ndarray,
) -> dict:
    """
    对单个参数做一维扫描，计算对应 R₀。
    用于论文中的 R₀ 参数敏感性表。

    Args:
        p:          基础参数
        param_name: 待扫描的参数名（ModelParams 字段名）
        values:     扫描值数组

    Returns:
        {'values': [...], 'R0': [...], 'HIT': [...]}
    """
    r0_list  = []
    hit_list = []
    for v in values:
        p_new = p.update(**{param_name: float(v)})
        r0    = compute_R0(p_new)
        r0_list.append(r0)
        hit_list.append(compute_herd_immunity_threshold(r0))

    return {
        "param":  param_name,
        "values": values.tolist(),
        "R0":     r0_list,
        "HIT":    hit_list,
    }


def r0_heatmap_data(
    p: "ModelParams",
    param_x: str,
    param_y: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """
    对两个参数做二维网格扫描，返回 R₀ 矩阵。
    用于论文中的 R₀ 热力图（如 β₀ × γ 扫描）。

    Returns:
        R₀ 矩阵，形状 (len(y_vals), len(x_vals))
    """
    Z = np.zeros((len(y_vals), len(x_vals)))
    for j, xv in enumerate(x_vals):
        for i, yv in enumerate(y_vals):
            p_new  = p.update(**{param_x: float(xv), param_y: float(yv)})
            Z[i, j] = compute_R0(p_new)
    return Z
