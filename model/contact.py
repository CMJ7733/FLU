"""
model/contact.py — 接触矩阵工具函数
=====================================
提供 2×2 接触矩阵的构建、归一化、场所分解等功能。
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import ModelParams


# ── 场所接触权重（默认，与 config.yaml 一致） ───────────────────────────────
DEFAULT_LOCATION_WEIGHTS: dict[str, float] = {
    "dorm":      0.30,   # 宿舍
    "classroom": 0.25,   # 教室
    "canteen":   0.10,   # 食堂
    "outdoor":   0.35,   # 课外/运动场所
}


def build_contact_matrix(p: "ModelParams") -> np.ndarray:
    """返回 2×2 接触矩阵（直接从 ModelParams 构建）。"""
    return p.contact_matrix()


def reciprocity_check(C: np.ndarray, N: np.ndarray, tol: float = 0.5) -> bool:
    """
    检查接触矩阵的互惠性：N[i] * C[i,j] ≈ N[j] * C[j,i]
    （群体 i 中的个体接触群体 j 的总次数，应等于反向）
    返回 True 表示互惠性基本满足。
    """
    n = len(N)
    ok = True
    for i in range(n):
        for j in range(i+1, n):
            lhs = N[i] * C[i, j]
            rhs = N[j] * C[j, i]
            if abs(lhs - rhs) / (max(lhs, rhs) + 1e-9) > tol:
                ok = False
    return ok


def symmetrize_contact(C: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    对接触矩阵做互惠性对称化：
        C_sym[i,j] = (N[i]*C[i,j] + N[j]*C[j,i]) / (2*N[i])
    """
    n = len(N)
    C_sym = C.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                total = N[i] * C[i, j] + N[j] * C[j, i]
                C_sym[i, j] = total / (2 * N[i])
    return C_sym


def effective_contact_matrix(
    p: "ModelParams",
    location_weights: dict[str, float] | None = None,
    location_reductions: dict[str, float] | None = None,
) -> np.ndarray:
    """
    计算干预后的有效接触矩阵。

    Args:
        p: 模型参数
        location_weights: 各场所接触权重（归一化）
        location_reductions: 各场所接触减少比例 {场所: 减少比例 ∈ [0,1]}

    Returns:
        2×2 有效接触矩阵
    """
    if location_weights is None:
        location_weights = DEFAULT_LOCATION_WEIGHTS
    if location_reductions is None:
        location_reductions = {}

    # 计算综合接触减少系数
    total_weight = sum(location_weights.values())
    reduction = sum(
        location_weights.get(loc, 0) * r
        for loc, r in location_reductions.items()
    ) / total_weight

    C = p.contact_matrix()
    return C * (1.0 - reduction)


def contact_matrix_from_literature() -> tuple[np.ndarray, str]:
    """
    基于 Prem et al. (2017) 中国接触矩阵的校园简化版本。
    返回 2×2 矩阵（学生/教职工）和来源说明。

    参考值来源：
    - Prem K, Cook AR, Jit M. Projecting social contact matrices in 152 countries.
      PLoS Comput Biol. 2017.
    - 校园环境放大系数（参考 Cauchemez et al. 2011 学校传播研究）
    """
    # 学生（18–25岁）与教职工（30–55岁）的日均接触次数
    # 校园环境比一般社区接触更密集（宿舍、课堂）
    C = np.array([
        [18.0, 2.0],   # 学生与学生/教职工的接触
        [2.0,  8.0],   # 教职工与学生/教职工的接触
    ])
    source = "基于 Prem et al. 2017 中国矩阵校园调整版"
    return C, source


# ── 场所分解辅助函数 ─────────────────────────────────────────────────────────

def c11_by_venue(p: "ModelParams") -> dict[str, float]:
    """返回各场所的 c11 值（宿舍/教室/食堂/户外）。"""
    return p.c11_by_venue()


def beta_mod_by_venue(p: "ModelParams") -> dict[str, float]:
    """返回各场所的 β 修饰系数（相对基础 β 的倍率）。"""
    return p.beta_mod_by_venue()
