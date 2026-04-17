"""
model — 两群体 SEIQR 传染病动力学模型核心包
==========================================
子模块:
    params      ModelParams 参数管理
    seasonal    β(t) 季节函数 + 学期调制 c(t)
    contact     接触矩阵工具
    ode_system  SEIQR ODE 方程组
    solver      数值求解封装
    r0          次代矩阵法 R₀ 计算
"""

from .params     import ModelParams
from .seasonal   import beta_t, contact_t, beta_series
from .ode_system import seiqr_rhs
from .solver     import solve_seiqr
from .r0         import compute_R0, compute_herd_immunity_threshold

__all__ = [
    "ModelParams",
    "beta_t", "contact_t", "beta_series",
    "seiqr_rhs",
    "solve_seiqr",
    "compute_R0", "compute_herd_immunity_threshold",
]
