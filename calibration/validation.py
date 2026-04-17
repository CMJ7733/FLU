"""
calibration/validation.py — 模型误差评估
==========================================
计算模型与观测数据之间的 RMSE、MAPE、R² 等指标，
并输出验证图（模拟 vs 实际）。
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """模型验证指标容器。"""
    rmse:          float   # 均方根误差
    mae:           float   # 平均绝对误差
    mape:          float   # 平均绝对百分比误差（%）
    r_squared:     float   # 决定系数 R²
    pearson_r:     float   # Pearson 相关系数
    n_points:      int     # 验证数据点数
    peak_error_days: float # 峰值时间误差（天）
    peak_rate_err:   float # 峰值感染率误差

    def summary(self) -> str:
        return (
            f"RMSE={self.rmse:.5f}  MAE={self.mae:.5f}  "
            f"MAPE={self.mape:.2f}%  R²={self.r_squared:.4f}  "
            f"r={self.pearson_r:.4f}"
        )


def compute_metrics(
    sim_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    t_col: str = "t_sim",
    obs_col: str = "h3n2_pos_rate",
    sim_t_col: str = "t",
    sim_y_col: str = "I_rate",
    scale: float = 1.0,
) -> ValidationResult:
    """
    计算仿真与观测数据之间的误差指标。

    Args:
        sim_df: solve_seiqr 输出（含 t 和 I_rate 列）
        obs_df: 观测数据（含 t_sim 和 h3n2_pos_rate 列）

    Returns:
        ValidationResult
    """
    obs_clean = obs_df.dropna(subset=[t_col, obs_col]).copy()
    obs_t = obs_clean[t_col].values.astype(float)
    obs_y = obs_clean[obs_col].values.astype(float)

    # 将仿真结果插值到观测时间点
    interp = interp1d(
        sim_df[sim_t_col].values,
        sim_df[sim_y_col].values,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    sim_y = interp(obs_t) * scale

    n = len(obs_y)
    residuals = sim_y - obs_y

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae  = float(np.mean(np.abs(residuals)))

    # MAPE（避免除以零）
    mask_pos = obs_y > 0.01   # 仅在流行周（阳性率>1%）计算 MAPE，排除近零分母
    mape = float(np.mean(np.abs(residuals[mask_pos] / obs_y[mask_pos])) * 100) \
           if mask_pos.sum() > 0 else np.nan

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((obs_y - obs_y.mean())**2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))

    # Pearson r
    if np.std(obs_y) > 0 and np.std(sim_y) > 0:
        pearson_r = float(np.corrcoef(obs_y, sim_y)[0, 1])
    else:
        pearson_r = np.nan

    # 峰值误差
    obs_peak_idx = int(np.argmax(obs_y))
    sim_peak_idx = int(np.argmax(sim_y))
    peak_error_days = float(obs_t[sim_peak_idx] - obs_t[obs_peak_idx]) \
                      if n > 0 else np.nan
    peak_rate_err   = float(sim_y[obs_peak_idx] - obs_y[obs_peak_idx]) \
                      if n > 0 else np.nan

    result = ValidationResult(
        rmse=rmse, mae=mae, mape=mape, r_squared=r2,
        pearson_r=pearson_r, n_points=n,
        peak_error_days=peak_error_days, peak_rate_err=peak_rate_err,
    )
    log.info(f"验证指标: {result.summary()}")
    return result


def cross_validate(
    obs_df: pd.DataFrame,
    p_init,
    bounds: dict | None = None,
    n_folds: int = 5,
    t_col: str = "t_sim",
    y_col: str = "h3n2_pos_rate",
) -> list[ValidationResult]:
    """
    k 折时序交叉验证（按时间顺序划分，不打乱）。

    Returns:
        各折的 ValidationResult 列表
    """
    from .fitting import fit_model
    from model.solver import solve_seiqr

    obs_clean = obs_df.dropna(subset=[t_col, y_col]).sort_values(t_col).copy()
    n = len(obs_clean)
    fold_size = n // n_folds
    results = []

    for k in range(n_folds):
        train = obs_clean.iloc[:fold_size * (k + 1)].copy()
        test  = obs_clean.iloc[fold_size * (k + 1): fold_size * (k + 2)].copy()

        if len(train) < 5 or len(test) < 2:
            continue

        fit = fit_model(train, p_init, bounds=bounds, max_nfev=500,
                        t_col=t_col, y_col=y_col)
        if not fit.success:
            continue

        p_best = p_init.update(**fit.params_best)
        t_max  = int(test[t_col].max()) + 1
        sim_df = solve_seiqr(p_best, t_eval=np.arange(0, t_max + 1, dtype=float))
        vr     = compute_metrics(sim_df, test, t_col=t_col, obs_col=y_col)
        results.append(vr)
        log.info(f"Fold {k+1}/{n_folds}: {vr.summary()}")

    return results
