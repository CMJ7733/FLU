"""
calibration/bootstrap.py — Bootstrap 参数置信区间
===================================================
对观测数据进行 n 次有放回重采样，每次重新拟合模型参数，
从重采样分布估计参数的 95% 置信区间和模拟轨迹置信带。
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .fitting import fit_model, FitResult

log = logging.getLogger(__name__)


def bootstrap_params(
    obs_df: pd.DataFrame,
    p_init,
    bounds: dict | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "leastsq",
    random_seed: int = 42,
    t_col: str = "t_sim",
    y_col: str = "h3n2_pos_rate",
) -> dict:
    """
    Bootstrap 参数置信区间估计。

    Args:
        obs_df:       观测数据
        p_init:       初始模型参数
        bounds:       参数拟合边界
        n_bootstrap:  Bootstrap 重采样次数
        ci_level:     置信水平（默认 0.95）
        method:       lmfit 求解方法
        random_seed:  随机种子（保证可重复性）

    Returns:
        dict:
            param_ci:    各参数的 CI，{param: (lo, hi, mean, std)}
            param_samples: 各参数样本数组 {param: np.ndarray}
            success_rate: 拟合收敛率
    """
    rng = np.random.default_rng(random_seed)
    obs_clean = obs_df.dropna(subset=[t_col, y_col]).copy()
    n_obs = len(obs_clean)

    if n_obs < 5:
        log.warning("观测数据太少，无法做 Bootstrap")
        return {}

    param_samples = {name: [] for name in (bounds or {
        "beta0": None, "delta1": None, "delta2": None,
        "phi1": None, "phi2": None, "alpha": None,
    }).keys()}
    n_success = 0

    log.info(f"Bootstrap 开始：{n_bootstrap} 次重采样，观测点数 = {n_obs}")

    for i in range(n_bootstrap):
        # 有放回重采样
        idx      = rng.integers(0, n_obs, size=n_obs)
        obs_boot = obs_clean.iloc[idx].copy()

        result = fit_model(
            obs_boot, p_init,
            bounds=bounds, method=method,
            max_nfev=500, t_col=t_col, y_col=y_col,
        )

        if result.success:
            n_success += 1
            for name in param_samples:
                if name in result.params_best:
                    param_samples[name].append(result.params_best[name])

        if (i + 1) % 100 == 0:
            log.info(f"  Bootstrap: {i+1}/{n_bootstrap}  成功率={n_success/(i+1):.1%}")

    # 转为 numpy 数组
    param_samples = {k: np.array(v) for k, v in param_samples.items() if len(v) > 0}

    # 计算 CI
    alpha_ci = (1 - ci_level) / 2
    param_ci = {}
    for name, samples in param_samples.items():
        if len(samples) > 0:
            lo  = float(np.percentile(samples, alpha_ci * 100))
            hi  = float(np.percentile(samples, (1 - alpha_ci) * 100))
            mu  = float(np.mean(samples))
            std = float(np.std(samples))
            param_ci[name] = {"lo": lo, "hi": hi, "mean": mu, "std": std}

    success_rate = n_success / n_bootstrap
    log.info(f"Bootstrap 完成：收敛率 = {success_rate:.1%}")

    return {
        "param_ci":      param_ci,
        "param_samples": param_samples,
        "success_rate":  success_rate,
        "n_bootstrap":   n_bootstrap,
        "ci_level":      ci_level,
    }


def bootstrap_trajectory(
    obs_df: pd.DataFrame,
    p_init,
    bounds: dict | None = None,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    method: str = "leastsq",
    random_seed: int = 42,
    t_col: str = "t_sim",
    y_col: str = "h3n2_pos_rate",
) -> dict:
    """
    Bootstrap 模拟轨迹置信带。
    对每次重采样得到的最优参数，运行 ODE 得到 I(t) 轨迹，
    最终返回逐时刻的百分位数包络。

    Returns:
        dict:
            t:          时间轴
            median:     中位数轨迹
            ci_lo:      置信下界
            ci_hi:      置信上界
            trajectories: 全部轨迹矩阵
    """
    from model.solver import solve_seiqr

    rng = np.random.default_rng(random_seed)
    obs_clean = obs_df.dropna(subset=[t_col, y_col]).copy()
    n_obs = len(obs_clean)

    t_max = int(obs_clean[t_col].max()) + 1
    t_sim = np.arange(0, t_max + 1, dtype=float)

    trajectories = []
    n_success    = 0

    log.info(f"Bootstrap 轨迹：{n_bootstrap} 次")

    for i in range(n_bootstrap):
        idx      = rng.integers(0, n_obs, size=n_obs)
        obs_boot = obs_clean.iloc[idx].copy()

        result = fit_model(
            obs_boot, p_init,
            bounds=bounds, method=method,
            max_nfev=500, t_col=t_col, y_col=y_col,
        )

        if result.success:
            p_best = p_init.update(**result.params_best)
            df_sim = solve_seiqr(p_best, t_eval=t_sim)
            trajectories.append(df_sim["I_rate"].values)
            n_success += 1

    if n_success == 0:
        log.error("所有 Bootstrap 拟合均失败")
        return {}

    traj_matrix = np.array(trajectories)
    alpha_ci = (1 - ci_level) / 2

    return {
        "t":            t_sim,
        "median":       np.median(traj_matrix, axis=0),
        "ci_lo":        np.percentile(traj_matrix, alpha_ci * 100, axis=0),
        "ci_hi":        np.percentile(traj_matrix, (1-alpha_ci) * 100, axis=0),
        "trajectories": traj_matrix,
        "n_success":    n_success,
        "ci_level":     ci_level,
    }
