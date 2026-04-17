"""
calibration/fitting.py — 参数拟合
===================================
使用 lmfit 最小化模拟感染率 I(t)/N 与 WHO FluNet H3N2 周阳性率的 RMSE。

待拟合参数：beta0, delta1, delta2, phi1, phi2, alpha
固定参数：sigma, gamma, gamma_Q, p_iso, N1, N2 等（来自文献）

拟合目标：
    residuals[i] = sim_I_rate(t_i) - obs_pos_rate[i]
    最小化 Σ residuals²（加权最小二乘）
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

try:
    from lmfit import Parameters, minimize, fit_report
    _HAS_LMFIT = True
except ImportError:
    _HAS_LMFIT = False
    from scipy.optimize import least_squares

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    """参数拟合结果容器。"""
    params_best:  dict        # 最优参数字典
    rmse:         float       # 均方根误差
    r_squared:    float       # 拟合 R²
    n_obs:        int         # 观测数据点数
    success:      bool        # 拟合是否收敛
    message:      str         # 拟合状态信息
    report:       str = ""    # lmfit 完整报告
    rho:          float = 1.0 # 观测比例（I_rate → FluNet 阳性率的换算系数）


def _build_lmfit_params(p_init, bounds: dict) -> "Parameters":
    """构建 lmfit Parameters 对象。"""
    lm_params = Parameters()
    for name, (lo, hi) in bounds.items():
        val = getattr(p_init, name, (lo + hi) / 2)
        lm_params.add(name, value=val, min=lo, max=hi)
    return lm_params


def _residuals_func(lm_params, obs_t, obs_y, p_base, t_sim):
    """计算残差向量（lmfit 的目标函数）。"""
    from model.params import ModelParams
    from model.solver import solve_seiqr

    # 从 lmfit 参数更新模型参数
    updates = {name: lm_params[name].value for name in lm_params}
    p = p_base.update(**updates)

    # 求解 ODE
    df_sim = solve_seiqr(p, t_eval=t_sim)

    # 插值到观测时间点
    sim_I_rate = df_sim["I_rate"].values
    sim_t      = df_sim["t"].values

    interp = interp1d(sim_t, sim_I_rate, kind="linear",
                      bounds_error=False, fill_value=0.0)
    sim_at_obs = interp(obs_t)

    # 加权：高阳性率周给更大权重
    weights = np.sqrt(obs_y + 0.01)
    return weights * (sim_at_obs - obs_y)


def fit_model(
    obs_df: pd.DataFrame,
    p_init,
    bounds: dict | None = None,
    method: str = "leastsq",
    max_nfev: int = 1000,
    t_col: str = "t_sim",
    y_col: str = "h3n2_pos_rate",
) -> FitResult:
    """
    拟合模型参数至观测数据。

    Args:
        obs_df:   观测数据 DataFrame（含 t_sim 和 h3n2_pos_rate 列）
        p_init:   ModelParams 初始参数
        bounds:   参数边界 {param: (lo, hi)}，None 时使用默认边界
        method:   lmfit 求解方法（leastsq / nelder / powell）
        max_nfev: 最大函数评估次数
        t_col:    时间列名（观测数据中）
        y_col:    目标列名（观测数据中）

    Returns:
        FitResult 实例
    """
    if bounds is None:
        bounds = {
            "beta0":  (0.003, 0.08),
            "delta1": (0.00, 0.70),
            "delta2": (0.00, 0.80),
            "phi1":   (-np.pi, np.pi),
            "phi2":   (-np.pi, np.pi),
            "alpha":  (0.01, 0.80),
        }

    # 准备观测数据
    obs_clean = obs_df.dropna(subset=[t_col, y_col]).copy()
    obs_t = obs_clean[t_col].values.astype(float)
    obs_y = obs_clean[y_col].values.astype(float)

    if len(obs_t) < 5:
        log.warning("观测数据点数过少（< 5），无法拟合")
        return FitResult(
            params_best=p_init.to_dict(), rmse=np.nan,
            r_squared=np.nan, n_obs=len(obs_t),
            success=False, message="数据点不足"
        )

    # 模拟时间轴（覆盖观测范围）
    t_sim = np.arange(0, obs_t.max() + 1, dtype=float)

    if _HAS_LMFIT:
        lm_params = _build_lmfit_params(p_init, bounds)
        result = minimize(
            _residuals_func,
            lm_params,
            args=(obs_t, obs_y, p_init, t_sim),
            method=method,
            max_nfev=max_nfev,
        )
        best = {name: result.params[name].value for name in result.params}
        resid = result.residual / np.sqrt(obs_y + 0.01)  # 去权后残差
        rmse  = float(np.sqrt(np.mean(resid**2)))
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((obs_y - obs_y.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        report = fit_report(result) if hasattr(result, "params") else ""
        success = result.success if hasattr(result, "success") else True
        msg = getattr(result, "message", "lmfit 完成")

    else:
        # fallback: scipy.optimize.least_squares
        log.warning("lmfit 未安装，使用 scipy.optimize.least_squares")
        param_names = list(bounds.keys())
        x0    = [getattr(p_init, n, (lo+hi)/2) for n, (lo, hi) in bounds.items()]
        lo_b  = [lo for lo, hi in bounds.values()]
        hi_b  = [hi for lo, hi in bounds.values()]

        def scipy_resid(x):
            upd = dict(zip(param_names, x))
            p = p_init.update(**upd)
            from model.solver import solve_seiqr
            df_sim = solve_seiqr(p, t_eval=t_sim)
            interp = interp1d(df_sim["t"].values, df_sim["I_rate"].values,
                              kind="linear", bounds_error=False, fill_value=0.0)
            sim_at_obs = interp(obs_t)
            weights = np.sqrt(obs_y + 0.01)
            return weights * (sim_at_obs - obs_y)

        res = least_squares(scipy_resid, x0, bounds=(lo_b, hi_b), max_nfev=max_nfev)
        best = dict(zip(param_names, res.x))
        resid = res.fun / np.sqrt(obs_y + 0.01)
        rmse  = float(np.sqrt(np.mean(resid**2)))
        ss_tot = np.sum((obs_y - obs_y.mean())**2)
        r2 = 1 - np.sum(resid**2) / max(ss_tot, 1e-12)
        report = ""
        success = res.success
        msg = res.message

    # 计算观测比例 rho：将模型 I_rate 映射到 FluNet 阳性率的最优线性缩放
    # rho* = argmin_ρ Σ(ρ·sim - obs)² = Σ(sim·obs) / Σ(sim²)
    rho_val = 1.0
    try:
        from model.solver import solve_seiqr as _solve_rho
        _p_rho = p_init.update(**{k: v for k, v in best.items() if hasattr(p_init, k)})
        _df_rho = _solve_rho(_p_rho, t_eval=t_sim)
        _interp_rho = interp1d(_df_rho["t"].values, _df_rho["I_rate"].values,
                               kind="linear", bounds_error=False, fill_value=0.0)
        _sim_rho = _interp_rho(obs_t)
        _denom = float(np.dot(_sim_rho, _sim_rho))
        if _denom > 1e-12:
            rho_val = float(np.clip(np.dot(_sim_rho, obs_y) / _denom, 0.01, 1.5))
    except Exception:
        pass

    log.info(f"拟合完成: RMSE={rmse:.5f}  R²={r2:.4f}  success={success}  rho={rho_val:.3f}")
    log.info(f"最优参数: {best}")

    return FitResult(
        params_best=best,
        rmse=rmse,
        r_squared=float(r2),
        n_obs=len(obs_t),
        success=success,
        message=msg,
        report=report,
        rho=rho_val,
    )


def prepare_obs_timeseries(
    weekly_df: pd.DataFrame,
    p_start,
    year: int | None = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    """
    将 FluNet 周度数据转换为以"模拟天"为索引的时间序列。

    Args:
        weekly_df:  data/processed/weekly_positivity.csv 内容
        p_start:    ModelParams（用于 t_start_doy）
        year:       仅取特定流行年度（flu_year 字段）；None 则使用全部年份
        aggregate:  True 时对多年数据按周次取均值，避免多年叠加导致 R² 崩溃

    Returns:
        DataFrame 含列: t_sim（模拟内天数）, h3n2_pos_rate
    """
    df = weekly_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if year is not None:
        df = df[df["flu_year"] == year].copy()

    if len(df) == 0:
        raise ValueError(f"flu_year={year} 无数据")

    # 计算相对于模拟起始日的天数
    start_doy = p_start.t_start_doy
    df["doy"] = df["date"].dt.dayofyear
    df["t_sim"] = ((df["doy"] - start_doy) % 365).astype(float)

    # 只保留模拟时间范围内的数据
    df = df[(df["t_sim"] >= 0) & (df["t_sim"] <= p_start.t_days)].copy()

    # 多年均值聚合：按7天为一个 bin 取均值，消除年际噪声
    if aggregate and year is None:
        df["t_sim_bin"] = (df["t_sim"] / 7).round() * 7
        df = (
            df.groupby("t_sim_bin", as_index=False)
            .agg(h3n2_pos_rate=("h3n2_pos_rate", "mean"))
            .rename(columns={"t_sim_bin": "t_sim"})
        )
        df["date"] = pd.NaT
        log.info(f"多年均值聚合：{len(df)} 个周次均值数据点（原始 {len(weekly_df)} 周）")

    df = df.sort_values("t_sim").reset_index(drop=True)
    return df[["t_sim", "h3n2_pos_rate", "date"]].dropna(subset=["t_sim", "h3n2_pos_rate"])
