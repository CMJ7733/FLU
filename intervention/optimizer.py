"""
intervention/optimizer.py — 防控方案多目标优化
================================================
目标函数：
    F = w₁ × AR + w₂ × PIP + w₃ × Cost

    AR   = 累计发病率（相对于基准）
    PIP  = 峰值感染压力（相对于基准）
    Cost = 综合成本评分 [0,1]

优化策略：
    1. 参数网格扫描（暴力搜索，简单可靠，适合 4–5 维）
    2. scipy.optimize.differential_evolution（全局优化）

网格扫描策略：
    - 对每类措施取 5 个离散级别（0, 0.25, 0.5, 0.75, 1.0）
    - 7 个参数 × 5 级别 = 5^7 ≈ 78,125 种组合
    - 筛选条件：Cost ≤ 阈值（如 0.5）
    - 排序目标：加权得分 F 最小
"""

from __future__ import annotations
import logging
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .measures import InterventionBundle, apply_interventions, NO_INTERVENTION
from .cost_model import compute_cost

log = logging.getLogger(__name__)


def _evaluate_bundle(bundle: InterventionBundle, base_params, scenario) -> dict:
    """对单个干预方案运行 ODE 并返回目标量。"""
    from model.solver import solve_seiqr, extract_summary

    p = scenario.apply_to(base_params)
    p = apply_interventions(p, bundle)
    df = solve_seiqr(p)
    summ = extract_summary(df, p)
    cost = compute_cost(bundle)

    return {
        "attack_rate":   summ["total_attack_rate"],
        "peak_I_rate":   summ["peak_I_rate"],
        "peak_day":      summ["peak_day"],
        "cost_score":    cost,
        **bundle.to_dict(),
    }


def objective_function(
    bundle: InterventionBundle,
    base_params,
    scenario,
    baseline_AR: float,
    baseline_PIP: float,
    weights: dict | None = None,
) -> float:
    """
    计算加权目标函数 F。

    Args:
        bundle:       干预措施
        base_params:  ModelParams 基础参数
        scenario:     Scenario 实例
        baseline_AR:  无干预时的累计发病率
        baseline_PIP: 无干预时的峰值感染率
        weights:      {'attack_rate': w1, 'peak_pressure': w2, 'cost': w3}

    Returns:
        F（越小越好）
    """
    if weights is None:
        weights = {"attack_rate": 0.50, "peak_pressure": 0.30, "cost": 0.20}

    result = _evaluate_bundle(bundle, base_params, scenario)

    ar_norm  = result["attack_rate"] / max(baseline_AR, 1e-6)
    pip_norm = result["peak_I_rate"] / max(baseline_PIP, 1e-6)
    cost     = result["cost_score"]

    F = (weights["attack_rate"]   * ar_norm +
         weights["peak_pressure"] * pip_norm +
         weights["cost"]          * cost)
    return F


def grid_search(
    base_params,
    scenario,
    n_levels:   int = 3,
    cost_limit: float = 0.60,
    weights:    dict | None = None,
    output_dir: str | Path | None = None,
    top_k:      int = 20,
) -> pd.DataFrame:
    """
    网格扫描优化。

    Args:
        base_params:  ModelParams 基础参数
        scenario:     Scenario 实例
        n_levels:     每个控制变量的离散级别数
        cost_limit:   成本上限过滤（[0,1]）
        weights:      目标函数权重
        output_dir:   结果保存路径
        top_k:        返回前 k 个方案

    Returns:
        DataFrame，按目标函数 F 排序
    """
    from model.solver import solve_seiqr, extract_summary

    if weights is None:
        weights = {"attack_rate": 0.50, "peak_pressure": 0.30, "cost": 0.20}

    # 计算基准（无干预）
    p_base = scenario.apply_to(base_params)
    df_base = solve_seiqr(p_base)
    summ_base = extract_summary(df_base, p_base)
    baseline_AR  = summ_base["total_attack_rate"]
    baseline_PIP = summ_base["peak_I_rate"]

    log.info(f"基准: AR={baseline_AR:.4f}, PeakI={baseline_PIP:.4f}")

    # 离散化控制变量
    levels = np.linspace(0, 1, n_levels)
    param_names = [
        "mask_level", "ventilation", "vaccination",
        "isolation_rate", "online_teaching", "activity_limit", "disinfection"
    ]

    # 总组合数
    total = n_levels ** len(param_names)
    log.info(f"网格搜索：{n_levels}^{len(param_names)} = {total} 种组合")

    records = []
    skipped = 0

    for combo in product(levels, repeat=len(param_names)):
        bundle = InterventionBundle(
            mask_level=combo[0],
            ventilation=combo[1],
            vaccination=combo[2],
            isolation_rate=combo[3],
            online_teaching=combo[4],
            activity_limit=combo[5],
            disinfection=combo[6],
        )

        cost = compute_cost(bundle)
        if cost > cost_limit:
            skipped += 1
            continue

        try:
            rec = _evaluate_bundle(bundle, base_params, scenario)
            ar_norm  = rec["attack_rate"] / max(baseline_AR, 1e-6)
            pip_norm = rec["peak_I_rate"] / max(baseline_PIP, 1e-6)
            F = (weights["attack_rate"]   * ar_norm +
                 weights["peak_pressure"] * pip_norm +
                 weights["cost"]          * cost)

            rec["F_objective"]      = F
            rec["AR_reduction_pct"] = (1 - ar_norm) * 100
            rec["PIP_reduction_pct"]= (1 - pip_norm) * 100
            records.append(rec)
        except Exception as e:
            log.debug(f"组合 {combo} 求解失败: {e}")

        if (len(records) + skipped) % 1000 == 0:
            log.info(f"  进度: {len(records)+skipped}/{total}  有效={len(records)}")

    log.info(f"完成：{len(records)} 个有效方案（跳过 {skipped} 个超成本）")

    if len(records) == 0:
        log.error("没有找到满足条件的方案！请放宽 cost_limit。")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("F_objective").reset_index(drop=True)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path / f"grid_search_{scenario.name[:4]}.csv", index=False)
        df.head(top_k).to_csv(out_path / f"top{top_k}_{scenario.name[:4]}.csv", index=False)
        log.info(f"结果已保存至 {out_path}")

    return df.head(top_k) if top_k else df


def differential_evolution_optimize(
    base_params,
    scenario,
    weights:    dict | None = None,
    maxiter:    int = 100,
    popsize:    int = 10,
    seed:       int = 42,
) -> dict:
    """
    使用 scipy differential_evolution 做连续空间全局优化。
    适合需要精细最优解的场景（比网格搜索更精确但更慢）。
    """
    from scipy.optimize import differential_evolution
    from model.solver import solve_seiqr, extract_summary

    if weights is None:
        weights = {"attack_rate": 0.50, "peak_pressure": 0.30, "cost": 0.20}

    # 基准
    p_base = scenario.apply_to(base_params)
    df_base = solve_seiqr(p_base)
    summ_base = extract_summary(df_base, p_base)
    baseline_AR  = summ_base["total_attack_rate"]
    baseline_PIP = summ_base["peak_I_rate"]

    bounds = [(0.0, 1.0)] * 7

    def obj(x):
        bundle = InterventionBundle(
            mask_level=x[0], ventilation=x[1], vaccination=x[2],
            isolation_rate=x[3], online_teaching=x[4],
            activity_limit=x[5], disinfection=x[6],
        )
        cost = compute_cost(bundle)
        try:
            rec = _evaluate_bundle(bundle, base_params, scenario)
            ar_n  = rec["attack_rate"] / max(baseline_AR, 1e-6)
            pip_n = rec["peak_I_rate"] / max(baseline_PIP, 1e-6)
            return (weights["attack_rate"] * ar_n +
                    weights["peak_pressure"] * pip_n +
                    weights["cost"] * cost)
        except Exception:
            return 1e6

    result = differential_evolution(
        obj, bounds,
        maxiter=maxiter, popsize=popsize, seed=seed,
        workers=1, polish=True,
        callback=lambda xk, convergence: log.info(f"  DE: F={obj(xk):.5f}")
    )

    x_opt = result.x
    best_bundle = InterventionBundle(
        mask_level=x_opt[0], ventilation=x_opt[1], vaccination=x_opt[2],
        isolation_rate=x_opt[3], online_teaching=x_opt[4],
        activity_limit=x_opt[5], disinfection=x_opt[6],
    )
    best_rec = _evaluate_bundle(best_bundle, base_params, scenario)

    log.info(f"DE 优化完成: F={result.fun:.5f}  AR={best_rec['attack_rate']:.4f}")
    log.info(f"最优方案: {best_bundle}")

    return {
        "bundle":     best_bundle,
        "result":     best_rec,
        "F_opt":      result.fun,
        "de_result":  result,
        "baseline_AR": baseline_AR,
    }
