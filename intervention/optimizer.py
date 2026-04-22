"""
intervention/optimizer.py — 防控方案多目标优化
================================================
NSGA-II 三目标优化：
    F₁ = 累计感染率 AR（相对于无干预基准，归一化）
    F₂ = 经济成本评分 econ_score ∈ [0,1]
    F₃ = 教学干扰评分 teaching_score ∈ [0,1]

    三目标独立最小化，输出 Pareto 前沿。

辅助排序用加权综合指标：
    F_objective = w₁ × AR_norm + w₂ × econ_score + w₃ × teaching_score

优化策略：
    1. 参数网格扫描（暴力搜索，简单可靠，适合 4–5 维）
    2. scipy.optimize.differential_evolution（全局优化）
    3. NSGA-II 多目标遗传算法（pymoo，直接输出 Pareto 前沿）
"""

from __future__ import annotations
import logging
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .measures import InterventionBundle, apply_interventions, NO_INTERVENTION
from .cost_model import compute_cost, compute_economic_score, compute_teaching_score

log = logging.getLogger(__name__)


def _evaluate_bundle(bundle: InterventionBundle, base_params, scenario) -> dict:
    """对单个干预方案运行 ODE 并返回目标量（含三独立成本指标）。"""
    from model.solver import solve_seiqr, extract_summary

    p = scenario.apply_to(base_params)
    p = apply_interventions(p, bundle)
    df = solve_seiqr(p)
    summ = extract_summary(df, p)

    return {
        "attack_rate":      summ["total_attack_rate"],
        "peak_I_rate":      summ["peak_I_rate"],
        "peak_day":         summ["peak_day"],
        "cost_score":       compute_cost(bundle),           # 综合（向后兼容）
        "econ_score":       compute_economic_score(bundle),  # 纯经济成本
        "teaching_score":   compute_teaching_score(bundle),  # 纯教学干扰
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
    计算加权目标函数 F（三目标加权综合）。

    Args:
        bundle:       干预措施
        base_params:  ModelParams 基础参数
        scenario:     Scenario 实例
        baseline_AR:  无干预时的累计发病率
        baseline_PIP: 无干预时的峰值感染率（仅供兼容，不再用于目标）
        weights:      {'attack_rate': w1, 'econ': w2, 'teaching': w3}

    Returns:
        F（越小越好）
    """
    if weights is None:
        weights = {"attack_rate": 0.50, "econ": 0.25, "teaching": 0.25}

    result = _evaluate_bundle(bundle, base_params, scenario)

    ar_norm  = result["attack_rate"] / max(baseline_AR, 1e-6)

    F = (weights["attack_rate"] * ar_norm +
         weights.get("econ", 0.25) * result["econ_score"] +
         weights.get("teaching", 0.25) * result["teaching_score"])
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
        weights = {"attack_rate": 0.50, "econ": 0.25, "teaching": 0.25}

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
            F = (weights["attack_rate"]  * ar_norm +
                 weights.get("econ", 0.25) * rec["econ_score"] +
                 weights.get("teaching", 0.25) * rec["teaching_score"])

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


# ── NSGA-II 多目标优化 ─────────────────────────────────────────────────────

_U_FIELDS = [
    "mask_level", "ventilation", "vaccination",
    "isolation_rate", "online_teaching", "activity_limit", "disinfection"
]


def _bundle_from_vec(x) -> InterventionBundle:
    """将长度 7 的决策向量还原为 InterventionBundle（顺序与 _U_FIELDS 一致）。"""
    return InterventionBundle(
        mask_level     = float(x[0]),
        ventilation    = float(x[1]),
        vaccination    = float(x[2]),
        isolation_rate = float(x[3]),
        online_teaching= float(x[4]),
        activity_limit = float(x[5]),
        disinfection   = float(x[6]),
    )


def nsga2_optimize(
    base_params,
    scenario,
    pop_size:   int = 60,
    n_gen:      int = 50,
    cost_limit: float = 0.65,
    weights:    dict | None = None,
    seed:       int = 42,
    output_dir: str | Path | None = None,
    top_k:      int = 20,
    return_full: bool = False,
) -> pd.DataFrame:
    """
    使用 NSGA-II 做 3 目标最小化（三目标独立，无约束）：
        F₁ = AR_norm     累计感染率（相对基准）
        F₂ = econ_score  经济成本评分 [0,1]
        F₃ = teaching_score 教学干扰评分 [0,1]

    Args:
        base_params:  ModelParams 基础参数
        scenario:     Scenario 实例
        pop_size:     种群大小
        n_gen:        演化代数
        cost_limit:   保留参数（不再作为约束，仅用于过滤输出中极端高成本解）
        weights:      用于二次排序的加权 F 权重
        seed:         随机种子
        output_dir:   结果保存路径
        top_k:        按加权 F 升序返回的前 k 个方案

    Returns:
        DataFrame（top_k 条，含 econ_score / teaching_score 列）
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from model.solver import solve_seiqr, extract_summary

    if weights is None:
        weights = {"attack_rate": 0.50, "econ": 0.25, "teaching": 0.25}

    # 基准（无干预）
    p_base = scenario.apply_to(base_params)
    df_base = solve_seiqr(p_base)
    summ_base = extract_summary(df_base, p_base)
    baseline_AR  = summ_base["total_attack_rate"]
    baseline_PIP = summ_base["peak_I_rate"]

    log.info(f"基准: AR={baseline_AR:.4f}, PeakI={baseline_PIP:.4f}")
    log.info(f"NSGA-II: pop={pop_size}, n_gen={n_gen}, 3-obj(AR,econ,teaching)")

    class _InterventionProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=7, n_obj=3, n_ieq_constr=1,
                xl=np.zeros(7), xu=np.ones(7),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            bundle = _bundle_from_vec(x)
            try:
                rec = _evaluate_bundle(bundle, base_params, scenario)
            except Exception as e:
                log.debug(f"NSGA-II 评估失败 x={x}: {e}")
                out["F"] = [1e3, 1.0, 1.0]
                out["G"] = [1.0]
                return

            ar_norm = rec["attack_rate"] / max(baseline_AR, 1e-6)
            out["F"] = [ar_norm, rec["econ_score"], rec["teaching_score"]]
            out["G"] = [rec["cost_score"] - cost_limit]

    problem   = _InterventionProblem()
    algorithm = NSGA2(pop_size=pop_size)

    res = pymoo_minimize(
        problem, algorithm,
        termination=("n_gen", n_gen),
        seed=seed, verbose=False, save_history=False,
    )

    # 收集最终种群 + Pareto 前沿，去重后排序
    rows: list[dict] = []
    seen: set[tuple] = set()

    def _ingest(x_vec):
        bundle = _bundle_from_vec(x_vec)
        key = tuple(round(v, 4) for v in x_vec)
        if key in seen:
            return key
        seen.add(key)
        try:
            rec = _evaluate_bundle(bundle, base_params, scenario)
        except Exception:
            return key
        ar_norm  = rec["attack_rate"] / max(baseline_AR, 1e-6)
        pip_norm = rec["peak_I_rate"] / max(baseline_PIP, 1e-6)
        F = (weights["attack_rate"]         * ar_norm +
             weights.get("econ", 0.25)      * rec["econ_score"] +
             weights.get("teaching", 0.25)  * rec["teaching_score"])
        rec["mask_level"]      = bundle.mask_level
        rec["ventilation"]     = bundle.ventilation
        rec["vaccination"]     = bundle.vaccination
        rec["isolation_rate"]  = bundle.isolation_rate
        rec["online_teaching"] = bundle.online_teaching
        rec["activity_limit"]  = bundle.activity_limit
        rec["disinfection"]    = bundle.disinfection
        rec["_dedup_key"]        = key
        rec["F_objective"]       = F
        rec["AR_reduction_pct"]  = (1 - ar_norm) * 100
        rec["PIP_reduction_pct"] = (1 - pip_norm) * 100
        rec["_pareto_rank0"]     = False
        rows.append(rec)
        return key

    # Pareto 前沿（rank 0）先入表
    pareto_X = res.X
    if pareto_X is not None and pareto_X.size > 0:
        if pareto_X.ndim == 1:
            pareto_X = pareto_X.reshape(1, -1)
        pareto_keys = {tuple(round(v, 4) for v in x_vec) for x_vec in pareto_X}
        for x_vec in pareto_X:
            _ingest(x_vec)
    else:
        pareto_keys = set()

    # 最终种群（含被支配解，可观察多样性）
    X_final = res.pop.get("X") if res.pop is not None else None
    if X_final is not None:
        for x_vec in X_final:
            _ingest(x_vec)

    for rec in rows:
        if rec["_dedup_key"] in pareto_keys:
            rec["_pareto_rank0"] = True

    if not rows:
        log.error("NSGA-II 未产生可行解！请增加 n_gen 或 pop_size。")
        return pd.DataFrame()

    df_full   = pd.DataFrame(rows).drop(columns=["_dedup_key"]).sort_values("F_objective").reset_index(drop=True)
    df_pareto = df_full[df_full["_pareto_rank0"]].sort_values("F_objective").reset_index(drop=True)

    log.info(f"NSGA-II 完成：可行解 {len(df_full)} 个，Pareto 前沿 {len(df_pareto)} 个")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tag = scenario.name[:4]
        df_full.to_csv(out_path / f"nsga2_full_{tag}.csv", index=False)
        df_pareto.to_csv(out_path / f"nsga2_pareto_{tag}.csv", index=False)
        df_full.head(top_k).to_csv(out_path / f"top{top_k}_{tag}.csv", index=False)
        log.info(f"结果已保存至 {out_path}")

    df_top = df_full.head(top_k) if top_k else df_full
    if return_full:
        return df_top, df_full
    return df_top
