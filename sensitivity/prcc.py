"""
sensitivity/prcc.py — PRCC 偏秩相关系数敏感性分析
====================================================
使用 SALib 的 Latin Hypercube 采样（LHS）生成参数组合，
对每组参数运行 ODE 模型，计算目标量，
最后通过偏秩相关（PRCC）量化各参数对目标量的影响。

目标量（outputs）：
    peak_infection_rate  峰值感染率（I_total/N 最大值）
    total_attack_rate    累计发病率（仿真结束时 R/(N-R0)）
    peak_timing_days     峰值时间（天）

参数范围：默认为基础值 ± 30%（在 config.yaml 中配置）
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from SALib.sample import latin
    from SALib.analyze import rbd_fast
    _HAS_SALIB = True
except ImportError:
    try:
        from SALib.sample.latin import sample as latin_sample
        _HAS_SALIB = True
    except ImportError:
        _HAS_SALIB = False
        log.warning("SALib 未安装，将使用简化 PRCC 实现。pip install SALib")


def _build_problem(p_base, param_variation: float = 0.30) -> dict:
    """
    构建 SALib 问题字典。
    参数范围：当前值 × (1 ± param_variation)。
    """
    params_to_analyze = {
        "beta0":        (p_base.beta0,   "基础传播系数 β₀"),
        "sigma":        (p_base.sigma,   "潜伏转感染率 σ"),
        "gamma":        (p_base.gamma,   "康复率 γ"),
        "alpha":        (p_base.alpha,   "病例隔离率 α"),
        "p_iso":        (p_base.p_iso,   "隔离成功比例 p_iso"),
        "delta1":       (p_base.delta1,  "年季节振幅 δ₁"),
        "delta2":       (p_base.delta2,  "半年季节振幅 δ₂"),
        "vax_coverage": (p_base.vax_coverage, "疫苗接种率"),
        "c11":          (p_base.c11,     "学生-学生接触率 c₁₁"),
        "c12":          (p_base.c12,     "学生-教职工接触率 c₁₂"),
    }

    names  = []
    bounds = []
    labels = {}

    for name, (base_val, label) in params_to_analyze.items():
        lo = base_val * (1 - param_variation)
        hi = base_val * (1 + param_variation)
        # 防止下界小于0（对于率参数）
        lo = max(lo, 1e-6)
        hi = max(hi, lo + 1e-6)
        names.append(name)
        bounds.append([lo, hi])
        labels[name] = label

    problem = {
        "num_vars": len(names),
        "names":    names,
        "bounds":   bounds,
    }
    return problem, labels


def _lhs_sample(problem: dict, n_samples: int, seed: int = 42) -> np.ndarray:
    """Latin Hypercube 采样。"""
    if _HAS_SALIB:
        try:
            from SALib.sample import latin as lhs_module
            return lhs_module.sample(problem, n_samples, seed=seed)
        except Exception:
            pass
        try:
            from SALib.sample.latin import sample
            return sample(problem, n_samples, seed=seed)
        except Exception:
            pass

    # fallback: 纯 numpy LHS
    log.warning("使用 numpy 简化 LHS 采样")
    rng = np.random.default_rng(seed)
    n_vars = problem["num_vars"]
    result = np.zeros((n_samples, n_vars))
    for j, (lo, hi) in enumerate(problem["bounds"]):
        cuts = np.linspace(0, 1, n_samples + 1)
        u    = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(u)
        result[:, j] = lo + u * (hi - lo)
    return result


def _evaluate_samples(
    param_matrix: np.ndarray,
    problem: dict,
    p_base,
) -> dict[str, np.ndarray]:
    """
    对每组参数样本运行 ODE，返回目标量字典。
    """
    from model.solver import solve_seiqr, extract_summary

    n = param_matrix.shape[0]
    outputs = {
        "peak_infection_rate": np.zeros(n),
        "total_attack_rate":   np.zeros(n),
        "peak_timing_days":    np.zeros(n),
    }

    log.info(f"PRCC 评估：{n} 组参数样本")
    for i in range(n):
        # 更新参数
        updates = dict(zip(problem["names"], param_matrix[i]))
        # 确保 c21 = c12（互惠对称）
        if "c12" in updates:
            updates["c21"] = updates["c12"]
        try:
            p = p_base.update(**updates)
            df = solve_seiqr(p)
            summ = extract_summary(df, p)
            outputs["peak_infection_rate"][i] = summ["peak_I_rate"]
            outputs["total_attack_rate"][i]   = summ["total_attack_rate"]
            outputs["peak_timing_days"][i]    = summ["peak_day"]
        except Exception as e:
            log.debug(f"样本 {i} 求解失败: {e}")
            outputs["peak_infection_rate"][i] = np.nan
            outputs["total_attack_rate"][i]   = np.nan
            outputs["peak_timing_days"][i]    = np.nan

        if (i + 1) % 100 == 0:
            log.info(f"  进度: {i+1}/{n}")

    return outputs


def _compute_prcc_scipy(
    X: np.ndarray,
    Y: np.ndarray,
    param_names: list[str],
) -> pd.DataFrame:
    """
    用 scipy 实现 PRCC（偏秩相关系数）。
    通过排秩后做线性回归残差的相关性计算。
    """
    from scipy import stats

    n, k = X.shape
    # 排秩
    X_rank = np.column_stack([stats.rankdata(X[:, j]) for j in range(k)])
    Y_rank = stats.rankdata(Y)

    prcc_vals = []
    pvals     = []

    for j in range(k):
        # 对 X[:,j] 对其余 X 做回归取残差
        other = np.delete(X_rank, j, axis=1)
        ones  = np.ones((n, 1))
        Xo    = np.hstack([ones, other])
        # 最小二乘
        coef_x, *_ = np.linalg.lstsq(Xo, X_rank[:, j], rcond=None)
        coef_y, *_ = np.linalg.lstsq(Xo, Y_rank, rcond=None)
        res_x = X_rank[:, j] - Xo @ coef_x
        res_y = Y_rank         - Xo @ coef_y

        # Pearson 相关
        r, p = stats.pearsonr(res_x, res_y)
        prcc_vals.append(r)
        pvals.append(p)

    return pd.DataFrame({
        "parameter": param_names,
        "PRCC":      prcc_vals,
        "p_value":   pvals,
    })


def run_prcc_analysis(
    p_base,
    n_samples: int = 1000,
    param_variation: float = 0.30,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict:
    """
    完整 PRCC 敏感性分析流程。

    Args:
        p_base:          ModelParams 基础参数
        n_samples:       LHS 采样数（建议 ≥ 500）
        param_variation: 参数扫描范围（±比例）
        seed:            随机种子
        output_dir:      结果保存目录（None 则不保存）

    Returns:
        dict:
            problem:   SALib 问题字典
            samples:   参数样本矩阵
            outputs:   目标量字典
            prcc:      各目标量的 PRCC 表 {output_name: DataFrame}
    """
    problem, labels = _build_problem(p_base, param_variation)
    log.info(f"PRCC 分析：{problem['num_vars']} 个参数，{n_samples} 个样本")

    # 1. 采样
    X = _lhs_sample(problem, n_samples, seed=seed)

    # 2. 运行模型
    outputs = _evaluate_samples(X, problem, p_base)

    # 3. 计算 PRCC（剔除 NaN）
    valid = ~np.any(np.isnan(np.column_stack(list(outputs.values()))), axis=1)
    X_clean = X[valid]

    prcc_tables = {}
    for out_name, Y_all in outputs.items():
        Y_clean = Y_all[valid]
        if len(Y_clean) < 10:
            log.warning(f"{out_name}: 有效样本不足，跳过")
            continue
        df_prcc = _compute_prcc_scipy(X_clean, Y_clean, problem["names"])
        df_prcc["label"] = df_prcc["parameter"].map(labels)
        df_prcc = df_prcc.sort_values("PRCC", key=abs, ascending=False)
        prcc_tables[out_name] = df_prcc
        log.info(f"\n--- PRCC: {out_name} ---\n{df_prcc.to_string(index=False)}")

    # 4. 保存结果
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for out_name, df in prcc_tables.items():
            df.to_csv(out_path / f"prcc_{out_name}.csv", index=False)
        log.info(f"PRCC 结果已保存至 {out_path}")

    return {
        "problem":  problem,
        "labels":   labels,
        "samples":  X,
        "outputs":  outputs,
        "prcc":     prcc_tables,
        "n_valid":  int(valid.sum()),
    }
