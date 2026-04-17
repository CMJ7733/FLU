"""
sensitivity/sobol.py — Sobol 全局敏感性分析（可选）
=====================================================
使用 SALib 的 Saltelli 采样计算 Sobol 一阶指数 S1 和总效应指数 ST。
Sobol 方法比 PRCC 计算量更大（需要 N×(2D+2) 次模型运行），
但可捕捉参数交互效应。

建议在 n_samples=256 时运行，约需 256×22=5632 次 ODE 求解。
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze
    _HAS_SALIB_SOBOL = True
except ImportError:
    _HAS_SALIB_SOBOL = False
    log.warning("SALib Sobol 分析不可用，请确保安装了 SALib >= 1.4")


def run_sobol_analysis(
    p_base,
    n_samples: int = 256,
    param_variation: float = 0.30,
    seed: int = 42,
    output_name: str = "peak_infection_rate",
    output_dir: str | Path | None = None,
) -> Optional[pd.DataFrame]:
    """
    Sobol 全局敏感性分析。

    Args:
        p_base:          ModelParams 基础参数
        n_samples:       Saltelli 基础样本数 N（总运行次数 ≈ N×(2D+2)）
        param_variation: 参数范围 ±比例
        seed:            随机种子
        output_name:     目标量名称（peak_infection_rate / total_attack_rate / peak_timing_days）
        output_dir:      结果保存目录

    Returns:
        DataFrame（含 S1, ST 及各自 CI），或 None（SALib 不可用时）
    """
    if not _HAS_SALIB_SOBOL:
        log.error("SALib 未安装或版本不支持 Sobol，请 pip install SALib")
        return None

    from sensitivity.prcc import _build_problem, _evaluate_samples

    problem, labels = _build_problem(p_base, param_variation)
    n_params = problem["num_vars"]
    total_runs = n_samples * (2 * n_params + 2)
    log.info(f"Sobol 分析：{n_params} 参数，N={n_samples}，共 {total_runs} 次 ODE 求解")

    # Saltelli 采样
    X = saltelli.sample(problem, n_samples, calc_second_order=False, seed=seed)

    # 运行模型
    outputs = _evaluate_samples(X, problem, p_base)
    Y = outputs.get(output_name, outputs[list(outputs.keys())[0]])

    # 处理 NaN
    nan_mask = np.isnan(Y)
    if nan_mask.sum() > 0:
        Y[nan_mask] = np.nanmean(Y)
        log.warning(f"替换了 {nan_mask.sum()} 个 NaN 值")

    # Sobol 分析
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                print_to_console=False, seed=seed)

    df = pd.DataFrame({
        "parameter": problem["names"],
        "label":     [labels[n] for n in problem["names"]],
        "S1":        Si["S1"],
        "S1_conf":   Si["S1_conf"],
        "ST":        Si["ST"],
        "ST_conf":   Si["ST_conf"],
    })
    df = df.sort_values("ST", ascending=False).reset_index(drop=True)

    log.info(f"\n--- Sobol 分析: {output_name} ---\n{df.to_string(index=False)}")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path / f"sobol_{output_name}.csv", index=False)
        log.info(f"Sobol 结果已保存至 {out_path}")

    return df
