"""
sensitivity/oat.py — One-At-a-Time 单参数敏感性扫描
=====================================================
对每个参数在 base×(1±variation) 范围内均匀取点，保持其余参数不变，
运行 ODE 记录 AR 与峰值感染率，用于绘制二维敏感性曲线。

与 PRCC 互补：PRCC 给出全局偏秩相关强度（多参数同时扰动），
OAT 给出单参数局部响应形状（非线性/单调性/临界点）。
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .prcc import _build_problem

log = logging.getLogger(__name__)


def run_oat_sensitivity(
    p_base,
    param_variation: float = 0.30,
    n_points: int = 31,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    单参数扫描敏感性分析。

    Args:
        p_base:          ModelParams 基准参数
        param_variation: 变动范围（±比例，默认 ±30%）
        n_points:        每参数取点数
        output_dir:      保存 CSV 的目录（None 则不保存）

    Returns:
        (results, labels)
        results[name] = DataFrame 列：value, attack_rate, peak_I_rate, peak_day
        labels[name]  = 中文标签（带单位符号）
    """
    from model.solver import solve_seiqr, extract_summary

    problem, labels = _build_problem(p_base, param_variation=param_variation)

    results: dict[str, pd.DataFrame] = {}

    for name, (lo, hi) in zip(problem["names"], problem["bounds"]):
        values = np.linspace(lo, hi, n_points)
        ar_list, peak_list, day_list = [], [], []

        for v in values:
            updates = {name: float(v)}
            if name == "c12":
                updates["c21"] = float(v)  # 保持对称
            try:
                p = p_base.update(**updates)
                df = solve_seiqr(p)
                summ = extract_summary(df, p)
                ar_list.append(summ["total_attack_rate"])
                peak_list.append(summ["peak_I_rate"])
                day_list.append(summ["peak_day"])
            except Exception as e:
                log.debug(f"OAT {name}={v}: {e}")
                ar_list.append(np.nan)
                peak_list.append(np.nan)
                day_list.append(np.nan)

        df_out = pd.DataFrame({
            "value":          values,
            "attack_rate":    ar_list,
            "peak_I_rate":    peak_list,
            "peak_day":       day_list,
        })
        results[name] = df_out

        log.info(
            f"OAT [{name:14s}] AR ∈ [{np.nanmin(ar_list):.3f}, {np.nanmax(ar_list):.3f}]  "
            f"Peak ∈ [{np.nanmin(peak_list):.3f}, {np.nanmax(peak_list):.3f}]"
        )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            df.to_csv(out_path / f"oat_{name}.csv", index=False)
        log.info(f"OAT 结果已保存至 {out_path}")

    return results, labels
