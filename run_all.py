"""
run_all.py — 一键运行全流程
=============================
执行顺序：
    Step 0: 初始化参数（从 config.yaml + 季节参数 JSON）
    Step 1: 基础 ODE 求解与 R₀ 计算
    Step 2: 参数拟合（若 FluNet 数据已下载）
    Step 3: Bootstrap 置信带
    Step 4: 模型验证（RMSE/R²）
    Step 5: PRCC 敏感性分析
    Step 6: 三场景 × 无干预基准对比
    Step 7: 防控方案优化（场景二/三）
    Step 8: 生成全部论文图表

运行方式：
    python run_all.py                     # 全流程（含数据下载）
    python run_all.py --skip-data         # 跳过数据下载步骤
    python run_all.py --skip-calibration  # 跳过参数拟合（使用 config.yaml 默认值）
    python run_all.py --fast              # 快速模式（减少 Bootstrap/PRCC 样本数）
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── 路径配置 ────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "output" / "run_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_all")


def parse_args():
    parser = argparse.ArgumentParser(description="H3N2 校园传播动力学建模全流程")
    parser.add_argument("--skip-data",        action="store_true",
                        help="跳过 WHO FluNet 数据下载步骤")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="跳过参数拟合（使用 config.yaml 默认参数）")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过 PRCC 敏感性分析（节省时间）")
    parser.add_argument("--skip-optimization",action="store_true",
                        help="跳过防控方案网格优化（节省时间）")
    parser.add_argument("--fast",             action="store_true",
                        help="快速模式：减少样本数（Bootstrap n=100, PRCC n=200）")
    return parser.parse_args()


# ── Step 0: 初始化参数 ──────────────────────────────────────────────────────

def load_params(fast: bool = False):
    """从 config.yaml 和季节参数 JSON 加载模型参数。"""
    from model.params import ModelParams

    config_path = ROOT / "config.yaml"
    if config_path.exists():
        p = ModelParams.from_yaml(config_path)
        log.info(f"参数从 {config_path.name} 加载")
    else:
        p = ModelParams()
        log.warning("config.yaml 不存在，使用内置默认值")

    # 用 FluNet 季节参数更新（若存在）
    sp_path = DATA_DIR / "seasonal_params.json"
    if sp_path.exists():
        with open(sp_path) as f:
            sp = json.load(f)
        p = p.update(
            delta1=sp.get("delta1", p.delta1),
            delta2=sp.get("delta2", p.delta2),
            phi1=sp.get("phi1_rad", p.phi1),
            phi2=sp.get("phi2_rad", p.phi2),
        )
        log.info(f"季节参数已从 {sp_path.name} 更新: δ₁={p.delta1:.3f}, δ₂={p.delta2:.3f}")

    log.info(f"参数概览: {p}")
    return p


# ── Step 1: 基础求解与 R₀ ───────────────────────────────────────────────────

def step_baseline(p):
    """运行基础 SEIQR 求解，计算 R₀。"""
    from model.solver import solve_seiqr, extract_summary
    from model.r0 import compute_R0, compute_herd_immunity_threshold, r0_heatmap_data

    log.info("=" * 60)
    log.info("Step 1: 基础模型求解")
    log.info("=" * 60)

    df = solve_seiqr(p)
    summ = extract_summary(df, p)

    R0 = compute_R0(p)
    HIT = compute_herd_immunity_threshold(R0)

    log.info(f"R₀ = {R0:.4f}  HIT = {HIT:.2%}")
    log.info(f"峰值: {summ['peak_I_total']:.0f} 人  "
             f"第 {summ['peak_day']:.0f} 天  "
             f"感染率 {summ['peak_I_rate']:.2%}")
    log.info(f"累计发病率: {summ['total_attack_rate']:.2%}")

    # 图表1: 基础流行曲线
    from plots.epidemic_curve import plot_epidemic_curve, plot_beta_seasonal
    plot_epidemic_curve(
        df, title=f"H3N2 校园传播基础模型（R₀={R0:.2f}）",
        output_path=FIG_DIR / "01_epidemic_curve_baseline.png"
    )
    plot_beta_seasonal(
        df, output_path=FIG_DIR / "01_beta_seasonal.png"
    )

    # R₀ 热力图（β₀ × γ）
    from plots.sensitivity_plot import plot_r0_heatmap
    x_vals = np.linspace(0.10, 0.60, 30)
    y_vals = np.linspace(0.10, 0.50, 25)
    Z = r0_heatmap_data(p, "beta0", "gamma", x_vals, y_vals)
    plot_r0_heatmap(
        Z, x_vals, y_vals,
        param_x="beta0", param_y="gamma",
        title=f"R₀ 参数热力图（α={p.alpha:.2f}, p_iso={p.p_iso:.2f}）",
        output_path=FIG_DIR / "01_r0_heatmap.png"
    )

    return df, summ, R0


# ── Step 2: 参数拟合 ────────────────────────────────────────────────────────

def step_calibration(p, fast: bool = False):
    """用 FluNet 数据拟合模型参数。"""
    from calibration.fitting import fit_model, prepare_obs_timeseries

    log.info("=" * 60)
    log.info("Step 2: 参数拟合")
    log.info("=" * 60)

    weekly_path = DATA_DIR / "weekly_positivity.csv"
    if not weekly_path.exists():
        log.warning("weekly_positivity.csv 不存在，跳过拟合")
        return p, None

    obs_weekly = pd.read_csv(weekly_path)
    obs_df = prepare_obs_timeseries(obs_weekly, p)

    if len(obs_df) < 5:
        log.warning("有效观测数据点不足，跳过拟合")
        return p, None

    result = fit_model(
        obs_df, p,
        method="leastsq",
        max_nfev=500 if fast else 1000,
    )

    # 无论是否收敛都使用最优参数（"Tolerance too small" 只是无法估计误差棒，最优点仍有效）
    safe_params = {k: v for k, v in result.params_best.items() if hasattr(p, k)}
    p_cal = p.update(**safe_params)
    if result.success:
        log.info(f"拟合成功: RMSE={result.rmse:.5f}  R²={result.r_squared:.4f}")
    else:
        log.warning(f"拟合未收敛: {result.message}（仍使用最优参数）")
        log.info(f"最优 RMSE={result.rmse:.5f}  R²={result.r_squared:.4f}")
    # 保存最优参数
    param_path = DATA_DIR / "best_params.json"
    param_path.write_text(
        json.dumps(result.params_best, indent=2, ensure_ascii=False)
    )
    log.info(f"最优参数已保存 → {param_path}")
    return p_cal, result


# ── Step 3: Bootstrap 置信带 ────────────────────────────────────────────────

def step_bootstrap(p, obs_df, fast: bool = False):
    """Bootstrap 参数置信区间与轨迹置信带。"""
    from calibration.bootstrap import bootstrap_params, bootstrap_trajectory

    log.info("=" * 60)
    log.info("Step 3: Bootstrap 置信区间")
    log.info("=" * 60)

    n_boot = 100 if fast else 500
    ci_result = bootstrap_params(obs_df, p, n_bootstrap=n_boot)

    if ci_result:
        log.info("参数 95% CI:")
        for name, ci in ci_result.get("param_ci", {}).items():
            log.info(f"  {name}: [{ci['lo']:.4f}, {ci['hi']:.4f}]  mean={ci['mean']:.4f}")

    traj = bootstrap_trajectory(obs_df, p, n_bootstrap=min(n_boot, 200))
    return ci_result, traj


# ── Step 4: 模型验证 ────────────────────────────────────────────────────────

def step_validation(p, obs_df, ci_band=None, rho: float = 1.0):
    """计算验证指标，输出验证图。"""
    from model.solver import solve_seiqr
    from calibration.validation import compute_metrics
    from plots.epidemic_curve import plot_epidemic_curve

    log.info("=" * 60)
    log.info("Step 4: 模型验证")
    log.info("=" * 60)

    df_sim = solve_seiqr(p)
    metrics = compute_metrics(df_sim, obs_df, scale=rho)
    log.info(f"验证指标: {metrics.summary()}")

    # 验证图
    plot_epidemic_curve(
        df_sim, obs_df=obs_df, ci_band=ci_band,
        title=f"模型验证：模拟 vs WHO FluNet 数据（RMSE={metrics.rmse:.4f}, R²={metrics.r_squared:.3f}）",
        output_path=FIG_DIR / "04_validation.png"
    )
    return metrics


# ── Step 5: PRCC 敏感性分析 ─────────────────────────────────────────────────

def step_sensitivity(p, fast: bool = False):
    """运行 PRCC 敏感性分析，输出 Tornado 图。"""
    from sensitivity.prcc import run_prcc_analysis
    from plots.sensitivity_plot import plot_prcc_tornado, plot_r0_sensitivity_line
    from model.r0 import r0_sensitivity_table

    log.info("=" * 60)
    log.info("Step 5: PRCC 敏感性分析")
    log.info("=" * 60)

    n_samples = 200 if fast else 1000
    prcc_result = run_prcc_analysis(
        p, n_samples=n_samples,
        output_dir=ROOT / "output" / "sensitivity"
    )

    for out_name, df_prcc in prcc_result.get("prcc", {}).items():
        plot_prcc_tornado(
            df_prcc, output_name=out_name,
            title="PRCC 敏感性分析（龙卷风图）",
            output_path=FIG_DIR / f"05_prcc_{out_name}.png"
        )
        log.info(f"PRCC Tornado 图已保存: 05_prcc_{out_name}.png")

    # R₀ 单参数敏感性图
    from plots.sensitivity_plot import plot_r0_sensitivity_line
    beta_vals = np.linspace(0.05, 0.70, 50)
    r0_tbl = r0_sensitivity_table(p, "beta0", beta_vals)
    plot_r0_sensitivity_line(
        np.array(r0_tbl["values"]), r0_tbl["R0"], r0_tbl["HIT"],
        param_label="基础传播系数 β₀",
        base_val=p.beta0,
        output_path=FIG_DIR / "05_r0_beta0_sensitivity.png"
    )

    return prcc_result


# ── Step 6: 三场景基准对比 ──────────────────────────────────────────────────

def step_scenarios(p):
    """三场景无干预基准对比。"""
    from intervention.scenarios import SCENARIOS
    from model.solver import solve_seiqr, extract_summary
    from plots.intervention_compare import plot_scenario_comparison

    log.info("=" * 60)
    log.info("Step 6: 三场景基准对比")
    log.info("=" * 60)

    sim_results = {}
    summaries   = {}

    for sc_name, scenario in SCENARIOS.items():
        p_sc = scenario.apply_to(p)
        df   = solve_seiqr(p_sc)
        summ = extract_summary(df, p_sc)
        sim_results[scenario.name] = df
        summaries[sc_name] = summ
        log.info(
            f"[{sc_name}] 峰值: {summ['peak_I_rate']:.2%}  "
            f"第 {summ['peak_day']:.0f} 天  AR={summ['total_attack_rate']:.2%}"
        )

    plot_scenario_comparison(
        sim_results, metric="I_rate",
        title="三场景流感传播对比（无干预）",
        output_path=FIG_DIR / "06_scenario_comparison.png"
    )
    return sim_results, summaries


# ── Step 7: 防控方案优化 ────────────────────────────────────────────────────

def step_optimization(p, fast: bool = False):
    """对场景二/三进行防控方案网格优化。"""
    from intervention.scenarios import SCENARIOS
    from intervention.optimizer import grid_search
    from plots.intervention_compare import plot_intervention_heatmap, plot_pareto_frontier

    log.info("=" * 60)
    log.info("Step 7: 防控方案优化")
    log.info("=" * 60)

    n_levels = 3 if fast else 5

    for sc_name in ["outbreak", "cluster"]:
        scenario = SCENARIOS[sc_name]
        log.info(f"\n优化场景: {scenario.name}")

        df_opt = grid_search(
            p, scenario,
            n_levels=n_levels,
            cost_limit=0.65,
            output_dir=ROOT / "output" / "optimization",
            top_k=20,
        )

        if df_opt.empty:
            continue

        # 为热力图添加方案名
        df_opt["scheme_name"] = [f"方案{i+1}" for i in range(len(df_opt))]

        plot_intervention_heatmap(
            df_opt,
            title=f"防控效果热力图（{scenario.name}）",
            output_path=FIG_DIR / f"07_heatmap_{sc_name}.png"
        )

        if "cost_score" in df_opt.columns and "AR_reduction_pct" in df_opt.columns:
            plot_pareto_frontier(
                df_opt,
                title=f"成本-效益 Pareto 前沿（{scenario.name}）",
                output_path=FIG_DIR / f"07_pareto_{sc_name}.png"
            )

        log.info(f"Top-5 方案:")
        for _, row in df_opt.head(5).iterrows():
            log.info(
                f"  F={row.get('F_objective', 0):.4f}  "
                f"AR降低={row.get('AR_reduction_pct', 0):.1f}%  "
                f"Cost={row.get('cost_score', 0):.3f}"
            )

    return True


# ── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("H3N2 上海高校传播动力学建模 — 全流程启动")
    log.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # Step 0: 加载参数
    p = load_params(fast=args.fast)

    # (可选) Step -1: 数据下载
    if not args.skip_data:
        weekly_path = DATA_DIR / "weekly_positivity.csv"
        if not weekly_path.exists():
            log.info("正在运行 data/fetch_flunet.py 获取 WHO 数据...")
            import subprocess
            result = subprocess.run(
                [sys.executable, str(ROOT / "data" / "fetch_flunet.py")],
                capture_output=False, text=True
            )
            if result.returncode != 0:
                log.warning("数据下载失败，将使用默认参数继续")
        else:
            log.info(f"数据文件已存在: {weekly_path.name}，跳过下载")

    # Step 1: 基础求解
    df_baseline, summ_baseline, R0 = step_baseline(p)

    # Step 2: 参数拟合
    obs_df = None
    fit_result = None
    if not args.skip_calibration:
        weekly_path = DATA_DIR / "weekly_positivity.csv"
        if weekly_path.exists():
            from calibration.fitting import prepare_obs_timeseries
            obs_weekly = pd.read_csv(weekly_path)
            obs_df = prepare_obs_timeseries(obs_weekly, p)
            if len(obs_df) >= 5:
                p, fit_result = step_calibration(p, fast=args.fast)
            else:
                log.warning("观测数据不足，跳过拟合")
        else:
            log.warning("weekly_positivity.csv 不存在，跳过拟合（请先运行数据下载步骤）")

    # Step 3+4: Bootstrap + 验证（需要观测数据）
    obs_rho = fit_result.rho if (fit_result is not None and hasattr(fit_result, "rho")) else 1.0
    if obs_df is not None and len(obs_df) >= 5:
        ci_result, traj = step_bootstrap(p, obs_df, fast=args.fast)
        ci_band = traj if traj else None
        step_validation(p, obs_df, ci_band=ci_band, rho=obs_rho)
    else:
        log.info("跳过 Bootstrap 和验证步骤（无观测数据）")

    # Step 5: 敏感性分析
    if not args.skip_sensitivity:
        step_sensitivity(p, fast=args.fast)

    # Step 6: 三场景对比
    sim_results, summaries = step_scenarios(p)

    # Step 7: 防控优化
    if not args.skip_optimization:
        step_optimization(p, fast=args.fast)

    log.info("\n" + "=" * 70)
    log.info("全流程完成！")
    log.info(f"图表输出目录: {FIG_DIR}")
    log.info(f"R₀ = {R0:.4f}   峰值发病率 = {summ_baseline['peak_I_rate']:.2%}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
