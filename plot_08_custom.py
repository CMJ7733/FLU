"""
plot_08_custom.py — 手动配置干预方案，生成 08 干预前后对比图
================================================================

使用说明：
    1. 在 SCENARIOS 配置区设置场景和 u 值
    2. 修改 OUTPUT_PATH 指定输出路径
    3. 运行：python plot_08_custom.py

场景配置说明：
    - scenario_key: "baseline" / "outbreak" / "cluster"（决定初始感染人数 I0_1/I0_2）
    - t_start_doy:  305=冬季(11月), 121=夏季(5月)
    - u1~u7:        各干预措施强度 [0, 1]，0=不实施，1=全强度
"""

import sys
from pathlib import Path

# ── 手动配置区 ───────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name":        "场景一：常态散发（冬季）",
        "scenario_key": "baseline",
        "t_start_doy":  305,
        "u1_mask":        1.0,
        "u2_ventilation": 1.0,
        "u3_vaccination": 1.0,
        "u4_isolation":   1.0,
        "u5_online":      1.0,
        "u6_activity":    1.0,
        "u7_disinfect":   1.0,
    },
    {
        "name":        "场景二：局部散发暴发（冬季）",
        "scenario_key": "outbreak",
        "t_start_doy":  305,
        "u1_mask":        1.0,
        "u2_ventilation": 1.0,
        "u3_vaccination": 1.0,
        "u4_isolation":   1.0,
        "u5_online":      1.0,
        "u6_activity":    1.0,
        "u7_disinfect":   1.0,
    },
    {
        "name":        "场景三：宿舍聚集疫情（冬季）",
        "scenario_key": "cluster",
        "t_start_doy":  305,
        "u1_mask":        1.0,
        "u2_ventilation": 1.0,
        "u3_vaccination": 1.0,
        "u4_isolation":   1.0,
        "u5_online":      1.0,
        "u6_activity":    1.0,
        "u7_disinfect":   1.0,
    },
]

OUTPUT_PATH = "output/figures/08_custom.png"
TITLE       = "最优防控方案干预前后感染曲线对比"

# ── 以下无需修改 ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.params import ModelParams
from model.solver import solve_seiqr, extract_summary
from intervention.scenarios import SCENARIOS as SCENARIO_DEFS
from intervention.measures import InterventionBundle, apply_interventions
from plots.intervention_compare import COLORS, setup_style


def run_single_scenario(cfg: dict) -> dict:
    """运行单个场景，返回 comparison 字典条目。"""
    sc_key  = cfg["scenario_key"]
    doy     = cfg["t_start_doy"]
    u_vals  = {
        "mask_level":      cfg["u1_mask"],
        "ventilation":     cfg["u2_ventilation"],
        "vaccination":     cfg["u3_vaccination"],
        "isolation_rate":  cfg["u4_isolation"],
        "online_teaching": cfg["u5_online"],
        "activity_limit":  cfg["u6_activity"],
        "disinfection":    cfg["u7_disinfect"],
    }

    # 加载参数（从 config.yaml + best_params.json）
    p = ModelParams.from_yaml(ROOT / "config.yaml")
    bp_path = ROOT / "data" / "processed" / "best_params.json"
    if bp_path.exists():
        best = json.load(open(bp_path))
        p = p.update(**{k: v for k, v in best.items() if hasattr(p, k)})

    # 应用场景（修改 t_start_doy 和 I0）
    scenario_def = SCENARIO_DEFS[sc_key]
    p_sc = scenario_def.apply_to(p)
    p_sc._t_start_doy = doy

    # 无干预基准
    df_before = solve_seiqr(p_sc)
    summ_b    = extract_summary(df_before, p_sc)

    # 干预后
    bundle    = InterventionBundle(**u_vals)
    p_int     = apply_interventions(p_sc, bundle)
    df_after  = solve_seiqr(p_int)
    summ_a    = extract_summary(df_after, p_int)

    ar_b  = summ_b["total_attack_rate"]
    ar_a  = summ_a["total_attack_rate"]
    pk_b  = summ_b["peak_I_rate"]
    pk_a  = summ_a["peak_I_rate"]

    print(f"[{cfg['name']}]")
    print(f"  u = ({bundle.mask_level:.1f}, {bundle.ventilation:.1f}, "
          f"{bundle.vaccination:.1f}, {bundle.isolation_rate:.1f}, "
          f"{bundle.online_teaching:.1f}, {bundle.activity_limit:.1f}, "
          f"{bundle.disinfection:.1f})")
    print(f"  AR:   {ar_b:.1%} → {ar_a:.1%}  (↓{(1-ar_a/ar_b)*100:.1f}%)")
    print(f"  Peak: {pk_b:.1%} → {pk_a:.1%}  (↓{(1-pk_a/pk_b)*100:.1f}%)")

    return {
        cfg["name"]: {
            "before": df_before,
            "after":  df_after,
            "bundle": bundle,
            "ar_before":   ar_b,
            "ar_after":    ar_a,
            "peak_before": pk_b,
            "peak_after":  pk_a,
        }
    }


def plot_comparison(comparison: dict, title: str, output_path):
    """绘制干预前后对比图（直接复制 intervention_compare.py 核心逻辑）。"""
    setup_style()

    names = list(comparison.keys())
    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    axes = axes[0]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    for ax, name in zip(axes, names):
        d    = comparison[name]
        df_b = d["before"]
        df_a = d["after"]
        metric = "I_rate"

        ax.plot(df_b["t"], df_b[metric],
                color=COLORS["baseline"], linewidth=2.2, linestyle="--",
                label="无干预", alpha=0.9)
        ax.plot(df_a["t"], df_a[metric],
                color=COLORS["danger"], linewidth=2.5, linestyle="-",
                label="干预方案", alpha=0.95)

        ib = int(df_b[metric].values.argmax())
        ia = int(df_a[metric].values.argmax())
        ax.scatter([df_b["t"].iloc[ib]], [df_b[metric].iloc[ib]],
                   color=COLORS["baseline"], s=40, zorder=5)
        ax.scatter([df_a["t"].iloc[ia]], [df_a[metric].iloc[ia]],
                   color=COLORS["danger"], s=50, zorder=5)

        ar_b, ar_a = d["ar_before"], d["ar_after"]
        pk_b, pk_a = d["peak_before"], d["peak_after"]
        ar_a = round(ar_a, 4)
        pk_a = round(pk_a, 4)
        ar_red = (1 - ar_a / max(ar_b, 1e-9)) * 100
        pk_red = (1 - pk_a / max(pk_b, 1e-9)) * 100
        bundle = d["bundle"]
        u_text = (
            f"u = ({bundle.mask_level:.1f}, {bundle.ventilation:.1f}, "
            f"{bundle.vaccination:.1f}, {bundle.isolation_rate:.1f}, "
            f"{bundle.online_teaching:.1f}, {bundle.activity_limit:.1f}, "
            f"{bundle.disinfection:.1f})"
        )
        info = (
            f"AR:   {ar_b:.1%} → {ar_a:.1%}  (↓{ar_red:.1f}%)\n"
            f"Peak: {pk_b:.1%} → {pk_a:.1%}  (↓{pk_red:.1f}%)\n"
            f"{u_text}"
        )
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="white", edgecolor=COLORS["baseline"], alpha=0.9))

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("模拟天数（天）")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.set_ylabel("感染率")
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n图片已保存: {output_path}")


def main():
    print("=" * 60)
    print("08 图生成脚本（手动配置 u 值）")
    print("=" * 60)

    comparison = {}
    for cfg in SCENARIOS:
        comparison.update(run_single_scenario(cfg))

    print()
    plot_comparison(comparison, TITLE, OUTPUT_PATH)


if __name__ == "__main__":
    main()
