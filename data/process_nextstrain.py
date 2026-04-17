#!/usr/bin/env python3
"""
data/process_nextstrain.py — P2/P3 Nextstrain H3N2 数据处理
=============================================================
P2 (高优先)：从 h3n2_global_ha.json 提取上海菌株 metadata
    → data/processed/shanghai_monthly_counts.csv
    → data/processed/shanghai_seasonal_intensity.csv

P3 (中优先)：从 h3n2_global_ha_tip-frequencies.json 提取支系频率
    → data/processed/clade_frequencies_wide.csv

实际文件与需求文档映射：
    metadata.json     → h3n2_global_ha.json      （Auspice v2 树形结构）
    frequencies.json  → h3n2_global_ha_tip-frequencies.json

运行方式：
    python data/process_nextstrain.py
    python data/process_nextstrain.py --skip-p3
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import date, timedelta
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────
RAW_DIR  = Path("data/raw/h3n2")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

TREE_FILE  = RAW_DIR / "h3n2_global_ha.json"
FREQ_FILE  = RAW_DIR / "h3n2_global_ha_tip-frequencies.json"
FLUNET_FILE = PROC_DIR / "weekly_positivity.csv"

# ── 上海 division 白名单 ──────────────────────────────────
SHANGHAI_DIVISIONS = {"Shanghai", "Shanghai municipality"}

# ── COVID 封控期 (上海 2022-03 ~ 2022-06) ────────────────
COVID_LOCKDOWN = {"2022-03", "2022-04", "2022-05", "2022-06"}

# ── 支系聚合规则（按 Shanghai 实际分布，4 个主组） ─────────
# 反映上海 H3N2 历史演化：旧株 → 3C.2a → 2a 系 → 1a.1
def _group_clade(clade: str) -> str:
    """将 Nextstrain 细分支系聚合到 4 个主要支系组。"""
    if not clade or clade == "unassigned":
        return "unclassified"
    if clade.startswith("1a"):
        return "1a.1"          # 2022 年起主导（最新）
    if clade in ("2a.3a.1", "2a.1b") or clade.startswith("3C.2a1b"):
        return "2a.3a.1+1b"    # 2019–2022 过渡期
    if clade.startswith("3C.2a"):
        return "3C.2a_branch"  # 2014–2020 主导
    if clade.startswith("3C"):
        return "3C_early"      # 2014 年前旧株
    return "other"


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════

def _get_tips(node: dict, out: list) -> None:
    """递归提取 Auspice v2 树的所有叶节点（tip）。"""
    if "children" not in node:
        out.append(node)
    else:
        for child in node["children"]:
            _get_tips(child, out)


def _decimal_to_date(decimal_year: float) -> date:
    """将小数年（如 2022.5）转换为 date 对象。"""
    year = int(floor(decimal_year))
    remainder = decimal_year - year
    day_of_year = int(remainder * 365)
    return date(year, 1, 1) + timedelta(days=day_of_year)


def _parse_date_precision(num_date_node: dict) -> tuple[str, str]:
    """
    从 num_date 节点推断日期精度并返回 (year_month_str, precision)。

    Auspice 规则：
    - 仅年份  (YYYY)     → confidence 区间跨度 > 0.9 年
    - 年-月   (YYYY-MM)  → confidence 跨度 0.07–0.9 年（约 1 个月）
    - 完整日期(YYYY-MM-DD)→ confidence 跨度 < 0.07 年（< 1 个月）

    返回
    -------
    year_month : str  'YYYY-MM' 格式（精度不足时取中值推算）
    precision  : str  'year-only' | 'year-month' | 'full'
    """
    value = num_date_node.get("value", 0.0)
    conf  = num_date_node.get("confidence", [value, value])
    span  = conf[1] - conf[0]

    d = _decimal_to_date(float(value))
    ym = d.strftime("%Y-%m")

    if span > 0.9:
        precision = "year-only"
    elif span > 0.07:
        precision = "year-month"
    else:
        precision = "full"

    return ym, precision


# ═══════════════════════════════════════════════════════════
#  P2：上海菌株 metadata 提取
# ═══════════════════════════════════════════════════════════

def process_p2(tree_file: Path = TREE_FILE) -> None:
    """P2 主流程：上海月度菌株计数 + 流行季强度统计。"""
    log.info("=== P2: 上海菌株 metadata 处理 ===")
    log.info(f"加载树文件: {tree_file.name}")
    with open(tree_file, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    # ── Step 1: 探测结构 ──────────────────────────────────
    top_keys = list(tree_data.keys())
    log.info(f"  顶层 keys: {top_keys}")
    # 已确认：Auspice v2 格式 → version / meta / tree

    # ── Step 2: 展开叶节点 ───────────────────────────────
    tips: list[dict] = []
    _get_tips(tree_data["tree"], tips)
    log.info(f"  叶节点总数: {len(tips):,}")

    # ── Step 3: 构建 DataFrame ───────────────────────────
    records = []
    for tip in tips:
        name  = tip["name"]
        attrs = tip.get("node_attrs", {})

        division = attrs.get("division", {}).get("value", "")
        country  = attrs.get("country",  {}).get("value", "")
        source   = attrs.get("source",   {}).get("value", "")
        clade    = attrs.get("clade_membership", {}).get("value", "unassigned")
        subclade = attrs.get("subclade", {}).get("value", "")
        region   = attrs.get("region",   {}).get("value", "")
        host     = attrs.get("host",     {}).get("value", "")

        num_date_node = attrs.get("num_date", {})
        if num_date_node:
            year_month, precision = _parse_date_precision(num_date_node)
        else:
            # 回退：用 year_month 字段
            year_month = attrs.get("year_month", {}).get("value", "")
            precision  = "year-month" if year_month else "unknown"

        records.append({
            "strain":         name,
            "year_month":     year_month,
            "date_precision": precision,
            "division":       division,
            "country":        country,
            "source":         source,
            "clade":          clade,
            "subclade":       subclade,
            "region":         region,
            "host":           host,
        })

    df_all = pd.DataFrame(records)
    log.info(f"  DataFrame shape: {df_all.shape}")

    # ── Step 4: 地理过滤（上海） ─────────────────────────
    mask_div = df_all["division"].isin(SHANGHAI_DIVISIONS)
    mask_src = df_all["source"].str.contains("Shanghai", na=False)
    df_sh = df_all[mask_div | mask_src].copy()
    log.info(f"  上海过滤后: {len(df_sh):,} 条  "
             f"(division匹配: {mask_div.sum()}, source匹配: {mask_src.sum()})")

    # ── Step 5: 去重（同一菌株名只保留首条） ─────────────
    n_before = len(df_sh)
    df_sh = df_sh.drop_duplicates(subset=["strain"], keep="first").copy()
    n_after = len(df_sh)
    log.info(f"  去重: {n_before} → {n_after}  (去除 {n_before - n_after} 条重复)")

    # ── Step 6: 日期处理 ─────────────────────────────────
    # year_month 已是 'YYYY-MM' 格式（或空）
    df_sh["collection_date"] = pd.to_datetime(
        df_sh["year_month"] + "-15",   # 取月中作为代表日
        format="%Y-%m-%d",
        errors="coerce"
    )
    n_na_dates = df_sh["collection_date"].isna().sum()
    if n_na_dates:
        log.warning(f"  {n_na_dates} 条记录日期解析失败，将被排除")

    # ── Step 7: 仅使用精度 ≥ YYYY-MM 的记录做月度聚合 ──
    df_monthly_src = df_sh[
        df_sh["date_precision"].isin(["year-month", "full"]) &
        df_sh["collection_date"].notna()
    ].copy()
    log.info(f"  可用于月度聚合: {len(df_monthly_src):,} 条  "
             f"(排除 year-only: {(df_sh['date_precision']=='year-only').sum()} 条)")

    # ── Step 8: COVID 封控标注 ───────────────────────────
    df_monthly_src["year_month_str"] = df_monthly_src["collection_date"].dt.strftime("%Y-%m")
    df_monthly_src["covid_flag"] = df_monthly_src["year_month_str"].isin(COVID_LOCKDOWN)

    # ── Step 9: 月度聚合 ─────────────────────────────────
    month_grp = df_monthly_src.groupby("year_month_str").agg(
        strain_count=("strain", "count"),
        covid_flag=("covid_flag", "max"),
    ).reset_index().rename(columns={"year_month_str": "year_month"})

    month_grp["year_month_dt"] = pd.to_datetime(month_grp["year_month"] + "-01")
    month_grp = month_grp.sort_values("year_month_dt")

    # 季节标注（上海双峰规律）
    def _season_label(ym: str) -> str:
        try:
            m = int(ym.split("-")[1])
        except Exception:
            return "off-season"
        if m in (1, 2, 3, 4):
            return "winter-spring"
        if m in (6, 7, 8, 9):
            return "summer"
        return "off-season"

    month_grp["season_label"] = month_grp["year_month"].apply(_season_label)

    out_monthly = PROC_DIR / "shanghai_monthly_counts.csv"
    month_grp[["year_month", "strain_count", "season_label", "covid_flag"]].to_csv(
        out_monthly, index=False
    )
    log.info(f"  月度计数已保存 → {out_monthly}  ({len(month_grp)} 行)")

    # ── Step 10: 流行季强度统计 ──────────────────────────
    df_monthly_src["flu_year"] = df_monthly_src["collection_date"].dt.year
    df_monthly_src["month"]    = df_monthly_src["collection_date"].dt.month

    season_rows = []
    for (flu_year, season_type), grp in df_monthly_src.groupby(["flu_year", "season_label"]) if False else []:
        pass  # placeholder

    # 按年 + 季节分组，找峰值月份
    df_monthly_src["season_period"] = df_monthly_src.apply(
        lambda r: f"{r['flu_year']}-{'winter' if r['month'] <= 6 else 'summer'}",
        axis=1
    )
    # 精确区分：1-4月为winter-spring，6-9月为summer
    df_monthly_src["season_period"] = df_monthly_src.apply(
        lambda r: (
            f"{r['flu_year']}-winter-spring"   if r['month'] in (1,2,3,4) else
            f"{r['flu_year']}-summer"          if r['month'] in (6,7,8,9) else
            "off-season"
        ),
        axis=1
    )

    intensity_rows = []
    for season_key, grp_s in df_monthly_src[df_monthly_src["season_period"] != "off-season"] \
            .groupby("season_period"):
        monthly_c = grp_s.groupby("year_month_str")["strain"].count()
        peak_month = monthly_c.idxmax() if len(monthly_c) else ""
        intensity_rows.append({
            "season":        season_key,
            "strain_count":  int(grp_s["strain"].count()),
            "peak_month":    peak_month,
        })

    df_intensity = pd.DataFrame(intensity_rows).sort_values("season")
    out_intensity = PROC_DIR / "shanghai_seasonal_intensity.csv"
    df_intensity.to_csv(out_intensity, index=False)
    log.info(f"  流行季强度已保存 → {out_intensity}  ({len(df_intensity)} 行)")

    # ── Step 11: 与 FluNet 相关性检验 ───────────────────
    if FLUNET_FILE.exists():
        flunet = pd.read_csv(FLUNET_FILE, parse_dates=["date"])
        flunet["year_month"] = flunet["date"].dt.strftime("%Y-%m")
        flunet_m = flunet.groupby("year_month")["h3n2_pos_rate"].mean().reset_index()
        merged = month_grp.merge(flunet_m, on="year_month", how="inner")
        if len(merged) >= 5:
            r, p = pearsonr(merged["strain_count"], merged["h3n2_pos_rate"])
            log.info(f"  与 FluNet 相关性: Pearson r={r:.3f}  p={p:.4f}  n={len(merged)}")
            if r < 0.5:
                log.warning("  r < 0.5：上海菌株计数与 FluNet 阳性率趋势不一致，论文中需解释（采样偏差/监测强度差异）")
        else:
            log.warning("  共同时间点不足 (<5)，跳过相关性检验")
    else:
        log.warning("  weekly_positivity.csv 不存在，跳过 FluNet 相关性检验")

    log.info("=== P2 完成 ===")
    return df_sh  # 供 P3 复用


# ═══════════════════════════════════════════════════════════
#  P3：支系频率提取
# ═══════════════════════════════════════════════════════════

def process_p3(
    df_strain_meta: pd.DataFrame | None = None,
    freq_file: Path = FREQ_FILE,
    tree_file: Path = TREE_FILE,
) -> None:
    """P3 主流程：支系频率时间序列 → 宽表 CSV。"""
    log.info("=== P3: 支系频率数据处理 ===")
    log.info(f"加载频率文件: {freq_file.name}")
    with open(freq_file, "r", encoding="utf-8") as f:
        freq_data = json.load(f)

    # ── Step 1: 结构识别 ─────────────────────────────────
    all_keys = list(freq_data.keys())
    non_strain_keys = [k for k in all_keys if not k.startswith("A/")]
    strain_keys     = [k for k in all_keys if k.startswith("A/")]
    log.info(f"  总 keys: {len(all_keys)}  菌株 keys: {len(strain_keys)}  "
             f"其他 keys: {non_strain_keys}")
    # 菌株级别频率（非支系级别），需要用树的 clade_membership 做聚合

    pivots = freq_data.get("pivots", [])
    if not pivots:
        log.error("  未找到 pivots 字段，退出 P3")
        return
    log.info(f"  Pivots: {len(pivots)} 个时间点  "
             f"范围 [{pivots[0]:.3f}, {pivots[-1]:.3f}]")

    # 时间轴转换：小数年 → date
    pivot_dates = [_decimal_to_date(p) for p in pivots]
    pivot_series = pd.to_datetime(pivot_dates)

    # ── Step 2: 获取全量菌株的 clade 归属（必须用完整树）──
    # 注意：df_strain_meta 只含上海菌株，不足以覆盖全球 2480 株
    log.info("  加载完整树以建立全局 clade 映射...")
    with open(tree_file, "r", encoding="utf-8") as f:
        tree_data = json.load(f)
    tips: list[dict] = []
    _get_tips(tree_data["tree"], tips)
    strain_clade_map = {
        tip["name"]: tip.get("node_attrs", {})
                        .get("clade_membership", {})
                        .get("value", "unassigned")
        for tip in tips
    }
    log.info(f"  clade 映射表: {len(strain_clade_map):,} 条（全树）")

    # ── Step 3: 按支系聚合频率 ───────────────────────────
    clade_groups = {
        "1a.1":          np.zeros(len(pivots)),
        "2a.3a.1+1b":    np.zeros(len(pivots)),
        "3C.2a_branch":  np.zeros(len(pivots)),
        "3C_early":      np.zeros(len(pivots)),
        "other":         np.zeros(len(pivots)),
        "unclassified":  np.zeros(len(pivots)),
    }

    n_matched = 0
    n_missing = 0
    for strain_name, freq_entry in freq_data.items():
        if strain_name in ("pivots", "generated_by") or not strain_name.startswith("A/"):
            continue
        freqs = freq_entry.get("frequencies", [])
        if len(freqs) != len(pivots):
            continue
        clade_raw = strain_clade_map.get(strain_name, "unassigned")
        group     = _group_clade(clade_raw)
        if group in clade_groups:
            clade_groups[group] += np.array(freqs, dtype=float)
            n_matched += 1
        else:
            n_missing += 1

    log.info(f"  频率聚合: 匹配 {n_matched:,} 株  未匹配 {n_missing:,} 株")

    # ── Step 4: 总频率验证 ────────────────────────────────
    total_freq = sum(clade_groups.values())
    max_dev = float(np.max(np.abs(total_freq - 1.0)))
    log.info(f"  各支系频率之和: 最大偏差 = {max_dev:.4f}  "
             + ("✓ 正常" if max_dev < 0.10 else "⚠ 偏差 >10%，有未归类支系"))

    # 将未归类部分归入 unclassified
    residual = np.maximum(0, 1.0 - total_freq)
    clade_groups["unclassified"] += residual

    # ── Step 5: 构建宽表 DataFrame ───────────────────────
    df_wide = pd.DataFrame({"date": pivot_series})
    for group_name, freqs in clade_groups.items():
        col = "clade_" + group_name.replace(".", "_").replace("+", "_")
        df_wide[col] = freqs

    df_wide["h3n2_total"] = total_freq.clip(0, None)

    # ── Step 6: 月度重采样 ───────────────────────────────
    df_wide.set_index("date", inplace=True)
    df_monthly = df_wide.resample("ME").mean().reset_index()
    df_monthly.rename(columns={"date": "month_end"}, inplace=True)
    df_monthly.insert(0, "date", df_monthly["month_end"].dt.to_period("M").astype(str))
    df_monthly.drop(columns=["month_end"], inplace=True)

    out_wide = PROC_DIR / "clade_frequencies_wide.csv"
    df_monthly.to_csv(out_wide, index=False)
    log.info(f"  宽表已保存 → {out_wide}  ({len(df_monthly)} 行 × {len(df_monthly.columns)} 列)")

    # ── Step 7: 夏季 H3N2 主导验证 ───────────────────────
    df_monthly["month_int"] = pd.to_datetime(df_monthly["date"] + "-01").dt.month
    summer = df_monthly[df_monthly["month_int"].isin([6, 7, 8])]

    log.info("\n  夏季（6–8月）H3N2 支系占比分析：")
    clade_cols = [c for c in df_monthly.columns if c.startswith("clade_")]
    for yr in range(2015, 2025):
        yr_data = summer[summer["date"].str.startswith(str(yr))]
        if yr_data.empty:
            continue
        total = float(yr_data["h3n2_total"].mean())
        h3n2_dom = total > 0.60
        dominant_clade = None
        if len(clade_cols) > 0:
            clade_means = yr_data[clade_cols].mean()
            dominant_clade = clade_means.idxmax().replace("clade_", "")
        flag = "✓ H3N2主导" if h3n2_dom else "— "
        log.info(f"    {yr}: H3N2总频率={total:.2f}  {flag}  主导支系={dominant_clade}")

    log.info("=== P3 完成 ===")


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Nextstrain H3N2 数据处理 (P2/P3)")
    parser.add_argument("--skip-p2", action="store_true", help="跳过 P2 metadata 处理")
    parser.add_argument("--skip-p3", action="store_true", help="跳过 P3 频率处理")
    parser.add_argument(
        "--tree",
        default=str(TREE_FILE),
        help=f"Auspice v2 树文件路径（默认: {TREE_FILE}）"
    )
    parser.add_argument(
        "--freq",
        default=str(FREQ_FILE),
        help=f"tip-frequencies 文件路径（默认: {FREQ_FILE}）"
    )
    args = parser.parse_args()

    tree_file = Path(args.tree)
    freq_file = Path(args.freq)

    if not tree_file.exists():
        log.error(f"树文件不存在: {tree_file}")
        sys.exit(1)

    df_meta = None
    if not args.skip_p2:
        df_meta = process_p2(tree_file=tree_file)

    if not args.skip_p3:
        if not freq_file.exists():
            log.error(f"频率文件不存在: {freq_file}")
        else:
            process_p3(df_meta, freq_file=freq_file, tree_file=tree_file)

    log.info("\n=== 全部完成 ===")
    log.info(f"输出目录: {PROC_DIR.resolve()}")


if __name__ == "__main__":
    main()
