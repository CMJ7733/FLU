"""
WHO FluNet VIW_FNT.csv 中国 H3N2 数据处理脚本
===============================================
支持两种工作模式：
  1. 本地模式（默认）：读取 data/raw/VIW_FNT.csv（WHO FluNet 完整导出文件）
  2. 下载模式（--download）：从 WHO API 下载后处理

VIW_FNT.csv 为 WHO FluNet 官方全球数据导出格式（53列），
本脚本自动筛选中国（CHN）数据并提取 H3N2 核心指标。

运行方式：
    python data/fetch_flunet.py                          # 使用 data/raw/VIW_FNT.csv
    python data/fetch_flunet.py --input 自定义路径.csv
    python data/fetch_flunet.py --download               # 从 WHO API 下载
    python data/fetch_flunet.py --year-from 2015 --year-to 2024

输出文件：
    data/processed/weekly_positivity.csv   中国 H3N2 周度阳性率时间序列
    data/processed/seasonal_params.json   双谐波季节参数（δ₁, δ₂, φ₁, φ₂）
    data/processed/seasonal_summary.csv   年度峰值汇总
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("flunet")

# ── 路径配置 ────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── WHO FluNet 下载地址（备用） ──────────────────────────────────────────────
FLUNET_CSV_URL = (
    "https://frontdoor-l4uikgap6gz3m.azurefd.net"
    "/FLUMART/GetFluMart?SurvStat=FluNet"
    "&ISOCountryCode=CHN&csv=1"
)

# ── VIW_FNT.csv 字段映射（53列完整格式）────────────────────────────────────
# 保留建模所需字段，忽略其余53列中的非流感指标
VIW_FNT_FIELD_MAP = {
    # 时空标识
    "WHOREGION":              "who_region",
    "FLUSEASON":              "flu_season",
    "HEMISPHERE":             "hemisphere",
    "ISO_YEAR":               "iso_year",
    "ISO_WEEK":               "iso_week",
    "ISO_WEEKSTARTDATE":      "week_start_date",   # 直接使用，无需推算
    "COUNTRY_CODE":           "country_code",       # 过滤用：CHN
    "COUNTRY_AREA_TERRITORY": "country_name",
    "ORIGIN_SOURCE":          "origin_source",
    # 检测量
    "SPEC_RECEIVED_NB":       "spec_received",
    "SPEC_PROCESSED_NB":      "spec_processed",
    # 甲型流感分型
    "AH1N12009":              "ah1_2009",     # A(H1N1)pdm09 — 2009年后主力H1N1
    "AH1":                    "ah1_old",      # 季节性H1N1（2009年前，通常为空）
    "AH3":                    "ah3",          # A(H3N2) ← 核心建模目标
    "AH5":                    "ah5",
    "AH7N9":                  "ah7n9",
    "ANOTSUBTYPED":           "a_unsubtyped",
    "ANOTSUBTYPABLE":         "a_not_subtypable",
    # 乙型流感（B/Victoria 拆分为4个删除型子列，需合并）
    "BVIC_2DEL":              "bvic_2del",    # B/Victoria 双缺失突变
    "BVIC_3DEL":              "bvic_3del",    # B/Victoria 三缺失突变
    "BVIC_NODEL":             "bvic_nodel",   # B/Victoria 无缺失
    "BVIC_DELUNK":            "bvic_delunk",  # B/Victoria 缺失型未知
    "BYAM":                   "b_yamagata",   # B/Yamagata（注意：非BYAMAGATA）
    "BNOTDETERMINED":         "b_nd",
    # 汇总
    "INF_A":                  "inf_a",
    "INF_B":                  "inf_b",
    "INF_ALL":                "inf_all",
    "INF_NEGATIVE":           "inf_negative",
    "ILI_ACTIVITY":           "ili_activity",
}

# 需要数值化的关键列（clean后名称）
NUMERIC_COLS = [
    "spec_received", "spec_processed",
    "ah1_2009", "ah1_old", "ah3", "ah5", "ah7n9",
    "a_unsubtyped", "a_not_subtypable",
    "bvic_2del", "bvic_3del", "bvic_nodel", "bvic_delunk",
    "b_yamagata", "b_nd",
    "inf_a", "inf_b", "inf_all", "inf_negative",
]

# ── 数据加载 ────────────────────────────────────────────────────────────────

def load_viwfnt(path: Path) -> pd.DataFrame:
    """
    加载 VIW_FNT.csv 文件。

    自动处理：
    - UTF-8 / GBK 编码
    - 列名前后空格
    - 空值（''）→ NaN
    """
    log.info(f"加载 VIW_FNT 文件: {path.name}  ({path.stat().st_size // 1024} KB)")
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                low_memory=False,
                na_values=["", " ", "N/A", "NA", "nan"],
                keep_default_na=True,
                on_bad_lines="warn",
            )
            df.columns = df.columns.str.strip()
            log.info(f"  编码={enc}  行数={len(df):,}  列数={len(df.columns)}")
            return df
        except UnicodeDecodeError as e:
            log.debug(f"  {enc} 解码失败: {e}")
            continue
    raise RuntimeError(f"无法解码文件: {path}")


def download_flunet(url: str, out_path: Path, timeout: int = 90) -> pd.DataFrame:
    """从 WHO FluNet API 下载全量数据（备用模式）。"""
    import requests
    log.info(f"下载 WHO FluNet 数据 → {out_path.name}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    log.info(f"下载完成 ({out_path.stat().st_size // 1024} KB)")
    return pd.read_csv(out_path, low_memory=False, na_values=["", " "])

# ── 数据清洗 ────────────────────────────────────────────────────────────────

def clean_viwfnt(
    df: pd.DataFrame,
    country_code: str = "CHN",
    year_from: int = 2015,
    year_to: int = 2024,
) -> pd.DataFrame:
    """
    清洗 VIW_FNT.csv 数据，提取中国 H3N2 核心指标。

    处理步骤：
      1. 重命名字段（保留已知列）
      2. 筛选目标国家（COUNTRY_CODE = CHN）
      3. 筛选时间范围（year_from ≤ ISO_YEAR ≤ year_to）
      4. 解析日期列（优先用 ISO_WEEKSTARTDATE，回退推算）
      5. 数值化所有量化字段（空值→NaN）
      6. 合并 B/Victoria 四个子列 → b_victoria 合计
      7. 合并 AH1 系列 → ah1_total
      8. 计算 H3N2 周阳性率及相关比率
      9. 重建完整周度索引，线性插值补缺（最多连续3周）
      10. 标注流行季 & 双峰标签（上海气候）
    """
    # ── Step 1: 重命名已知字段 ────────────────────────────────────────────
    rename_map = {k: v for k, v in VIW_FNT_FIELD_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    log.info(f"字段映射: {len(rename_map)}/{len(VIW_FNT_FIELD_MAP)} 列匹配")

    # 保留有用的列
    keep = [v for v in VIW_FNT_FIELD_MAP.values() if v in df.columns]
    df = df[keep].copy()

    # ── Step 2: 筛选目标国家 ──────────────────────────────────────────────
    if "country_code" not in df.columns:
        raise ValueError("country_code 列不存在，请确认 COUNTRY_CODE 在 VIW_FNT.csv 中")
    n_before = len(df)
    df = df[df["country_code"].str.upper().str.strip() == country_code.upper()].copy()
    log.info(f"国家过滤 ({country_code}): {n_before:,} → {len(df):,} 行")

    if len(df) == 0:
        raise ValueError(f"过滤后无数据：COUNTRY_CODE={country_code} 在文件中不存在")

    # ── Step 3: 时间范围过滤 ──────────────────────────────────────────────
    df["iso_year"] = pd.to_numeric(df["iso_year"], errors="coerce")
    df["iso_week"] = pd.to_numeric(df["iso_week"], errors="coerce")
    df = df.dropna(subset=["iso_year", "iso_week"])
    df = df[df["iso_year"].between(year_from, year_to)].copy()
    df = df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    log.info(f"时间过滤 ({year_from}–{year_to}): {len(df)} 周")

    # ── Step 4: 解析日期 ──────────────────────────────────────────────────
    # 优先使用 ISO_WEEKSTARTDATE（VIW_FNT 直接提供，格式 YYYY-MM-DD）
    if "week_start_date" in df.columns:
        df["date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
        n_parsed = df["date"].notna().sum()
        log.info(f"日期解析（ISO_WEEKSTARTDATE）: {n_parsed}/{len(df)} 行成功")
    else:
        log.warning("week_start_date 列不存在，回退到 ISO 年+周推算")
        df["date"] = df.apply(_iso_to_date, axis=1)

    df = df.dropna(subset=["date"])

    # ── Step 5: 数值化关键字段 ────────────────────────────────────────────
    existing_num = [c for c in NUMERIC_COLS if c in df.columns]
    for c in existing_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Step 6: 合并 B/Victoria 四子列 → b_victoria ──────────────────────
    # VIW_FNT 新格式将 B/Victoria 拆为 4 个删除型亚群，需合并为单一列
    bvic_cols = [c for c in ["bvic_2del", "bvic_3del", "bvic_nodel", "bvic_delunk"]
                 if c in df.columns]
    if bvic_cols:
        df["b_victoria"] = df[bvic_cols].fillna(0).sum(axis=1)
        df["b_victoria"] = df["b_victoria"].where(
            df[bvic_cols].notna().any(axis=1), np.nan
        )
        log.info(f"B/Victoria 合并: {bvic_cols} → b_victoria")
    elif "bvictoria" in df.columns:
        df["b_victoria"] = df["bvictoria"]   # 旧格式兼容
    else:
        df["b_victoria"] = np.nan

    # ── Step 7: 合并 AH1 系列 → ah1_total ────────────────────────────────
    # 2009年后 AH1N12009 是主力，AH1（季节性H1N1）通常为0或空
    ah1_cols = [c for c in ["ah1_2009", "ah1_old"] if c in df.columns]
    if ah1_cols:
        df["ah1_total"] = df[ah1_cols].fillna(0).sum(axis=1)
        df["ah1_total"] = df["ah1_total"].where(
            df[ah1_cols].notna().any(axis=1), np.nan
        )
    else:
        df["ah1_total"] = np.nan

    # ── Step 8: 计算 H3N2 核心比率 ───────────────────────────────────────
    ah3  = df.get("ah3",          pd.Series(np.nan, index=df.index))
    spec = df.get("spec_processed", pd.Series(np.nan, index=df.index))
    inf_all = df.get("inf_all",   pd.Series(np.nan, index=df.index))
    inf_a = df.get("inf_a",       pd.Series(np.nan, index=df.index))
    # H3N2 / 总检测数（周阳性率）
    df["h3n2_pos_rate"] = np.where(
        spec > 0, ah3 / spec, np.nan
    )
    # H3N2 / 所有流感阳性（H3N2 在流感中的占比）
    df["h3n2_share"] = np.where(
        inf_all > 0, ah3 / inf_all, np.nan
    )
    # 所有流感阳性率
    df["flu_pos_rate"] = np.where(
        spec > 0, inf_all / spec, np.nan
    )
    # 流感A阳性率
    df["inf_a_rate"] = np.where(
        spec > 0, inf_a / spec, np.nan
    )
    # 标本阳性率（含所有呼吸道病原）
    total_tested = spec.fillna(0)
    df["overall_pos_rate"] = np.where(
        total_tested > 0,
        inf_all / total_tested,
        np.nan
    )

    # ── Step 9: 重建完整周度索引，插值补缺 ───────────────────────────────
    df = df.set_index("date")
    # 以 Monday 为周起点（与 ISO_WEEKSTARTDATE 一致）
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="W-MON",
    )
    df = df.reindex(full_idx)

    # 线性插值（最多连续补 3 个缺失周）
    for col in ["h3n2_pos_rate", "h3n2_share", "flu_pos_rate",
                "inf_a_rate", "overall_pos_rate"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .interpolate(method="linear", limit=3, limit_direction="forward")
                .fillna(0.0)
            )
    # 原始计数列：仅填 NaN → 0（不插值，避免虚增计数）
    for col in ["spec_processed", "ah3", "inf_all", "ah1_total", "b_victoria"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # iso_year / iso_week 补全（从索引推算）
    df["iso_year"] = df.index.year
    df["iso_week"] = df.index.isocalendar().week.values

    # ── Step 10: 流行季 & 双峰标签 ───────────────────────────────────────
    # 上海定义：10月起计入下一流感年度（与国内监测年度一致）
    df["flu_year"] = df.index.map(
        lambda d: d.year if d.month < 10 else d.year + 1
    )
    # 双峰标签（上海气候特征：冬峰 week 1–12 / 夏峰 week 24–36）
    iso_wk = df.index.isocalendar().week
    df["peak_label"] = "off-peak"
    df.loc[iso_wk.between(1, 12),  "peak_label"] = "winter-peak"
    df.loc[iso_wk.between(24, 36), "peak_label"] = "summer-peak"

    df = df.reset_index().rename(columns={"index": "date"})
    mean_h3n2 = df["h3n2_pos_rate"].mean()
    log.info(
        f"清洗完成: {len(df)} 周数据，"
        f"H3N2 均值阳性率={mean_h3n2:.4f}，"
        f"缺失周已插值"
    )
    return df


def _iso_to_date(row) -> pd.Timestamp:
    """从 iso_year + iso_week 推算该周周一日期（回退方案）。"""
    try:
        return datetime.fromisocalendar(int(row["iso_year"]), int(row["iso_week"]), 1)
    except (ValueError, TypeError):
        return pd.NaT

# ── 季节参数估算 ────────────────────────────────────────────────────────────

def compute_seasonal_params(df: pd.DataFrame) -> dict:
    """
    双谐波回归估算 β(t) 的季节参数（以周为时间单位）。

    模型：y(t) = a₀ + a₁cos(2πt/52) + b₁sin(2πt/52)
                     + a₂cos(4πt/52) + b₂sin(4πt/52) + ε

    返回：
        beta0_proxy  背景阳性水平 a₀
        delta1       年振幅 / a₀（相对振幅）
        delta2       半年振幅 / a₀
        phi1_rad     年相位（弧度）
        phi2_rad     半年相位（弧度）
        fit_r2       拟合决定系数 R²
    """
    from numpy.linalg import lstsq

    y = df["h3n2_pos_rate"].fillna(0.0).values
    T = len(y)
    t = np.arange(T, dtype=float)

    X = np.column_stack([
        np.ones(T),
        np.cos(2 * np.pi * t / 52), np.sin(2 * np.pi * t / 52),
        np.cos(4 * np.pi * t / 52), np.sin(4 * np.pi * t / 52),
    ])
    coef, *_ = lstsq(X, y, rcond=None)
    a0, a1, b1, a2, b2 = coef

    if abs(a0) < 1e-8:
        a0 = max(y.mean(), 1e-4)

    delta1 = float(np.clip(np.sqrt(a1**2 + b1**2) / abs(a0), 0, 1))
    delta2 = float(np.clip(np.sqrt(a2**2 + b2**2) / abs(a0), 0, 1))
    phi1   = float(np.arctan2(-b1, a1))
    phi2   = float(np.arctan2(-b2, a2))

    y_pred = X @ coef
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    params = {
        "beta0_proxy": float(a0),
        "delta1":      delta1,
        "delta2":      delta2,
        "phi1_rad":    phi1,
        "phi2_rad":    phi2,
        "fit_r2":      r2,
        "n_weeks":     int(T),
        "year_range":  f"{int(df['iso_year'].min())}–{int(df['iso_year'].max())}",
    }
    log.info(
        f"季节参数: δ₁={delta1:.3f}  δ₂={delta2:.3f}  "
        f"φ₁={phi1:.3f}rad  φ₂={phi2:.3f}rad  R²={r2:.3f}"
    )
    return params

# ── 年度汇总统计 ────────────────────────────────────────────────────────────

def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按流行年度汇总峰值、谷值、持续时间统计。"""
    records = []
    for flu_year, grp in df.groupby("flu_year"):
        winter = grp[grp["peak_label"] == "winter-peak"]
        summer = grp[grp["peak_label"] == "summer-peak"]

        # 峰值周日期
        w_peak_date = (winter["date"].iloc[winter["h3n2_pos_rate"].values.argmax()]
                       if len(winter) > 0 else pd.NaT)
        s_peak_date = (summer["date"].iloc[summer["h3n2_pos_rate"].values.argmax()]
                       if len(summer) > 0 else pd.NaT)

        records.append({
            "flu_year":               int(flu_year),
            "winter_peak_rate":       round(winter["h3n2_pos_rate"].max(), 4) if len(winter) > 0 else 0.0,
            "winter_peak_date":       str(w_peak_date)[:10] if pd.notna(w_peak_date) else "",
            "summer_peak_rate":       round(summer["h3n2_pos_rate"].max(), 4) if len(summer) > 0 else 0.0,
            "summer_peak_date":       str(s_peak_date)[:10] if pd.notna(s_peak_date) else "",
            "mean_annual_rate":       round(grp["h3n2_pos_rate"].mean(), 4),
            "h3n2_dominant_weeks":    int((grp["h3n2_pos_rate"] > 0.05).sum()),
            "total_spec_processed":   int(grp["spec_processed"].sum()),
            "total_ah3":              int(grp["ah3"].sum()),
        })
    return pd.DataFrame(records)

# ── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WHO FluNet VIW_FNT.csv 中国 H3N2 数据处理",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", metavar="CSV_PATH",
        default=str(RAW_DIR / "VIW_FNT.csv"),
        help="VIW_FNT.csv 文件路径（默认: data/raw/VIW_FNT.csv）",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="从 WHO API 下载数据（网络不稳定时不推荐）",
    )
    parser.add_argument("--country",   default="CHN",    help="ISO3 国家代码（默认: CHN）")
    parser.add_argument("--year-from", type=int, default=2015)
    parser.add_argument("--year-to",   type=int, default=2024)
    args = parser.parse_args()

    # ── 获取原始数据 ──────────────────────────────────────────────────────
    if args.download:
        raw_path = RAW_DIR / "VIW_FNT_downloaded.csv"
        df_raw = download_flunet(FLUNET_CSV_URL, raw_path)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error(f"文件不存在: {input_path}")
            log.error("请将 VIW_FNT.csv 放入 data/raw/ 目录，或指定 --input 路径")
            log.error("也可使用 --download 从 WHO API 下载")
            sys.exit(1)
        df_raw = load_viwfnt(input_path)

    log.info(f"原始数据: {len(df_raw):,} 行 × {len(df_raw.columns)} 列")

    # ── 清洗 ──────────────────────────────────────────────────────────────
    df_clean = clean_viwfnt(
        df_raw,
        country_code=args.country,
        year_from=args.year_from,
        year_to=args.year_to,
    )

    # ── 保存清洗数据 ──────────────────────────────────────────────────────
    out_path = PROC_DIR / "weekly_positivity.csv"
    df_clean.to_csv(out_path, index=False)
    log.info(f"周度数据已保存 → {out_path}")

    # ── 季节参数估算 ──────────────────────────────────────────────────────
    params = compute_seasonal_params(df_clean)
    param_path = PROC_DIR / "seasonal_params.json"
    param_path.write_text(json.dumps(params, indent=2, ensure_ascii=False))
    log.info(f"季节参数已保存 → {param_path}")

    # ── 年度汇总 ──────────────────────────────────────────────────────────
    summary = seasonal_summary(df_clean)
    summary_path = PROC_DIR / "seasonal_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"年度汇总已保存 → {summary_path}")

    # ── 终端输出 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"处理完成：{args.country} H3N2 数据 {args.year_from}–{args.year_to}")
    print("=" * 65)
    print("\n【季节参数（可更新至 config.yaml）】")
    print(json.dumps(params, indent=2, ensure_ascii=False))
    print("\n【年度峰值汇总】")
    print(summary.to_string(index=False))
    print(f"\n输出文件目录: {PROC_DIR}")


if __name__ == "__main__":
    main()
