"""
plots/_style.py — 统一图表样式配置
=====================================
所有绘图函数在初始化时调用 setup_style()。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

# 颜色方案
COLORS = {
    "student":    "#2196F3",   # 学生群体 — 蓝色
    "staff":      "#FF7043",   # 教职工群体 — 橙红
    "total":      "#333333",   # 总量 — 深灰
    "observed":   "#4CAF50",   # 观测数据 — 绿色
    "baseline":   "#9E9E9E",   # 基准/对照 — 灰色
    "ci":         "#90CAF9",   # 置信带 — 浅蓝
    "accent1":    "#7C4DFF",   # 强调色1 — 紫
    "accent2":    "#00BCD4",   # 强调色2 — 青
    "warn":       "#FFC107",   # 警示色 — 黄
    "danger":     "#F44336",   # 危险色 — 红
}

PALETTE_INTERVENTIONS = [
    "#1976D2", "#43A047", "#FB8C00", "#8E24AA",
    "#00897B", "#E53935", "#F9A825",
]


def _find_chinese_font() -> str | None:
    """查找系统中可用的中文字体。"""
    candidates = [
        "SimHei", "Microsoft YaHei", "STSong", "FangSong",  # Windows
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",           # Linux
        "PingFang SC", "Heiti SC",                            # macOS
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return None


def setup_style(font: str | None = None) -> None:
    """配置 matplotlib 全局样式。"""
    warnings.filterwarnings("ignore", category=UserWarning)

    # 中文字体
    if font is None:
        font = _find_chinese_font()

    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    else:
        # 回退：使用 DejaVu，标签改为英文
        plt.rcParams["font.family"] = ["DejaVu Sans"]

    plt.rcParams.update({
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "lines.linewidth":    2.0,
        "axes.prop_cycle":    matplotlib.cycler(color=list(COLORS.values())[:7]),
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })

    matplotlib.rcParams["axes.unicode_minus"] = False
