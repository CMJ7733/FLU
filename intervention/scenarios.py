"""
intervention/scenarios.py — 三场景定义
========================================
场景一（常态）：正常开学期间，散发病例，常规监测
场景二（散发）：出现明确传播链，启动应急响应
场景三（聚集）：单一场所（宿舍/班级）出现聚集性疫情
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from model.params import ModelParams


@dataclass
class Scenario:
    """场景定义容器。"""
    name:        str
    description: str
    I0_1:        int    = 5     # 初始感染学生数
    I0_2:        int    = 1     # 初始感染教职工数
    t_days:      int    = 180   # 模拟时长（天）
    t_start_doy: int    = 305   # 起始日（一年中第几天）
    # 场景特定参数覆盖（None = 使用 ModelParams 默认值）
    beta0_override:   Optional[float] = None
    alpha_override:   Optional[float] = None
    cluster_factor:   float = 1.0  # 局部聚集传播放大系数（作用于 c₁₁）

    def apply_to(self, p: ModelParams) -> ModelParams:
        """将场景设置应用到 ModelParams，返回新实例。"""
        updates = {
            "I0_1":        self.I0_1,
            "I0_2":        self.I0_2,
            "t_days":      self.t_days,
            "t_start_doy": self.t_start_doy,
        }
        if self.beta0_override is not None:
            updates["beta0"] = self.beta0_override
        if self.alpha_override is not None:
            updates["alpha"] = self.alpha_override
        if self.cluster_factor != 1.0:
            updates["c11"] = p.c11 * self.cluster_factor

        return p.update(**updates)


# ── 三场景定义 ───────────────────────────────────────────────────────────────

SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="场景一：常态散发",
        description=(
            "正常开学状态，H3N2 处于低流行水平。\n"
            "初始 5 名学生感染，常规监测，无额外干预。\n"
            "代表上海高校典型冬季流感季（非暴发年）。"
        ),
        I0_1=5, I0_2=1,
        t_days=180,
        t_start_doy=305,  # 11月1日
        beta0_override=None,   # 使用配置文件默认值
        alpha_override=None,
        cluster_factor=1.0,
    ),

    "outbreak": Scenario(
        name="场景二：局部散发暴发",
        description=(
            "检测到明确的传播链，单个班级/宿舍楼出现多例。\n"
            "初始 20 名学生感染，接触追踪刚刚启动。\n"
            "代表典型校园流感暴发事件初期（2019年上海某高校）。"
        ),
        I0_1=20, I0_2=2,
        t_days=120,
        t_start_doy=305,
        beta0_override=None,
        alpha_override=0.10,   # 尚未全面启动隔离，实际隔离率偏低
        cluster_factor=1.0,
    ),

    "cluster": Scenario(
        name="场景三：宿舍聚集疫情",
        description=(
            "单栋宿舍楼出现聚集性 H3N2 疫情，宿舍内密切接触放大传播。\n"
            "初始 30 名学生感染（集中在同一楼栋），宿舍接触率提升 50%。\n"
            "代表极端密集接触情景，用于压力测试防控方案。"
        ),
        I0_1=30, I0_2=1,
        t_days=90,
        t_start_doy=305,
        beta0_override=None,
        alpha_override=0.08,   # 聚集初期发现延迟，隔离率更低
        cluster_factor=1.50,   # 宿舍楼内接触率 ×1.5
    ),
}


def get_scenario(name: str) -> Scenario:
    """按名称获取场景（支持 baseline/outbreak/cluster）。"""
    if name not in SCENARIOS:
        raise KeyError(f"未知场景: {name!r}，可用: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> None:
    """打印所有场景摘要。"""
    for key, sc in SCENARIOS.items():
        print(f"\n[{key}] {sc.name}")
        print(f"  初始感染: 学生={sc.I0_1}, 教职工={sc.I0_2}")
        print(f"  模拟时长: {sc.t_days}天，起始DOY={sc.t_start_doy}")
        print(f"  描述: {sc.description.split(chr(10))[0]}")
