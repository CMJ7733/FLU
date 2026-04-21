"""
intervention/cost_model.py — 防控成本与教学影响量化
=====================================================
将 7 类干预措施的资源消耗和教学中断代价规范化为 [0,1] 区间，
用于多目标优化中的成本约束项。

成本分类：
    经济成本：采购口罩/消毒液、疫苗接种费用、通风设备改造等
    教学成本：线上教学技术支持、学生满意度损失、社团停办影响等

参考基准（上海高校，2023年价格水平）：
    - N95 口罩：约 3元/人/天，全员佩戴全学期约 54,000 元/天
    - 疫苗接种：约 100元/人，全员接种约 300 万元（一次性）
    - 通风改造：约 5–20 万元/楼栋（固定成本）
    - 线上教学：教务系统已具备，边际成本≈0，但教学质量损失显著
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .measures import InterventionBundle


# ── 各措施单位成本权重（总和归一化后用于综合评分） ──────────────────────────
# 数值代表"全强度实施该措施"对应的相对成本（0–1 区间）
UNIT_COSTS: dict[str, float] = {
    "mask_level":      0.05,   # 口罩：成本中等，可接受性高
    "ventilation":     0.01,   # 通风：一次性改造成本较高
    "vaccination":     0.20,   # 疫苗：大规模接种费用最高
    "isolation_rate":  0.20,   # 隔离强化：人力成本（宿管/医护）
    "online_teaching": 0.18,   # 线上教学：教学质量损失（无货币成本）
    "activity_limit":  0.02,   # 活动限制：学生体验损失
    "disinfection":    0.05,   # 消毒：材料成本低，人力成本低
}

# ── 教学干扰权重（影响正常教学秩序，归一化至总和=1.0） ──────────────────
# 设计原则：温和措施（口罩/通风/消毒）干扰极低，
#           破坏性措施（线上教学/活动限流/隔离）干扰显著更高。
#           这样优化器在"最小化教学干扰"目标压力下，
#           自然倾向于常态防控方案（只启用低干扰措施）。
TEACHING_DISRUPTION: dict[str, float] = {
    "mask_level":      0.01,   # 轻微不适，对教学影响极小
    "ventilation":     0.01,   # 几乎无教学干扰（基础设施改造）
    "vaccination":     0.06,   # 接种日需半天，组织动员有小影响
    "isolation_rate":  0.18,   # 隔离导致学生缺课，影响较大
    "online_teaching": 0.40,   # 线上教学是最主要的教学中断因素
    "activity_limit":  0.25,   # 课外活动停止严重影响校园生活质量
    "disinfection":    0.02,   # 偶尔占用教室时间，干扰很小
}


def _u_vals(bundle: "InterventionBundle") -> dict[str, float]:
    """提取 bundle 中各措施的控制变量值。"""
    return {
        "mask_level":      bundle.mask_level,
        "ventilation":     bundle.ventilation,
        "vaccination":     bundle.vaccination,
        "isolation_rate":  bundle.isolation_rate,
        "online_teaching": bundle.online_teaching,
        "activity_limit":  bundle.activity_limit,
        "disinfection":    bundle.disinfection,
    }


def compute_economic_score(bundle: "InterventionBundle") -> float:
    """
    纯经济成本评分（归一化至 [0, 1]）。

    公式：score = Σᵢ (UNIT_COSTS[i] × uᵢ) / Σᵢ UNIT_COSTS[i]

    Returns:
        经济成本评分 ∈ [0, 1]
    """
    u = _u_vals(bundle)
    raw = sum(UNIT_COSTS[k] * u[k] for k in UNIT_COSTS)
    return float(raw / sum(UNIT_COSTS.values()))


def compute_teaching_score(bundle: "InterventionBundle") -> float:
    """
    纯教学干扰评分（归一化至 [0, 1]）。

    公式：score = Σᵢ (TEACHING_DISRUPTION[i] × uᵢ) / Σᵢ TEACHING_DISRUPTION[i]

    Returns:
        教学干扰评分 ∈ [0, 1]
    """
    u = _u_vals(bundle)
    raw = sum(TEACHING_DISRUPTION[k] * u[k] for k in TEACHING_DISRUPTION)
    return float(raw / sum(TEACHING_DISRUPTION.values()))


def compute_cost(bundle: "InterventionBundle") -> float:
    """
    综合成本评分（经济50% + 教学50%，归一化至 [0, 1]）。
    保留用于向后兼容（grid_search 等旧接口）。

    Returns:
        综合成本评分 ∈ [0, 1]
    """
    return float(min(
        0.5 * compute_economic_score(bundle) + 0.5 * compute_teaching_score(bundle),
        1.0,
    ))


def compute_economic_cost_yuan(bundle: "InterventionBundle", n_students: int = 27000) -> float:
    """
    估算干预措施的实际经济成本（元/学期）。
    用于论文中的直观呈现（非优化用途）。

    Returns:
        经济成本估算（元）
    """
    SEMESTER_DAYS = 120  # 学期约120天

    cost = 0.0

    # u1: 口罩 — 约3元/人/天（外科口罩），全校
    if bundle.mask_level > 0:
        mask_rate = 3.0 if bundle.mask_level >= 0.8 else 1.5  # N95 vs 外科
        cost += bundle.mask_level * mask_rate * n_students * SEMESTER_DAYS

    # u2: 通风改造 — 约10万元/楼栋，假设30栋楼
    if bundle.ventilation > 0:
        cost += bundle.ventilation * 100_000 * 30

    # u3: 疫苗接种 — 约100元/人
    if bundle.vaccination > 0:
        target_rate = 0.50
        new_vax = bundle.vaccination * (target_rate - 0.0032)
        cost += max(new_vax, 0) * n_students * 100

    # u4: 隔离强化 — 约50元/人/天（宿管+医护人力），按隔离率估算
    if bundle.isolation_rate > 0:
        est_iso_persons = n_students * 0.05 * bundle.isolation_rate  # 估计5%处于隔离
        cost += est_iso_persons * 50 * SEMESTER_DAYS

    # u5: 线上教学 — 技术成本约5万元/学期，主要是教学质量损失（无货币化）
    if bundle.online_teaching > 0:
        cost += bundle.online_teaching * 50_000

    # u6: 活动限制 — 损失学生活动经费约5元/人/天
    if bundle.activity_limit > 0:
        cost += bundle.activity_limit * 5 * n_students * SEMESTER_DAYS * 0.1

    # u7: 消毒 — 约0.5元/m²/次，假设10万m²，每周两次
    if bundle.disinfection > 0:
        cost += bundle.disinfection * 0.5 * 100_000 * 2 * (SEMESTER_DAYS // 7)

    return cost


def cost_effectiveness_ratio(
    attack_rate_reduction: float,
    cost_score: float,
) -> float:
    """
    成本效益比 = 发病率降低 / 成本评分。
    用于比较不同方案的效益-成本权衡。
    越大表示单位成本节省的发病更多。
    """
    if cost_score < 1e-6:
        return float("inf")
    return attack_rate_reduction / cost_score
