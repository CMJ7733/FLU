"""
intervention/measures.py — 7类防控措施参数化
=============================================
每项措施以控制变量 u ∈ [0, 1] 表示强度：
    u = 0  未启用
    u = 1  全强度实施

效果通过乘数因子作用于模型参数（β₀, α, 接触矩阵），
保证生物学合理性且可叠加。

参考文献：
    1. Cowling BJ et al. Facemasks and Hand Hygiene to Prevent Influenza.
       Ann Intern Med. 2009. (口罩效果)
    2. Li Y et al. Role of Ventilation in Airborne Disease Transmission.
       Indoor Air. 2021. (通风效果)
    3. Iuliano AD et al. Estimates of global seasonal influenza-associated
       respiratory mortality. Lancet. 2018. (疫苗效果)
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field

from model.params import ModelParams

# ── 场所接触权重（归一化至 1.0） ────────────────────────────────────────────
LOCATION_WEIGHTS: dict[str, float] = {
    "dorm":      0.30,   # 宿舍：密切接触场所，传播风险最高
    "classroom": 0.25,   # 教室：次高风险，口罩+通风改善明显
    "canteen":   0.10,   # 食堂：短时接触，相对低风险
    "outdoor":   0.35,   # 课外/活动：接触次数多但持续时间短
}


@dataclass
class InterventionBundle:
    """
    7类防控措施的控制变量集合。
    各变量 ∈ [0, 1]，0=未实施，1=全强度。

    u1 mask_level:      口罩佩戴 → 降低 β₀
    u2 ventilation:     通风改善 → 降低室内 β₀
    u3 vaccination:     疫苗接种 → 提升 vax_coverage
    u4 isolation_rate:  隔离强化 → 提升 α（p_iso 不变）
    u5 online_teaching: 线上教学 → 减少教室接触
    u6 activity_limit:  社团限流 → 减少课外接触
    u7 disinfection:    环境消毒 → 轻微降低 β₀
    """
    mask_level:      float = 0.0  # u1
    ventilation:     float = 0.0  # u2
    vaccination:     float = 0.0  # u3
    online_teaching: float = 0.0  # u5
    activity_limit:  float = 0.0  # u6
    isolation_rate:  float = 0.0  # u4
    disinfection:    float = 0.0  # u7

    def to_dict(self) -> dict:
        return {
            "u1_mask":        self.mask_level,
            "u2_ventilation": self.ventilation,
            "u3_vaccination": self.vaccination,
            "u4_isolation":   self.isolation_rate,
            "u5_online":      self.online_teaching,
            "u6_activity":    self.activity_limit,
            "u7_disinfect":   self.disinfection,
        }

    def cost_score(self) -> float:
        """
        防控综合成本评分（0–1 规范化）。
        基于经济成本 + 教学中断程度的加权估算。
        """
        from .cost_model import compute_cost
        return compute_cost(self)

    def __repr__(self) -> str:
        active = [
            f"mask={self.mask_level:.1f}",
            f"vent={self.ventilation:.1f}",
            f"vax={self.vaccination:.1f}",
            f"iso={self.isolation_rate:.1f}",
            f"online={self.online_teaching:.1f}",
            f"activ={self.activity_limit:.1f}",
            f"disinfect={self.disinfection:.1f}",
        ]
        return f"InterventionBundle({', '.join(active)})"


# ── 无干预基准 ──────────────────────────────────────────────────────────────
NO_INTERVENTION  = InterventionBundle()

# ── 预设方案 ────────────────────────────────────────────────────────────────
PRESET_MILD = InterventionBundle(
    mask_level=0.5, ventilation=0.3, isolation_rate=0.3
)
PRESET_MODERATE = InterventionBundle(
    mask_level=0.8, ventilation=0.5, isolation_rate=0.6,
    activity_limit=0.4, disinfection=0.5
)
PRESET_STRONG = InterventionBundle(
    mask_level=1.0, ventilation=1.0, isolation_rate=1.0,
    online_teaching=0.7, activity_limit=0.8, disinfection=1.0,
    vaccination=0.5
)


def apply_interventions(
    base_params: ModelParams,
    bundle: InterventionBundle,
    location_weights: dict | None = None,
) -> ModelParams:
    """
    将干预措施叠加到基础参数，返回修改后的参数副本。
    所有效果以乘数形式作用，符合独立效果假设。

    效果公式（参考文献值）：
        u1 口罩：     β₀ × (1 - 0.17 × u1)
        u2 通风：     β₀ × (1 - 0.40 × u2 × 室内权重比例)
        u3 疫苗：     vax_coverage += u3 × (0.50 - coverage)  [上限 50%]
        u4 隔离强化： α × (1 + 3.0 × u4)                     [上限 1.0]
        u5 线上教学： c₁₁ × (1 - 0.25 × u5), c₁₂ × (1 - 0.15 × u5)
        u6 社团限流： c₁₁ × (1 - 0.35 × u6 × outdoor_weight)
        u7 消毒：     β₀ × (1 - 0.05 × u7)
    """
    if location_weights is None:
        location_weights = LOCATION_WEIGHTS

    p = deepcopy(base_params)

    # ── u1: 口罩佩戴 ──────────────────────────────────────────────────────
    # Cowling 2009: 外科口罩对飞沫/接触传播降低约 17%
    p = p.update(beta0=p.beta0 * (1.0 - 0.17 * bundle.mask_level))

    # ── u2: 通风改善 ──────────────────────────────────────────────────────
    # 仅作用于室内接触（宿舍+教室占总接触的 55%）
    indoor_share  = location_weights["dorm"] + location_weights["classroom"]
    vent_reduction = indoor_share * 0.40 * bundle.ventilation
    p = p.update(beta0=p.beta0 * (1.0 - vent_reduction))

    # ── u3: 疫苗接种 ──────────────────────────────────────────────────────
    # 将接种率提升至目标值（0.50），按 u3 比例线性插值
    vax_target    = 0.50
    vax_increment = bundle.vaccination * max(vax_target - p.vax_coverage, 0.0)
    p = p.update(vax_coverage=min(p.vax_coverage + vax_increment, 1.0))

    # ── u4: 隔离强化 ──────────────────────────────────────────────────────
    # 提升病例发现率（α），乘数 1+3u₄，上限 1.0
    new_alpha = min(p.alpha * (1.0 + 3.0 * bundle.isolation_rate), 1.0)
    p = p.update(alpha=new_alpha)

    # ── u5: 线上教学 ──────────────────────────────────────────────────────
    # 减少教室接触（学生间 c₁₁ 降低教室权重部分，学生-教职工 c₁₂ 减少一半效果）
    classroom_w   = location_weights["classroom"]
    c11_reduction = classroom_w * bundle.online_teaching
    c12_reduction = classroom_w * 0.5 * bundle.online_teaching
    p = p.update(
        c11=p.c11 * (1.0 - c11_reduction),
        c12=p.c12 * (1.0 - c12_reduction),
        c21=p.c21 * (1.0 - c12_reduction),
    )

    # ── u6: 社团/课外活动限流 ────────────────────────────────────────────
    outdoor_w     = location_weights["outdoor"]
    c11_outdoor   = outdoor_w * 0.70 * bundle.activity_limit
    p = p.update(c11=p.c11 * (1.0 - c11_outdoor))

    # ── u7: 环境消毒 ──────────────────────────────────────────────────────
    p = p.update(beta0=p.beta0 * (1.0 - 0.05 * bundle.disinfection))

    return p
