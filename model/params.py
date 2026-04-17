"""
model/params.py — ModelParams 参数管理
=======================================
所有模型参数的单一数据源。支持从 config.yaml 加载，
提供初始状态向量和接触矩阵的构建方法。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

import numpy as np

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class ModelParams:
    # ── 人口结构 ─────────────────────────────────────────────────────────────
    N1: int   = 27000      # 学生总数
    N2: int   = 3000       # 教职工总数

    # ── 核心传播参数 ──────────────────────────────────────────────────────────
    beta0:   float = 0.30  # 基础传播系数（/天）
    sigma:   float = 0.50  # 潜伏期转感染率（1/σ = 2天）
    gamma:   float = 0.25  # 感染者康复率（1/γ = 4天）
    gamma_Q: float = 0.20  # 隔离者康复率（1/γQ = 5天）
    alpha:   float = 0.15  # 病例隔离率
    p_iso:   float = 0.40  # 感染者被发现并成功隔离的比例
    omega:   float = 0.0   # 免疫衰退率（单季设0）

    # ── 疫苗参数 ──────────────────────────────────────────────────────────────
    vax_coverage: float = 0.0032  # 接种率（上海高校约 0.32%）
    vax_efficacy:  float = 0.27   # H3N2 疫苗有效率

    # ── 季节参数（谐波，由 FluNet 拟合更新） ─────────────────────────────────
    delta1: float = 0.30   # 年振幅（相对于 β₀）
    delta2: float = 0.20   # 半年振幅（相对于 β₀）
    phi1:   float = 0.0    # 年相位（弧度）
    phi2:   float = 0.0    # 半年相位（弧度）

    # ── 接触矩阵元素（2×2 群体，单位：次/天） ────────────────────────────────
    c11: float = 18.0      # 学生-学生
    c12: float = 2.0       # 学生-教职工
    c21: float = 2.0       # 教职工-学生
    c22: float = 8.0       # 教职工-教职工

    # ── 初始条件 ──────────────────────────────────────────────────────────────
    I0_1: int = 5          # 初始感染学生数
    I0_2: int = 1          # 初始感染教职工数

    # ── 模拟时间配置 ──────────────────────────────────────────────────────────
    t_days:      int = 180  # 模拟天数
    t_start_doy: int = 305  # 起始日（一年中第几天，305=11月1日）

    # ── 便捷属性 ──────────────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        """总人口数。"""
        return self.N1 + self.N2

    @property
    def R0_approx(self) -> float:
        """R₀ 近似估算（均质混合假设，供快速检验）。"""
        mu = self.gamma * (1 - self.p_iso) + self.alpha * self.p_iso
        return self.beta0 * (self.c11 / self.N1) * self.N1 / mu

    def initial_state(self) -> np.ndarray:
        """
        构建10维初始状态向量:
            [S1, E1, I1, Q1, R1,  S2, E2, I2, Q2, R2]
        疫苗接种者直接进入 R 舱（简化处理）。
        """
        vacc_frac = self.vax_coverage * self.vax_efficacy
        S0_1 = self.N1 * (1 - vacc_frac) - self.I0_1
        R0_1 = self.N1 * vacc_frac        # 已接种 → 等效免疫
        S0_2 = self.N2 * (1 - vacc_frac) - self.I0_2
        R0_2 = self.N2 * vacc_frac

        S0_1 = max(S0_1, 0.0)
        S0_2 = max(S0_2, 0.0)

        return np.array([
            S0_1, 0.0, float(self.I0_1), 0.0, R0_1,   # 学生群体
            S0_2, 0.0, float(self.I0_2), 0.0, R0_2,   # 教职工群体
        ], dtype=float)

    def contact_matrix(self) -> np.ndarray:
        """返回 2×2 日均有效接触矩阵。"""
        return np.array([
            [self.c11, self.c12],
            [self.c21, self.c22],
        ], dtype=float)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return asdict(self)

    def update(self, **kwargs) -> "ModelParams":
        """返回更新了指定字段的新实例（不修改原对象）。"""
        from copy import copy
        p = copy(self)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
            else:
                raise ValueError(f"ModelParams 没有字段: {k!r}")
        return p

    # ── 类方法：从外部文件加载 ────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelParams":
        """从 config.yaml 的 model 节加载参数。"""
        if not _HAS_YAML:
            raise ImportError("请先安装 pyyaml: pip install pyyaml")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg.get("model", {}))

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelParams":
        """从 JSON 文件加载参数（用于校准结果持久化）。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_seasonal_params(
        cls,
        seasonal_json: str | Path,
        **overrides,
    ) -> "ModelParams":
        """
        从 data/processed/seasonal_params.json 读取谐波参数，
        合并到默认 ModelParams 中。

        seasonal_params.json 字段映射:
            beta0_proxy → 不直接使用（仅供参考）
            delta1      → delta1
            delta2      → delta2
            phi1_rad    → phi1
            phi2_rad    → phi2
        """
        with open(seasonal_json, encoding="utf-8") as f:
            sp = json.load(f)

        kwargs = {
            "delta1": sp.get("delta1", 0.30),
            "delta2": sp.get("delta2", 0.20),
            "phi1":   sp.get("phi1_rad", 0.0),
            "phi2":   sp.get("phi2_rad", 0.0),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelParams(N1={self.N1}, N2={self.N2}, "
            f"β₀={self.beta0:.3f}, σ={self.sigma:.3f}, γ={self.gamma:.3f}, "
            f"α={self.alpha:.3f}, p_iso={self.p_iso:.3f}, "
            f"δ₁={self.delta1:.3f}, δ₂={self.delta2:.3f})"
        )
