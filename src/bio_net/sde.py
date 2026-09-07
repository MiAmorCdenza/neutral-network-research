# src/bio_net/sde.py
# 内核模块（12-repo-layout §3：kernel 三件套之一，不 import 任何其他包模块）
# 规格来源：docs/specs/00-kernel.md §K1（总纲 SDE，Song et al. 2021 落地）

"""K1 总纲 SDE 骨干（00-kernel.md）。

前向（加噪/抽象/压缩）：dx = f(x,t)dt + g(t)dw
反向（去噪/推断/生成）：dx = [f(x,t) − g(t)²·s_θ(x,t)]dt + g(t)dw̄

本文件实现 K.F1–K.F7（V0 函数集，07-build-order §8）。
当前进度：K.F1 sde_marginal。
"""

import torch

__all__ = ["sde_marginal"]

# ── K.F1 参数（00-kernel.md §K.F1）───────────────────────────────────────────
_EPS = 1e-6  # t=0 时 std 的钳制下限（规格原文："钳制 ≥EPS=1e-6"）

# VP 默认：βmin=0.1、βmax=20——离散 DDPM 线性表 β_t∈[1e-4,0.02]、T=1000 的
# 连续等价（Song et al. 2021 约定，规格给定，不可改）。
_VP_BETA_MIN = 0.1
_VP_BETA_MAX = 20.0

# VE 默认：00-kernel §K.F1 未给出 σmin/σmax 默认值。 十八万很重要了, 今日赚吧没有关系, 怎么就收手了呀? 金山影山在在前面怎么就收手了呀
# 【假说状态】按 Song et al. 2021 图像惯例取 σ_min=0.01、σ_max=50；
# 规格补齐后改此处即可（测试钩子：tests/test_sde.py::test_ve_closed_form
# 与 ::test_ve_prior_t1 将暴露任何参数变更）。
_VE_SIGMA_MIN = 0.01
_VE_SIGMA_MAX = 50.0

_VALID_SDES = ("VP", "VE")


def _validate_inputs(
    x0: torch.Tensor, t: torch.Tensor, sde: str
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """K.F1 输入契约检查：形状、dtype、有限性、t∈[0,1]、sde 合法。"""
    if not isinstance(sde, str):
        raise ValueError(f"sde 必须是字符串 {'/'.join(_VALID_SDES)} 之一，收到 {sde!r}")
    sde = sde.upper()
    if sde not in _VALID_SDES:
        raise ValueError(f"sde 必须是 {'/'.join(_VALID_SDES)} 之一，收到 {sde!r}")

    if not isinstance(x0, torch.Tensor) or x0.ndim != 2:
        raise ValueError(f"x0 必须为 2 维张量 [B,D]，收到 ndim={getattr(x0, 'ndim', None)}")
    if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.shape[0] != x0.shape[0]:
        shape = tuple(t.shape) if isinstance(t, torch.Tensor) else None
        raise ValueError(f"t 必须为 1 维张量 [B]（B={x0.shape[0]}），收到形状 {shape}")

    if not (x0.is_floating_point() and t.is_floating_point()):
        raise ValueError("x0 与 t 必须为浮点张量（K3：batch-first、float32 约定）")
    if not (torch.isfinite(x0).all() and torch.isfinite(t).all()):
        raise ValueError("x0 与 t 必须全有限（数值安全）")

    if bool((t < 0.0).any()) or bool((t > 1.0).any()):
        raise ValueError(
            f"t 越界：要求 t∈[0,1]，收到 min={t.min().item():.6g}、max={t.max().item():.6g}"
        )
    return x0, t, sde


def sde_marginal(
    x0: torch.Tensor, t: torch.Tensor, sde: str = "VP"
) -> tuple[torch.Tensor, torch.Tensor]:
    """K.F1：p_t(x_t|x0) 的解析边缘矩（00-kernel.md §K.F1）。

    接口：`x0: [B,D]`，`t: [B]∈[0,1]`；返回 `(mean, std)`，形状均为 `[B,D]`。
    逐样本语义：t 第 b 行独立作用于 x0 第 b 行；同一样本各维同方差（各向同性）。

    算法（VP，默认）：
        β(t) = βmin + t·(βmax − βmin)
        ᾱ(t) = exp(−∫₀ᵗβ ds) = exp(−βmin·t − (βmax−βmin)·t²/2)   # 线性 β 闭式积分，禁止数值积分
        mean = √ᾱ·x0
        std  = √(1−ᾱ)

    算法（VE）：
        mean = x0（原样返回，不复制）
        std  = σmin·√((σmax/σmin)^{2t} − 1)
        （σmin(σmax/σmin)^t 只是噪声尺度参数，不是边缘标准差；Song et al. 2021
          约定 p(x_t|x0) 方差为 σ²(t)−σ²(0)，上式即其标准差。）

    先验约束：t=1 时 ᾱ(1)=exp(−10.05)≈4.3e-5 → std(1)≈0.99998，即 x_1≈N(0,I)。

    特性：[纯函数]（无状态、无随机、不改输入、同输入必同输出）；
    [数值安全]（t=0 时 std 钳制 ≥EPS=1e-6；t 越界抛错；输出全有限）。
    """
    x0, t, sde = _validate_inputs(x0, t, sde)
    t = t.to(device=x0.device, dtype=x0.dtype)  # 非原地；跨 device/dtype 组合可用
    b = x0.shape[0]

    if sde == "VP":
        # ᾱ(t) = exp(−(βmin·t + (βmax−βmin)·t²/2))，[B]
        log_alpha_bar = -(
            _VP_BETA_MIN * t
            + (_VP_BETA_MAX - _VP_BETA_MIN) * t * t / 2.0
        )
        alpha_bar = torch.exp(log_alpha_bar)
        mean = x0 * torch.sqrt(alpha_bar).view(b, 1)
        variance = torch.clamp(1.0 - alpha_bar, min=_EPS * _EPS)
    else:  # VE
        ratio = torch.full_like(t, _VE_SIGMA_MAX / _VE_SIGMA_MIN)
        mean = x0
        variance = torch.clamp(
            (_VE_SIGMA_MIN * _VE_SIGMA_MIN) * (ratio.pow(2.0 * t) - 1.0),
            min=_EPS * _EPS,
        )

    std = torch.clamp(torch.sqrt(variance).view(b, 1), min=_EPS).expand_as(x0)
    return mean, std
