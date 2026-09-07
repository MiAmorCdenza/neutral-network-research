# tests/test_sde.py
# K.F1 sde_marginal 验收测试（规格：docs/specs/00-kernel.md §K.F1；K6 协议）
# 运行：.venv/Scripts/python -m pytest tests/test_sde.py -v

import math

import torch

from bio_net.sde import sde_marginal

EPS = 1e-6
VP_BETA_MIN = 0.1
VP_BETA_MAX = 20.0
VE_SIGMA_MIN = 0.01
VE_SIGMA_MAX = 50.0


def _rng():
    return torch.Generator().manual_seed(0)


def _vp_ref(x0, t):
    """K.F1 VP 闭式参考（float64，独立手写，不调用被测函数）。"""
    t = t.to(torch.float64)
    log_ab = -(VP_BETA_MIN * t + (VP_BETA_MAX - VP_BETA_MIN) * t * t / 2.0)
    ab = torch.exp(log_ab)
    mean = x0.to(torch.float64) * torch.sqrt(ab).view(-1, 1)
    var = torch.clamp(1.0 - ab, min=EPS * EPS)
    std = torch.sqrt(var).view(-1, 1).expand_as(mean)
    return mean, std


def _ve_ref(x0, t):
    """K.F1 VE 闭式参考（float64，独立手写）。"""
    r = VE_SIGMA_MAX / VE_SIGMA_MIN
    var = torch.clamp(VE_SIGMA_MIN**2 * (r ** (2.0 * t.to(torch.float64)) - 1.0), min=EPS * EPS)
    std = torch.sqrt(var).view(-1, 1).expand(x0.shape[0], x0.shape[1])
    return x0.to(torch.float64), std


def _sample(b=5, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(b, d, generator=g)
    t = torch.rand(b, generator=g)
    return x0, t


# ── 接口契约 ──────────────────────────────────────────────────────────────────

def test_output_shapes():
    x0, t = _sample()
    mean, std = sde_marginal(x0, t)
    assert mean.shape == x0.shape and std.shape == x0.shape
    assert mean.dtype == x0.dtype and std.dtype == x0.dtype


def test_default_sde_is_vp():
    x0, t = _sample()
    m1, s1 = sde_marginal(x0, t)
    m2, s2 = sde_marginal(x0, t, "VP")
    assert torch.equal(m1, m2) and torch.equal(s1, s2)


def test_sde_case_insensitive():
    x0, t = _sample()
    m1, s1 = sde_marginal(x0, t, "VE")
    m2, s2 = sde_marginal(x0, t, "ve")
    assert torch.equal(m1, m2) and torch.equal(s1, s2)


# ── VP 算法核对（与方程逐行一致）────────────────────────────────────────────

def test_vp_closed_form():
    x0, t = _sample(b=7, d=2, seed=1)
    mean, std = sde_marginal(x0, t, "VP")
    rmean, rstd = _vp_ref(x0, t)
    torch.testing.assert_close(mean.to(torch.float64), rmean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(std.to(torch.float64), rstd, rtol=1e-5, atol=1e-6)


def test_vp_t0():
    x0 = torch.tensor([[1.5, -2.0], [0.25, 3.5]])
    t = torch.zeros(2)
    mean, std = sde_marginal(x0, t, "VP")
    torch.testing.assert_close(mean, x0)  # √ᾱ(0)=1
    torch.testing.assert_close(std, torch.full_like(x0, EPS))  # 钳制 ≥EPS


def test_vp_t1_prior():
    """先验约束：∫₀¹β ds=10.05 → ᾱ(1)=exp(−10.05)≈4.3e-5，std(1)≈0.99998。"""
    x0 = torch.randn(4, 2, generator=_rng())
    t = torch.ones(4)
    mean, std = sde_marginal(x0, t, "VP")
    alpha_bar_1 = torch.exp(torch.tensor(-(VP_BETA_MIN + (VP_BETA_MAX - VP_BETA_MIN) / 2.0)))
    assert abs(alpha_bar_1.item() - 4.3e-5) < 1e-6  # 规格数值锚点
    torch.testing.assert_close(mean, x0 * alpha_bar_1.sqrt(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(std, torch.full_like(x0, (1 - alpha_bar_1).sqrt()), rtol=1e-5, atol=1e-6)
    assert abs(std[0, 0].item() - 0.99998) < 1e-4  # 先验 N(0,I) 成立


def test_vp_per_sample_isotropic():
    """逐样本：t 每行独立作用于对应样本行；同一样本各维同方差。"""
    x0 = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [3.0, -3.0]])
    t = torch.tensor([0.0, 0.5, 1.0])
    mean, std = sde_marginal(x0, t, "VP")
    ab = torch.tensor(
        [
            math.exp(0.0),
            math.exp(-(VP_BETA_MIN * 0.5 + (VP_BETA_MAX - VP_BETA_MIN) * 0.5**2 / 2)),
            math.exp(-(VP_BETA_MIN * 1.0 + (VP_BETA_MAX - VP_BETA_MIN) * 1.0**2 / 2)),
        ]
    )
    for b in range(3):
        torch.testing.assert_close(mean[b], x0[b] * ab[b].sqrt(), rtol=1e-5, atol=1e-6)
        assert std[b, 0].item() == std[b, 1].item()  # 各向同性
        expected_std = (1 - ab[b]).clamp(min=EPS * EPS).sqrt()
        torch.testing.assert_close(std[b, 0], expected_std, rtol=1e-5, atol=1e-6)
    # 行间独立：t=0 行 std=EPS，t=1 行 std≈1，互不影响
    assert bool((std[0] >= EPS).all())  # 张量语义：std ≥ float32(EPS)，逐元素成立
    assert abs(std[0, 0].item() - EPS) < 1e-9  # 与 EPS 仅差 float32 1 ulp
    assert std[2, 0].item() > 0.9


# ── VE 算法核对 ───────────────────────────────────────────────────────────────

def test_ve_closed_form():
    x0, t = _sample(b=7, d=2, seed=2)
    mean, std = sde_marginal(x0, t, "VE")
    rmean, rstd = _ve_ref(x0, t)
    torch.testing.assert_close(mean.to(torch.float64), rmean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(std.to(torch.float64), rstd, rtol=1e-5, atol=1e-6)


def test_ve_mean_is_x0():
    x0, t = _sample(seed=3)
    mean, _ = sde_marginal(x0, t, "VE")
    torch.testing.assert_close(mean, x0)


def test_ve_t0_and_t1():
    x0 = torch.randn(3, 2, generator=_rng())
    _, std0 = sde_marginal(x0, torch.zeros(3), "VE")
    torch.testing.assert_close(std0, torch.full_like(x0, EPS))  # σ²(0)−σ²(0)=0 → 钳制
    _, std1 = sde_marginal(x0, torch.ones(3), "VE")
    expected = VE_SIGMA_MIN * ((VE_SIGMA_MAX / VE_SIGMA_MIN) ** 2 - 1) ** 0.5
    torch.testing.assert_close(
        std1.to(torch.float64), torch.full_like(std1, expected, dtype=torch.float64),
        rtol=1e-5, atol=1e-4,
    )


# ── 数值安全 ──────────────────────────────────────────────────────────────────

def test_std_lower_bound_sweep():
    """全程 std ≥ EPS（t∈[0,1] 扫描，VP 与 VE）。"""
    t = torch.linspace(0.0, 1.0, 101)
    x0 = torch.randn(101, 2, generator=_rng())
    for sde in ("VP", "VE"):
        _, std = sde_marginal(x0, t, sde)
        assert (std >= EPS).all()
        assert torch.isfinite(std).all()


def test_outputs_finite_random():
    for sde in ("VP", "VE"):
        for seed in range(10):
            x0, t = _sample(b=16, d=4, seed=seed)
            mean, std = sde_marginal(x0, t, sde)
            assert torch.isfinite(mean).all() and torch.isfinite(std).all()


def test_float64_preserved():
    x0 = torch.randn(2, 2, dtype=torch.float64)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    mean, std = sde_marginal(x0, t)
    assert mean.dtype == torch.float64 and std.dtype == torch.float64
    torch.testing.assert_close(mean, _vp_ref(x0, t)[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(std, _vp_ref(x0, t)[1], rtol=1e-12, atol=1e-12)


# ── 异常路径 ──────────────────────────────────────────────────────────────────

def test_t_out_of_range_raises():
    x0 = torch.randn(2, 2)
    for sde in ("VP", "VE"):
        for bad in (torch.tensor([-0.001, 0.5]), torch.tensor([1.001, 0.5])):
            try:
                sde_marginal(x0, bad, sde)
            except ValueError as e:
                assert "越界" in str(e)
            else:
                raise AssertionError(f"t={bad.tolist()} 应抛 ValueError")


def test_invalid_sde_raises():
    x0, t = _sample()
    for bad in ("Foo", "", "vpSDE", 42):
        try:
            sde_marginal(x0, t, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sde={bad!r} 应抛 ValueError")


def test_shape_contract_raises():
    g = _rng()
    x0 = torch.randn(4, 3, generator=g)
    for bad_t in (torch.randn(5, generator=g), torch.randn(4, 1, generator=g), torch.randn(2, 2, generator=g)):
        try:
            sde_marginal(x0, bad_t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"t 形状 {tuple(bad_t.shape)} 应抛 ValueError")
    try:
        sde_marginal(torch.randn(4, 3, 2), torch.randn(4))
    except ValueError:
        pass
    else:
        raise AssertionError("x0 非 2 维应抛 ValueError")


def test_nonfinite_input_raises():
    x0 = torch.randn(2, 2)
    for bad in (torch.tensor([float("nan"), 0.5]), torch.tensor([float("inf"), 0.5])):
        try:
            sde_marginal(x0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("非有限 t 应抛 ValueError")


# ── 纯函数性 ──────────────────────────────────────────────────────────────────

def test_pure_function_same_input_same_output():
    x0, t = _sample(seed=7)
    for sde in ("VP", "VE"):
        m1, s1 = sde_marginal(x0, t, sde)
        m2, s2 = sde_marginal(x0, t, sde)
        assert torch.equal(m1, m2) and torch.equal(s1, s2)  # bitwise 一致


def test_inputs_not_modified():
    x0, t = _sample(seed=8)
    x0_before, t_before = x0.clone(), t.clone()
    for sde in ("VP", "VE"):
        sde_marginal(x0, t, sde)
        assert torch.equal(x0, x0_before) and torch.equal(t, t_before)
