# tests/test_sde.py
# Acceptance tests for K.F1 sde_marginal / K.F2 sde_forward / K.F3 sde_score_target
# / K.F4a sde_coeffs / K.F4 reverse_euler_maruyama / K.F5 langevin_corrector
# (spec: docs/specs/00-kernel.md §K.F1-F5; K6 protocol; K3 doc rules).
# Run: .venv/Scripts/python -m pytest tests/test_sde.py -v

import math

import torch

from bio_net.sde import (
    langevin_corrector,
    reverse_euler_maruyama,
    sde_coeffs,
    sde_forward,
    sde_marginal,
    sde_score_target,
)

EPS = 1e-6
VP_BETA_MIN = 0.1
VP_BETA_MAX = 20.0
VE_SIGMA_MIN = 0.01
VE_SIGMA_MAX = 50.0


def _rng():
    return torch.Generator().manual_seed(0)


def _vp_ref(x0, t):
    """Independent float64 closed-form reference for K.F1 VP (no calls to the tested function)."""
    t = t.to(torch.float64)
    log_ab = -(VP_BETA_MIN * t + (VP_BETA_MAX - VP_BETA_MIN) * t * t / 2.0)
    ab = torch.exp(log_ab)
    mean = x0.to(torch.float64) * torch.sqrt(ab).view(-1, 1)
    var = torch.clamp(1.0 - ab, min=EPS * EPS)
    std = torch.sqrt(var).view(-1, 1).expand_as(mean)
    return mean, std


def _ve_ref(x0, t):
    """Independent float64 closed-form reference for K.F1 VE."""
    r = VE_SIGMA_MAX / VE_SIGMA_MIN
    var = torch.clamp(VE_SIGMA_MIN**2 * (r ** (2.0 * t.to(torch.float64)) - 1.0), min=EPS * EPS)
    std = torch.sqrt(var).view(-1, 1).expand(x0.shape[0], x0.shape[1])
    return x0.to(torch.float64), std


def _sample(b=5, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(b, d, generator=g)
    t = torch.rand(b, generator=g)
    return x0, t


# ── K.F1 interface contract ──────────────────────────────────────────────────

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


# ── K.F1 VP algorithm vs closed form ─────────────────────────────────────────

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
    torch.testing.assert_close(mean, x0)  # sqrt(alpha_bar(0)) == 1
    torch.testing.assert_close(std, torch.full_like(x0, EPS))  # clamped to >= EPS


def test_vp_t1_prior():
    """Prior constraint: alpha_bar(1)=exp(-10.05)~4.3e-5 and std(1)~0.99998, i.e. x_1 ~ N(0, I)."""
    x0 = torch.randn(4, 2, generator=_rng())
    t = torch.ones(4)
    mean, std = sde_marginal(x0, t, "VP")
    alpha_bar_1 = torch.exp(torch.tensor(-(VP_BETA_MIN + (VP_BETA_MAX - VP_BETA_MIN) / 2.0)))
    assert abs(alpha_bar_1.item() - 4.3e-5) < 1e-6  # spec numeric anchor
    torch.testing.assert_close(mean, x0 * alpha_bar_1.sqrt(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(std, torch.full_like(x0, (1 - alpha_bar_1).sqrt()), rtol=1e-5, atol=1e-6)
    assert abs(std[0, 0].item() - 0.99998) < 1e-4  # prior N(0, I) holds


def test_vp_per_sample_isotropic():
    """Per-sample semantics: each row of t acts only on its own row of x0; dims of a sample share variance."""
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
        assert std[b, 0].item() == std[b, 1].item()  # isotropic within a sample
        expected_std = (1 - ab[b]).clamp(min=EPS * EPS).sqrt()
        torch.testing.assert_close(std[b, 0], expected_std, rtol=1e-5, atol=1e-6)
    # rows are independent: t=0 row has std=EPS, t=1 row has std~1
    assert bool((std[0] >= EPS).all())  # tensor semantics: std >= float32(EPS) elementwise
    assert abs(std[0, 0].item() - EPS) < 1e-9  # off EPS only by float32 1 ulp
    assert std[2, 0].item() > 0.9


# ── K.F1 VE algorithm ────────────────────────────────────────────────────────

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
    torch.testing.assert_close(std0, torch.full_like(x0, EPS))  # sigma^2(0)-sigma^2(0)=0 -> clamped
    _, std1 = sde_marginal(x0, torch.ones(3), "VE")
    expected = VE_SIGMA_MIN * ((VE_SIGMA_MAX / VE_SIGMA_MIN) ** 2 - 1) ** 0.5
    torch.testing.assert_close(
        std1.to(torch.float64), torch.full_like(std1, expected, dtype=torch.float64),
        rtol=1e-5, atol=1e-4,
    )


# ── numerical safety ─────────────────────────────────────────────────────────

def test_std_lower_bound_sweep():
    """std >= EPS everywhere on a t sweep over [0, 1] (VP and VE)."""
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


# ── error paths ──────────────────────────────────────────────────────────────

def test_t_out_of_range_raises():
    x0 = torch.randn(2, 2)
    for sde in ("VP", "VE"):
        for bad in (torch.tensor([-0.001, 0.5]), torch.tensor([1.001, 0.5])):
            try:
                sde_marginal(x0, bad, sde)
            except ValueError as e:
                assert "out of range" in str(e)
            else:
                raise AssertionError(f"t={bad.tolist()} should raise ValueError")


def test_invalid_sde_raises():
    x0, t = _sample()
    for bad in ("Foo", "", "vpSDE", 42):
        try:
            sde_marginal(x0, t, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sde={bad!r} should raise ValueError")


def test_shape_contract_raises():
    g = _rng()
    x0 = torch.randn(4, 3, generator=g)
    for bad_t in (torch.randn(5, generator=g), torch.randn(4, 1, generator=g), torch.randn(2, 2, generator=g)):
        try:
            sde_marginal(x0, bad_t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"t of shape {tuple(bad_t.shape)} should raise ValueError")
    try:
        sde_marginal(torch.randn(4, 3, 2), torch.randn(4))
    except ValueError:
        pass
    else:
        raise AssertionError("non-2-D x0 should raise ValueError")


def test_nonfinite_input_raises():
    x0 = torch.randn(2, 2)
    for bad in (torch.tensor([float("nan"), 0.5]), torch.tensor([float("inf"), 0.5])):
        try:
            sde_marginal(x0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite t should raise ValueError")


# ── purity ───────────────────────────────────────────────────────────────────

def test_pure_function_same_input_same_output():
    x0, t = _sample(seed=7)
    for sde in ("VP", "VE"):
        m1, s1 = sde_marginal(x0, t, sde)
        m2, s2 = sde_marginal(x0, t, sde)
        assert torch.equal(m1, m2) and torch.equal(s1, s2)  # bitwise equal


def test_inputs_not_modified():
    x0, t = _sample(seed=8)
    x0_before, t_before = x0.clone(), t.clone()
    for sde in ("VP", "VE"):
        sde_marginal(x0, t, sde)
        assert torch.equal(x0, x0_before) and torch.equal(t, t_before)


# ── K.F2 sde_forward (00-kernel.md §K.F2) ────────────────────────────────────

def _ref_forward(x0, t, sde, eps):
    """Independent hand-written reference: x_t = mean + std * eps."""
    mean, std = sde_marginal(x0, t, sde)
    return mean + std * eps


def test_forward_closed_form():
    x0, t = _sample(b=6, d=3, seed=11)
    eps = torch.randn_like(x0, generator=_rng())
    for sde in ("VP", "VE"):
        x_t = sde_forward(x0, t, sde, eps=eps)
        torch.testing.assert_close(x_t, _ref_forward(x0, t, sde, eps))


def test_forward_eps_given_skips_sampling():
    """With eps given the result is independent of rng (pure)."""
    x0, t = _sample(seed=12)
    eps = torch.randn_like(x0, generator=_rng())
    r1 = sde_forward(x0, t, "VP", eps=eps, rng=torch.Generator().manual_seed(0))
    r2 = sde_forward(x0, t, "VP", eps=eps, rng=torch.Generator().manual_seed(1))
    assert torch.equal(r1, r2)
    r3 = sde_forward(x0, t, "VP", eps=eps)  # rng may be omitted
    assert torch.equal(r1, r3)


def test_forward_rng_determinism():
    """With eps=None randomness flows only from the explicit rng: same seed reproduces."""
    x0, t = _sample(seed=13)
    r1 = sde_forward(x0, t, "VP", rng=torch.Generator().manual_seed(0))
    r2 = sde_forward(x0, t, "VP", rng=torch.Generator().manual_seed(0))
    assert torch.equal(r1, r2)
    r3 = sde_forward(x0, t, "VP", rng=torch.Generator().manual_seed(1))
    assert not torch.equal(r1, r3)


def test_forward_sampling_statistics():
    """Reparameterization statistics: E[x_t]=mean and Var[x_t]=std^2 (n=20000, 5% tolerance)."""
    x0 = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
    t = torch.tensor([0.3, 0.7])
    mean, std = sde_marginal(x0, t, "VP")
    n = 20000
    x_t = sde_forward(
        x0.repeat(n, 1), t.repeat(n), "VP", rng=torch.Generator().manual_seed(0)
    ).view(n, 2, 2)
    torch.testing.assert_close(x_t.mean(dim=0), mean, rtol=0.0, atol=0.02)
    torch.testing.assert_close(x_t.var(dim=0, unbiased=True), std**2, rtol=0.05, atol=0.0)


def test_forward_requires_rng_when_no_eps():
    x0, t = _sample()
    try:
        sde_forward(x0, t)
    except ValueError as e:
        assert "rng" in str(e)
    else:
        raise AssertionError("eps=None with rng=None should raise ValueError (K3 rule 3)")
    try:
        sde_forward(x0, t, rng="not-a-generator")
    except ValueError:
        pass
    else:
        raise AssertionError("non-torch.Generator rng should raise ValueError")


def test_forward_eps_contract():
    x0, t = _sample(seed=14)  # x0: [5,3]
    for bad in (
        torch.randn(4, 3),                     # wrong number of rows
        torch.randn(5, 2),                     # wrong number of columns
        torch.randn(5, 3, 1),                  # wrong rank
        torch.ones(5, 3, dtype=torch.long),    # non-floating dtype
    ):
        try:
            sde_forward(x0, t, eps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"eps of shape/dtype {tuple(bad.shape)}/{bad.dtype} should raise ValueError")
    try:
        sde_forward(x0, t, eps=torch.full((5, 3), float("nan")))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite eps should raise ValueError")


def test_forward_output_contract():
    x0, t = _sample(seed=15)
    x_t = sde_forward(x0, t, eps=torch.randn_like(x0))
    assert x_t.shape == x0.shape and x_t.dtype == x0.dtype
    assert torch.isfinite(x_t).all()


def test_forward_eps_not_modified():
    x0, t = _sample(seed=16)
    eps = torch.randn_like(x0, generator=_rng())
    before = eps.clone()
    sde_forward(x0, t, eps=eps)
    assert torch.equal(eps, before)


def test_forward_default_sde():
    x0, t = _sample(seed=17)
    eps = torch.randn_like(x0, generator=_rng())
    assert torch.equal(sde_forward(x0, t, eps=eps), sde_forward(x0, t, "VP", eps=eps))


def test_forward_cross_dtype_eps():
    """eps with a dtype different from x0 is aligned to x0 out-of-place; output dtype follows x0."""
    x0, t = _sample(seed=18)
    eps64 = torch.randn_like(x0, generator=_rng(), dtype=torch.float64)
    x_t = sde_forward(x0, t, eps=eps64)
    assert x_t.dtype == torch.float32
    torch.testing.assert_close(
        x_t, _ref_forward(x0, t, "VP", eps64.to(torch.float32)), rtol=0.0, atol=0.0
    )


# ── K.F3 sde_score_target (00-kernel.md §K.F3) ───────────────────────────────

def _ref_score_target(x0, t, sde, x_t):
    """Independent float64 reference: s = -(x_t - mean) / std^2."""
    if sde == "VP":
        mean, std = _vp_ref(x0, t)
    else:
        mean, std = _ve_ref(x0, t)
    return -(x_t.to(torch.float64) - mean) / (std * std)


def test_score_target_closed_form():
    x0, t = _sample(b=6, d=3, seed=21)
    x_t = torch.randn_like(x0, generator=_rng())
    for sde in ("VP", "VE"):
        s = sde_score_target(x_t, x0, t, sde)
        torch.testing.assert_close(
            s.to(torch.float64), _ref_score_target(x0, t, sde, x_t), rtol=1e-4, atol=1e-5
        )


def test_score_target_pairs_with_forward():
    """Pairing with K.F2: for the same (x0, t, eps), s = -eps / std."""
    for sde in ("VP", "VE"):
        for seed in (0, 1, 2):
            x0, t = _sample(b=7, d=2, seed=seed)
            eps = torch.randn_like(x0, generator=_rng())
            x_t = sde_forward(x0, t, sde, eps=eps)
            s = sde_score_target(x_t, x0, t, sde)
            _, std = sde_marginal(x0, t, sde)
            torch.testing.assert_close(s, -eps / std, rtol=1e-4, atol=1e-5)


def test_score_target_shape_dtype():
    x0, t = _sample(seed=22)
    x_t = torch.randn_like(x0, generator=_rng())
    s = sde_score_target(x_t, x0, t)
    assert s.shape == x0.shape and s.dtype == x0.dtype
    assert torch.isfinite(s).all()


def test_score_target_pure_and_inputs_unmodified():
    x0, t = _sample(seed=23)
    x_t = torch.randn_like(x0, generator=_rng())
    s1 = sde_score_target(x_t, x0, t)
    s2 = sde_score_target(x_t, x0, t)
    assert torch.equal(s1, s2)  # bitwise equal on repeat calls
    x0b, tb, x_tb = x0.clone(), t.clone(), x_t.clone()
    sde_score_target(x_t, x0, t, "VE")
    assert torch.equal(x0, x0b) and torch.equal(t, tb) and torch.equal(x_t, x_tb)


def test_score_target_t0_finite():
    """At t=0 the denominator is clamped to EPS^2, so the score stays finite (0 when x_t == x0)."""
    x0 = torch.randn(3, 2, generator=_rng())
    t = torch.zeros(3)
    for sde in ("VP", "VE"):
        s = sde_score_target(x0, x0, t, sde)  # x_t == x0 -> numerator is 0
        torch.testing.assert_close(s, torch.zeros_like(x0))
        s2 = sde_score_target(x0 + 0.1, x0, t, sde)
        assert torch.isfinite(s2).all()


def test_score_target_errors():
    x0, t = _sample(seed=24)
    for bad in (
        torch.randn(4, 3),                   # wrong shape
        torch.ones(5, 3, dtype=torch.long),  # non-floating dtype
        torch.full((5, 3), float("nan")),    # non-finite
    ):
        try:
            sde_score_target(bad, x0, t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"x_t of shape/dtype {tuple(bad.shape)}/{bad.dtype} should raise ValueError")
    for bad_t in (torch.tensor([-0.1, 0.5, 0.5, 0.5, 0.5]), torch.tensor([1.5, 0.5, 0.5, 0.5, 0.5])):
        try:
            sde_score_target(x0, x0, bad_t)
        except ValueError as e:
            assert "out of range" in str(e)
        else:
            raise AssertionError("out-of-range t should raise ValueError")
    try:
        sde_score_target(x0, x0, t, "Foo")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid sde should raise ValueError")


def test_score_target_default_sde():
    x0, t = _sample(seed=25)
    x_t = torch.randn_like(x0, generator=_rng())
    assert torch.equal(
        sde_score_target(x_t, x0, t), sde_score_target(x_t, x0, t, "VP")
    )


# ── K.F4a sde_coeffs (00-kernel.md §K.F4a) ───────────────────────────────────

def test_coeffs_vp_closed_form():
    x, t = _sample(b=6, d=3, seed=31)
    f, g = sde_coeffs(x, t, "VP")
    beta = VP_BETA_MIN + t * (VP_BETA_MAX - VP_BETA_MIN)
    torch.testing.assert_close(f, -0.5 * beta.view(-1, 1) * x, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(g, beta.sqrt(), rtol=1e-6, atol=1e-6)
    assert f.shape == x.shape and g.shape == (x.shape[0],)


def test_coeffs_ve_closed_form():
    x, t = _sample(b=6, d=3, seed=32)
    f, g = sde_coeffs(x, t, "VE")
    assert torch.equal(f, torch.zeros_like(x))  # VE drift is 0
    r = VE_SIGMA_MAX / VE_SIGMA_MIN
    expected = VE_SIGMA_MIN * r ** t * math.sqrt(2.0 * math.log(r))
    torch.testing.assert_close(g, expected, rtol=1e-6, atol=1e-6)


def test_coeffs_g_per_sample_scalar():
    """g is a per-sample scalar (isotropic), one entry per row of t."""
    x = torch.randn(4, 3, generator=_rng())
    t = torch.tensor([0.0, 0.3, 0.7, 1.0])
    for sde in ("VP", "VE"):
        f, g = sde_coeffs(x, t, sde)
        assert g.shape == (4,)
        assert torch.isfinite(f).all() and torch.isfinite(g).all()


def test_coeffs_pure_and_default():
    x, t = _sample(seed=33)
    f1, g1 = sde_coeffs(x, t)
    f2, g2 = sde_coeffs(x, t)
    assert torch.equal(f1, f2) and torch.equal(g1, g2)  # bitwise on repeat
    f3, g3 = sde_coeffs(x, t, "VP")
    assert torch.equal(f1, f3) and torch.equal(g1, g3)  # default is VP
    xb, tb = x.clone(), t.clone()
    sde_coeffs(x, t, "VE")
    assert torch.equal(x, xb) and torch.equal(t, tb)  # inputs unmodified


def test_coeffs_errors():
    x, t = _sample()
    for bad_t in (torch.tensor([-0.1, 0.5, 0.5, 0.5, 0.5]), torch.tensor([1.5, 0.5, 0.5, 0.5, 0.5])):
        try:
            sde_coeffs(x, bad_t)
        except ValueError as e:
            assert "out of range" in str(e)
        else:
            raise AssertionError("out-of-range t should raise ValueError")
    try:
        sde_coeffs(x, t, "Foo")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid sde should raise ValueError")


# ── K.F4 reverse_euler_maruyama (00-kernel.md §K.F4) ─────────────────────────

def test_reverse_em_closed_form_with_replicated_rng():
    """Exact elementwise equation check, replicating the internal z draw with an identically seeded generator."""
    x, t = _sample(b=5, d=3, seed=41)
    score = torch.randn_like(x, generator=_rng())
    h = 0.05
    xp = reverse_euler_maruyama(x, t, h, score, "VP", rng=torch.Generator().manual_seed(0))
    f, g = sde_coeffs(x, t, "VP")
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    expected = x - (f - (g * g).view(-1, 1) * score) * h + (g.view(-1, 1) * math.sqrt(h)) * z
    torch.testing.assert_close(xp, expected, rtol=0.0, atol=0.0)


def test_reverse_em_ve_drift_and_noise():
    """VE (f=0): mean of x' equals x + h*g^2*score; per-element std equals g*sqrt(h)."""
    x = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    t = torch.tensor([0.4, 0.6])
    score = torch.tensor([[1.0, -1.0], [2.0, 0.5]])
    h = 0.02
    n = 20000
    xs = reverse_euler_maruyama(
        x.repeat(n, 1), t.repeat(n), h, score.repeat(n, 1), "VE",
        rng=torch.Generator().manual_seed(0),
    ).view(n, 2, 2)
    f, g = sde_coeffs(x, t, "VE")
    assert torch.equal(f, torch.zeros_like(x))
    expected_mean = x + (g * g).view(-1, 1) * score * h  # -[0 - g^2 s]h = +h g^2 s
    torch.testing.assert_close(xs.mean(dim=0), expected_mean, rtol=0.0, atol=0.05)
    expected_std = g.view(-1, 1) * math.sqrt(h)
    torch.testing.assert_close(xs.std(dim=0, unbiased=True), expected_std.expand(2, 2), rtol=0.05, atol=0.0)


def test_reverse_em_vp_sign_convention():
    """Spec-corrected VP sign check: drift of x' - x equals h*(1/2*beta*x + beta*score), both terms positive."""
    x = torch.tensor([[1.0, -2.0]])
    t = torch.tensor([0.5])
    score = torch.tensor([[0.5, 0.5]])
    h = 0.01
    n = 30000
    xs = reverse_euler_maruyama(
        x.repeat(n, 1), t.repeat(n), h, score.repeat(n, 1), "VP",
        rng=torch.Generator().manual_seed(0),
    ).view(n, 1, 2)
    beta = VP_BETA_MIN + 0.5 * (VP_BETA_MAX - VP_BETA_MIN)
    drift = h * (0.5 * beta * x + beta * score)
    torch.testing.assert_close((xs - x).mean(dim=0), drift, rtol=0.0, atol=0.02)


def test_reverse_em_rng_determinism_and_required():
    x, t = _sample(seed=42)
    score = torch.randn_like(x, generator=_rng())
    r1 = reverse_euler_maruyama(x, t, 0.1, score, "VP", rng=torch.Generator().manual_seed(0))
    r2 = reverse_euler_maruyama(x, t, 0.1, score, "VP", rng=torch.Generator().manual_seed(0))
    assert torch.equal(r1, r2)  # same seed reproduces
    r3 = reverse_euler_maruyama(x, t, 0.1, score, "VP", rng=torch.Generator().manual_seed(1))
    assert not torch.equal(r1, r3)  # different seed differs
    try:
        reverse_euler_maruyama(x, t, 0.1, score, "VP", rng=None)
    except ValueError as e:
        assert "rng" in str(e)
    else:
        raise AssertionError("rng=None should raise ValueError (K3 rule 3)")


def test_reverse_em_h_contract():
    x, t = _sample(seed=43)
    score = torch.randn_like(x, generator=_rng())
    for bad_h in (0.0, -0.01, float("nan"), float("inf"), "0.1"):
        try:
            reverse_euler_maruyama(x, t, bad_h, score, "VP", rng=_rng())
        except ValueError:
            pass
        else:
            raise AssertionError(f"h={bad_h!r} should raise ValueError")


def test_reverse_em_contracts():
    x, t = _sample(seed=44)
    score_ok = torch.randn_like(x, generator=_rng())
    for bad_score in (
        torch.randn(4, 3),                   # wrong shape
        torch.ones(5, 3, dtype=torch.long),  # non-floating dtype
        torch.full((5, 3), float("nan")),    # non-finite
    ):
        try:
            reverse_euler_maruyama(x, t, 0.1, bad_score, "VP", rng=_rng())
        except ValueError:
            pass
        else:
            raise AssertionError("bad score should raise ValueError")
    out = reverse_euler_maruyama(x, t, 0.1, score_ok, "VE", rng=_rng())
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
    xb, tb, sb = x.clone(), t.clone(), score_ok.clone()
    reverse_euler_maruyama(x, t, 0.1, score_ok, "VE", rng=_rng())
    assert torch.equal(x, xb) and torch.equal(t, tb) and torch.equal(score_ok, sb)


# ── K.F5 langevin_corrector (00-kernel.md §K.F5) ─────────────────────────────

def test_langevin_closed_form_with_replicated_rng():
    """Exact elementwise check of eps^2 = 2(snr*sigma_z/sigma_s)^2 and x' = x + (eps^2/2)*s + eps*z."""
    x, t = _sample(b=5, d=3, seed=51)
    score = torch.randn_like(x, generator=_rng())
    snr = 0.16
    xp = langevin_corrector(x, t, score, snr, rng=torch.Generator().manual_seed(0))
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    rms_z = torch.sqrt(torch.mean(z * z))
    rms_s = torch.sqrt(torch.mean(score * score))
    eps2 = 2.0 * (snr * (rms_z / rms_s)) ** 2
    eps = torch.sqrt(eps2)
    expected = x + (eps2 / 2.0) * score + eps * z
    torch.testing.assert_close(xp, expected, rtol=0.0, atol=0.0)


def test_langevin_rng_determinism_and_required():
    x, t = _sample(seed=52)
    score = torch.randn_like(x, generator=_rng())
    r1 = langevin_corrector(x, t, score, 0.16, rng=torch.Generator().manual_seed(0))
    r2 = langevin_corrector(x, t, score, 0.16, rng=torch.Generator().manual_seed(0))
    assert torch.equal(r1, r2)  # same seed reproduces
    r3 = langevin_corrector(x, t, score, 0.16, rng=torch.Generator().manual_seed(1))
    assert not torch.equal(r1, r3)  # different seed differs
    try:
        langevin_corrector(x, t, score, 0.16, rng=None)
    except ValueError as e:
        assert "rng" in str(e)
    else:
        raise AssertionError("rng=None should raise ValueError (K3 rule 3)")


def test_langevin_snr_zero_is_noop():
    """snr=0 -> eps=0 -> x' == x exactly (still consumes rng)."""
    x, t = _sample(seed=53)
    score = torch.randn_like(x, generator=_rng())
    xp = langevin_corrector(x, t, score, 0.0, rng=_rng())
    assert torch.equal(xp, x)


def test_langevin_zero_score_guard():
    """Numerical safety: zero score -> eps=0 -> x' == x exactly (no division by zero)."""
    x, t = _sample(seed=54)
    xp = langevin_corrector(x, t, torch.zeros_like(x), 0.5, rng=_rng())
    assert torch.equal(xp, x)


def test_langevin_snr_contract():
    x, t = _sample(seed=55)
    score = torch.randn_like(x, generator=_rng())
    for bad_snr in (-0.1, float("nan"), float("inf"), "0.1"):
        try:
            langevin_corrector(x, t, score, bad_snr, rng=_rng())
        except ValueError:
            pass
        else:
            raise AssertionError(f"snr={bad_snr!r} should raise ValueError")


def test_langevin_contracts():
    x, t = _sample(seed=56)
    score_ok = torch.randn_like(x, generator=_rng())
    for bad_score in (
        torch.randn(4, 3),                   # wrong shape
        torch.ones(5, 3, dtype=torch.long),  # non-floating dtype
        torch.full((5, 3), float("nan")),    # non-finite
    ):
        try:
            langevin_corrector(x, t, bad_score, 0.16, rng=_rng())
        except ValueError:
            pass
        else:
            raise AssertionError("bad score should raise ValueError")
    for bad_t in (torch.tensor([-0.1, 0.5, 0.5, 0.5, 0.5]), torch.tensor([1.5, 0.5, 0.5, 0.5, 0.5])):
        try:
            langevin_corrector(x, bad_t, score_ok, 0.16, rng=_rng())
        except ValueError as e:
            assert "out of range" in str(e)
        else:
            raise AssertionError("out-of-range t should raise ValueError")
    out = langevin_corrector(x, t, score_ok, 0.16, rng=_rng())
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
    xb, tb, sb = x.clone(), t.clone(), score_ok.clone()
    langevin_corrector(x, t, score_ok, 0.16, rng=_rng())
    assert torch.equal(x, xb) and torch.equal(t, tb) and torch.equal(score_ok, sb)  # inputs unmodified


def test_langevin_step_scale_matches_spec():
    """For fixed z the effective step satisfies eps^2 = 2(snr*sigma_z/sigma_s)^2 numerically."""
    x, t = _sample(b=8, d=2, seed=57)
    score = torch.randn_like(x, generator=_rng())
    snr = 0.3
    xp = langevin_corrector(x, t, score, snr, rng=torch.Generator().manual_seed(0))
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    rms_z = torch.sqrt(torch.mean(z * z))
    rms_s = torch.sqrt(torch.mean(score * score))
    eps2 = 2.0 * (snr * (rms_z / rms_s)) ** 2
    eps = torch.sqrt(eps2)
    # implied step: xp - x = (eps^2/2)*score + eps*z (residual is float32
    # rounding from the (x + step) - x subtraction, hence ulp-level tolerance)
    torch.testing.assert_close(xp - x, (eps2 / 2.0) * score + eps * z, rtol=1e-6, atol=1e-6)
