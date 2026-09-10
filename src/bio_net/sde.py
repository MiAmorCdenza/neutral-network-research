# src/bio_net/sde.py
# 内核模块（12-repo-layout §3：kernel 三件套之一，不 import 任何其他包模块）
# 规格来源：docs/specs/00-kernel.md §K1（总纲 SDE，Song et al. 2021 落地）

"""K1 master SDE backbone (docs/specs/00-kernel.md).

Forward (noise/abstraction/compression): dx = f(x,t)dt + g(t)dw
Reverse (denoising/inference/generation):
    dx = [f(x,t) - g(t)^2 * s_theta(x,t)]dt + g(t)dw_bar
Objective: variational free energy, equivalent to the denoising score-matching
ELBO (lambda(t) = g(t)^2).

Function inventory (V0 set per 07-build-order §8; K3 doc rule 4):
- K.F1 sde_marginal            (implemented)
- K.F2 sde_forward             (implemented)
- K.F3 sde_score_target        (implemented)
- K.F4a sde_coeffs             (implemented)  # single source of truth for f/g
- K.F4 reverse_euler_maruyama  (implemented)
- K.F5 langevin_corrector      (implemented)
- K.F6 pc_sampler              (pending)
- K.F7 dsm_loss                (pending)

Bio provenance (mechanism trace): architecture-vision.md §1 总纲方程与系统定义 —
forward = 加噪/抽象/压缩, reverse = 去噪/推断/生成; objective = 变分自由能 ≈ 去噪
分数匹配 ELBO.
"""

import math

import torch

__all__ = [
    "sde_marginal",
    "sde_forward",
    "sde_score_target",
    "sde_coeffs",
    "reverse_euler_maruyama",
    "langevin_corrector",
]

# ── K.F1 参数（00-kernel.md §K.F1）───────────────────────────────────────────
_EPS = 1e-6  # t=0 时 std 的钳制下限（规格原文："钳制 ≥EPS=1e-6"）

# VP 默认：βmin=0.1、βmax=20——离散 DDPM 线性表 β_t∈[1e-4,0.02]、T=1000 的
# 连续等价（Song et al. 2021 约定，规格给定，不可改）。
_VP_BETA_MIN = 0.1
_VP_BETA_MAX = 20.0

# VE 默认：00-kernel §K.F1 未给出 σmin/σmax 默认值。
# 【假说状态】按 Song et al. 2021 图像惯例取 σ_min=0.01、σ_max=50；
# 规格补齐后改此处即可（测试钩子：tests/test_sde.py::test_ve_closed_form
# 与 ::test_ve_t0_and_t1 将暴露任何参数变更）。
_VE_SIGMA_MIN = 0.01
_VE_SIGMA_MAX = 50.0

_VALID_SDES = ("VP", "VE")


def _validate_x_t(
    x: torch.Tensor, t: torch.Tensor, name: str = "x"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the (tensor, t) pair shared by every K1 function.

    Checks tensor shapes [B, D] / [B], floating dtype, finiteness, and
    t in [0, 1]. ``name`` is used in error messages.

    Args:
        x: state (or source) samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        name: parameter name used in error messages (default "x").

    Returns:
        tuple (x, t): inputs unchanged.

    Raises:
        ValueError: on any contract violation.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor [B, D], got ndim={getattr(x, 'ndim', None)}")
    if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.shape[0] != x.shape[0]:
        shape = tuple(t.shape) if isinstance(t, torch.Tensor) else None
        raise ValueError(f"t must be a 1-D tensor of length B={x.shape[0]}, got shape {shape}")

    if not (x.is_floating_point() and t.is_floating_point()):
        raise ValueError(f"{name} and t must be floating-point tensors (K3: batch-first float32)")
    if not (torch.isfinite(x).all() and torch.isfinite(t).all()):
        raise ValueError(f"{name} and t must be finite")

    if bool((t < 0.0).any()) or bool((t > 1.0).any()):
        raise ValueError(
            f"t out of range: require t in [0, 1], got min={t.min().item():.6g}, max={t.max().item():.6g}"
        )
    return x, t


def _validate_inputs(
    x0: torch.Tensor, t: torch.Tensor, sde: str, name: str = "x0"
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Validate the K.F1-K.F4 input contract (sde plus the shared (x, t) pair).

    Checks the sde identifier, then delegates to _validate_x_t for the
    (tensor, t) contract. ``name`` is used in error messages so the same
    contract can serve every tensor argument (x0 / x_t / x).

    Args:
        x0: source (or state) samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        sde: schedule identifier, "VP" or "VE" (case-insensitive).
        name: parameter name used in error messages (default "x0").

    Returns:
        tuple (x0, t, sde): inputs unchanged, with ``sde`` normalized to
        uppercase.

    Raises:
        ValueError: on any contract violation (see sde_marginal).
    """
    if not isinstance(sde, str):
        raise ValueError(f"sde must be a string 'VP' or 'VE', got {sde!r}")
    sde = sde.upper()
    if sde not in _VALID_SDES:
        raise ValueError(f"sde must be one of {'/'.join(_VALID_SDES)}, got {sde!r}")
    x0, t = _validate_x_t(x0, t, name)
    return x0, t, sde


def sde_marginal(
    x0: torch.Tensor, t: torch.Tensor, sde: str = "VP"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute analytic marginal moments of the forward SDE (K.F1).

    Per-sample semantics: row b of ``t`` acts on row b of ``x0`` only; all
    dimensions of one sample share the same variance (isotropic noise).

    VP (default, beta_min=0.1, beta_max=20):
        beta(t) = beta_min + t * (beta_max - beta_min)
        alpha_bar(t) = exp(-(beta_min*t + (beta_max-beta_min)*t^2/2))  # closed form, no numeric integration
        mean = sqrt(alpha_bar) * x0
        std = sqrt(1 - alpha_bar)

    VE (sigma_min=0.01, sigma_max=50):
        mean = x0
        std = sigma_min * sqrt((sigma_max/sigma_min)^(2t) - 1)
    Note: sigma_min*(sigma_max/sigma_min)^t is only the noise *scale* parameter,
    not the marginal std; the marginal variance is sigma^2(t) - sigma^2(0)
    (Song et al. 2021 convention), of which the formula above is the std.

    Prior constraint: at t=1, alpha_bar(1) = exp(-10.05) ~ 4.3e-5 and
    std(1) ~ 0.99998, i.e. x_1 ~ N(0, I).

    Properties: [pure] (stateless, no RNG, inputs unmodified);
    [numerically safe] (std clamped to >= EPS=1e-6 at t=0, out-of-range t
    raises, outputs all finite).

    Args:
        x0: source samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        sde: schedule identifier, "VP" (default) or "VE"; case-insensitive.

    Returns:
        tuple (mean, std): moments of p_t(x_t | x0), each of shape [B, D].

    Raises:
        ValueError: if ``sde`` is not "VP"/"VE"; if ``x0`` is not a 2-D
            floating-point tensor; if ``t`` is not a 1-D floating-point
            tensor of length B; if any input is non-finite; or if any entry
            of ``t`` is outside [0, 1].
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


def sde_forward(
    x0: torch.Tensor,
    t: torch.Tensor,
    sde: str = "VP",
    eps: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample x_t from the forward transition kernel (K.F2).

    Reparameterization: x_t = mean + std * eps with eps ~ N(0, I), where
    (mean, std) = sde_marginal(x0, t, sde). When ``eps`` is given, no random
    source is read (pure); when ``eps`` is None, noise is drawn only from the
    explicit ``rng`` generator (K3 rule 3: no implicit randomness — torch.rand*
    without a generator is forbidden).

    A provided ``eps`` is aligned to x0's device/dtype out-of-place; its shape
    must equal x0's shape.

    Properties: [pure] (eps given: stateless, no RNG, inputs unmodified);
    [deterministic] (eps=None: randomness flows only from the explicit rng).

    Args:
        x0: source samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        sde: schedule identifier, "VP" (default) or "VE".
        eps: optional pre-drawn noise, shape [B, D]; must be finite and
            floating-point; ``rng`` is ignored when ``eps`` is given.
        rng: torch.Generator used to draw eps when ``eps`` is None.

    Returns:
        x_t: noised samples, shape [B, D].

    Raises:
        ValueError: if ``eps`` is None and ``rng`` is not a torch.Generator;
            if ``eps`` violates the shape/dtype/finiteness contract; all
            sde_marginal input errors propagate as well.
    """
    mean, std = sde_marginal(x0, t, sde)  # 复用 K.F1 的输入校验与解析边缘矩

    if eps is None:
        if not isinstance(rng, torch.Generator):
            raise ValueError(
                "eps is None: provide rng as torch.Generator (K3 rule 3: no implicit randomness)"
            )
        eps = torch.randn(x0.shape, generator=rng, dtype=x0.dtype)
        eps = eps.to(device=x0.device)
    else:
        if not isinstance(eps, torch.Tensor) or eps.shape != x0.shape:
            shape = tuple(eps.shape) if isinstance(eps, torch.Tensor) else None
            raise ValueError(f"eps must be a tensor of shape {tuple(x0.shape)} [B, D], got {shape}")
        if not eps.is_floating_point():
            raise ValueError("eps must be a floating-point tensor (K3: float32)")
        if not torch.isfinite(eps).all():
            raise ValueError("eps must be finite")
        eps = eps.to(device=x0.device, dtype=x0.dtype)  # 非原地对齐

    return mean + std * eps


def sde_score_target(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    sde: str = "VP",
) -> torch.Tensor:
    """Compute the analytic denoising score-matching target (K.F3).

    Formula: s = -(x_t - mean) / std^2, where (mean, std) =
    sde_marginal(x0, t, sde) are the analytic moments of p_t(x_t | x0); this
    is d/dx_t log p_t(x_t | x0) in closed form.

    Paired with K.F2: for the same (x0, t, eps) with x_t = mean + std * eps,
    s = -eps / std exactly.

    Properties: [pure] (stateless, no RNG, inputs unmodified);
    numerically safe (std is clamped to >= EPS=1e-6 by sde_marginal, so the
    denominator std^2 >= 1e-12; x_t is validated for finiteness).

    Args:
        x_t: noised samples, shape [B, D].
        x0: source samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        sde: schedule identifier, "VP" (default) or "VE".

    Returns:
        s: score target, shape [B, D].

    Raises:
        ValueError: if ``x_t`` is not a finite floating-point tensor of shape
            [B, D]; all sde_marginal input errors propagate as well.
    """
    mean, std = sde_marginal(x0, t, sde)  # 复用 K.F1 校验与解析矩（std ≥ EPS）

    if not isinstance(x_t, torch.Tensor) or x_t.shape != x0.shape:
        shape = tuple(x_t.shape) if isinstance(x_t, torch.Tensor) else None
        raise ValueError(f"x_t must be a tensor of shape {tuple(x0.shape)} [B, D], got {shape}")
    if not x_t.is_floating_point():
        raise ValueError("x_t must be a floating-point tensor (K3: float32)")
    if not torch.isfinite(x_t).all():
        raise ValueError("x_t must be finite")
    x_t = x_t.to(device=x0.device, dtype=x0.dtype)  # 非原地对齐

    return -(x_t - mean) / (std * std)


def sde_coeffs(
    x: torch.Tensor, t: torch.Tensor, sde: str = "VP"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the drift and diffusion coefficients of the SDE (K.F4a).

    Single source of truth for f/g (spec rule): K.F4 reverse steps and the
    K.F7 weighting lambda(t) = g(t)^2 must call this function instead of
    re-deriving the formulas locally.

    VP (default, beta_min=0.1, beta_max=20):
        beta(t) = beta_min + t * (beta_max - beta_min)
        f(x, t) = -1/2 * beta(t) * x    (shape [B, D])
        g(t) = sqrt(beta(t))            (shape [B])
    VE (sigma_min=0.01, sigma_max=50):
        f(x, t) = 0                     (shape [B, D])
        g(t) = sigma_min * (sigma_max/sigma_min)^t * sqrt(2 * ln(sigma_max/sigma_min))
                                        (shape [B]; from d sigma^2/dt = 2 sigma^2 ln r)

    Properties: [pure]; f is per-sample per-dimension; g is a per-sample
    scalar (isotropic noise, broadcast to D dims by the caller).

    Args:
        x: state samples, shape [B, D].
        t: noise time in [0, 1], shape [B].
        sde: schedule identifier, "VP" (default) or "VE".

    Returns:
        tuple (f, g): drift of shape [B, D] and diffusion of shape [B].

    Raises:
        ValueError: if ``x``/``t`` violate the shape/dtype/finiteness
            contract, if ``sde`` is invalid, or if any entry of ``t`` is
            outside [0, 1].
    """
    x, t, sde = _validate_inputs(x, t, sde, name="x")
    t = t.to(device=x.device, dtype=x.dtype)  # 非原地对齐

    if sde == "VP":
        beta = _VP_BETA_MIN + t * (_VP_BETA_MAX - _VP_BETA_MIN)  # [B]
        f = -0.5 * beta.view(-1, 1) * x
        g = torch.sqrt(beta)
    else:  # VE
        log_ratio = math.log(_VE_SIGMA_MAX / _VE_SIGMA_MIN)
        ratio_t = torch.full_like(t, _VE_SIGMA_MAX / _VE_SIGMA_MIN).pow(t)
        f = torch.zeros_like(x)
        g = _VE_SIGMA_MIN * ratio_t * math.sqrt(2.0 * log_ratio)
    return f, g


def reverse_euler_maruyama(
    x: torch.Tensor,
    t: torch.Tensor,
    h: float,
    score: torch.Tensor,
    sde: str = "VP",
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Take one reverse Euler-Maruyama step (K.F4).

    Reverse-time update for x_{t-h} from x_t (Anderson 1982 reverse SDE):

        x' = x - [f(x,t) - g(t)^2 * score] * h + g(t) * sqrt(h) * z,
        z ~ N(0, I),  with f, g from sde_coeffs (K.F4a, single source of truth).

    Sign convention (spec-corrected): ``h > 0`` is the reverse step size; the
    caller advances t downwards (K.F6 steps t: 1 -> 0 equally spaced with
    h = 1/n_steps). For VP this reduces to x + h*(1/2*beta*x + beta*score):
    both drift terms point along the score (toward the data manifold) and
    counteract contraction.

    Properties: [deterministic] (given rng; z is drawn only from the explicit
    generator, K3 rule 3).

    Args:
        x: current state, shape [B, D].
        t: current noise time in [0, 1], shape [B].
        h: positive reverse step size.
        score: score estimate s_theta(x, t), shape [B, D].
        sde: schedule identifier, "VP" (default) or "VE".
        rng: torch.Generator used to draw z.

    Returns:
        x': next state x_{t-h}, shape [B, D].

    Raises:
        ValueError: if ``h`` is not a positive finite number; if ``rng`` is
            not a torch.Generator; if ``score`` violates the
            shape/dtype/finiteness contract; all sde_coeffs input errors
            propagate as well.
    """
    x, t, sde = _validate_inputs(x, t, sde, name="x")
    t = t.to(device=x.device, dtype=x.dtype)  # 非原地对齐

    if not isinstance(h, (int, float)) or isinstance(h, bool) or not math.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite step size, got {h!r}")
    h = float(h)

    if not isinstance(score, torch.Tensor) or score.shape != x.shape:
        shape = tuple(score.shape) if isinstance(score, torch.Tensor) else None
        raise ValueError(f"score must be a tensor of shape {tuple(x.shape)} [B, D], got {shape}")
    if not score.is_floating_point():
        raise ValueError("score must be a floating-point tensor (K3: float32)")
    if not torch.isfinite(score).all():
        raise ValueError("score must be finite")
    score = score.to(device=x.device, dtype=x.dtype)  # 非原地对齐

    if not isinstance(rng, torch.Generator):
        raise ValueError("rng must be a torch.Generator (K3 rule 3: no implicit randomness)")

    f, g = sde_coeffs(x, t, sde)  # K.F4a：f/g 单一事实来源
    b = x.shape[0]
    z = torch.randn(x.shape, generator=rng, dtype=x.dtype)
    z = z.to(device=x.device)

    drift = (f - (g * g).view(b, 1) * score) * h
    noise = (g.view(b, 1) * math.sqrt(h)) * z
    return x - drift + noise


def langevin_corrector(
    x: torch.Tensor,
    t: torch.Tensor,
    score: torch.Tensor,
    snr: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Take one Langevin corrector step (K.F5).

    Annealed Langevin dynamics with an SNR-tuned step size (Song et al. 2021
    corrector, NCSN update form):

        z ~ N(0, I)
        eps^2 = 2 * (snr * sigma_z / sigma_s)^2
        x' = x + (eps^2 / 2) * score + eps * z

    where sigma_z / sigma_s is the empirical norm ratio of the current noise
    draw z and the score (RMS over all elements, a single scalar per call).

    Note: ``t`` is validated for the [B] in [0, 1] contract but does not
    enter the update (the corrector is time-agnostic; the parameter is kept
    per the spec signature for interface consistency).

    Numerical safety guard: if the score RMS is <= EPS (score effectively
    zero), eps is set to 0, so x' = x exactly.

    Properties: [deterministic] (given rng; z is drawn only from the explicit
    generator, K3 rule 3).

    Args:
        x: current state, shape [B, D].
        t: current noise time in [0, 1], shape [B] (validated, unused).
        score: score estimate s_theta(x, t), shape [B, D].
        snr: signal-to-noise ratio (non-negative finite scalar) that scales
            the step size.
        rng: torch.Generator used to draw z.

    Returns:
        x': corrected state, shape [B, D].

    Raises:
        ValueError: if ``snr`` is negative or non-finite; if ``rng`` is not
            a torch.Generator; if ``x``/``t``/``score`` violate the
            shape/dtype/finiteness contract.
    """
    x, t = _validate_x_t(x, t, "x")

    if not isinstance(snr, (int, float)) or isinstance(snr, bool) or not math.isfinite(snr) or snr < 0:
        raise ValueError(f"snr must be a non-negative finite number, got {snr!r}")

    if not isinstance(score, torch.Tensor) or score.shape != x.shape:
        shape = tuple(score.shape) if isinstance(score, torch.Tensor) else None
        raise ValueError(f"score must be a tensor of shape {tuple(x.shape)} [B, D], got {shape}")
    if not score.is_floating_point():
        raise ValueError("score must be a floating-point tensor (K3: float32)")
    if not torch.isfinite(score).all():
        raise ValueError("score must be finite")
    score = score.to(device=x.device, dtype=x.dtype)  # 非原地对齐

    if not isinstance(rng, torch.Generator):
        raise ValueError("rng must be a torch.Generator (K3 rule 3: no implicit randomness)")

    z = torch.randn(x.shape, generator=rng, dtype=x.dtype)
    z = z.to(device=x.device)

    rms_z = torch.sqrt(torch.mean(z * z))        # σ_z：z 的经验范数（全局标量）
    rms_s = torch.sqrt(torch.mean(score * score))  # σ_s：score 的经验范数（全局标量）
    if bool(rms_s <= _EPS):
        eps2 = torch.zeros_like(rms_z)  # 数值安全：score≈0 → ε=0 → x'=x
    else:
        eps2 = 2.0 * (snr * (rms_z / rms_s)) ** 2  # ε² = 2·(snr·σ_z/σ_s)²
    eps = torch.sqrt(eps2)

    return x + (eps2 / 2.0) * score + eps * z
