# V0 函数级规格说明书（Architect Spec v0.1）——【已废止】

> ⚠️ 本文件已被 `docs/specs/` 目录取代（保真优先的全量规格）。其中仍有效的部分（验收协议、无梯度纪律、三条纪律）已并入 `docs/specs/00-kernel.md` §K3/K6。请以 `docs/specs/` 为唯一权威。

- 版本：v0.1（废止）
- 定位：architecture-vision.md 的 V0 最小闭环 + project-decomposition.md 的模块卡片，细化到**函数级**。每个函数给出：接口（签名/形状/语义）、算法（精确方程）、必要特性（可测试的性质）、验收测试（提交给我时须附）。
- 使用方式：按 §3 的实现顺序逐函数开发。每完成一个函数，提交（函数编号 + 代码 + 单测输出）给我验收；通过后进入下一个。除本文件外不需要其他设计资料。
- 编号体系：模块 M0–M8；函数 F01–F36。函数编号即实现顺序参考，依赖关系见 §3 DAG。
- 与卡片注册表的映射：M1=数据、M2=V0-01 种子展开、M3 元件库=V0-02/04、M4=V0-03 皮层块、M5=V0-05 学习规则、M6=自校准、M7/M8=调度与验收台（V0-00）。

---

## 0. 范围与技术裁决

| 裁决项 | 决定 | 理由 |
|---|---|---|
| 语言/框架 | Python 3.10+，PyTorch ≥2.0，NumPy | 复用成熟扩散栈（D2）；手动参数更新方便；单开发者可维护 |
| 去噪框架 | DDPM 离散步，x₀-参数化（网络直接预测 x₀），T=200，线性 β | 标准、稳定、误差 δ=x̂₀−x₀ 直接可用 |
| 生物路径梯度边界 | 全程 `torch.no_grad()` + `param.data` 手动更新；**禁止 `.backward()`** | 三因素规则的真实形态；与基线隔离，可审计 |
| 反向传播允许范围 | 仅 F30 基线训练（对照用） | D4-1 需要 backprop 基线 |
| 误差总线 | d = Pᵀδ，P∈R^{D×k} 冻结正交；M_l = B_l d，B_l∈R^{N_l×k} 冻结随机；oracle 模式 k=D, P=I | D3（低维投影）+ Lillicrap FA + e-prop；k 扫描=D4-6 |
| 基准数据 | 2D 高斯混合（8 分量），有解析分数（F05，可选诊断用） | 训练分钟级、可解析验证、种子统计易算 |
| 资格迹 | 逐层缓存 E←λE+mean_b(z⊗a)，λ=0.9，每 K=10 步结算 | 5.2 节时钟域：T0 递推/T1 结算；D4-4 延迟实验 |
| 稀疏性 | 硬阈值 + 除法归一化竞争，活跃率目标 0.15∈[0.1,0.2]（D4-5），阈值经校准函数设定 | L0 硬阈值 + Carandini-Heeger |
| 快慢双通路 | 快支 W_fast 全剂量；慢支 W_slow 初始 1%（剂量调整先行），慢状态 s 为层内 EMA 缓存 | L1 ★★"快慢双通路先行"；异时性原则 |
| V0 裁剪 | 偏置/增益/阈值固定不学习；无 STP/NMDA/Pr/秩调度/三锁代码（V1+）；无 L3–L9 | V0 验收只覆盖三因素闭环（D4） |
| 设备 | dtype=float32；CPU/GPU 均可；随机性全部经显式 rng 参数 | 确定性验收要求 |

## 1. 全局约定

### 1.1 记号与形状

- B=batch，D=数据维（V0 固定 2），N_l=第 l 隐层宽度，L=隐层层数（V0 默认 2），E=时间嵌入维（默认 32），T=去噪步数（默认 200），k=误差总线维（默认 8，可扫描），K=结算周期（默认 10）。
- 张量一律 **batch-first**；元素 dtype float32。
- `Tensor[a,b]` 表示形状 (a,b)。`z⊗a` 表示外积 batch 平均：`(1/B)Σ_b z_b a_bᵀ`。
- 参数命名：W=权重，b=偏置，θ=阈值，σ=归一化分母偏置，s=慢状态，E=资格迹，M=学习信号，d=总线信号。

### 1.2 函数性质标记（每个函数必须满足其标记）

- **[纯函数]**：无隐藏状态、不读全局 RNG、同输入必同输出、不改输入（out-of-place）。
- **[确定性]**：给定 (输入, rng) 结果可复现；随机性只从显式 rng 流出。
- **[无梯度]**：函数体内不产生 autograd 图；断言输入无 `grad_fn`；不得调用 `.backward()`。
- **[冻结]**：所涉张量 `requires_grad=False` 且在训练中永不被修改。
- **[数值安全]**：分母钳制下限、输出全有限（测试含 inf/nan 检查）。

### 1.3 时钟域约定

| 时钟 | 触发 | 载体 |
|---|---|---|
| T0 每推理步 | 每次 net 前向（bio 模式） | F23 资格迹递推、慢状态 s 递推 |
| T1 每 K 步 | 训练循环每 K 步 | F27 settle 结算 |
| T2 阶段级 | Phase 1 自校准 | F28 |
| T4 构建期 | 种子展开一次 | F06 |

### 1.4 三条纪律（违反即退回）

1. **生物路径零反向传播**：F22–F28 及 F29（bio 模式）内出现任何 `.backward()` / `loss.backward()` / 参数 `.grad` 依赖，一律退回。
2. **基线隔离**：反向传播代码只允许出现在 `scheduler.py` 的 F30 与 `eval.py` 中基线相关函数。
3. **无隐式随机**：除 F01/F02 外，任何函数不得调用 `torch.rand/randn/randint` 而不带 generator 参数（或通过显式 rng 实参）。

### 1.5 提交与验收协议（我们的工作契约）

- **提交单位**：单个函数（或自然成组的 2–3 个函数）。提交内容：
  1. 函数编号（如 F13）；
  2. 代码文件路径与函数定位（文件名:行号）；
  3. 单元测试文件路径 + **完整测试运行输出**；
  4. 特性自检表（按该函数"必要特性"逐条打勾或说明）。
- **验收方式**：我核对 ① 签名与接口契约一致；② 算法实现与本节方程逐行一致；③ 测试断言覆盖全部"必要特性"；④ 跑测试输出确认通过。返回"通过"或"退回+具体原因"。
- **模块级闸门**：一个模块内函数全部通过后做一次集成验收（见各模块"集成验收"条）。
- **V0 总闸门**：见 §5，全部 D4 指标 + 对照曲线通过才算 V0 完成。

## 2. 仓库结构与配置对象

```
project/
├── docs/
│   ├── architecture-vision.md
│   ├── neural-construction-iteration.md
│   ├── project-decomposition.md
│   └── v0-function-spec.md            # 本文件
└── src/bio_net/
    ├── __init__.py
    ├── config.py      # NetConfig/LearnConfig/TrainConfig/DataConfig/NetSpec
    ├── infra.py       # F01 F02
    ├── data.py        # F03 F04 F05
    ├── seed.py        # F06 F07
    ├── components.py  # F08–F17（元件库）
    ├── assemblies.py  # F18–F21（组合体）
    ├── learning.py    # F22–F27（学习规则）
    ├── self_calib.py  # F28
    ├── scheduler.py   # F29 F30
    └── eval.py        # F31–F36（验收台）
tests/                 # 每模块一个测试文件 + 集成测试
```

### 2.1 配置对象（config.py，dataclass，全默认值）

```python
@dataclass(frozen=True)
class DataConfig:
    n_comp: int = 8          # 混合分量数
    dim: int = 2             # D
    spread: float = 2.0      # 分量均值半径 U(-spread, spread)
    comp_scale: float = 0.15 # 分量标准差

@dataclass(frozen=True)
class NetConfig:            # 粗结构：全网络统一（通道化）
    in_dim: int = 2         # D
    hidden: tuple = (64, 64)    # 隐层宽度 N_l，L=len(hidden)
    emb_dim: int = 32       # E
    n_steps: int = 200      # T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    norm_bandwidth: float = 0.3   # 除法归一化竞争核带宽
    target_active: float = 0.15   # D4-5 活跃率目标
    slow_alpha: float = 0.1       # 慢通路 EMA
    w_slow_scale: float = 0.01    # 慢支初始剂量（快慢分离）
    init_scale: float = 0.05      # W 初始化标准差

@dataclass(frozen=True)
class LearnConfig:
    lr: float = 1e-3
    trace_decay: float = 0.9      # λ
    settle_every: int = 10        # K
    bus_dim: int = 8              # k
    bus_mode: str = 'broadcast'   # 'broadcast' | 'oracle'
    wd: float = 0.0               # 权重衰减（代谢正则代理，V0 默认关）

@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20_000
    batch: int = 128
    seed: int = 0
```

### 2.2 NetSpec（expand 的输出 / build 的输入）

```python
@dataclass
class NetSpec:
    cfg: NetConfig
    coarse: dict   # 通道化粗结构（与 seed 无关）：层表、激活形式、归一化形式、嵌入方案
    tensors: dict  # 细参数张量（seed 决定取值）：W_fast_l, W_slow_l, b_l, gain_l,
                   #   theta_l, sigma_l（l=1..L）, W_head, b_head
```

## 3. 函数总表与实现顺序

实现顺序 = 拓扑序（依赖先行）+ 验收台纯函数先行。F 编号即建议顺序。

| F# | 函数 | 模块/文件 | 依赖 | D4 覆盖 |
|---|---|---|---|---|
| F01 | set_global_seed | M0 infra | – | 全局 |
| F02 | make_rng | M0 infra | – | 全局 |
| F03 | make_mixture | M1 data | F02 | – |
| F04 | mixture_sample | M1 data | F03 | – |
| F05 | mixture_score | M1 data（可选） | F03 | 诊断 |
| F32 | compute_metrics（MMD 纯部分） | M8 eval | F04 | D4-1 口径 |
| F08 | diffusion_schedule | M3 components | – | – |
| F09 | time_embed | M3 components | – | – |
| F06 | expand | M2 seed | F02 | D4-2 |
| F07 | build | M2 seed | F06 | D4-2 |
| F10 | divisive_normalize | M3 components | F07(形状) | D4-5 |
| F11 | hard_threshold | M3 components | – | D4-5 |
| F12 | calibrate_thresholds | M3 components | F11 | D4-5 |
| F13 | forward_noise | M3 components | F08 | – |
| F14 | posterior_mean | M3 components | F08 | – |
| F15 | train_error | M3 components | – | – |
| F16 | implicit_error | M3 components | F14 | 诊断 |
| F17 | oja_step | M3 components | – | D4-3 |
| F18 | DualPathLayer | M4 assemblies | F10 F11 | – |
| F19 | reset_dual_path | M4 assemblies | F18 | – |
| F20 | make_cortical_block | M4 assemblies | F18 F09 | D4-2(零数据前向) |
| F21 | net_forward | M4 assemblies | F20 | – |
| F22 | attach_learning | M5 learning | F21 | – |
| F23 | update_trace | M5 learning | F22 | D4-4 |
| F24 | error_bus | M5 learning | – | D4-6 |
| F25 | feedback_project | M5 learning | F24 | D4-6 |
| F26 | three_factor_update | M5 learning | – | D4-1 |
| F27 | settle | M5 learning | F23–F26 | D4-1/4/6 |
| F28 | self_calibrate | M6 self_calib | F17 F21 | D4-3 |
| F29 | train_loop（bio） | M7 scheduler | F04 F13 F15 F21–F27 | D4-1/5 |
| F30 | train_loop_baseline | M7 scheduler | F04 F13 F15 F21 | D4-1 对照 |
| F31 | sample_reverse | M8 eval | F14 F21 | D4-1 采样 |
| F33 | bus_scan | M8 eval | F29 F30 F32 | D4-6 |
| F34 | lag_ablation | M8 eval | F29 F32 | D4-4 |
| F35 | seed_sweep | M8 eval | F06 F07 F21 F32 | D4-2 |
| F36 | selfcalib_ablation | M8 eval | F28 F29 F32 | D4-3 |

依赖 DAG（同层可并行）：F01/F02 → F03/F04 → F32(纯) ‖ F08/F09 → F06/F07 → F10/F11/F12/F13/F14/F15/F16/F17 → F18/F19 → F20/F21 → F22 → F23/F24/F25/F26 → F27 → F28/F29/F30 → F31 → F33/F34/F35/F36。

---

## 4. 分模块函数规格

### M0 公共基础设施（infra.py）

**F01 `set_global_seed(seed: int) -> None`**
- 接口：`seed: int`（任意整数）；无返回值。
- 语义：设置 torch/numpy/python 全局随机种子；仅供训练入口调用一次。
- 算法：依次调用 `torch.manual_seed(seed)`、`np.random.seed(seed % 2**32)`、`random.seed(seed)`。
- 必要特性：[确定性] 同 seed 两次调用后采样序列一致。
- 验收测试：`tests/test_infra.py::test_set_global_seed`——两次 set 后 `torch.randn(10)` 相等。

**F02 `make_rng(seed: int | None = None) -> torch.Generator`**
- 接口：`seed: int | None`（None 时取全局当前状态派生）；返回 `torch.Generator`（CPU）。
- 算法：`gen = torch.Generator(); if seed is not None: gen.manual_seed(seed)`。
- 必要特性：[确定性] 同 seed 返回同行为生成器；[纯函数] 无副作用。
- 验收测试：两个 `make_rng(7)` 生成序列相等。

### M1 数据（data.py）

**F03 `make_mixture(rng: torch.Generator, cfg: DataConfig) -> Mixture`**
- 接口：返回 `Mixture` 具名对象，字段：`means: Tensor[K,D]`、`scale: float`、`weights: Tensor[K]`（等权）、`cfg`。
- 算法：`means ~ U(-spread, spread)` 自 rng 采样；`scale = comp_scale`。
- 必要特性：[确定性]；K=cfg.n_comp；均值落在 `[-spread, spread]^D`。
- 验收测试：同 rng 两次构造 → 字段逐元素相等；形状断言。

**F04 `mixture_sample(m: Mixture, rng: torch.Generator, n: int) -> Tensor[n,D]`**
- 算法：按权重选分量（等权=均匀），`x = mean_c + scale·ε`，ε~N(0,I) 自 rng。
- 必要特性：[确定性][纯函数]；输出有限。
- 验收测试：n 次采样经验均值 ≈ 各分量均值均值（容差 3σ/√n）；同 rng 结果复现。

**F05 `mixture_score(m: Mixture, x: Tensor[*,D]) -> Tensor[*,D]`（可选，诊断用）**
- 算法：GMM 解析分数 ∇log p(x) = −Σ_c γ_c(x)(x−mean_c)/scale²，γ_c = 后验归属权重（softmax of log N(x; mean_c, scale²I)）。
- 必要特性：[纯函数]；数值安全（log-sum-exp）。
- 验收测试：有限差分核对（h=1e-4，误差 <1e-2）。

### M2 种子展开（seed.py）

**F06 `expand(seed: int, cfg: NetConfig) -> NetSpec`**
- 接口：`seed: int`、`cfg`；返回 `NetSpec`。
- 算法（退化路径：固定手工图 + 通道化采样）：
  1. 粗结构 `coarse`（**与 seed 无关，全网络统一**）：层表 [(D,E+? )→N_1→…→N_L→D]；激活形式 `threshold`；归一化形式 `divisive`；嵌入方案 `sinusoidal`；快慢通路结构（每隐层 fast+slow 两支）；
  2. 细参数（seed 决定，个体化）：`W_fast_l ~ N(0, init_scale²) [N_l, N_{l-1}]`（l=1 时输入维 D+E）；`W_slow_l = w_slow_scale · N(0, init_scale²)`；`b_l ~ U(-0.1, 0.1)`；`gain_l = 1 + U(-0.05, 0.05)`；`theta_l` 初始 0（由 F12 校准覆盖）；`sigma_l ~ U(0.1, 0.5)`；`W_head ~ N(0, init_scale²) [D, N_L]`、`b_head=0`。
  3. 全部随机量出自 `make_rng(seed)`，采样顺序固定（先在规格中列出的先采样）。
- 必要特性：[确定性] 同 (seed,cfg) → bitwise 相同 NetSpec；不同 seed → `coarse` 完全相同、`tensors` 不同；无全局状态。
- 验收测试：`test_expand_determinism`、`test_coarse_seed_invariant`、`test_spec_shapes`（全部张量形状断言）。

**F07 `build(spec: NetSpec) -> nn.Module`**
- 接口：输入 NetSpec；返回装配好的网络（结构见 M4）。
- 算法：把 spec.tensors 装配为 DualPathLayer×L + 头部 Linear（权重/偏置按 spec 复制，`requires_grad=True`——供基线用；bio 路径手动更新）。
- 必要特性：纯装配（不采样随机数）；`build(expand(seed,cfg))` 两次 → 结构一致、参数值一致（deepcopy 比较）。
- 验收测试：`test_build_rebuild`、`test_param_values_match_spec`。

### M3 元件库（components.py）

**F08 `diffusion_schedule(n_steps: int, beta_start: float, beta_end: float) -> DDPM`**
- 接口：返回具名对象 `DDPM(betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, posterior_coef1, posterior_coef2, posterior_var)`，各为 `Tensor[n_steps]`（1-索引语义：下标 t 对应步 t，t=1..T）。
- 算法：`beta_t = beta_start + (t−1)/(T−1)·(beta_end−beta_start)`（线性）；`alpha=1−beta`；`alpha_bar_t = cumprod(alpha)`；`posterior_coef1_t = sqrt(alpha_bar_{t−1})·beta_t/(1−alpha_bar_t)`；`posterior_coef2_t = sqrt(alpha_t)·(1−alpha_bar_{t−1})/(1−alpha_bar_t)`；`posterior_var_t = (1−alpha_bar_{t−1})/(1−alpha_bar_t)·beta_t`（t=1 时 alpha_bar_0=1）。
- 必要特性：[纯函数]；beta 严格递增且 ∈[1e-4, 0.02]；alpha_bar 递减；数值安全。
- 验收测试：手算 t=1、t=T 与闭式比对；单调性断言。

**F09 `time_embed(t: Tensor[B], n_steps: int, dim: int) -> Tensor[B,E]`**
- 算法：正弦位置嵌入（冻结，无参数）：`e[2i]=sin(t·ω_i)`，`e[2i+1]=cos(t·ω_i)`，`ω_i = 1/10000^(2i/E)`，t 为 [0,T] 连续值，E 为偶数。
- 必要特性：[纯函数][冻结]；E 为奇数时抛错。
- 验收测试：形状；t=0 与 t=T 输出不同且有限；与手写公式比对。

**F10 `divisive_normalize(x: Tensor[B,N], sigma: Tensor[N], kernel: Tensor[N,N] | None = None, bandwidth: float = 0.3) -> Tensor[B,N]`**
- 算法：`y_i = x_i / (σ_i + Σ_j K_ij·relu(x_j))`；K 为固定竞争核：`K_ij ∝ exp(−d_ij²/(2b²))` 行归一（`d_ij = |i−j|/(N−1)`，b=bandwidth）；kernel=None 时内部按带宽构造（不参与梯度）。
- 必要特性：[数值安全] 分母 ≥ max(σ_min, 1e-6)；缩放不变性：f(a·x)=f(x)（a>0）；可微（供基线反向传播）；输入为 None 时视为全连接核（由调用方决定）。
- 验收测试：缩放不变性（a=2 时输出相等）；分母下界；有限性；梯度存在（oracle 用）。

**F11 `hard_threshold(y: Tensor[B,N], theta: Tensor[N], straight_through: bool = False) -> Tensor[B,N]`**
- 算法：`z_i = y_i·1{y_i > θ_i}`（单侧硬阈值）。
- 必要特性：稀疏性可调（由 θ 决定活跃率）；`straight_through=True` 时梯度按 1{y>θ} 直通（仅基线用）；[数值安全]。
- 验收测试：符号正确性（手造样例逐元素比对）；STE 梯度等于指示函数。

**F12 `calibrate_thresholds(net, rng, target_active: float, n_samples: int = 1024) -> None`**
- 接口：对 net 各隐层，用零外部数据（纯噪声输入）前向 n_samples 次，收集每单元 y 分布；设置 `theta_i = y 分布的第 (1−target_active) 分位`。
- 必要特性：[确定性]；校准后活跃率 ∈ [target_active−0.05, target_active+0.05]；只写 theta 缓冲。
- 验收测试：校准后实测活跃率落在容差带（D4-5 口径）。

**F13 `forward_noise(x0: Tensor[B,D], t: Tensor[B,1]|Tensor[B], ddpm: DDPM, eps: Tensor[B,D] | None = None, rng: torch.Generator | None = None) -> Tensor[B,D]`**
- 算法：q(x_t|x0) 重参数化：`x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε`；eps=None 时自 rng 采样。
- 必要特性：[纯函数]（eps 给定时）；eps=None 时 [确定性]（依赖 rng）；t 越界（<1 或 >T）抛错。
- 验收测试：E[x_t]=√ᾱ_t·x0、Var=1−ᾱ_t（各 10⁴ 样本统计，容差 5%）；给定 eps 逐元素比对。

**F14 `posterior_mean(x_t: Tensor[B,D], t: Tensor[B], x0_hat: Tensor[B,D], ddpm: DDPM) -> Tensor[B,D]`**
- 算法：`μ_θ = coef1_t·x0_hat + coef2_t·x_t`（coef 取自 DDPM 对象，见 F08）。
- 必要特性：[纯函数]。
- 验收测试：t=T 时 μ_θ≈coef1_T·x̂₀+coef2_T·x_T 逐元素比对；t=1 时 μ_θ 与 x̂₀ 偏差 = (1−coef2_1)·(x̂₀−x_t)。

**F15 `train_error(x0_hat: Tensor[B,D], x0: Tensor[B,D]) -> Tensor[B,D]`**
- 算法：`δ = x0_hat − x0`。
- 说明（D2 等价性）：后验均值误差 μ_θ−μ̃ = coef1_t·(x̂₀−x₀)，coef1_t>0，故 δ 与 D2 定义的"x̂₀ 与后验均值之差"方向一致，仅差正标量——V0 训练主路径用 δ（监督信号），F16 提供 D2 原口径。
- 必要特性：[纯函数]。
- 验收测试：方向等价性（μ_θ−μ̃ 与 δ 成比例，对随机输入验证）。

**F16 `implicit_error(x_t: Tensor[B,D], t: Tensor[B], x0_hat: Tensor[B,D], ddpm: DDPM) -> Tensor[B,D]`**
- 算法：`δ_impl = x0_hat − posterior_mean(x_t, t, x0_hat, ddpm)`（D2 原文口径；推断期惊异度/无监督路径用，V0 主训练不用）。
- 必要特性：[纯函数]。
- 验收测试：闭式一致性；t→1 时 δ_impl→(1−coef2_1)(x̂₀−x_t)。

**F17 `oja_step(W: Tensor[N_out,N_in], x: Tensor[B,N_in], y: Tensor[B,N_out], eta: float) -> Tensor[N_out,N_in]`**
- 接口：返回新 W（out-of-place）。
- 算法（Oja 规则）：`ΔW = η·(yᵀx/B − mean_b(y²)[:,None]·W)`；随后行归一：`W_i ← W_i / max(‖W_i‖₂, 1e-6)`。
- 必要特性：[纯函数][无梯度]；行范数上界 1（归一后）；不修改输入。
- 验收测试：迭代后行范数→1（收敛性冒烟）；纯函数性（两次调用同输出、输入不变）。

### M4 组合体（assemblies.py）

**F18 `DualPathLayer`（类规格）**
- 接口：
  ```python
  class DualPathLayer(nn.Module):
      def __init__(self, in_dim, out_dim, cfg: NetConfig, spec_tensors: dict, key: int)
      def forward(self, x: Tensor[B,N_in], mode: str, trace_sink=None) -> Tensor[B,N_out]
      def reset(self) -> None
  ```
- 内部：`W_fast, W_slow`（requires_grad=True）、`b, gain, theta, sigma`（固定）、缓冲 `s: Tensor[N_in]`（慢积分器，初始 0）、竞争核 K（构造自 bandwidth）。
- 算法：`h_fast = x@W_fastᵀ + b`；慢支：`s ← (1−α)s + α·mean_b(x)`（mode∈{'bio','baseline'} 时更新，`eval` 模式冻结；更新在 no_grad 下进行，s 作为常量输入）；`h_slow = s@W_slowᵀ`；`h = (h_fast + h_slow)·gain`；`y = divisive_normalize(h, sigma, K)`；`z = hard_threshold(y, theta, straight_through=(mode=='baseline'))`；若 trace_sink 非空则缓存 (a=x, z) 供 F23。
- 必要特性：mode='eval' 时 s 不变（确定性推断）；[数值安全]；slow 支初始剂量 = w_slow_scale（快慢分离）。
- 验收测试：eval 模式两次前向（中间 reset 不调用）同输入同输出；s 更新只发生在 bio/baseline；输出形状与稀疏性。

**F19 `reset_dual_path(net) -> None`**
- 算法：遍历所有 DualPathLayer 调用 `reset()`（s←0）。
- 必要特性：调用后 eval 模式前向与"新建同参数网络"前向逐位一致。
- 验收测试：重置前后输出一致性对比。

**F20 `make_cortical_block(spec: NetSpec) -> nn.Module`（即 build 的装配主体）**
- 结构（折叠式，D2）：`emb = time_embed(t, T, E)` → `x_in = concat(x_t, emb) [B, D+E]` → L×DualPathLayer（l=1 输入维 D+E，其余 N_{l−1}）→ 头部 `x0_hat = z_L @ W_headᵀ + b_head`。
- 必要特性：与 spec 一一对应（F07 契约）；零外部数据可前向（D4-2 后半）。
- 验收测试：`build(expand(s,cfg))(randn(B,D), t, 'eval')` 输出有限且形状 [B,D]；跨 10 种子全部可前向。

**F21 `net_forward(net, x_t: Tensor[B,D], t: Tensor[B], mode: str, ls=None) -> Tensor[B,D]`**
- 接口：`mode ∈ {'bio','baseline','eval'}`；`ls` 为 LearningState（F22 产物），bio 模式下逐层缓存 (a,z) 写入 ls.cache。
- 算法：见 F20 前向链；bio 模式调用方必须包在 `torch.no_grad()` 内（本函数不自行包裹——由 F29 负责，保证"谁训练谁声明"）。
- 必要特性：三种模式行为差异仅限"慢状态更新 / 直通梯度 / 缓存写入"，输出数值在 eval 与 bio 下一致（同参数同输入）。
- 验收测试：三模式输出一致性；ls 缓存键完整（每层 a/z）。

### M5 学习规则（learning.py）—— V0 灵魂，★★★

**F22 `attach_learning(net, cfg: LearnConfig, rng: torch.Generator) -> LearningState`**
- 接口：返回 `LearningState`，字段：`P: Tensor[D,k]`（冻结正交）、`B_l: dict[l, Tensor[N_l,k]]`（冻结随机）、`E_l: dict[l, Tensor[N_l,N_{l-1}]]`（资格迹，初 0）、`E_head: Tensor[D,N_L]`、`cache: dict`（前向缓存）、`cfg`。
- 算法：`P = QR(N(0,1)[D,k]).Q`（列正交，PᵀP=I_k）；`bus_mode='oracle'` 时 k=D、P=I；`B_l ~ N(0, 1/k)`；所有 P/B 置 `requires_grad=False`。
- 必要特性：[确定性][冻结] P/B 永不被修改；k≤D 时 PᵀP=I_k（正交性）。
- 验收测试：正交性断言；requires_grad=False；同 rng 复现。

**F23 `update_trace(ls: LearningState, net, trace_decay: float) -> None`**
- 算法（T0 递推，[无梯度]）：对每隐层 l：`E_l ← λ·E_l + mean_b(z_l ⊗ a_l)`（a=层输入、z=阈值后输出，取自 ls.cache）；头部：`E_head ← λ·E_head + mean_b(δ ⊗ z_L)`（δ 由 F27 传入缓存，见 F27）。
- 必要特性：[无梯度]；λ=1 时 E 单调不减（逐元素）；缓存缺失时抛错（先 forward 后 trace）。
- 验收测试：单步手算比对（构造 2×2 样例，E' = λE + 外积精确相等）；no_grad 断言。

**F24 `error_bus(ls: LearningState, delta: Tensor[B,D]) -> Tensor[B,k]`**
- 算法（T1 输入，[无梯度]）：`d = delta @ P`（即 d_b = Pᵀδ_b）；oracle 模式 P=I → d=δ。
- 必要特性：[纯函数]；‖d‖≤‖δ‖（正交投影，k≤D 时）；k=D 时（oracle）d=δ。
- 验收测试：逐元素比对 Pᵀδ；范数不等式；oracle 恒等。

**F25 `feedback_project(ls: LearningState, d: Tensor[B,k]) -> dict[str, Tensor]`**
- 算法（[无梯度]）：`M_l = d @ B_lᵀ`（[B,N_l]，逐单元标量信号）；`M_head = d @ Pᵀ`（[B,D]）。返回 dict：{'l1':…, 'l2':…, 'head':…}。
- 必要特性：[纯函数]；B/P 冻结不参与；M 值有限。
- 验收测试：形状契约；B 不被修改（前后 deepcopy 比对）。

**F26 `three_factor_update(ls, net, M: dict, lr: float, wd: float) -> None`**
- 算法（T1 结算，[无梯度]，仅 `param.data` 原地更新）：`ΔW_l = lr·(mean_b(M_l)[:,None]·E_l) − lr·wd·W_l`；`W_l.data += ΔW_l`；头部同式（用 E_head、M_head）。
- 必要特性：[无梯度]（参数 `.grad` 全程为 None）；纯三因素：ΔW 只依赖 (M, E, W)，不依赖任何逐权重梯度；偏置/增益/阈值不被修改。
- 验收测试：回放核对——记录结算前 (E, M, W)，外部按公式重算 ΔW，与结算后 W−W_before 逐元素相等；偏置等不变。

**F27 `settle(ls, net, delta: Tensor[B,D], cfg: LearnConfig) -> None`**
- 接口：每 K 步调用一次（由 F29 触发）；`delta` 为当前步 F15 输出。
- 算法：① 将 `mean_b(δ)` 写入 ls.cache（供 F23 头部资格迹）；② `d = error_bus(ls, delta)`；③ `M = feedback_project(ls, d)`；④ `three_factor_update(ls, net, M, cfg.lr, cfg.wd)`。
- 必要特性：整个函数无 `.backward()`；可重复调用（幂等性不要求，但每调用一次恰好结算一次）。
- 验收测试：与 F23–F26 单元测试联合回放（集成闸门 M5）：构造已知前向缓存，手动执行 23→24→25→26 与调用 settle 结果逐位一致。

### M6 自校准（self_calib.py）

**F28 `self_calibrate(net, ls, rng, cfg: TrainConfig, n_steps: int = 2000, eta: float = 1e-2) -> None`**
- 算法（Phase 1，零外部数据，[无梯度]）：每步：`x ~ N(0,I) [B,D]`（自产生活动，视网膜波类比）→ 前向（mode='bio'，不结算）→ 对每隐层快支执行 Oja：`W_fast = oja_step(W_fast, a_l, z_l, eta)`（慢支与头部不动）。只改 `W_fast.data`。
- 必要特性：[确定性][无梯度]；输入零外部数据（函数不接受数据集参数）；不改结构（层数/宽度不变）；行范数归一。
- 验收测试：训练前后结构不变；W_fast 行范数≈1；no_grad 断言；同 rng 两次结果一致。

### M7 调度器（scheduler.py）

**F29 `train_loop(net, ls, m: Mixture, cfg: TrainConfig, lcfg: LearnConfig, ncfg: NetConfig, rng) -> TrainReport`**
- 算法（bio 模式，T0/T1 时钟）：
  ```
  ddpm = diffusion_schedule(...)
  calibrate_thresholds(net, rng, ncfg.target_active)     # 一次性（T4）
  for step in 1..cfg.steps:
      x0 = mixture_sample(m, rng, batch); t ~ U{1..T}; x_t = forward_noise(x0, t, ddpm, rng=rng)
      with torch.no_grad(): x0_hat = net_forward(net, x_t, t, 'bio', ls)
      delta = train_error(x0_hat, x0)                     # 无梯度
      update_trace(ls, net, lcfg.trace_decay)             # T0
      if step % K == 0: settle(ls, net, delta, lcfg)      # T1
      记录: step, loss=mean(delta²) EMA, 活跃率均值
  ```
  返回 `TrainReport(loss_curve, active_curve, final_ema_loss, cfg)`。
- 必要特性：全程仅 F29 有权调用 no_grad 包裹前向；参数 `.grad` 全程 None；[确定性]（同 seed 全曲线复现）；每 K 步恰好结算一次（报告含结算计数）。
- 验收测试：冒烟——500 步 loss 下降；确定性（同 seed 两次跑 loss 曲线逐点相等）；结算计数 = steps//K。

**F30 `train_loop_baseline(net, m, cfg, ncfg, rng) -> TrainReport`**
- 算法：与 F29 同采样管线；`x0_hat = net_forward(net, x_t, t, 'baseline')`；`loss = mean(train_error²)`；`loss.backward()`；`Adam(lr=1e-3)` 更新。**本函数是全项目唯一允许反向传播的训练入口。**
- 必要特性：与 F29 同种子同数据管线（公平对照）；不使用 ls/资格迹。
- 验收测试：冒烟 loss 下降；与 F29 采样序列一致（前 100 步 x_t 相同）。

### M8 验收台（eval.py）

**F31 `sample_reverse(net, ddpm, rng, n: int, ncfg) -> Tensor[n,D]`**
- 算法：`x_T ~ N(0,I)`；for t=T..1：`x0_hat = net_forward(net, x_t, t, 'eval')`；`μ = posterior_mean(x_t, t, x0_hat, ddpm)`；`x_{t−1} = μ + (t>1)·√β̃_t·z`（z~N(0,I) 自 rng）。
- 必要特性：[确定性]（给定 rng）；eval 模式（慢状态冻结）；输出有限。
- 验收测试：确定性复现；样本落于合理范围（|x|≤5，混合数据尺度下）。

**F32 `compute_metrics(samples: Tensor[n,D], m: Mixture, ref: Tensor[n,D] | None = None, ...) -> dict`**
- 算法：① MMD²（RBF 核，带宽=两样本集合并 median 启发式，无偏 U-统计）——`samples` vs 真实混合的独立样本集；② 活跃率/稀疏统计（传入 net 时）；③ 种子粗统计（F35 调用）：`{活跃率均值, 隐层激活归一化协方差前 5 特征值占比, 输出边缘方差}` 及其跨种子 CV=std/mean。
- 必要特性：[纯函数]（MMD 部分）；MMD≥0；两相同样本集 MMD≈0。
- 验收测试：自比较 MMD≈0；不同分布 MMD>0（混合 vs 单高斯）。

**F33 `bus_scan(net_spec_fn, m, base_cfg, k_values=(1,2,4,8,16,D)) -> dict[k, final_loss]`**
- 算法：对每个 k：attach_learning(k) → F29 训练（固定精简步数，默认 5000）→ 记录 final_ema_loss；oracle（k=D,P=I）为参照。输出扫描曲线；**验收判据：找到拐点 k\*，k≥k\* 时 loss ≤ 1.1×oracle**（信息瓶颈定位，D4-6）。
- 必要特性：[确定性]；对照 oracle 在同一配置下运行。
- 验收测试：曲线单调趋势（loss 随 k 非增，容差噪声）；oracle 项存在。

**F34 `lag_ablation(net_spec_fn, m, cfg, K_values=(1,10,50,100)) -> dict[K, final_loss]`**
- 算法：同 F29，改 settle_every=K。**验收判据：loss(K=10) ≤ 1.1×loss(K=1)**（资格迹桥接有效，D4-4）。
- 验收测试：K 与结算计数一致；结果字典键完整。

**F35 `seed_sweep(seeds=(0..9), cfg, rng) -> dict`**
- 算法：每 seed：expand→build→（随机输入）前向→统计粗结构指标（F32 第③组）。**验收判据：≥10 种子，各指标 CV ≤10%**（通道化验证，D4-2）。
- 验收测试：粗结构（层表/激活/归一化形式）逐种子相等；统计 CV 断言。

**F36 `selfcalib_ablation(net_spec_fn, m, cfg, seed) -> dict`**
- 算法：同 seed 下两条训练曲线：无自校准 vs F28 后训练。**验收判据：收敛步数（loss EMA 首次低于阈值）减少 ≥20%，或最终 loss 有可测下降**（D4-3）。
- 验收测试：两组曲线长度一致；gain 计算正确。

---

## 5. V0 总闸门（全局验收）

全部通过才宣布 V0 完成。每项附对照，口径与 D4 一致：

| # | 指标 | 判据 | 数据来源 |
|---|---|---|---|
| 1 | 训练可行性 | 完整步数下 F29 收敛；final loss 与 F30 基线相对差距 ≤30% | F29/F30 |
| 2 | 种子稳定性 | ≥10 种子粗统计 CV≤10%；零外部数据可前向 | F35、F06/F07 测试 |
| 3 | 自校准增益 | 收敛步数 ↓≥20% 或最终指标可测提升 | F36 |
| 4 | 资格迹延迟 | loss(K=10) ≤ 1.1×loss(K=1) | F34 |
| 5 | 稀疏性 | 全程活跃率 ∈[0.10, 0.20] 且不损指标 1 | F29 active_curve |
| 6 | 误差总线信息量 | k 扫描拐点存在：k≥k\* 时 ≤1.1×oracle | F33 |

交付物：`docs/v0-report.md`——六项指标表 + 各曲线（loss/活跃率/bus 扫描/lag）+ 对照说明。

## 6. V1–V8 模块分割预告（函数级规格待各版启动时产出）

| 版本 | 新增模块（预留位置） | 启动时按本方法论先写其函数级规格 |
|---|---|---|
| V1 通信细化 | components 增 STP(u,x)/NMDA 门控/Pr 随机门控/电突触 skip | 验收：STP 降高频伪影、Pr 提升缺失输入鲁棒性 |
| V2 介观束 | learning 增秩调度(T3)；locks/ 三锁协议代码化 | 验收：长程一致性、锁定稳定性、逐个解锁 |
| V3 容积场 | 新模块 modulation（T2 EMA + FiLM 注入） | 验收：多样性、空间可控性 |
| V4 结构演化 | 新模块 structure（静默突触池/健康度/软凋亡/回收池，T3/T4） | 验收：参数下降不损性能、新任务适配更快 |
| V5 价值层 | 新模块 value（TD/RPE 结算、难度轴、Go/NoGo、评估器可重指向） | 验收：稀疏奖励可训练、伪影步抑制、评估器可替换 |
| V6 时间轴 | 新模块 rhythm（相位编码/CFC/通信窗口/分频，T0 扩展） | 验收：多任务并行、预测/误差分频增益 |
| V7 行动闭环 | 新模块 cerebellum（颗粒扩展层/浦肯野读出 + 传出拷贝） | 验收：自生成成分被正确条件化 |
| V8 睡眠巩固 | 新模块 sleep（三嵌套回放 + SHY，T4） | 验收：灾难性遗忘缓解、稳定性提升 |

规则重申：每版启动前先产出其函数级规格（接口/算法/特性/验收），再动手写代码——与 V0 相同的契约。
