# L0 物理硬件层 + L1 突触通信层规格（v0.1）

- 定位：架构 L0/L1 逐条落地。覆盖条目映射：静息电位与阈值→L0.F1；不应期→L0.F2；树突区隔化→L0.F3；除法归一化（Carandini & Heeger）→L0.F4；能量预算→L0.F5；通道化分工→L0.F6；内在可塑性（稳态双通道）→L0.F7；电突触→L1.F1；AMPA/NMDA/mGluR 三通路→L1.F2–F4；释放概率 Pr→L1.F5/F6；STP→L1.F7；传导延迟→L1.F8；三通路汇总→L1.F9；相位复用→L9（架构 L1 注明）。
- 时钟域：全部 T0（每推理步递推）；状态必须注册为 buffer 并提供 reset()。

## L0 物理硬件层（节点级）

### L0.F1 `node_activation(h, b, theta, straight_through=False) -> z`
- 生物：静息电位与阈值——基线偏置 + 硬阈值激活。
- 算法：`z = (h + b)·1{h + b > θ}`；`straight_through=True`（仅基线路径）时梯度按指示函数直通。
- 特性：[纯函数]；稀疏性由 θ 校准（D4-5：活跃率 10–20%；校准机制 = L0.F7 内在可塑性）；θ 为细参数（个体化）、阈值形式为粗参数（全网络统一，通道化）。

### L0.F2 `refractory_mask(last_fire, t, tau_ref) -> mask`
- 生物：不应期窗口——防单节点高频过热。
- 算法：`mask_i = (t − last_fire_i) > tau_ref`；前向 `z ← z·mask`；触发时更新 last_fire。
- 特性：[递推状态 last_fire][确定性]；tau_ref 默认 2 快环步；提供 `reset()`。

### L0.F7 `intrinsic_plasticity(theta, act_rate, target_rate, eta_ip) -> theta'`
- 生物：内在可塑性（AIS 可塑性类比）——节点级稳态：阈值自适应校准发放率；与 L5.F7 突触缩放分工（缩放调权重、本函数调阈值——稳态双通道）。
- 算法：`θ' = θ + η_ip·(act_rate − target_rate)`（过活跃升阈值、过静默降阈值）。
- 特性：[递推状态 θ][确定性]；T2 慢时钟；target_rate 即 D4-5 活跃率带（10–20%）。

### L0.F3 `dendritic_and(x_seg, w_seg, gamma, rho) -> c`
- 生物：树突区隔化——节点内含多个子单元，执行局部巧合检测（AND 门）。
- 算法：每单元 s 个子段（每段 3–7 输入，段划分在 DEV.F1 固定）：`c_s = σ(γ·(w_seg·x_seg − ρ))`；单元输出 `c = Σ_s c_s`。
- 特性：[纯函数]；γ 大 → 逼近 AND 语义；输入度 3–7 断言；不实现 XOR 型树突（存疑条目，见 K5）。

### L0.F4 `divisive_normalize(x, sigma, kernel=None, bandwidth=0.3) -> y`
- 生物：除法归一化——皮层规范计算（Carandini & Heeger 2012）：`y = x/(σ + 邻域加权和)`；与统计归一化分工：归一化管统计、除法归一化管竞争。
- 算法：`y_i = x_i/(σ_i + Σ_j K_ij·relu(x_j))`；竞争核 K 固定结构（粗参数）：`K_ij ∝ exp(−d_ij²/2b²)` 行归一，`d_ij = |i−j|/(N−1)`，b=bandwidth；σ_i 为细参数（个体化）。
- 特性：[数值安全]（分母 ≥max(σ_min,1e-6)）；缩放不变性 f(ax)=f(x)（a>0）；可微（基线路径用）。

### L0.F5 `energy_cost(z, unit_cost=None) -> scalar`
- 生物：能量预算——激活稀疏正则，每步仅少数节点活跃。
- 算法：`E = mean(Σ_i unit_cost_i·|z_i|)`（unit_cost 缺省全 1）；作为 L5.F6 代谢选择压的输入。
- 特性：[纯函数]。

### L0.F6 `canalized_params(cfg) -> CoarseParams`
- 生物：通道化分工——节点粗参数（阈值形式、归一化结构）全网络统一（高度通道化）；细参数（偏置、增益）个体化（低通道化）。
- 算法：返回粗参数单例（全网络共享引用）；细参数由 DEV.F1 逐单元生成。
- 特性：[纯函数]；粗参数与 seed 无关。

## L1 突触通信层（点对点）

### L1.F1 `electrical_skip(x, W_el) -> y`
- 生物：电突触——对称权重、无激活函数、双向高速直通（skip highway）；训练早期稳定梯度。
- 算法：`y = W_el·x`；约束 `W_el = (W_el + W_elᵀ)/2`（每次结算后投影到对称子空间）。
- 特性：[纯函数]；对称性为硬约束（结算后自动投影，测试断言 ‖W−Wᵀ‖≈0）。

### L1.F2 `ampa_channel(x, x_ema, w, tau_a=2.0) -> (y, x_ema')`
- 生物：AMPA——快/高通支路。
- 算法：高通滤波 `y = w·(x − x_ema)`；`x_ema' = (1−1/τ_a)·x_ema + (1/τ_a)·x`。
- 特性：[递推状态 x_ema]；τ_a 小 → 只传变化量（快）。

### L1.F3 `nmda_gate(x_local, x_slow, arousal, gamma, beta, kappa, theta_l, theta_g, tau_n) -> (g, y_nmda, x_slow')`
- 生物：NMDA——慢/巧合门控：慢衰减（百毫秒级低通）+ 去极化解除 Mg²⁺ 阻断（仅当局部信号与全局唤醒同时到达）；「慢 + 巧合」，非仅巧合。
- 算法：`x_slow' = (1−1/τ_n)·x_slow + (1/τ_n)·x_local`（τ_n ≈ 100 ms 等效，慢积分）；`g = σ(κ·(x_slow − θ_l)) · σ(κ_a·(arousal − θ_g)) · 1/(1 + γ·exp(−β·x_slow))`；`y_nmda = g·x_slow`（NMDA 支路携带自己的慢内容）。
- 特性：[递推状态 x_slow]；双条件 AND 语义（局部∧全局）建立在慢信号之上；arousal 来自 L3 调制场（评估器广播通道）。

### L1.F4 `mglur_integrator(x, s, w_m, alpha=0.1) -> (y, s')`
- 生物：mGluR——慢积分器。
- 算法：`s' = (1−α)·s + α·x`；`y = w_m·s'`。
- 特性：[递推状态 s]；低通语义（GPCR 高亲和类比：慢、积分）。

### L1.F5 `release_probability(h, phi) -> p`
- 生物：释放概率 Pr——历史依赖随机门控。
- 算法：`p = σ(φ·h)`；`h = EMA(近期释放掩码)`（释放史缓存）；φ 可学习。
- 特性：[纯函数]（h 为输入）；p∈(0,1)。

### L1.F6 `stochastic_release(x, p, rng, straight_through=False) -> (y, mask)`
- 算法：训练 `mask ~ Bernoulli(p)`，`y = x·mask`；eval 用期望 `y = x·p`；基线 STE 直通。
- 特性：[确定性(rng)]；训练/推断语义分离（探索在训练、确定在推断——主动推断原则 3 的微观形态）。

### L1.F7 `stp_step(u, x_res, a, U, tau_f, tau_d) -> (y, u', x_res')`
- 生物：短时程可塑性 STP——连接级囊泡池状态变量 (u, x)，频率依赖的易化/耗竭。
- 算法（Tsodyks–Markram 率版）：
  - 易化：`u' = u + (U−u)/τ_f + U·(1−u)·a`
  - 耗竭：`x_res' = x_res − u'·x_res·a + (1−x_res)/τ_d`
  - 输出：`y = u'·x_res'·a`
- 特性：[递推状态 u, x_res][数值安全]（∈[0,1] 断言）；τ_f<τ_d 易化型、τ_f>τ_d 耗竭型（配置）。

### L1.F8 `delay_line(x_hist, w_logits) -> x_delayed`
- 生物：传导延迟——可学习 tau，动态感受野。
- 算法：环缓冲 τ_max 步；软延迟 `x_delayed = Σ_τ softmax(w_logits)_τ·x[t−τ]`；w_logits 可学习。
- 特性：[递推状态 x_hist]；默认 logits 锐化于 τ=0（延迟关闭、剂量可调——异时性）。

### L1.F9 `synapse_forward(x, st) -> y`（组合件）
- 算法：三通路并行汇总 + 电突触直通：
  `y = L1.F1(x,W_el) + L1.F2(ampa) + L1.F3(nmda 支路 y_nmda) + L1.F6(stp∘pr∘delay∘mglur)`
- 特性：各通路独立开关（注册表 enabled）；状态聚合在 `SynapseState`（u/x_res/s/x_ema/h/x_hist），reset() 全清。
- 集成验收：单通路关闭 = 该项贡献为零（消融钩子）。

### L1.F10 `phase_multiplex` — 占位：相位复用移至 L9 时间轴层统一处理（架构 L1 原文注明）。
