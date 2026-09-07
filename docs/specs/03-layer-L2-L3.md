# L2 介观束层 + L3 全局状态层规格（v0.1）

- 定位：架构 L2/L3 逐条落地。覆盖条目映射：白质束低秩主干→L2.F1；束内梯度投影→L2.F2；自适应髓鞘化（McKenzie 2014）→L2.F3；髓鞘双职能→L2.F3+锁；布线成本（小世界/rich club）→L2.F4；电场耦合→L2.F5；空间调制场→L3.F1/F2；受体亚型成对调制→L3.F3；HPA 负反馈→L3.F4；昼夜节律→L3.F5；评估器接口（GPCR 低通）→L3.F6。
- 时钟域：L2 训练 T1 / 秩调度 T3；L3 全 T2（每 N 步 EMA）+ T5（昼夜）。

## L2 介观束层（骨架）

### L2.F1 `low_rank_tract(x, U, V, W0, alpha, r) -> y`
- 生物：白质束 = 低秩主干（LoRA 式），秩 = 带宽。
- 算法：`y = W0·x + (α/r)·U·(Vᵀx)`；U∈[N,r]、V∈[N_in,r]；r 由 L2.F3 调度。
- 特性：[纯函数]；W0 冻结时仅 U,V 可塑——「带宽锁」的机制载体（L6.F2 髓鞘锁锁 r、锁 U,V）。

### L2.F2 `tract_grad_projection(g, r, c) -> g'`
- 生物：束内梯度投影——防单权重发散。
- 算法：按束分块：`g_block ← g_block·min(1, c/‖g_block‖)`（c 默认 1.0）。
- 特性：[纯函数]；仅限束内参数（U,V）使用。

### L2.F3 `myelin_rank_schedule(singular_energy, r, tau_hi, tau_lo, r_min, r_max) -> r'`
- 生物：自适应髓鞘化——按互信息动态增缩秩（McKenzie 2014）；**髓鞘双职能**：高速硬件 + 可塑性锁（关键期刹车之一）。
- 算法：MI 代理 = 束内谱能量占比 `ρ_r = Σ_{i≤r} σ_i² / Σσ_i²`（σ_i 为束内激活协方差奇异值）；`ρ_r > τ_hi → r+1`；`ρ_r < τ_lo → r−1`；否则保持。
- 特性：[确定性]；每 K epoch（T3）；锁定（髓鞘锁）时 r 冻结、本函数跳过；r∈[r_min, r_max]。

### L2.F4 `wire_cost(W, positions, lambda_sw, lambda_rc) -> scalar`
- 生物：布线成本正则——小世界 + rich club 拓扑先验。
- 算法：`L = λ_sw·Σ_ij W_ij²·d_ij − λ_rc·Σ_{hub 对} W_ij²`；d_ij=‖p_i−p_j‖（p 来自 DEV.F2）；hub=度前 10%。
- 特性：[纯函数]；作为 L5 代谢/结构选择压的附加项（鼓励局部连接 + hub 互联）。

### L2.F5 `ephaptic_coupling(x, C, eps)`（可选，默认 enabled=False）
- 生物：电场耦合（架构标注可选）。
- 算法：`z += ε·C·x`，C 为小幅度耦合矩阵（冻结）。
- 特性：[纯函数]；默认关闭（无指标增益不引入——版本门考核）。

## L3 全局状态层（广播）

### L3.F1 `modulation_field_update(G, source, tau_G) -> G'`
- 生物：空间调制场（hormone map）——慢动力学，独立时间常数 tau_state，EMA 每 N 步更新（T2）。
- 算法：`G' = (1−1/τ_G)·G + (1/τ_G)·source`；source = 评估器输出/激活统计的低分辨率投影（L3.F6 提供）。
- 特性：[递推状态 G]；G 形状 = 低分辨率场（如 4×4×c），远小于单元数——「慢环粗略而全局」。

### L3.F2 `filminject(x, G, upsample, gamma_fn, beta_fn) -> z`
- 生物：低分辨率场上采样后 FiLM/仿射注入；不同位置浓度不同。
- 算法：`m_i = upsample(G)[pos_i]`（双线性）；`z = gamma_fn(m_i)⊙x + beta_fn(m_i)`；gamma_fn/beta_fn 为可学习仿射。
- 特性：[纯函数]；位置异质性（同层不同单元受不同浓度调制）。

### L3.F3 `paired_receptor_modulation(m, w) -> (gamma_plus, gamma_minus)`
- 生物：受体亚型成对调制——同一调制信号产生方向相反的增益对。
- 算法：`γ_+ = 1 + tanh(w·m)`；`γ_− = 1 − tanh(w·m)`；分别注入成对通道（如兴奋支/抑制支）。
- 特性：[纯函数]；配对约束：`γ_+ − 1 = −(γ_− − 1)`（测试断言）。

### L3.F4 `hpa_feedback(var_local, theta_hpa, kappa) -> neg_mod`
- 生物：负反馈（HPA 类比）——激活方差超阈值区域施加局部负调制。
- 算法：`neg_mod_i = κ·max(0, var_i − θ_hpa)`（var_i 为单元近期激活方差 EMA）；注入 G 或直接降增益。
- 特性：[纯函数]；自稳态——方差回落到阈值以下时负调制消失。

### L3.F5 `circadian_gate(t, tau_circ, phase, amplitude) -> c`
- 生物：昼夜节律——最慢门控（学习率/探索率周期调制）。
- 算法：`c(t) = 1 + A·cos(2π·t/τ_circ + φ)`；`lr_eff = lr·c`；探索温度 `/c`。
- 特性：[纯函数]；T5 时钟（训练周期级）；A=0 时退化恒等（剂量可调）。

### L3.F6 `evaluator_broadcast(value_signal, tau_bc) -> source`
- 生物：**评估器接口**——调制场是价值层（L4）向全局广播的通道；GPCR 高亲和 → 低通积分器语义。
- 算法：`source = lowpass(value_signal, tau_bc)`（L4 输出的效价/难度轴信号的低通投影）。
- 特性：[纯函数]；本函数是 L4→L3 的唯一官方通道（弱调控连接：接口不敏感于内部参数）。
