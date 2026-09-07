# L4 价值决策层 + L5 学习规则层规格（v0.1）

- 定位：架构 L4/L5 逐条落地。覆盖条目映射：RPE 好坏轴→L4.F1/F2；难度轴（努力/不确定/EVC）→L4.F3/F4；5-HT 惩罚抑制→L4.F5；Go/NoGo 双门控→L4.F6；MF/MB 双系统→L4.F7；评估器可重指向→L4.F8；脱钩监控→L4.F9；三因素→L5.F1；资格迹（单/双时间常数）→L5.F2/F3；五选择压→L5.F4–F7/F10/F11；BCM 元可塑性→L5.F8；Vogels 抑制可塑性→L5.F9；逆行信号→L5.F10；结算汇总→L5.F12。
- 价值理论（生物学文档 10.5）：价值 = 关于参照系统持续存在的预测；五层接地链中本层实现情感/联想层。**好坏轴与难度轴是同一评估器的两个正交预测轴**。
- 时钟域：L4 T1（每去噪步/每序列结算）；L5 递推 T0、结算 T1、缩放 T2。

## L4 价值决策层（评估器）

### L4.F1 `value_estimate(V, s) -> v`
- 生物：价值 = 对"结果是否有利于参照系统持续存在"的预测（情感/联想层信号）。
- 算法：`v = V(s)`，V 为价值头（MLP，输入网络全局状态摘要 s，输出标量效价）。
- 特性：[纯函数]；实现 Evaluator 接口（L4.F8）——任何评估器必须满足此签名（可重指向的前提）。

### L4.F2 `td_error(r, v, v_next, gamma) -> delta_v`
- 生物：多巴胺 RPE——结果比预期好/坏多少（时序差分误差，逐去噪步结算）。
- 算法：`δ_v = r + γ·v_next − v`；**V0 中 r := r_energy = reward_influx − cost_ema·dt**（能量流入超预期；接地形态见 10-energy-budget §2/§2.5，cost_ema 为能量池状态）。
- 特性：[纯函数]；δ_v 是奖惩选择压（L5.F5）的调制信号 M 之一。

### L4.F3 `difficulty_axis(pe, pe_var_ema, control_cost, weights) -> (effort, u_unexp, u_exp)`
- 生物：难度轴三信号——努力成本（前扣带回/中脑边缘多巴胺）、意外不确定（蓝斑 NE）、预期不确定（ACh，Yu & Dayan 2005）。
- 算法：
  - `effort = ‖control‖²`（控制/动作代价）
  - `u_unexp = EMA(‖pe‖)`（当前预测误差幅值——意外不确定）
  - `u_exp = EMA(‖pe‖²) − ‖EMA(pe)‖²`（误差方差——预期不确定）
- 特性：[纯函数][递推 EMA 状态]；u_unexp 与 u_exp 语义正交（测试：恒定误差下 u_exp→0 而 u_unexp 非零）。

### L4.F4 `evc(expected_value, cost, uncertainty, w) -> control_value`
- 生物：控制的期望价值（EVC，Shenhav 2013）——「值不值得尝试」的复合预测。
- 算法：`EVC = w_b·E[v] − w_c·effort − w_u·(u_unexp + u_exp)`。
- 特性：[纯函数]；输出门控"是否投入控制/探索"（L4.F6 的输入）。

### L4.F5 `serotonin_modulate(ser, ltd_gain, temperature) -> (ltd_mult, temp')`
- 生物：5-HT → 局部 LTD + 保守采样（惩罚/行为抑制）。
- 算法：`ltd_mult = 1 + ser·ltd_gain`（放大负向更新）；`temp' = temperature/(1 + ser)`（采样温度下降=保守）。
- 特性：[纯函数]；**ser 来源定义：ser = relu(−δ_v)**（负 RPE 即惩罚代理；外部惩罚事件经事件系统归一后亦汇入）。痛感是可调制的预测——存疑条目处理。

### L4.F6 `gono_gate(values, beta) -> (p_go, p_nogo)`
- 生物：基底节直接/间接通路双门控（Go/NoGo）。
- 算法：`p_go = σ(β·(v − v̄))`（直接通路：放大选中项）；`p_nogo = 1 − p_go`（间接通路：抑制竞争项）；作用于候选动作/采样分布。
- 特性：[纯函数]；p_go+p_nogo=1（配对断言）。

### L4.F7 `dual_controller(mf_value, mb_value, u_exp) -> value_mix`
- 生物：model-free（习惯）与 model-based（规划）双系统。
- 算法：`value_mix = w_mb·mb + (1−w_mb)·mf`，`w_mb = σ(κ·(u_exp − θ))`（预期不确定高时倾向 model-based）。
- 特性：[纯函数]；w_mb∈(0,1)。

### L4.F8 `evaluator_registry`（模块接口，非单函数）
- 生物：**评估器可重指向**——评估器是可替换模块，文化层可安装新价值源（神经元回收原理；新价值源不是新机制，是旧通道的新内容）。
- 接口：`install_evaluator(name, ev: Evaluator)`；`get_evaluator() -> Evaluator`；`Evaluator` 契约 = `value(s) -> ValueSignal(valence, effort, u_unexp, u_exp)`。
- 特性：替换时旧评估器不删除（退化性：备用评估器保留，可回切）。

### L4.F9 `decoupling_monitor(valence_stream, ground_stream, window, threshold) -> alert`
- 生物：**脱钩监控**——检测评估器与接地层的漂移（成瘾/人口转变类脱钩的架构级检测器）。
- 算法：滑窗相关系数 `ρ = corr(valence, ground)`；`ρ < threshold` 持续 window 步 → alert（评估器失效标记，触发元选择/重指向）。
- 特性：[确定性]；**ground_stream 唯一来源 = E.F8 ground_signal（能量轨迹）**——与 10-energy-budget §2 对齐；接地层是代谢层（10.5.1），脱钩监控对代谢轨迹测漂移，不是对原始任务奖励。

## L5 学习规则层（贯穿全部）——V0 灵魂的完整五选择压形态

### L5.F1 `three_factor(m, e, eta) -> dw`
- 生物：三因素规则：`Δw = η·全局调制(误差/奖励)·资格迹(t)`（Gerstner 2018）。
- 算法：`dw = η·m·e`（逐元素）。
- 特性：[纯函数]；m 为标量或逐单元向量（广播调制），e 为资格迹。

### L5.F2 `eligibility_trace(pre, post, lam) -> E'`
- 生物：资格迹——每连接的近端活动指数衰减缓存（生物版「缓存梯度待结算」）。
- 算法：`E' = λ·E + mean_b(post ⊗ pre)`。
- 特性：[无梯度][递推状态]；λ∈(0,1)；结算延迟 K 步的性能损失 ≤10% 是它的验收（D4-4）。

### L5.F3 `eligibility_dual(pre, post, tau_f, tau_s) -> (E_f', E_s')`
- 生物：双时间常数资格迹 → STDP 时序核（保留时序语义的通道，dt 阶梯 5–10ms 档）。
- 算法：双缓冲不同 λ（λ_f=e^{−Δt/τ_f}，λ_s=e^{−Δt/τ_s}）各自按 L5.F2 递推；结算取差：`e_stdp = E_f − c·E_s`。
- 特性：[递推状态]；提供时序窗口测试钩子（pre-before-post → 增强，post-before-pre → 减弱）。

### L5.F4 `hebbian(z, a, eta_h) -> dw`（活动竞争）
- 生物：活动竞争 = Hebb——时间相关性（一起放电一起存活）。
- 算法：`dw = η_h·(z⊗a − mean 正则)`。
- 特性：[纯函数]。

### L5.F5 `rpe_settle(delta_v, E, eta_r) -> dw`（奖惩）
- 生物：奖惩 = RPE 结算——预测误差（结果比预期好/坏多少）。
- 算法：`dw = η_r·δ_v·E`（δ_v 来自 L4.F2；价值结算的本地代理）。
- 特性：[纯函数]。

### L5.F6 `metabolic_regularization(W, zbar, eta_m) -> dw`（代谢）
- 生物：代谢 = 稀疏/能量正则——表征的能耗（L0.F5 的导数项）。
- 算法：`dw = −η_m·‖z̄‖₁·W`（活动越贵惩罚越强；z̄ 为单元活动率 EMA）。
- 特性：[纯函数][递推 z̄]。

### L5.F7 `synaptic_scaling(W, r_actual, r_target, gamma, tau) -> W'`（稳态）
- 生物：稳态 = 突触缩放——活动率目标（Turrigiano 1998）。
- 算法：`W' = W·(r_target/max(r_actual,ε))^γ`；慢（T2），r_actual 为长时程率 EMA。
- 特性：[纯函数]；乘法性（保符号、保相对结构）。

### L5.F8 `bcm_slide(theta, z2, tau) -> theta'` + `bcm_update(w, a, z, theta, eta) -> w'`（元可塑性）
- 生物：BCM 滑动阈值——低活动易 LTP、高活动易 LTD（Bienenstock 1982）。
- 算法：`θ' = θ + (z² − θ)/τ`；`Δw = η·z·a·(z − θ)`。
- 特性：[递推状态 θ][纯函数]；z>θ 增强、z<θ 削弱（符号测试）。

### L5.F9 `vogels_update(w_i, z_e, z_i, rho0, eta_i) -> w_i'`（抑制性可塑性）
- 生物：E-I 平衡局部规则（Vogels 2011）——把后突触率推向目标 ρ0。
- 算法：`Δw_i = η_i·(z_i·z_e − ρ0·z_i)`。
- 特性：[纯函数]；仅作用于 Dale 标签为抑制的连接（DEV.F1）。

### L5.F10 `retrograde_signal(post_signal) -> retro`
- 生物：逆行信号——post→pre 广播（连接级元数据通道）。
- 算法：`retro_j = post_signal_j`（后突触单元向全部前突触广播）；作为营养竞争（连接级存活信号）的输入：`survival_ij = EMA(retro_j·pre_i)`。
- 特性：[纯函数][递推 survival]。

### L5.F11 `five_pressures(dw_hebb, dw_rpe, dw_meta, dw_homeo, dw_nutri, weights) -> dw_total`
- 生物：五选择压 = 一个被代理量的五个本地代理——统一不在同一函数，而在同一接地链 + 同一数学形式（对参照的预测误差：RPE 是时间差分版、营养是连接版、稳态是速率版）。
- 算法：`dw_total = w1·dw_hebb + w2·dw_rpe + w3·dw_meta + w4·dw_homeo + w5·dw_nutri`。
- 特性：[纯函数]；权重可配（剂量调整，异时性）。

### L5.F12 `settle_learning(net, ls, signals, cfg) -> None`（结算汇总，T1 每 K 步）
- 算法：资格迹（L5.F2/F3 递推已在 T0 完成）→ 五选择压各自 Δw（L5.F4–F11）→ 汇总（L5.F11）→ 三锁检查（L6.F2：锁定参数跳过）→ `param.data` 更新 → 电突触对称投影（L1.F1）→ 束内梯度投影（L2.F2）。
- 特性：[无梯度]（全程 no_grad，参数 `.grad` 恒 None）；结算计数报告；回放核对：外部按方程重算 ΔW 与 `param.data` 变化逐元素相等（集成验收闸门）。
