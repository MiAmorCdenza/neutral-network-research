# L8 行动闭环层（含小脑模块）+ L9 时间轴层规格（v0.1）

- 定位：架构 L8/L9 与第 9 节小脑规格逐条落地。覆盖条目映射：传出拷贝→L8.F1；前向模型→L8.F2/L8C；CPG→L8.F3；主动推断→L8.F4；小脑十二组件→L8C.F1–F8；频段层级→L9.F1；相位编码→L9.F2/F3；CFC→L9.F4；通信窗口→L9.F5；预测/误差分频→L9.F6；相位门控可塑性→L9.F7；dt 阶梯→L9.F8。
- 双引擎分工（10.D1）：皮层引擎 = 层级预测编码主干（全局语义 + 局部细节）；小脑引擎 = 并行前向模型 + 误差总线（快速预测、局部时序修正、学习信号源）。**功能切分，不按时间切分。**

## L8 行动闭环层（主动推断）

### L8.F1 `efference_copy(a) -> a_cond`
- 生物：传出拷贝——动作副本作为条件通道注入生成器。
- 算法：`a_cond = FiLM/拼接嵌入(a)`；注入生成器各层（自己引起的「噪声」作为条件而非去噪对象——认知原则 3）。
- 特性：[纯函数]；无 a 时退化为零条件（恒等）。

### L8.F2 `forward_model(net, s, a) -> s_hat`
- 生物：前向模型——预测动作后果（小脑式模块）。
- 算法：`s_hat = L8C 通路(s, a_cond)`；详细规格见 L8C 组。
- 特性：[纯函数]；与皮层主干并行（双引擎）。

### L8.F3 `cpg_phase(phases, omega, K_c, dt, rng) -> phases'`
- 生物：中央模式发生器——节律基底（出生即工作的先天回路）。
- 算法（Kuramoto）：`φ_i' = φ_i + dt·(ω_i + (K_c/N)·Σ_j sin(φ_j − φ_i)) + σ_c·√dt·N(0,1)`。
- 特性：[确定性(rng)]；同步性随 K_c 上升（序参量测试）。

### L8.F4 `expected_free_energy(predictions, preferences, ambiguity, w) -> G`
- 生物：主动推断——感知-行动循环，最小化预期自由能（规划 = 在流形上的反向扩散）。
- 算法：`G = −w_p·E_q[ln p(o|pref)] + w_a·H[p(o|s)]`（pragmatic 项 + epistemic 项：偏好驱动 + 信息寻求）。
- 特性：[纯函数]；动作选择 = 对 G 的局部下降（探索-选择代替精确编程的 L8 形态）。

## L8C 小脑模块（架构 §9 规格落地，双引擎之并行前向模型）

组件映射表（架构 9.2）：苔藓纤维=条件/状态输入（含传出拷贝通道）；颗粒层=稀疏扩展重编码层；平行纤维=扩展特征大扇入；浦肯野=巨型扇入线性读出（预测头）；爬行纤维=误差总线；复杂尖峰=低频结算；简单尖峰=快速推断（stop-gradient 分离）；资格迹=各块本地衰减缓存；高尔基负反馈=扩展层稀疏约束；核-橄榄负反馈=误差总线稳态调节；斑马纹旁矢状带=模块化误差通道（MoE）。

### L8C.F1 `granule_expand(x, G, sparsity_target) -> z`
- 生物：颗粒层——每单元仅 3–7 输入；随机或可学习；除法归一化保稀疏。
- 算法：固定随机稀疏矩阵 G（默认 V0：随机固定零训练成本；V1 验证可学习）→ 除法归一化（L0.F4）→ 稀疏控制（L8C.F6 动态阈值）。
- 特性：[冻结(默认)]；输入度 3–7 断言；扩展比 10–100×（宽扁平层）。

### L8C.F2 `purkinje_readout(z, W_p) -> y_hat`
- 生物：浦肯野细胞——巨型扇入线性读出（预测头）；本地误差驱动调权。
- 算法：`y_hat = W_p·z`（线性，扇入 = 全扩展层）。
- 特性：[纯函数]；W_p 用本地误差（爬行纤维信号 × 资格迹）更新，不经反向传播。

### L8C.F3 `climbing_fiber(delta, rpe, w) -> err_bus`
- 生物：爬行纤维——误差总线；同时携带运动误差与 RPE（Kostadinov 2019；Sendhilnathan 2020），误差总线与价值通道可合并。
- 算法：`err_bus = w·delta + (1−w)·rpe`（低维投影见 K1/误差总线；广播到所有块）。
- 特性：[纯函数]；w 可配置（运动误差与价值混合比）。

### L8C.F4 `complex_spike_settle(t, K) -> settle_flag`
- 生物：复杂尖峰（~1 Hz）——低频结算信号；每 K 步或每序列结算一次。
- 算法：`settle_flag = (t mod K == 0)`；结算频率与资格迹衰减时间常数匹配（9.3 关键决策）。
- 特性：[纯函数]。

### L8C.F5 `simple_spike_readout(y_hat) -> out`
- 生物：简单尖峰——快速推断输出；与学习通道分离（stop-gradient）。
- 算法：`out = y_hat.detach()`（推断通道无梯度泄漏）。
- 特性：[无梯度边界]；测试断言：简单尖峰输出不带 grad_fn。

### L8C.F6 `golgi_feedback(z, target_sparsity) -> theta'`
- 生物：高尔基负反馈——扩展层稀疏约束（动态阈值）。
- 算法：`θ' = θ + η_g·(actual_sparsity − target_sparsity)`（过密则升阈值）。
- 特性：[递推状态 θ][纯函数]。

### L8C.F7 `nucleo_olivary_homeostasis(err_bus_norm, tau) -> scale`
- 生物：核-橄榄负反馈——误差总线稳态调节，防误差信号爆炸。
- 算法：`scale = τ/EMA(err_bus_norm)`（幅值归一）。
- 特性：[递推状态 EMA][纯函数]。

### L8C.F8 `zebrin_channels(err, gating) -> errs_masked`
- 生物：斑马纹旁矢状带——模块化误差通道（MoE 式并行块，模块间竞争）。
- 算法：err 按 gating 分配到 k 个模块通道（top-m 竞争选通），逐模块独立误差——**D3 风险项（广播瓶颈）的结构性缓解**。
- 特性：[纯函数]；模块数可配置；每块误差只回授本块（区室化）。

## L9 时间轴层（贯穿全部，T0 递推）

### L9.F1 `band_oscillators(freqs, dt) -> phases`
- 生物：**频段层级**——频率 = 空间尺度（Buzsáki & Draguhn 2004）：慢节律全局调度（theta 类比 = 序列帧），快节律局部执行（gamma 类比 = 帧内地址）。
- 算法：各频段相位累加器 `φ += 2π·f·dt`；默认频段 {theta 4-8Hz, gamma 30-80Hz, ripple 100-250Hz}（按待测假说标注，见 K5）。
- 特性：[递推状态][确定性]。

### L9.F2 `phase_precession(phi_base, pos, kappa) -> phi`
- 生物：**相位编码**——时间即地址（相位进动：发放相位逐周期提前，相位编码野内位置）。
- 算法：`φ_i = φ_base + κ·pos_i`（相位随位置线性提前）。
- 特性：[纯函数]；kappa 决定分辨率（位置↔相位双射）。

### L9.F3 `theta_sequence(seq, n_lookahead, phase) -> lookahead`
- 生物：theta 序列——每个周期把前方路径压缩为前瞻扫描（预测的物理形态）。
- 算法：`lookahead(t) = readout(seq, pos + v·phase/2π·n_lookahead)`（未来轨迹压缩进当下周期）。
- 特性：[纯函数]；前瞻窗口 = 帧长/载波比（约一个数量级）。

### L9.F4 `pac_coupling(phi_theta, a_gamma, A) -> a_gamma'`
- 生物：**CFC 多路复用**——慢相位调制快幅度（同一批单元并行传输多层信息；theta:gamma 约 5-7 自相似嵌套）。
- 算法：`a_gamma' = (1 + A·cos φ_theta)·a_gamma`。
- 特性：[纯函数]；A∈[0,1]（调制深度）。

### L9.F5 `phase_window(dphi, kappa, b) -> g`
- 生物：**通信窗口**——相干路由（CTC）：不改连接改相位；节律性抑制制造通信窗口。
- 算法：`g = σ(κ·cos Δφ + b)`（有效连接 = 解剖连接 × 相位窗口）。
- 特性：[纯函数]；同一发送者可用不同相位对不同接收者说话（选择性路由）。

### L9.F6 `spectral_split(signal, tau) -> (pred, err)`
- 生物：**预测/误差分频**——预测走 beta/alpha 类低频流（自上而下），误差走 gamma 类高频流（自下而上）（Bastos 2012）；与总纲推断双流直接对接。
- 算法：`pred = lowpass(signal, τ)`；`err = signal − pred`。
- 特性：[递推状态]；pred+err=signal（恒等断言）；pred 慢而全局、err 快而局部。

### L9.F7 `phase_gated_lr(phi, eta, a) -> eta_t`
- 生物：**相位门控可塑性**——何时更新由相位决定（theta 峰 LTP/谷 LTD，Huerta & Lisman 1995）。
- 算法：`η_t = η·(1 + a·cos φ)`（峰增强、谷抑制/反转）。
- 特性：[纯函数]；a=0 退化为常数学习率。

### L9.F8 `dt_ladder(cfg) -> ladder`
- 生物：**dt 阶梯与嵌套时间结构**——保留哪个机制，就把等效时间分辨率压到其下；快环精确而稀疏，慢环粗略而全局；慢层永远不在快 dt 上积分。
- 算法：机制→dt 档映射（架构 5.1 表）：全尖峰波形 0.1ms｜涟漪回放 1-2ms｜gamma 相位 2.5-5ms｜NMDA/STDP 窗口 5-10ms｜theta 帧 25ms；断言：每个保留机制 ≥10 积分步解析；绑定到 K2 时钟域。
- 特性：[确定性]；违反解析约束时抛配置错误（保真原则 2 的硬检查）。
