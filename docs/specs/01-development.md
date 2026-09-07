# 发育框架与统一迭代调度规格（Development Spec v0.1）

- 定位：架构 §0-B（种子-展开-通道化）、§4 发育时间表、§3 统一迭代算法，及生物学文档 §6/§6.5（发育框架理论）的函数级落地。
- 覆盖条目映射：种子-展开-通道化（6.5A/C）→ DEV.F1/F5；位置信息与化学亲和（6.5B）→ DEV.F2/F4；时间身份（6.5C）→ DEV.F3；自发生活动校准（6.5D）→ DEV.F6–F8；统一迭代算法（架构 §3）→ DEV.F9。

## DEV1 种子与展开（T4 构建期一次）

### DEV.F1 `expand(seed, cfg) -> NetSpec`
- 接口：`seed: int`，`cfg: DevelopConfig`；返回 `NetSpec{cfg, coarse, tensors, positions, birth_order, dale_labels}`。
- 算法：粗结构（层表/激活形式/归一化结构/嵌入方案/连接先验）由 **master_rng（跨种子固定）** 确定——通道化保证粗结构跨种子大致确定（架构原则 5）；细参数（b, gain, σ, θ, 位置 p_i, 出生时间 τ_i, Dale 标签：80% 兴奋/20% 抑制）由 **seed_rng** 个体化（低通道化）。
- 特性：[确定性]；同 seed 同 spec（bitwise）；不同 seed：coarse 完全一致、tensors 不同；无全局状态。
- 验收测试：determinism / coarse-seed-invariant / 形状断言 / Dale 比例断言。

### DEV.F2 `position_field(n_units, gradient_spec, rng) -> Tensor[n_units, dim]`
- 生物：位置信息——形态发生素梯度为每个单元提供坐标（Wolpert 法国国旗模型；BMP/Wnt/Shh/FGF 信号中心）。
- 算法：`p_i = Σ_g a_g·σ(w_g·u_i) + b`（u_i 为单元索引空间坐标，多梯度叠加形成前后/内外轴），叠加小噪声。
- 特性：[纯函数][确定性(rng)]；输出 ∈ [0,1]^dim。

### DEV.F3 `temporal_identity(birth_order, n_stages) -> Tensor[n_units, n_stages]`
- 生物：时间身份——TF 级联按出生顺序贴时间标签（Hb→Kr→Pdm→Cas；皮层六层内向外）。
- 算法：`label_i = onehot(bucket(birth_order, n_stages))`；出生顺序决定层归属（先出生→深层）。
- 特性：[纯函数]；bucket 单调——出生序与层序严格对应。

### DEV.F4 `chemoaffinity_connectivity(pos, labels, sigma_d, sigma_l, rng) -> Tensor[N,N] bool`
- 生物：化学亲和——Eph/ephrin 梯度生成拓扑投射；生长锥只响应局部化学信号，无全局蓝图。
- 算法：`p_ij ∝ exp(−‖p_i−p_j‖²/2σ_d²)·exp(−‖l_i−l_j‖²/2σ_l²)`；仅在局部邻域（top-m 候选）内采样邻接。
- 特性：[确定性(rng)]；连接稀疏（度分布受 σ_d 控制）；对称候选（先采样后按方向拆分兴奋/抑制）。

### DEV.F5 `canalization_sample(master_rng, seed_rng, cfg) -> (coarse, fine)`
- 算法：粗参数（阈值形式、归一化结构、带宽）自 master_rng——跨种子同分布同实现；细参数自 seed_rng。确定性与差异性的分离发生在结构尺度（架构原则 5）。
- 特性：[确定性]。

## DEV2 自发校准（Phase 1，零外部数据）

### DEV.F6 `retinal_waves(rng, n, spatial_profile, n_waves, n_steps) -> Tensor[n_steps, n, ...]`
- 生物：视网膜波——视觉通路打开前的自产生活动（Meister 1991）；自发活动按 Hebb 预校准拓扑。
- 算法：`a(x,t) = Σ_k w_k·exp(−‖x − c_k(t)‖²/2σ_w²) + noise`，`c_k(t)` 为平滑随机游走（波前移动）。
- 特性：[确定性(rng)]；波前空间连续（相邻单元活动相关）——这是"结构可自发校准"的信息来源。

### DEV.F7 `linsker_hebb(W, x, y, eta, decay) -> W'`
- 生物：Linsker 1988——纯随机输入 + 局部 Hebb 自发涌现方向选择性。
- 算法：`ΔW = η·(y⊗x) − decay·W`；行归一 `W_i ← W_i/max(‖W_i‖,ε)`。
- 特性：[纯函数][无梯度]。

### DEV.F8 `self_calibrate(net, rng, cfg) -> None`
- 算法（Phase 1）：零外部数据；输入 = DEV.F6 视网膜波 + 纯噪声；规则 = DEV.F7（兴奋支）+ Oja（快支行归一）+ Vogels（L5.F9 抑制支，E-I 平衡预置）。只改 `W.data`，不动结构。
- 特性：[确定性][无梯度]；不接受数据集参数；结构不变（层数/宽度/邻接不变）。
- 验收：D4-3 口径——下游收敛步数 ↓≥20% 或最终指标可测提升（对照：无自校准）。

## DEV3 统一迭代算法调度（元算法，架构 §3）

### DEV.F9 `meta_iteration(net, registry, epoch_stats, cfg) -> Ops`
- 生物：过度产生 → 按价值选择 → 稳定化（锁定）→ 保留退化与再分配——在每个尺度上运行。
- 接口：输入本 epoch 统计（激活率/健康度/评估器打分/损失分布）；返回结构化操作列表 `Ops`，由 L6/L7 模块执行。
- 算法（四尺度）：

| 尺度 | 过度产生 | 选择（价值来源） | 稳定化 | 退化/再分配 |
|---|---|---|---|---|
| 结构/模块 | L6.F5 静默突触激活、L6.F10 分裂 | 健康度 + 评估器（L6.F6） | 三锁（L6.F2） | 软凋亡、回收池（L6.F7/F8） |
| 参数/权重 | 生成层注入多样性（噪声/剪接类比） | RPE/资格迹结算（L5.F5） | 突触缩放、EMA 固化（L5.F7） | SHY 软下调（L7.F7） |
| 程序/架构 | 元件重组（复制/扰动/异时性） | 评估器元选择 | 架构冻结（通道化） | 假基因化式弃用（enabled=False） |
| 文化/数据 | 提示、规范、数据多样性 | 文化选择 | 规范固化 | 规范替换 |

- 特性：[确定性]；每 epoch 一次（T3）；操作列表带预算约束（L6.F9：扩张必伴凋亡）。
