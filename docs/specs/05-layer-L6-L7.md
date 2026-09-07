# L6 结构演化层 + L7 维护层规格（v0.1）

- 定位：架构 L6/L7 逐条落地。覆盖条目映射：LTP/LTD→L6.F1；关键期三锁→L6.F2；关键期开闭调度→L6.F3；静默突触→L6.F4/F5；健康度/凋亡/回收池→L6.F6–F8；扩张/分裂/萌芽→L6.F9–F11；五通道留存→L6.F12；星形胶质→L7.F1；小胶质→L7.F2；少突胶质→L7.F3；睡眠三嵌套→L7.F4/F5；SWR 回放→L7.F6；SHY→L7.F7；类淋巴→L7.F8。
- 时钟域：L6 全 T3（每 M epoch）/ T4；L7 全 T4（每轮训练/睡眠）。

## L6 结构演化层（拓扑）

### L6.F1 `ltp_ltd(w, a, z, theta_bcm, eta) -> w'`
- 生物：功能可塑性（LTP/LTD，权重级）。
- 算法：复用 L5.F8 BCM 核（权重级可塑性统一走五选择压，本函数为结构层入口）。
- 特性：[纯函数]。

### L6.F2 `install_lock(module, kind)` / `remove_lock(module, kind)`
- 生物：**关键期三锁**（PNN/髓鞘/表观类比）——模块级可塑性锁，冗余且可逐个拆除（重开可塑性）。
- 三锁语义：
  - `param`（PNN 类比）：模块参数冻结——L5.F12 结算跳过该模块；
  - `band`（髓鞘类比）：束带宽锁——L2.F3 秩调度跳过，r、U、V 冻结；
  - `meta`（表观类比）：元参数/超参冻结——剂量表、注册表 enabled 字段冻结。
- 特性：锁状态持久化（注册表 K.F10）；锁定冗余（三锁互相独立）；解锁必须显式调用且可单锁拆除（验收：逐个解锁后该模块恢复可塑性）。
- 验收测试：锁后参数逐元素不变（跑 N 步结算比对）；解锁后恢复更新。

### L6.F3 `critical_period_state(ei_balance, experience_quality, min_epochs, cfg) -> window_state`
- 生物：**关键期开闭调度**——抑制就绪（E/I 达标）→ 窗口打开；经验质量触发 → 关闭（数据不足时延迟锁定，暗饲养类比）。
- 算法：`E/I_ratio = mean(ā_E)/max(mean(ā_I), ε)`（兴奋/抑制单元激活率 EMA 之比，Dale 标签来自 DEV.F1；target 默认 1.0，tol 默认 0.2）；`ready = |E/I_ratio − target| < tol`；`open = ready`；`close = open ∧ (quality > τ_q ∨ epochs > max_wait)`；quality = 验证增益/新颖度统计。
- 特性：[确定性]；返回 `{open, close, delay_reason}`；三锁安装（L6.F2）只在 close 时执行。

### L6.F4 `silent_pool(adjacency, health) -> candidates`
- 生物：元件库储备——静默突触池（预置零权重、可被激活的连接）。
- 算法：邻接中存在但 w=0 的连接构成候选池；按健康度/资格迹排序。
- 特性：[纯函数]；池大小受预算约束（L6.F9）。

### L6.F5 `awaken(candidates, eligibility, threshold, rng) -> activated`
- 生物：静默突触被活动激活（过度产生的结构侧）。
- 算法：候选资格迹 > threshold 者 `w ← w_init·N(1, σ)`（小剂量启动）；需预算批准。
- 特性：[确定性(rng)]；返回激活清单（供注册表留痕）。

### L6.F6 `health_score(act_rate, grad_contrib, value_eval, alpha) -> h`
- 生物：健康度评分 = 激活率 + 梯度贡献 + 价值评估器打分。
- 算法：`h_i = α1·ā_i + α2·|ḡ_i| + α3·v_i`（各分量先归一化到 [0,1]）。
- 特性：[纯函数]；h∈[0,1]。

### L6.F7 `soft_apoptosis(W, h, eta_apop, theta_apop) -> W'`
- 生物：结构凋亡/修剪——软凋亡（低健康度权重衰减至回收）。
- 算法：`W' = W·(1 − η_apop·relu(theta_apop − h))`。
- 特性：[纯函数]；软性（连续衰减，非硬删——退化与效率的受控通道）。

### L6.F8 `recycle_pool(removed_params) -> pool`
- 生物：回收池——凋亡参数与连接模板留存备用（退化性/储备原则）。
- 算法：入池存 (shape, init_stats, health_history)；出池供 L6.F5/F10 复用。
- 特性：[确定性]；池有容量上限（稳态预算）。

### L6.F9 `expansion_budget(n_apop, rho) -> n_max_new`
- 生物：稳态预算——扩张必伴凋亡。
- 算法：`n_max_new = ⌊ρ·n_apop⌋`（ρ 默认 1.0）。
- 特性：[纯函数]；所有结构新增（F5/F10/F11）必须先过此预算。

### L6.F10 `split_unit(w, noise_scale, rng) -> (w_a, w_b)`
- 生物：结构扩张/分裂——高损失/高互信息瓶颈处节点分裂（BDNF 类比）。
- 算法：`w_a = w`；`w_b = w·(1 + noise_scale·N(0,1))`（复制 + 剂量扰动）；连接复制减半以保总能量。
- 特性：[确定性(rng)]；默认 enabled=False（成人神经发生存疑条目，见 K5）。

### L6.F11 `sprout_bypass(W, mask, init_scale, rng) -> W'`
- 生物：旁路萌芽——高损失/高 MI 瓶颈处加并行通路（探索-选择布线）。
- 算法：瓶颈定位（单元健康度分布尾部）→ 新增小剂量并行连接。
- 特性：[确定性(rng)]；受 L6.F9 预算约束。

### L6.F12 `retention_channels`（模块接口）——**五通道留存读写接口**
- 生物：参数（权重）、结构（拓扑）、程序（架构描述）、文化（提示/规范/数据）、生态位（环境/工具）——五条不同速度、不同保真度的继承通道并行；子代优势的大头走快通道。
- 接口：`save_channel(channel, data)` / `load_channel(channel)`；channel ∈ {params, structure, program, culture, niche}。
- 语义：params=权重快照（快/高保真）；structure=邻接+锁状态（中）；program=种子+配置+注册表（慢）；culture=提示/规范/数据清单（快）；niche=环境/工具版本（慢）。
- 特性：版本化、可校验（校验和）；保存/载入往返一致性测试。

## L7 维护层（胶质 + 睡眠）

### L7.F1 `astrocyte_gain(var_i, target, a_g) -> gamma_i`
- 生物：星形胶质（三方突触）——增益缓冲 + 激活方差监控（自适应正则）。
- 算法：`γ_i = 1 + a_g·(target − var_i)`（方差超目标降增益）。
- 特性：[纯函数][递推 var_i]；γ_i>0 断言。

### L7.F2 `microglia_tag(health, threshold) -> prune_mask`
- 生物：小胶质——修剪执行者（补体标记执行）。
- 算法：`prune_mask = health < threshold`；标记清单交给 L6.F7 执行（标记与执行分离——受控退化）。
- 特性：[纯函数]。

### L7.F3 `oligo_myelin(rank_state, mi_estimate) -> rank_schedule`
- 生物：少突胶质——自适应髓鞘化（执行 L2.F3 的秩调度决策）。
- 算法：包装 L2.F3；锁定（band 锁）时返回空操作。
- 特性：[确定性]。

### L7.F4 `sleep_carriers(t, dt, cfg) -> (phi_slow, phi_spindle, phi_ripple)`
- 生物：睡眠节律硬件——慢波 <1Hz（皮层 up-down）、纺锤 7–15Hz（丘脑网状核）、涟漪 100–250Hz（海马）。
- 算法：相位累加器 `φ += 2π·f·dt`；`up_state = (φ_slow mod 2π) < π`。
- 特性：[递推状态相位][确定性]；频段参数可配置（40Hz 夹带等候选假说留钩子，见 K5）。

### L7.F5 `nested_gate(phi_slow, phi_spindle, phi_ripple, cfg) -> ripple_events`
- 生物：**睡眠三嵌套**——慢波 up-state 嵌套纺锤波，纺锤波嵌套涟漪，涟漪携带回放（Staresina 2015）。
- 算法：`event = up_state ∧ (|sin φ_spindle| > τ_sp) ∧ (|sin φ_ripple| > τ_rp)`。
- 特性：[纯函数]；事件密度随嵌套深度递减（θ:γ:ripple ≈ 5-7:5 自相似）。

### L7.F6 `swr_replay(net, buffer, events, cfg, rng) -> None`
- 生物：SWR 回放——涟漪窗口内离线回放训练（经验重放 + 压缩/巩固）。
- 算法：事件窗口内从经验 buffer 采样序列重放；低 lr 局部巩固结算（L5.F12 的睡眠档：仅 RPE+稳态分量，抑制 Hebb 分量）。
- 特性：[确定性(rng)]；只发生于 T4 睡眠轮；与在线训练参数隔离（训练与运行分离）。

### L7.F7 `shy_downscale(W, eta_shy) -> W'`
- 生物：睡眠 SHY（突触稳态假说，Tononi & Cirelli）——周期性全局权重软下调。
- 算法：`W' = W·(1 − η_shy)`（睡眠轮末执行）。
- 特性：[纯函数]；乘法性、全局统一剂量。

### L7.F8 `glymphatic_flush(traces, grads_cache, decay) -> None`
- 生物：类淋巴——梯度/激活「代谢废物」清理（软重置）。
- 算法：陈旧资格迹/梯度缓存/优化器状态按 decay 衰减或清空（睡眠轮末执行）。
- 特性：[确定性]；防止陈旧信号跨轮污染结算。
