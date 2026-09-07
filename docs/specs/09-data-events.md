# 数据（事件）子系统规格（Data & Events Spec v0.1）

- 定位：回答「数据（事件）如何获得、定义、输入、反馈」。数据模块 = 五通道中**文化通道**的运行时载体 + 感知-行动闭环的接口层。补全 specs/ 缺失的数据模块（取代旧 v0 规格 M1）。
- 核心立场：**网络的原生本体不是「数据集」而是「事件流」**——连续状态流形（总纲 M）上的离散观测事件；数据集只是事件流的录制回放这一退化情形。语义连续、离散涌现（认知原则 2）：事件是离散标记，网络从离散样本中推断连续动力学（这正是扩散去噪所做的桥接）。
- 关键区分：**事件的外延大于训练数据**——自发活动（Phase 1 视网膜波）、传出拷贝（L8.F1）、回放（L7.F6）、网络自生成样本（DATA.F12）都是事件，只是 src 不同。自己引起的「噪声」作为条件而非去噪对象（认知原则 3）。

## 1. 获得（Acquisition）——获得是动作，不是被动接收

### 1.1 事件来源（六类）

| src | 内容 | 阶段/接口 |
|---|---|---|
| external | 环境/传感器/API 流；数据集=录制事件的回放 | Phase 2 起；DATA.F3 |
| spontaneous | 自发流：视网膜波（DEV.F6）、CPG 节律（L8.F3）、纯噪声 | Phase 1；DEV.F6 |
| efference | 动作副本条件流（自己引起的变化） | V7；L8.F1 |
| replay | 缓冲回放（SWR，L7.F6） | Phase 4；DATA.F4 |
| cultural | 提示/演示/标注（文化通道直入） | 任意阶段 |
| self | 网络自生成样本（自我引导/自举） | V7；DATA.F12 |

### 1.2 获得机制

- **主动采样**：采样即动作——注意力/眼跳类比。下一个采样点/模态/区域的选择 = 最小化预期自由能（L8.F4）或最大化 EVC（L4.F4）；探索温度受昼夜节律（L3.F5）与 5-HT（L4.F5）调制。V0 退化版 = 均匀采样。
- **事件带宽**：获得率是设计参数（时钟域绑定：快事件 T0、价值结算事件 T1、结构事件 T3）。事件驱动 vs 批量密集两种形态共用同一 EventSource 接口（运行时策略 §2 的形态切换不换接口）。
- **缓冲与保留**：经验缓冲按新颖度/难度加权保留（过度产生→按价值选择在数据尺度的落点；DATA.F4）。

## 2. 定义（Definition）——流形优先的事件本体

### 2.1 事件 Schema（DATA.F1）

```python
@dataclass(frozen=True)
class Event:
    t: float            # 事件时间：所在时钟域内的时间戳/相位（时间即地址，L9）
    src: str            # external|spontaneous|efference|replay|cultural|self
    mod: str            # 模态标签（visual/auditory/proprio/...；V0: 'latent'）
    loc: Tensor[int]    # 空间地址（通道/位置索引；空 = 全局观测）
    x: Tensor           # 观测载荷 [D]
    cond: dict          # 条件通道（上下文/动作/提示；可空）
    val: float | None   # 价值标注（reward/label）——只进评估器，不进去噪目标
    meta: dict          # novelty/uncertainty/difficulty 估计（缓冲与课程用）
```

### 2.2 定义原则

1. **流形优先**：每个事件定义/投影为语义流形 M 上的一个点；各模态是 M 的**投影（视图）**，不是独立空间——跨模态融合 = 条件化多个视图，而非拼接多个世界。
2. **价值与感知分离**：`val`、`cond` 是通道不是目标——cond 注入生成器（条件化），val 只送 L4 评估器（价值是「关于持续存在的预测」，与感知目标正交）。
3. **离散标记、连续基底**：事件是离散标记；底层动力学连续（SDE）。事件频率决定 dt 阶梯的最低档（最快机制 ≥10 积分步解析）。
4. **时间即地址**：t 不是均匀索引而是相位/位置码（L9.F2 相位进动）；序列 = theta 帧，前瞻 = theta 序列。
5. **Schema 即文化通道制品**：版本化、可校验（L6.F12 culture 通道 save/load）；规范固化 = 元算法「稳定化」在数据尺度的形态；规范替换 = 其「退化/再分配」。

## 3. 输入（Input）——离散到连续的桥

### 3.1 输入管线

```
事件 → 编码（DATA.F5）→ 流形投影（DATA.F6）→ 噪声耦合（DATA.F8）→ 网络前向
              ↓ 并行                    ↓ 并行
        条件通道拆分（DATA.F7）→ cond 注入生成器 / val 送评估器
```

- **编码**（DATA.F5）：模态编码器 x→e（V0 恒等/线性；L8C 颗粒层是稀疏重编码的生物学完整形态）。
- **投影**（DATA.F6）：e → z∈M；缺失模态 = 部分观测（mask），在子流形上去噪——「不完整数据天然可训练」。
- **噪声耦合**（DATA.F8）：`x_t = mean(t)·z + std(t)·ε`（K.F2 包装）。**离散事件与连续流形的桥 = 加噪本身**：同一机制桥接「类别（离散）↔流形（连续）」与「观测（离散）↔动力学（连续）」。
- **条件注入**（DATA.F7）：cond（上下文/动作/提示）→ FiLM/拼接（L3.F2/L8.F1）；mod_mask → 输入门控；val → 仅 L4。

### 3.2 输入门控（生物过滤 = 鲁棒性）

| 门控 | 机制 | 作用 |
|---|---|---|
| 巧合门控 | NMDA（L1.F3）：局部信号 ∧ 全局唤醒 | 无关事件不入网络 |
| 相位窗口 | 通信窗口（L9.F5）：Δφ 决定有效输入 | 时间上错位的输入被抑制 |
| 随机门控 | Pr 释放概率（L1.F6） | 输入缺失鲁棒性（训练期随机丢事件） |
| 唤醒调制 | L3 调制场（评估器广播 L3.F6） | 价值相关的事件获得增益 |

## 4. 反馈（Feedback）——四条回路 + 两个闭环特例

```
                 ┌──────────────────────────────────────────────┐
                 ▼                                              │
 环境/自发/文化 ──事件──▶ 编码/投影 ──▶ M(z) ──噪声耦合──▶ 网络(去噪/推断)
   ▲                                            │                │
   │ 动作执行 (L8)                               │ x̂₀, 预测        │
   │                                            ▼                │
   └── 采样策略 ◀── EVC(L4.F4)/EFE(L8.F4) ◀── 规划=流形反向扩散     │
                                 │                              │
              误差 δ → 总线 → 资格迹结算 (L5) ── 学习回路 ──────────┤
              RPE → 评估器 (L4) → L3 调制场 ──── 价值回路 ──────────┤
              novelty/健康度 → L6 结构回路 ────────────────────────┤
              惊异度 → 缓冲优先级 → SWR 回放 (L7) ── 回放回路 ──────┘
```

- **学习回路**（误差反馈）：δ=x̂₀−x₀（F15）→ 误差总线 → 五选择压结算（L5.F12）→ 权重。反馈的本地代理 = 资格迹（延迟结算）。
- **价值回路**（评估器反馈）：结果 → RPE（L4.F2）→ 评估器更新；评估器经 L3.F6 广播 → 全局增益/唤醒调制输入。脱钩监控（L4.F9）检测评估器漂移 → 元选择（L4.F8）。
- **结构回路**（结构反馈）：novelty/健康度统计 → L6 静默突触激活/凋亡/分裂——数据驱动结构（DEV.F9 的数据尺度：过度产生=多样性，选择=文化选择，稳定=规范固化，退化=规范替换）。
- **回放回路**（缓冲反馈）：惊异度 → 保留优先级 → SWR 重放为内部事件（src='replay'）→ 再入网络。
- **闭环特例一：回放**——网络消费自己历史的重放（记忆巩固）。
- **闭环特例二：自生成**（DATA.F12）——网络自己的采样成为新事件源（自我引导；V7 具身闭环：动作 → 环境 → 事件 → 网络 → 动作）。这是认知原则 3「感知-行动-纠错循环」的完整闭合。

## 5. 发育时间表 × 事件流绑定

| 阶段 | 事件来源 | 反馈回路 | 温度/带宽 |
|---|---|---|---|
| Phase 0 种子 | 无（纯程序展开） | 无 | – |
| Phase 1 自发校准 | spontaneous | 局部 Hebb（无全局误差） | 高带宽、零外部 |
| Phase 2 经验塑形 | external + efference + cultural | 学习 + 价值 | 高多样性（关键期开放） |
| Phase 3 锁定 | 经验质量统计 | 结构（三锁安装） | 数据不足则延迟锁定（暗饲养） |
| Phase 4 维持 | replay + spontaneous | 回放 + SHY | 低带宽、巩固 |
| 持续 | external + self（主动采样） | 全回路 + 评估器元选择 | 昼夜节律调制 |

## 6. 函数规格（data.py）

### DATA.F1 `Event`（schema，见 §2.1）
- 特性：[frozen 不可变]；可序列化；schema 版本号字段；往返一致性（save/load 后逐字段相等）。

### DATA.F2 `EventSource`（接口）
- 接口：`sample(n, rng) -> list[Event]`；`stream(rng) -> Iterator[Event]`；`bandwidth: float`（事件率）；`clock: str`（时钟域）。
- 特性：[确定性(rng)]；事件驱动与批量密集共用此接口（形态切换不换接口）；source 注册表与 K4 同构。

### DATA.F3 `mixture_source(m: Mixture, rng, cfg) -> EventSource`（V0 环境）
- 算法：`x ~ 混合分布`；事件 = (t=0, src='external', mod='latent', loc=∅, x=x₀, cond={}, val=None, meta={novelty: −log p(x)})。
- 特性：[确定性]；novelty 用解析分数（F05 诊断通道）。

### DATA.F4 `replay_buffer(capacity, retention)`
- 接口：`push(events)`；`sample(n, rng)`；retention = novelty/difficulty 加权优先级（桶抽样）。
- 特性：[确定性]；容量上限（旧事件淘汰 = 数据尺度的选择）；供 L7.F6。

### DATA.F5 `event_encode(x, mod, encoders) -> e`
- 算法：按 mod 选择编码器；V0 默认恒等/线性。
- 特性：[纯函数]。

### DATA.F6 `manifold_project(e, mask) -> (z, mask)`
- 算法：e → z∈M；mask=0 的模态不参与（部分观测，子流形去噪）。
- 特性：[纯函数]；mask 全 1 = 完整观测；mask 全 0 抛错（无信息事件）。

### DATA.F7 `condition_channels(ev) -> (cond_vec, mod_mask, val_signal)`
- 算法：三通道拆分——cond_vec→生成器（L3.F2/L8.F1）；mod_mask→输入门控；val_signal→仅 L4。
- 特性：[纯函数]；val 绝不进入去噪路径（接口层硬隔离，测试断言）。

### DATA.F8 `noise_couple(z, t, sde, eps=None, rng=None) -> x_t`
- 算法：K.F2 包装（离散→连续之桥）。
- 特性：同 K.F2。

### DATA.F9 `acquisition_policy(state, evc, efe, temperature, rng) -> action`
- 算法：`action = argmin EFE（L8.F4）或 argmax EVC（L4.F4）` + temperature 噪声（探索）；temperature 受 L3.F5/L4.F5 调制。
- 特性：[确定性(rng)]；V0 退化版 = 均匀采样（temperature→∞）。

### DATA.F10 `feedback_router(delta, rpe, novelty, stats) -> routes`
- 算法：δ→L5 结算（学习）；rpe→L4（价值）；novelty/健康度→L6（结构）；惊异度→DATA.F4 优先级（回放）。
- 特性：[纯函数]；路由表可配置（弱调控连接：换目标不改事件方）。

### DATA.F11 `curriculum_schedule(difficulty, stage, cfg) -> temperature`
- 算法：难度轴（L4.F3）→ 采样温度/分布（易→难）；Phase 2 开放期高多样性、Phase 3 后收窄。
- 特性：[纯函数]。

### DATA.F12 `self_generation_source(net, sde, rng, cfg) -> EventSource`
- 算法：pc_sampler（K.F6）采样 → 事件（src='self'）。
- 特性：[确定性(rng)]；默认 enabled=False（V7 启用）。

### DATA.F13 `event_statistics(events, buffer) -> stats`
- 算法：novelty（−log p 或 ‖x−E[x]‖）；u_exp/u_unexp（L4.F3 口径）；difficulty（近期损失/错误率 EMA）。
- 特性：[纯函数][递推 EMA 状态]。

## 7. V0 落地与验收

- V0 接线：DATA.F3（MoG 源）→ DATA.F5/F6（恒等）→ DATA.F8（噪声耦合）→ 网络；反馈仅学习回路（δ→总线→结算）；val/cond/动作回路留空（接口就位、默认关闭）。
- 验收口径：
  1. [确定性]：同 seed 全事件流逐位复现；
  2. 部分观测：随机 mask 下训练可收敛（缺模态鲁棒性冒烟）；
  3. 回放回路：缓冲 push/sample 往返一致；retention 优先序正确；
  4. val 隔离：断言 val 通道在去噪路径中无梯度/无值传递；
  5. 反馈路由：δ 与 rpe 分别只到达 L5/L4（路由隔离测试）。
- V7 具身闭环接入时：仅新增 efference 源（L8.F1）与 acquisition_policy 的 EFE 模式——接口不变（形态承诺兑现的验证点）。
