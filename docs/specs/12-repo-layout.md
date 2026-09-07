# 代码目录树与函数位置规划（Repo Layout v0.1）

- 定位：specs/ 全部函数规格（00–11）的落位方案——目录树、文件名、函数→文件映射、import 依赖 DAG、版本归属。**取代 project-decomposition §8 的示意树**（其通道角色映射不变）。
- 硬约束来源：00-kernel K3（命名：函数 ID 形如 L0.F1；实现文件 src/bio_net/<layer>.py；状态一律 buffer）；08-runtime 阶段一（Python+PyTorch 密集形态；每层一个融合步函数）；三函数契约（init/step/update + schedule/lock）。

## 1. 目录树（通道角色标注）

    project/
    |-- README.md                      # 5 分钟启动指南
    |-- pyproject.toml                 # 依赖（生态位通道）
    |-- docs/                          # 文化通道：愿景/方法论（architecture-vision、neural-construction-iteration）
    |-- specs/                         # 结构通道：接口契约 00–12
    |   +-- cards/                     # 模块卡片注册表（每卡一文件，V0 序列见 §6）
    |-- src/bio_net/                   # 参数通道：全部机制实现（扁平包）
    |   |-- __init__.py                # 包门面 + 纪律执行点 + 版本锚（内容规格见 §8）
    |   |-- sde.py                     # K.F1–F7
    |   |-- clock.py                   # K.F8–F9 + nominal_dt 契约
    |   |-- registry.py                # K.F10 + MechanismSpec + LockState（锁持久化）
    |   |-- development.py             # DEV.F1–F9（种子展开 + 自校准 + 元迭代）
    |   |-- l0.py  …  l9.py            # 分层实现（映射见 §2）
    |   |-- l8c.py                     # L8C.F1–F8 小脑模块
    |   |-- data.py                    # DATA.F1–F13 事件子系统
    |   |-- energy.py                  # E.F1–F8 能量账本（接地层）
    |   |-- curriculum.py              # C.F1–F6 课程阶段机
    |   |-- net.py                     # NetSpec 组装 + 三函数契约分发（init/step/update/schedule/lock）
    |   +-- scheduler.py               # T0–T5 时钟域主循环（V0 训练循环入口）
    |-- eval/                          # 验收台：唯一允许 .backward() 的目录（K3 纪律 2）
    |   |-- acceptance.py              # D4 六指标
    |   |-- baseline_backprop.py       # backprop 基线
    |   |-- ablations.py               # 消融钩子驱动
    |   +-- run_v0.py                  # V0 验收运行脚本
    |-- tests/                         # 逐文件单测（函数 ID 1:1，文件名 = test_<file>.py）
    |-- configs/                       # 程序通道：种子/剂量/超参
    |   |-- v0.yaml                    # nominal_dt、网络规模、阈值、评估器配置
    |   +-- doses.yaml                 # 五选择压权重阶梯（V0 剂量协议，07-build-order）
    +-- locks/
        +-- lock_state.json            # 三锁登记（注册表持久化，L6.F2）

## 2. 函数 → 文件映射（版本归属）

| 文件 | 函数 | 时钟域 | 版本 |
|---|---|---|---|
| sde.py | K.F1 sde_marginal · F2 sde_forward · F3 sde_score_target · F4 reverse_euler_maruyama · F5 langevin_corrector · F6 pc_sampler · F7 dsm_loss | T0 | **V0** |
| clock.py | K.F8 Clock/NestedClock · F9 clock_domains + nominal_dt 契约 | 全部 | **V0** |
| registry.py | K.F10 register_mechanism + MechanismSpec + LockState | T3/T4 | **V0**（基础设施：L6.F2 锁持久化依赖） |
| development.py | DEV.F1 expand · F2 position_field · F3 temporal_identity · F4 chemoaffinity_connectivity · F5 canalization_sample | T4 | **V0** |
| development.py | DEV.F6 retinal_waves · F7 linsker_hebb · F8 self_calibrate | T2 | **V0**（Phase 1） |
| development.py | DEV.F9 meta_iteration | T3 | V4 |
| l0.py | L0.F1 node_activation · F2 refractory_mask · F3 dendritic_and · F4 divisive_normalize · F5 energy_cost · F6 canalized_params · F7 intrinsic_plasticity | T0（F7=T2） | **V0** |
| l1.py | L1.F1 electrical_skip · F2 ampa_channel · F3 nmda_gate · F4 mglur_integrator · F5 release_probability · F6 stochastic_release · F7 stp_step · F8 delay_line · F9 synapse_forward | T0 | **V0** |
| l1.py | L1.F10 phase_multiplex（占位） | T0 | V6（由 l9 实现） |
| l2.py | L2.F1 low_rank_tract · F2 tract_grad_projection | T1 | **V0**（最小形态） |
| l2.py | L2.F3 myelin_rank_schedule · F4 wire_cost · F5 ephaptic_coupling | T3 | V2（F5 默认关） |
| l3.py | L3.F1 modulation_field_update · F2 filminject · F3 paired_receptor_modulation · F4 hpa_feedback · F5 circadian_gate · F6 evaluator_broadcast | T2/T5 | V3 |
| l4.py | L4.F1 value_estimate · F2 td_error | T1 | **V0**（最小 TD 评估器） |
| l4.py | L4.F3 difficulty_axis · F4 evc · F5 serotonin_modulate · F6 gono_gate · F7 dual_controller · F8 evaluator_registry · F9 decoupling_monitor | T1/T2 | V5 |
| l5.py | L5.F1 three_factor · F2 eligibility_trace · F3 eligibility_dual · F4 hebbian · F5 rpe_settle · F6 metabolic_regularization · F7 synaptic_scaling · F8 bcm_slide/bcm_update · F9 vogels_update · F10 retrograde_signal · F11 five_pressures · F12 settle_learning | T0 递推/T1 结算/T2 | **V0**（完整五选择压） |
| l6.py | L6.F2 install_lock/remove_lock | T3 | **V0**（三锁） |
| l6.py | L6.F3 critical_period_state | T3 | V2 |
| l6.py | L6.F1 ltp_ltd · F4 silent_pool · F5 awaken · F6 health_score · F7 soft_apoptosis · F8 recycle_pool · F9 expansion_budget · F10 split_unit · F11 sprout_bypass · F12 retention_channels | T3/T4 | V4 |
| l7.py | L7.F1 astrocyte_gain · F2 microglia_tag · F3 oligo_myelin · F4 sleep_carriers · F5 nested_gate · F6 swr_replay · F7 shy_downscale · F8 glymphatic_flush | T4 | V8 |
| l8.py | L8.F1 efference_copy · F2 forward_model · F3 cpg_phase · F4 expected_free_energy | T0/T1 | V7 |
| l8c.py | L8C.F1 granule_expand · F2 purkinje_readout · F3 climbing_fiber · F4 complex_spike_settle · F5 simple_spike_readout · F6 golgi_feedback · F7 nucleo_olivary_homeostasis · F8 zebrin_channels | T0/T1 | V7 |
| l9.py | L9.F8 dt_ladder | T0 | **V0**（表 + 断言，供 K2 nominal_dt 契约查询） |
| l9.py | L9.F1 band_oscillators · F2 phase_precession · F3 theta_sequence · F4 pac_coupling · F5 phase_window · F6 spectral_split · F7 phase_gated_lr | T0 | V6 |
| data.py | DATA.F1 Event · F2 EventSource · F3 mixture_source · F5 event_encode · F6 manifold_project · F7 condition_channels · F8 noise_couple · F10 feedback_router | T0/T1 | **V0** |
| data.py | DATA.F4 replay_buffer | T4 | V8 |
| data.py | DATA.F9 acquisition_policy · F12 self_generation_source | T1 | V7 |
| data.py | DATA.F11 curriculum_schedule · F13 event_statistics | T2 | V5 |
| energy.py | E.F1 energy_pool · F2 energy_inflow · F3 energy_cost · F4 energy_settle · F5 energy_apoptosis · F6 energy_approve · F7 reward_to_energy · F8 ground_signal | T0/T1/T3 | **V0**（自 V0 起随 L5 结算启用，10-energy §5） |
| curriculum.py | C.F1 modality_stage（V0 恒等 'A' 占位） | T4 | **V0 占位** |
| curriculum.py | C.F2 cross_modal_align | T1 | 阶段 A（V0 后、V5 前） |
| curriculum.py | C.F3 concept_condition · F4 concept_readout · F5 install_value_source · F6 critical_period_stage | T1/T3 | V5（阶段 B） |
| net.py | NetSpec 组装（DEV.F1 产物 → 各层 state 装配）；step/update/schedule/lock 四路分发；SynapseState 聚合 | 全部 | **V0** |
| scheduler.py | T0 主循环（前向/递推）、T1 结算、T2 慢场、T3 结构、T4 周期、T5 昼夜的触发与调用表 | 全部 | **V0** |

## 3. import 依赖 DAG（无环 · 弱调控连接）

    sde / clock / registry（kernel：不 import 任何人）
        ▲
        |-- l0 · l1 · l2 · l3 · l9        （只 import kernel）
        |-- l5                            （import kernel + l2 投影；锁查询走 registry，不 import l6）
        |-- l4                            （import kernel + energy）
        |-- l6                            （import kernel + l5 BCM 核 + energy）
        |-- l7                            （import kernel + l5 睡眠档 + l2 秩调度）
        |-- l8 / l8c                      （import kernel + l0 归一化）
        |-- development                   （import kernel）
        |-- data                          （import kernel）
        +-- energy                        （import kernel + l0/l2 成本算子 + l5 资格迹只读）
              ▲
    curriculum（import kernel + energy 接地 + l4 注册表 + l6 锁计划）
    net（组装全部，四路分发）
    scheduler（import net/data/energy/curriculum，唯一主循环）

硬规则：

1. **无环**：任何反向 import 即退回（K6 协议第 4 点）；
2. **跨层只读白名单**：L5.F12 可读 L2.F2 投影与 registry 锁状态；energy 可读 l5 资格迹（同迹双币）；l6 可复用 l5.F8 BCM 核；l7.F3 包装 l2.F3；l8c.F1 复用 l0.F4。其余跨层通信一律走 state/signals（dict），不走 import；
3. **eval 隔离**：.backward() 只出现在 eval/（K3 纪律 2）；核心包内任何文件不得调用 autograd；
4. **区室化**：一机制一函数、一文件一层；层内重构不影响接口（弱调控连接）。

## 4. 每层文件模板（融合步函数边界，08-runtime §5.2）

每个 layer 文件必须导出：

- 规格函数（按 ID 命名，特性标记照抄）；
- init_layer(cfg) -> LayerState（T4 构建期；状态 dataclass 全 buffer；reset()）；
- step_layer(state, x, clocks) -> (y, state')（T0 融合步函数——唯一前向入口，内部按 L*.F9 式组合件编排）；
- update_layer(state, signals) -> state'（仅 T1+ 有结算的层：l2/l4/l5/l6/l7）；
- schedule_layer(state) -> ops / lock_layer(state, kind) / unlock_layer(state, kind)（仅 l2/l6/l7）。

net.py 只做四路分发（init/step/update/schedule/lock）与组装，不含机制逻辑——机制全部住在各自文件。

## 5. 调度器契约（scheduler.py 调用表）

| 时钟 | 周期 | 调用 |
|---|---|---|
| T0 | 每推理步 | 各层 step_layer（L0→L1→L2→主干，l9.F8 断言 dt 档） |
| T1 | 每 K 步 | l5 结算（settle_learning）+ l4 TD + E.F2/F4 + DATA.F10 路由 |
| T2 | 每 N 步 EMA | l3 场更新 · L0.F7 IP · L5.F7 缩放 · E.F3 记账 |
| T3 | 每 K epoch | L2.F3 秩调度 · L6 锁/结构 · DEV.F9 元迭代 · E.F5/F6 |
| T4 | 每轮训练 | l7 睡眠三嵌套/SHY/回放 · DEV.F1 展开（构建期一次） |
| T5 | 训练周期级 | l3 昼夜门控 · 学习率/探索率调制 |

## 6. specs/cards/ 卡序列（V0，执行顺序 = 开发计划）

| 卡 | 内容 | 落位 |
|---|---|---|
| v0-00 | eval 验收台骨架（D4 六指标 + backprop 基线） | eval/ |
| v0-01 | 种子展开（DEV.F1–F5） | development.py + configs/v0.yaml |
| v0-02 | 除法归一化（L0.F4） | l0.py |
| v0-03 | 折叠式皮层块（D2：主干 + 预测头 + 隐式误差） | net.py + l0/l1 |
| v0-04 | 快慢双通路（L1.F2/F3/F4） | l1.py |
| v0-05 | 资格迹 + 误差总线 + 三因素（L5.F2/F5/F1 + K.F7） | l5.py + sde.py |
| v0-06 | 自校准（DEV.F6–F8，Phase 1） | development.py |

## 7. 决策记录（与既有文档的关系）

- 本文件取代 project-decomposition §8 的示意树；通道角色映射不变（参数=代码提交、文化=docs、结构=specs、程序=configs/seed、生态位=pyproject）。
- **L5↔L6 循环的解法**：锁状态持久化在 registry（K.F10/LockState），l5 结算时只查 registry、不 import l6——import DAG 因此无环。
- **L9.F8 提前到 V0**：nominal_dt 契约（00-kernel K2）需要 dt 阶梯表，故 L9.F8（表 + 断言）归 V0 基础设施，其余 L9 仍在 V6。
- **K.F10 提前到 V0**：V0 含 L6.F2 三锁，锁需持久化载体，注册表随 V0 落地。
- 五选择压剂量阶梯（V0 剂量协议）由 configs/doses.yaml 承载，消融脚本在 eval/ablations.py。

## 8. src/bio_net/__init__.py 内容规格

定位：包的公共门面 + 纪律执行点 + 版本锚。**只允许急导入 kernel 三件套（sde/clock/registry）**；其余模块经 PEP 562 延迟访问——保证 import bio_net 零环、轻量（弱调控连接在包级的落点）。

### 8.1 必须导出（__all__ 白名单）

| 符号 | 来源 | 语义 |
|---|---|---|
| __version__ | 本文件 | 包版本（与 specs 版本同步：specs v0.1 → 0.1.0） |
| BUILD_SPECS | 本文件 | 构建依据快照 dict：{"architecture": "v0.2.2", "specs": "v0.1", "repo_layout": "v0.1"}——提交时与 docs/ 对照的版本锚 |
| register_mechanism / MechanismSpec / LockState | registry | K4 注册表与 L6.F2 锁状态的唯一入口（再导出） |
| CLOCK_DOMAINS / validate_nominal_dt | clock | T0–T5 常量与名义步长校验（config 校验用） |
| build_net / save_net / load_net | net（延迟） | 三函数契约的组装入口与五通道 params 序列化 |
| update_param | 本文件 | **唯一合法的权重修改器**：param.data += delta；断言 delta 无 grad、param.grad is None（K3 纪律 1 的机械执行点） |
| new_generator | 本文件 | 唯一合法的 generator 工厂（K3 纪律 3：无隐式随机） |
| bio_no_grad | 本文件 | torch.no_grad 的纪律化别名（T1+ 结算必须包住） |

### 8.2 纪律执行点（K3 三纪律的代码形态）

    from contextlib import contextmanager

    def update_param(param, delta):
        """T1+ 结算的唯一权重写入通道。"""
        assert not delta.requires_grad, "生物路径禁止梯度进入 delta"
        assert param.grad is None, "生物路径参数不得持有 grad（.backward() 只在 eval/）"
        param.data.add_(delta)

    def new_generator(seed: int) -> torch.Generator:
        """唯一 generator 工厂——包内禁止 torch.rand* 无 generator 调用。"""
        return torch.Generator().manual_seed(seed)

    @contextmanager
    def bio_no_grad():
        """结算/结构/维护通道的纪律上下文。"""
        with torch.no_grad():
            yield

### 8.3 延迟导入协议（PEP 562）

    import importlib

    _LAZY_MODULES = ("development", "l0", "l1", "l2", "l3", "l4", "l5", "l6",
                     "l7", "l8", "l8c", "l9", "data", "energy", "curriculum",
                     "net", "scheduler")
    _LAZY_SYMBOLS = {"build_net": "net", "save_net": "net", "load_net": "net"}

    def __getattr__(name):
        if name in _LAZY_MODULES:
            mod = importlib.import_module(f"bio_net.{name}")
            globals()[name] = mod
            return mod
        if name in _LAZY_SYMBOLS:
            mod = importlib.import_module(f"bio_net.{_LAZY_SYMBOLS[name]}")
            return getattr(mod, name)
        raise AttributeError(name)

规则：kernel 三件套（sde/clock/registry）在 __init__ 顶部**急导入**（它们是根，零内部依赖）；其余全部延迟。包内互引一律用 `from bio_net import l2`（走本协议）或 `import bio_net.l2`（标准子模块导入）；**禁止 from bio_net.l2 import xxx 直接抓函数跨层**（跨层白名单见 §3）。

### 8.4 版本同步规则

- 每次 specs/ 或 docs/ 结构性变更：更新 BUILD_SPECS 并 __version__ 小版本 +1；
- K6 提交时附 __version__ 与 BUILD_SPECS 快照（版本可追溯）；
- eval/ 启动时校验 BUILD_SPECS 与 docs/specs 文件头版本一致，不一致告警（不阻断）。

### 8.5 验收测试（tests/test_init.py）

1. import bio_net 后：sde/clock/registry 已在 sys.modules；l5 等延迟模块尚未加载（延迟导入生效）；
2. 白名单：__all__ 与 8.1 表一致；__getattr__ 未知名字抛 AttributeError；
3. update_param：delta 带 grad → 抛；param.grad 非 None → 抛；正常路径 param.data 变化正确且逐元素等于手工重算；
4. new_generator 确定性：同 seed 两个 generator 产出同序列；
5. bio_no_grad：退出后 torch.is_grad_enabled() 恢复原状态；
6. BUILD_SPECS 与 docs/specs 文件头版本对照（读文件断言）。
