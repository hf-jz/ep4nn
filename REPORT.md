# Cadence 判定与扩展报告

对象：微信公众号文章《彻底抛弃Transformer！302个神经元掀翻大模型》(图灵独白, 2026-09-25)
与 Pragma Research 开源神经系统架构 Cadence（本地 checkout `/Users/sheng/math/cadence`，v0.80.0+，
git head `0b0889e`；PyPI `cadence-net`）。
方法：读文章原文 + 读代码/文档/收据 + AutoCLI 联网核对 + 在本机跑实测（探针见 `probes/`）。

---

## 0 结论摘要

| 问题 | 判语 |
|---|---|
| 是否真的用线虫 302 神经元连接完成功能实现 | **接线是真的，功能不是从接线里长出来的**。拓扑、感觉/运动神经元映射逐条对得上；但符号、增益、响应律、身体映射、教学目标全部是人写的声明，且出生权重不是实测突触强度（0.50 版无安装参数接口）。 |
| 文章可信度 | **技术核心真，传播层失真**：授权协议写错（GPL→MIT）、数值未溯源（8.5e-12）、睡眠 8000 次→100% 语法在代码仓查无此数、AGI 定性被作者本人文档否定。 |
| 架构是否有提前优化的空间 | **有，且是数量级的**：(a) 稀疏图输运选择浪费 1.8x；(b) 不动点迭代未用加速，冷启动可省 2.5–4.2x；(c) 收缩证书用全局最坏界（本图为 q=88，实际局部谱半径 0.53），导致"算不动就拒绝"而不是"有预算保证地交付"。 |
| 能否扩展到更复杂控制任务 | **可以，但缺的是结构而不是旋钮**。实测：compose 出的均衡脑在 1 自由度连续控制上不跟踪，策略坍缩到单一动作（70–84%）；连续动作读出、目标端口、多时间尺度、长时程信用、学到的预测模型入环都是缺件。 |

成熟度判语：**概念层成熟 / 工程层半成熟（有 2–4x 余量）/ 控制层不成熟**。

---

## 1 文章逐条核实

证据来源标注为 `文件:位置`。文章原文来自浏览器抓取（`mp.weixin.qq.com/s/BMxzYd_MYRJh1nsTO3SdQg`）。

| # | 文章主张 | 代码/文档证据 | 判定 |
|---|---|---|---|
| 1 | 302 个线虫神经元、真实连接组、刚性掩码 | `cadence-examples/worm/data/SOURCES.md`：`c302_A_Full.net.nml` = 302 雌雄同体神经元、2,279 化学突触类、1,084 缝隙连接（openworm/c302 @`6cd861f8`，MIT），带 sha256；`worm/README.md:33,72-92`："每个神经元是一个单 patch column，声明输入恰为其突触，其他连接无法生长" | **真** |
| 2 | 气味 A→AWA、气味 B→AWC、伤害→ASH、前进 AVB/PVC、后退 AVA/AVD/AVE | `worm/data/params.json` `inputs`/`outputs` 与 `worm/README.md:87-91` 完全一致；本机从 `connectome.json` 复算：302 神经元，61 感觉/91 中间/118 运动/10 命令/22 咽部 | **真** |
| 3 | 学习只发生在既有突触的权重上 | `worm/README.md:69,83-86`：Cadence 0.50 无继承参数接口，出生权重=库的种子值×入连接数；实测突触计数与递质符号只用于绘图，"学习决定了连接做什么" | **真（但"既有突触权重"≠实测突触强度）** |
| 4 | 浏览器里边活边学，食物/针刺即课程 | `worm/web/brain.js`（0.50.0 参考修补的移植）+ `worm/tests/parity.mjs`（浏览器引擎 vs 库，3 次课程与整个生活，最坏相对差 **4.8e-8**） | **真** |
| 5 | "浏览器与 Python 相对误差 8.5e-12" | 仓库内找不到该数字；worm 的公开 parity 数字是 4.8e-8 | **无法溯源** |
| 6 | "以宽容的 MIT 协议完全开源" | `cadence/pyproject.toml:21-22` + `LICENSE`：**GPL-3.0**；MIT 的是归档示例仓 `cadence-examples/LICENSE`。第三方二手介绍（tau-home.com）也写成 MIT——错误正是从这里传播的 | **错** |
| 7 | 继承 Bengio 的均衡传播 (Equilibrium Propagation) | 论文 §1：数学成分的前例是坐标下降、近端推断学习、图上的预测编码、equilibrium learning[11,12]；库内契约是 **free/nudged 对比 + 参数锚定**（`docs/learning.md`）；302 线虫示例走的是 `observe_batch(source="estimate")` 的批量教学（`worm/README.md:106-125`），不是 EP 的 nudge 路径 | **半真（思想同族，非实现）** |
| 8 | "训练与推理没有区别" | `docs/world-model.md:141`：平静的脑贪心作答、什么都不学；青年/唤醒才采样、学习、写记忆。`Brain.live` 有 arousal 门 | **半真（无模式开关，但学习被事件门控）** |
| 9 | 睡眠 8000 次自洽做梦 → 语法归纳 100%（81.5%→90.5%→100%） | 睡眠/做梦机制真实存在（`docs/record-patch.md:339-395`；`cadence-examples/connect4/receipts/sleep-2026-09-22/REPORT.md`：切片符号准确率 0.55→0.755/0.80，一夜 4k 单元约 140–210 s/1000 行）。但 81.5/90.5/100 三个数字在库仓与示例仓都查不到（`benchmarks/keydoor/README.md:328` 的 0.815 是另一件事） | **机制真，数字无法核实** |
| 10 | "掀翻大模型/AGI 可行路径"、"智能不需要那么臃肿" | README:55 "Cadence is alpha research software"；`cadence-examples/README.md` "它们不建立生物保真、通用机器人能力或对 transformer 的匹配优势"；`docs/scaling.md:68` "没有建立通用的 Cadence 缩放律" | **被作者自己的文档否定** |
| 11 | 线虫在图里游动、身体由脑驱动 | `worm/README.md:68,102-104`：前进/后退读出只调制**工程控制器**；运动神经元不驱动肌肉，身体波与转向是身体自己的 | **半真（读数入环，执行外挂）** |
| 12 | 目前对外的线虫 demo | 公开 demo 页（`floatingpragma.io/demos/`）已无线虫：现在是斑马鱼眼动 + Platynereis 幼虫（`cadence-demos/connectome/`，2026-10-07 建）；线虫是 0.50.0 的版本化归档示例 | **文章说的 demo 已退役** |

---

## 2 判定：功能到底靠什么实现

把 302 线虫示例拆开看，各层来源如下（`worm/README.md:171-178` 的 "Supplied and learned" 是作者自己的口径）：

| 层 | 来源 | 说明 |
|---|---|---|
| 接线（谁连谁） | **实测** | c302 A_Full 化学突触 + 缝隙连接双向 + 自身上一 tick 的持续性项 |
| 递质符号 | 数据里有，但**不施加于脑** | 只决定绘图颜色；符号在库默认里来自 `sign` |
| 权重/偏置 | **学出来的** | 从种子值开始，12 次产前反射课 + 一生中的课程 |
| 神经元响应律 | 声明 | rate 单元，rectified sigmoid（`src/cadence/neuron.py:92`），非脉冲 |
| 增益 | 声明 | 默认 gain=1.0、dt=0.5 |
| 感觉映射 | 声明 | 哪个细胞闻什么 |
| 身体与运动 | **工程控制器** | 运动神经元不驱动肌肉 |
| 教学目标 | 声明 | `source="estimate"` 的构造目标（1.0 版承认这是估计而非观测见证） |
| 条件反射收据 | **尚缺** | `worm/README.md:192-194`：配对的/未配对/冻结/损毁对照的多种子收据"还没有" |

因此严格表述是：**这是一个在实测拓扑上做局部塑性学习的梯度速率网络**，不是"线虫神经系统实现的功能复现"。
"结构约束了它不能凭空长出通路"这句为真；但"生命如何感知世界，代码就如何运转"（文章原话）为假——感知、动力学、身体、目标都是人写的。

---

## 3 架构深入分析

### 3.1 三套并列实现，契约互不相同

| 车道 | 入口 | 数学对象 | 学习规则 |
|---|---|---|---|
| 默认脑 | `Brain.compose` / `live` / `step` | 神经电位 settling（式(10)） | free/nudged 对比 + 锚定（`src/cadence/learning.py:533`） |
| 记录补丁 | `RecordPatchNet` | 事件记录 + 慢权重 | delta 写记录 + 睡眠教学（`docs/record-patch.md`） |
| 群体求解器 | `cadence.experimental.equilibrium` | 群体能量式(2) | 局部坐标步 + 准入（论文 §3） |

三者共用"局部一致修补"的组织原则，但**证明与保证各自归属**，不能互相引用（论文 §1 "one hypothesis, two mathematical realizations"）。

### 3.2 本机实测的数值事实

环境：Python 3.13.2 + numpy 2.5.4 + scipy 1.18.1，`PYTHONPATH=<checkout>/src`，Apple 笔记本 CPU，单线程。
探针：`probes/probe_settle_cost.py`、`probe_accelerate.py`、`probe_endtoend.py`、`probe_certificate.py`、`probe_profile.py`。

**A. compose 出的脑，每个 tick 的成本**（`Brain.live` 一次 = 一次 settle + 奖励资格 + 记忆写）

| 结构 | 神经元 | 突触 | 每 tick | tick/s | 收敛步数 | 容差/残差 |
|---|---|---|---|---|---|---|
| modules=(16,8) | 44 | 524 | 1.83 ms | 547 | 32.5 | 0.003 / 4e-16 |
| modules=(64,32,16) | 140 | 6,028 | 2.00 ms | 499 | 28.3 | 0.003 / 3e-10 |
| modules=(256,128,64) | 524 | 88,588 | 4.36 ms | 229 | 32.0 | 0.003 / 1.6e-5 |

冷启动 vs 容差（140 神经元，`equilibrate`）：1e-3→32 步 / 1.04 ms；1e-4→64 步 / 2.01 ms；1e-5→96 步 / 3.01 ms；1e-6→128 步 / 4.52 ms。
同输入热启动：0 步 / 0.075 ms（`probe_settle_cost.py` "tolerance_sweep"）。**热启动余量已在，但输入一变就退回全价。**

**B. 302 线虫真连接组（从 `cadence-examples/worm/data/connectome.json` 重建）**

302 神经元 / 3,137 个有向突触类 / 10,133 接触 / 密度 3.44%（`probes/probe_worm_graph.py`）。
一次 settle 到 1e-4：32 步、残差 1.2e-5。

**C. 输运实现的选择与密度无关**（`probes/probe_transport_ab.py`，302 线虫图，同一驱动同一容差，300 次热身后计时）

| 量 | 块稠密（库默认；块项数 91,204 = 对 3,137 条边浪费 29x） | scipy CSR（`dense_limit=0`） | 比 |
|---|---|---|---|
| 单次输运调用 | 17.3 µs | 4.7 µs | 3.7x |
| 残差检查 | 36.3 µs | 22.1 µs | 1.6x |
| 32 步 chunk（热启动） | 880.6 µs → **27.5 µs/sweep** | 486.9 µs → **15.2 µs/sweep** | 1.81x |
| 一次完整 settle（同一 32 步） | 986.9 µs | 554.0 µs | **1.78x** |

另测（`probes/probe_profile.py` cProfile）：524 神经元 / 88,588 突触的 compose 图（块内密度 32%）反而是块稠密更快
（一次 32 步 chunk：72.8 µs/sweep）。**所以正确做法是按密度选路径，而不是按 `layout.size` 选**
（`src/cadence/brain.py:449`：`blocked = self.layout.size <= dense_limit * dense_limit`）。
注：早期 `probes/probe_certificate.py` 的 settle 计时（1.28 vs 6.26 ms）在同图上给出相反次序，是本进程内测量顺序造成的假象；
以本表的受控 A/B 为准。

**D. 全局收缩界 vs 局部（活动集）界**（`probes/probe_certificate.py`，302 图，库默认动力学 gain=1.0/dt=0.5）

| 量 | 值 |
|---|---|
| 全局最坏界 q = 1-dt+dt·max_i Σ_j|W_ij|·max s' | **88.0**（292/302 行 ≥1，证书失败） |
| 局部 q = 1-dt+dt·max_i Σ_j|W_ij|·s'_j(v_j)（实际状态） | 1.186（1/302 行 ≥1） |
| 局部雅可比 K=(1-dt)I+dt·W·diag(s') 的谱半径 | **0.529** |
| 加权见证（正向量 h 使 Kh ≤ αh） | α=0.696，h∈[2.89, 11.08] |

结论：论文的 Prop 3.8/3.11 在全局最坏界上对本图无效，但在**当前状态的活动集**上有证书。
这把"算不动就拒绝"变成"预算可预测"——控制回路需要的正是后者。

**E. 不动点迭代的加速余量**（`probes/probe_accelerate.py`，140 神经元，同一残差判据）

| 容差 | 朴素迭代 | Anderson(m=5, 单调保护) | 迭代比 | 时间比 |
|---|---|---|---|---|
| 1e-3 | 28 | 11 | 2.55x | 2.25x |
| 1e-5 | 79 | 19 | **4.16x** | 2.82x |
| 1e-6 | 104 | 30 | 3.47x | 2.54x |
| 热启动小扰动 @1e-5 | 32 | 12 | 2.67x | — |

端到端（`probe_endtoend.py`，把 `Brain.equilibrate` 换成 Anderson + 原求解器兜底）：80 个 tick 的 sweep 均值 28.8→22.25，资格全通过；
但**墙钟反而略慢**（1.98→2.14 ms/tick），原因是在 140 神经元规模上每 sweep 的 Python 调用开销占比高、保护步要额外一次映射。
⇒ 加速必须实现在库内部的热循环里（或配合 `CADENCE_FUSED`/numa、torch 驻留），在探针层套壳只能省迭代数、省不了时间。

### 3.3 提前优化清单（按 收益/风险 排序）

| 优先级 | 优化 | 实测收益 | 代价/风险 |
|---|---|---|---|
| P0 | **输运按密度选**（CSR vs 块稠密 vs reduceat 三选一，用 nnz/n² 与块填充率判定） | 稀疏连接组 1.8x/sweep；稠密图保持现状 | 小；需在 `Brain.__init__` 加一次密度估计与回归测试（fused 路径要保留） |
| P0 | **不动点迭代加速**（Anderson(m)/拟牛顿，保留原方程残差资格） | 冷启动迭代 2.5–4.2x | 中；必须带单调保护 + 原求解器兜底，且要复核 `_repeated_state` 停滞判据 |
| P0 | **活动集收缩证书 → 每 tick 预算界**（用 K=(1-dt)I+dt·W·diag(s') 的加权见证给 k≈log(ε/δ)/log(α)） | 本图 α=0.696 有证书（全局界 88 无证书）；把"拒绝"变成"预算+兜底" | 中；需定义证书失效时的降级语义 |
| P1 | 起用 `CADENCE_FUSED`（编译热循环）/ torch、mlx 后端 + 状态驻留 | 本机未装 numba（`_FUSED=false`），未量化；524 神经元起收益显著 | 低；后端切换需工作负载一致性检查（文档已要求） |
| P1 | 热启动复用（入不变则 0 sweep，已具备；补"变化小则短预算"的策略） | 稳态 tick 可从 32 sweep 降到个位数 | 低 |
| P2 | temporal 引擎的稠密消息 O(N·T·H³)（`docs/scaling.md:52-58`）改矩阵无关/低秩 | 未测 | 高；改的是另一条车道的数学 |

### 3.4 不是调参能解决的（结构问题）

1. **只有不动点**：`PatchNet._validate_structure` 要求 `ep_structure(...).compatible` 且无 adaptation（`src/cadence/patch.py:169-174`），而 `src/cadence/neuron.py:14-16` 明说 adaptation 是"让互抑连接组产生节律而非不动点"的机制——**节律被学习契约排除**。
2. **拒绝语义**：动作不收敛就不发（`docs/contracts.md`、`docs/continuous.md`），控制回路不能有"这一拍没有动作"。
3. **单标量奖励 + 离散动作槽**：见 §4.1 实测（策略坍缩）。
4. **反馈跨 tick**：一个 tick 内只有顺突触距离方向的连接是活状态，反向要等下一 tick（worm 的声明顺序 hack）——多时间尺度不是一等公民。

---

## 4 扩展到更复杂控制任务

### 4.1 先看现状：一个真实的负结果

探针 `probes/probe_control_plant.py`：被控对象 m·x″ = u − k·x − c·x′（k=2, c=1, m=1，欠阻尼），
观测 = [误差, 误差率, 参考, 1]，动作 = 5 档力（±4, ±2, 0），奖励 = 上一步实测结果（稠密/稀疏两种），
6000 个 tick，`Brain.compose(inputs=4, actions=5, modules=(64,32,16), arousal=True)`。

| 臂 | 平均 |误差| | 末四分之一 |误差| | 命中带(±0.25) 末四分之一 | 末四分之一动作分布 |
|---|---|---|---|---|
| 均匀随机 | 1.020 | 1.033 | 0.000 | [.225 .185 .194 .199 .197] |
| 冻结贪心（arousal 关） | 1.011 | 1.067 | 0.467 | [0 0 0 **1.0** 0] |
| `live`（唤醒门控学习，稀疏奖励） | 0.880 | 0.855 | 0.449 | [.045 .053 .038 **.824** .040] |
| `live`（稠密奖励，seed 5） | 0.923 | 0.919 | 0.209 | [.241 .017 .019 **.705** .017] |
| `live`（稠密奖励，seed 11） | 0.856 | 0.815 | 0.459 | [.043 **.843** .040 .039 .035] |
| `step`（每个结果都学，稠密奖励） | 1.016 | 1.007 | 0.000 | [.009 .004 **.980** .007] |

读法：**误差只降了约 13%（1.02→0.88），命中带没有一项超过冻结基线，末四分之一动作分布坍缩到单一动作（70–84%，`step` 臂 98%）。**
即：这个脑学到的不是控制器，而是"一直往一个方向推"。这与我给的奖励是否稠密无关（两个 seed 都坍缩到同一个极端力）。
原因是结构性的：动作是对已沉降状态取 argmax 的离散槽，奖励路径只带一个标量、资格窗口 12 步（`describe()` 的 `actor_eligibility_steps`），
而对象的闭环时间常数是 1–2 s（50–100 tick）——没有任何机制把"误差符号 → 反向动作"这条关系写进图里。

对照：库自己的控制向 demo **Rover Lab 用的是学到的后果模型 + 应用侧控制器**，其 README 明确写
"它不实现也不证明完整的 `Brain.compose` 记忆/修补协议"。⇒ 均衡脑**至今没有被证明能自己闭合控制回路**。

### 4.2 扩展设计（按依赖排序，前三项已给出可测指标）

**E1 数值与预算层（不改模型语义）**
- 密度感知输运 + 迭代加速 + 活动集收缩证书（§3.3 P0）。
- 新语义：`settle(deadline=...)` → 用证书给出 k 步预算；预算耗尽时**交付上一个合格动作**并打收据（`stale=True, age=Δs`），
  而不是抛"no action issued"。控制回路的正确性要求是"永远有动作 + 偏差可界"，不是"动作必须几何收敛"。
- 指标：tick 超时率、每 tick 步数 p95、跟踪误差、staleness 分布。

**E2 接口层：连续动作与目标端口**
- 连续动作读出：不用 argmax 槽，用电机群体的沉降状态经**声明度量**线性读出（`Brain` 已能读 motor 群体，缺的是"声明度量 + 平滑惩罚"）。
- 目标端口：把 setpoint 作为**边界量**（固定输入）而不是 nudge——对应 `TemporalPatchNet.plan` 里已有的 target 语义，
  但要让 `Brain.compose` 原生支持，使"同一个脑、不同目标"成立。
- 惩罚项：动作变化率进入奖励（现实控制的抖动代价），否则会学到 bang-bang。
- 指标：同一脑在 3 个不同 setpoint 上免重训的跟踪误差。

**E3 动力学层：从"不动点"升到"吸引子"**
- 打开 adaptation（`NeuronModel(adaptation=Adaptation(tau_steps, strength))`），把契约从"残差→0"换成
  "返回映射（Poincaré）收缩"：相位与周期可读，节律成为一等对象。
- 这一步是升维的关键：现在库只能回答"答案在哪"，不能回答"轨迹是什么"。复杂控制（腿式、抓取、振荡设定点）需要后者。
- **实测状态：未证实。** `probes/probe_rhythm.py` 在同一 302 图上用恒定驱动跑 1500 步，四组 adaptation 参数
  （τ=50/s=1、τ=20/s=2、τ=100/s=0.5 及关闭）全部收敛到不动点（残差 1e-14~3e-7，末段每 100 步活动位移 0）。
  即"打开 adaptation 就有节律"只是 `src/cadence/neuron.py:14-16` 的文档表述，不是这张图的行为；
  节律需要互抑结构 + 合适的增益/时间常数配比，本图（294/302 活跃、增益 1.0）不构成该条件。
- 数学上要重新证：参与式里 equilibrium 的 free/nudged 对比在有节律时给出什么量（很可能退化为相位响应曲线 PRC 的对偶）。

**E4 信用与分层**
- 资格窗口 12 → 与控制周期同阶（按 `gamma/lam` 或按采样周期声明），并让信用按区域记账（observers 的伴随遍历天然给出分层路径，代价是 O(depth)）。
- 分层 observers：上层读下层的误差与状态，只对下层"失败的那一类"发修补（对应文档里尚未实现的 failure-gated cheap operation）。

**E5 预测模型入环**
- `TemporalPatchNet` 已能学后果并私有规划，但不在 `Brain.compose` 里。扩展路径：把它做成脑内一个端口（同一耦合均衡），
  而不是旁边的独立引擎；先把 `O(N·T·H³)` 的稠密消息换成矩阵无关/低秩，否则长时间视界不可行。

**落地顺序建议**：E1（一周级，纯数值与接口）→ E2（接口，需重跑对照）→ E3/E5（研究级，需要新的数学契约）→ E4（随 E2/E3 一起）。

---

## 5 判语（是否成熟）

- **概念层：成熟。** 局部修补→全局均衡的构造清楚、契约严格（残差资格、拒绝不改状态、锚定变更、经验来源区分），
  且论文把"已证结论"和"私人想象"分得很干净（§12、`docs/missteps.md`）。
- **工程层：半成熟。** 数值路径有 2–4x 可拿的余量（稀疏输运、迭代加速、证书），而当前默认把稀疏图当稠密算、
  用全局最坏界换掉了可用的局部界；这不是理论缺陷，是实现选择。
- **控制层：不成熟。** 没有闭环控制的公开证据；库自己的控制 demo 用外部控制器；本机实测 compose 脑在 1-DOF 跟踪上不成立（策略坍缩）。
- **文章层面：核心真、传播失真。** 302 线虫接线、刚性掩码、边活边学、睡眠/做梦机制都真；
  协议（GPL→MIT）、数字（8.5e-12、100% 语法）、能力定性（"掀翻大模型/AGI 路径"）都不是代码支持的说法。
  作者自己的 README/MANIFESTO/示例仓 README 与文章口径冲突——**冲突的一侧是文章，不是代码**。

---

## 6 复现与文件

```bash
# 环境（一次性）
/opt/homebrew/bin/python3.13 -m venv ~/.hermes/cache/scratch/cad-venv
~/.hermes/cache/scratch/cad-venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy

cd /Users/sheng/math/ep4nn
export PYTHONPATH=/Users/sheng/math/cadence/src
P=~/.hermes/cache/scratch/cad-venv/bin/python
$P probes/probe_settle_cost.py     # tick 成本与容差
$P probes/probe_accelerate.py      # Anderson 加速
$P probes/probe_endtoend.py        # 端到端 tick 对比
$P probes/probe_certificate.py     # 全局界 vs 活动集证书
$P probes/probe_profile.py         # 块稠密 vs CSR 逐 sweep
$P probes/probe_worm_graph.py      # 302 线虫真连接组
$P probes/probe_transport_ab.py    # 块稠密 vs CSR 受控 A/B
$P probes/probe_control_plant.py   # 控制回路（负结果）
```

数据依赖：`data/worm_connectome.json`（sha256 `2a7b117f…`；从 `muellerberndt/cadence-examples` 的
`worm/data/connectome.json` 取，原始来源 openworm/c302 @`6cd861f8`，sha256 见 `worm/data/SOURCES.md`）。

外部核对（AutoCLI `google search`）：C. elegans 全脑建模的同期工作（Zhao 2024 数据驱动整合模型、
NeuroSimWorm 2025、OpenWorm）、连接组驱动机器人的前例（arXiv 2011.09057 "Linking the connectome to action"）、
以及二手介绍把 Cadence 写成 MIT 的传播源（tau-home.com）。

---

## 7 判语：这条仿生神经元路线能否实现 AGI

**不能——以当前形态、以及当前可见的数学路径看，它不构成通向 AGI 的路线。它可能是有价值的组件，不是路线。**

三条支撑：

1. **"均衡"本身不新增算力。** 均衡求解 + 隐式微分在深度学习里已有等价物（深度均衡模型一类）。把前向传播换成整图沉降，
   换掉的是"答案怎么形成、梯度怎么来"，不是"模型能表示什么"。当前可见的独占收益是**空间**上的信用局部性（一层不回传），
   而 AGI 缺的主要是**时间**上的信用分配与表示学习 —— 这两项这条路线没有给新东西：资格窗口 12 步（`describe()` 的
   `actor_eligibility_steps`），预测模型还挂在旁边一条车道上（`TemporalPatchNet` 未接入 `compose`）。
2. **把生物的结构当成算法。** 302 神经元的局部性之所以够用，是因为它面对的世界小（趋化、避害、进食），
   而不是因为它的学习规则强。生物脑的"局部"是 860 亿神经元 + 数十年发育 + 演化先验之上的局部；
   拿 302 当尺度论证，方向反了。库自己也写"生物名字只是软件角色的命名"，作者在 MANIFESTO 里明确
   把语言与可扩展性列为**尚未建立**的目标。
3. **实测不支持能力主张。** 本报告的测量：1 自由度连续跟踪上策略坍缩到单一动作（70–84%）、命中带不如冻结基线；
   1-DOF 的 4x 数值余量和 1.8x 输运余量说明工程层还没到位；只有不动点语义排除节律，硬性排除了一整类控制与认知动力学。
   作者自己的 `docs/scaling.md`：没有建立缩放律；示例仓 README：不建立生物保真、机器人能力或对 transformer 的匹配优势。

翻盘条件（可证伪，三者任一成立我都会改判）：
① 在匹配信息与算力下，某个**持续学习**基准上超过反向传播的在线学习器（保留旧技能 + 学新任务，不是单任务均值）；
② 语言不是 next-token 拟合，而是从一个生命经历里长出组合泛化（作者自己的第 173 号目标是这个）；
③ 宽度/经验/时间任一维上出现可测的缩放律。

现在的状态：三条都空。价值在于它反主流地认真做了**别的**事——在线适应、低功耗嵌入、可读沉降、不遗忘的世界模型
（Rover Lab 那条"学后果模型 + 外部控制器"才是真有产业形状的部分）。作为"掀翻大模型"的叙事是流量，
作为嵌入式持续学习的工具箱是正经活。
