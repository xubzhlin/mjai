# 四川麻将血战到底 RL 项目实施计划

> 项目：Sichuan Mahjong Self-Play RL (XueZhanDaoDi 血战到底)
> 架构：Rust 引擎核心 + PyTorch RL 训练 (PyO3 绑定)
> 参考：Suphx 论文、Mortal 项目、Mahjong-AI 项目

***

## 一、项目总览

### 1.1 目标

构建一个四川麻将（血战到底）强化学习自博弈智能体，包含：

1. **高性能麻将引擎**（Rust）：完整实现换三张、定缺、碰杠胡、血战到底、番型计分等全部规则
2. **神经网络模型**（PyTorch）：ResNet 特征提取器 + Dueling DQN 决策网络 + GRP 奖励预测器
3. **自博弈训练系统**：Server-Client 分布式架构，支持多客户端并行自博弈 + 在线训练
4. **评估与可视化**：对局回放、指标统计、模型对比

### 1.2 技术栈

| 层     | 技术                                            | 说明                      |
| ----- | --------------------------------------------- | ----------------------- |
| 引擎核心  | Rust + PyO3                                   | 高性能规则引擎，编译为 Python 扩展模块 |
| RL 训练 | PyTorch 2.x                                   | 策略网络、GRP 模型、训练循环        |
| 构建工具  | maturin                                       | Rust → Python 扩展模块编译    |
| 并行计算  | rayon (Rust) + torch.multiprocessing (Python) | 引擎并行 + 训练并行             |
| 配置管理  | TOML + dataclass                              | 超参数与训练配置                |

### 1.3 与参考项目的关键差异

| 方面   | 日麻 (Mortal/Suphx) | 川麻 (本项目)               |
| ---- | ----------------- | ---------------------- |
| 牌种   | 34种（含字牌）          | 27种（万筒条1-9，无字牌）        |
| 吃牌   | 有                 | **无**                  |
| 副露   | 吃/碰/杠             | **仅碰/杠**               |
| 胡牌后  | 局结束               | **血战到底，继续对局**          |
| 换三张  | 无                 | **有（骰子定向交换）**          |
| 定缺   | 无                 | **有（含天缺特例）**           |
| 流局   | 无人听牌              | **查花猪 + 查大叫**          |
| 决策模型 | 5个（弃/立直/吃/碰/杠）    | **6个（换三张/定缺/弃/碰/杠/胡）** |
| 特征维度 | 34维               | **27维**                |

***

## 二、目录结构

> **设计原则**：规则、评分、模型三大易变模块高度解耦，通过 trait/接口 + 配置文件驱动，便于独立调整。

```
e:\ai\mjai\
├── Cargo.toml                    # Rust workspace 根配置
├── pyproject.toml                # Python 构建配置 (maturin)
├── docs/
│   ├── plan.md                   # 本文件
│   └── rules/
│       └── 传统川麻规则.md        # 规则唯一来源（已有）
├── configs/                      # 所有配置集中管理
│   ├── engine.toml               # 引擎规则配置（番型表/封顶/底注/杠结算等）
│   ├── model.toml                # 模型结构配置（网络层数/通道数/决策头等）
│   ├── train.toml                # 训练超参数（lr/batch_size/epsilon等）
│   └── reward.toml               # 奖励配置（即时/终局/GRP权重等）
├── engine/                       # Rust 引擎核心（规则层，可独立调整）
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # #[pymodule] 入口
│       ├── py_helper.rs          # PyO3 辅助函数
│       ├── tile/                 # 牌定义（通用，不随规则变化）
│       │   ├── mod.rs
│       │   ├── tile.rs           # 牌枚举 (万筒条 1-9)
│       │   └── hand.rs           # 手牌结构
│       ├── algo/                 # 核心算法（通用，不随规则变化）
│       │   ├── mod.rs
│       │   ├── shanten.rs        # 听牌数计算
│       │   ├── winning.rs        # 胡牌判定（牌型结构判定）
│       │   └── counter.rs        # 预计算表初始化
│       ├── rules/                # ★ 规则层（可替换，trait 驱动）
│       │   ├── mod.rs            # RuleSet trait 定义
│       │   ├── fan.rs            # 番型计算（实现 FanRule trait）
│       │   ├── scoring.rs        # 计分规则（实现 ScoringRule trait）
│       │   ├── swap.rs           # 换三张规则 trait 实现（AI 选牌策略）
│       │   ├── missing.rs        # 定缺规则 trait 实现（AI 选缺策略）
│       │   └── presets/          # 预设规则集
│       │       ├── mod.rs
│       │       └── xue_zhan.rs   # 血战到底规则集（当前默认）
│       ├── state/                # 游戏状态管理
│       │   ├── mod.rs
│       │   ├── player.rs         # 玩家状态 (手牌/副露/缺门/弃牌)
│       │   └── action.rs         # 合法动作生成与验证
│       ├── arena/                # 对局模拟（调用 rules/ trait）
│       │   ├── mod.rs
│       │   ├── board.rs          # 牌桌 (牌墙/骰子/轮次)
│       │   └── game.rs           # 完整对局流程
│       ├── phases/               # 开局阶段（流程编排，调用 rules/ trait 获取 AI 策略）
│       │   ├── mod.rs
│       │   ├── deal.rs           # 配牌
│       │   ├── swap.rs           # 换三张流程（掷骰子定向 + 四家同步交换，AI 选牌委托给 rules/swap.rs）
│       │   └── missing.rs        # 定缺流程（AI 选缺委托给 rules/missing.rs）
│       └── observation/          # 特征编码（可替换编码器）
│           ├── mod.rs            # ObservationEncoder trait
│           └── encoder.rs        # 默认编码器实现
├── mjai/                         # Python AI 组件
│   ├── __init__.py
│   ├── models/                   # ★ 模型层（可独立替换/调整）
│   │   ├── __init__.py
│   │   ├── base.py               # ModelBase 抽象接口
│   │   ├── brain.py              # ResNet 特征提取器
│   │   ├── heads.py              # 决策头（Discard/Pong/Kong/Win/Swap/Missing）
│   │   ├── grp.py                # GRP 奖励预测器
│   │   └── factory.py            # 模型工厂（根据 config 创建模型）
│   ├── rewards/                  # ★ 奖励层（可独立调整）
│   │   ├── __init__.py
│   │   ├── base.py               # RewardCalculator 抽象接口
│   │   ├── immediate.py          # 即时奖励（杠/胡/花猪/大叫）
│   │   ├── terminal.py           # 终局奖励
│   │   ├── grp_reward.py         # GRP 辅助奖励
│   │   └── factory.py            # 奖励工厂（根据 config 创建）
│   ├── engine.py                 # 推理引擎（模型加载 + 动作选择）
│   ├── train.py                  # 训练主脚本
│   ├── buffer.py                 # 经验回放缓冲区
│   ├── opponent_pool.py          # 自博弈对手池管理
│   ├── server.py                 # 分布式训练服务端（可选）
│   ├── client.py                 # 自博弈训练客户端（可选）
│   ├── dataloader.py             # 牌谱数据加载
│   ├── config.py                 # 配置数据类（加载 configs/*.toml）
│   └── bots/
│       ├── __init__.py
│       ├── random_bot.py         # 随机基线
│       └── heuristic_bot.py      # 启发式基线
├── tests/                        # Rust 单元测试 + 集成测试
│   ├── engine/
│   │   ├── test_tile.rs
│   │   ├── test_shanten.rs
│   │   ├── test_winning.rs
│   │   ├── test_fan.rs           # 番型测试（验证规则可替换）
│   │   ├── test_scoring.rs       # 计分测试
│   │   ├── test_swap.rs
│   │   ├── test_missing.rs
│   │   ├── test_robbing_kong.rs  # 抢杠胡测试
│   │   └── test_game.rs
│   └── python/
│       ├── test_engine.py        # 引擎绑定测试
│       ├── test_model.py         # 模型测试（验证模型可替换）
│       ├── test_reward.py        # 奖励测试（验证奖励可替换）
│       └── test_selfplay.py      # 自博弈测试
├── replays/                      # 牌谱持久化存储（MJAI JSON 格式）
│   ├── iter_0010/                # 按模型迭代轮次保存
│   │   └── *.jsonl
│   └── eval/                     # 评估对局牌谱
│       └── *.jsonl
└── scripts/                      # 工具脚本
    ├── build.py                  # 编译引擎
    ├── train.py                  # 训练入口
    ├── eval.py                   # 评估入口
    └── replay.py                 # 对局回放
```

### 2.1 模块解耦设计

**Rust 引擎规则层（trait 驱动）**：

```rust
// rules/mod.rs — 规则 trait 定义

pub struct FanResult {           // fan.rs 输出：番数 + 额外底数
    pub fan: u32,                // 参与封顶的番数（已含所有乘法修饰）
    pub extra_bases: u32,        // 不受封顶限制的额外底数（目前只有自摸加底=1）
}

/// 杠类型（用于杠结算）
pub enum KongType { AnKan, MinKan, BuKan }
/// 副露类型（rules 层轻量视图，用于根的统计）
pub enum MeldKind { Pong, ExposedKong, ConcealedKong, AddKong }
pub struct MeldView { pub tile: Tile, pub kind: MeldKind }

pub trait RuleSet: Send + Sync {
    fn name(&self) -> &'static str;                                          // 规则集名称
    fn fan(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult;  // 番型计算（含根的副露杠统计）
    fn score_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;  // 胡牌得分
    fn score_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;  // 杠结算
    fn swap_rule(&self) -> &dyn SwapRule;                        // 换三张规则
    fn missing_rule(&self) -> &dyn MissingRule;                  // 定缺规则
}

pub trait FanRule { fn calculate(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult; }
pub trait ScoringRule {
    fn calculate_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;
    fn calculate_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;
}
pub trait SwapRule {
    fn determine_direction(&self, dice: u8) -> SwapDirection;
    fn select_swap_tiles(&self, hand: &Hand) -> Vec<Tile>;  // AI 选 3 张同花色换出牌
}
pub trait MissingRule {
    fn is_valid_missing(&self, hand: &Hand, suit: Suit) -> bool;
    fn select_missing_suit(&self, hand: &Hand) -> Suit;     // AI 选缺门花色
}

// rules/presets/xue_zhan.rs — 血战到底规则集实现
pub struct XueZhanRuleSet { /* 配置参数 */ }
impl RuleSet for XueZhanRuleSet { /* ... */ }
```

- 调整玩法规则：修改 `rules/presets/xue_zhan.rs` 或新建预设（如 `xue_zhan_lian_zhuang.rs`）
- 调整番型/评分：修改 `rules/fan.rs` 和 `rules/scoring.rs`，或通过 `configs/engine.toml` 调参数

**Python 模型层（工厂模式）**：

```python
# models/base.py — 模型抽象接口
class ModelBase(ABC):
    @abstractmethod
    def forward(self, obs: Tensor) -> Dict[str, Tensor]: ...
    @abstractmethod
    def get_action(self, obs: Tensor, legal_mask: Dict[str, Tensor]) -> Dict[str, int]: ...

# models/factory.py — 根据 config 创建模型
def create_model(config: ModelConfig) -> ModelBase:
    if config.arch == "resnet_dqn":
        return ResNetDQN(config)
    elif config.arch == "transformer_dqn":
        return TransformerDQN(config)  # 未来可扩展
```

- 调整模型结构：修改 `models/brain.py`（特征提取器）或 `models/heads.py`（决策头）
- 替换模型架构：在 `models/factory.py` 注册新架构，通过 `configs/model.toml` 切换

**Python 奖励层（策略模式）**：

```python
# rewards/base.py — 奖励计算抽象接口
class RewardCalculator(ABC):
    @abstractmethod
    def calculate(self, replay: Replay) -> List[float]: ...

# rewards/factory.py — 根据 config 创建
def create_reward_calculator(config: RewardConfig) -> RewardCalculator:
    return CompositeReward([
        ImmediateReward(config.immediate),
        TerminalReward(config.terminal),
        GRPReward(config.grp),
    ])
```

- 调整奖励设计：修改 `rewards/immediate.py` 或 `rewards/terminal.py`
- 调整奖励权重：通过 `configs/reward.toml` 调参，无需改代码

***

## 三、实施阶段

### Phase 1：Rust 引擎核心（阶段一）

**目标**：实现完整的血战到底麻将引擎，通过规则自检（3000局零违规、零和成立）。

#### 1.1 牌具与基础结构

**文件**：`engine/src/tile/tile.rs`, `engine/src/tile/hand.rs`

- 定义 `Tile` 枚举：万(Wan)、筒(Tong)、条(Tiao) 各 1-9，共 27 种
- `Tile` 方法：`suit()` → 花色, `number()` → 点数, `from_index()` → 索引转牌
- `Hand` 结构：手牌计数 `[u8; 27]`、缺门花色（**副露由 `PlayerState.melds` 独立管理**，Hand 仅管手牌）
- `Hand` 方法：`add_tile()`, `remove_tile()`, `count()`, `contains()`, `set_missing_suit()`, `hand_without_missing()` 等

#### 1.2 核心算法

**文件**：`engine/src/algo/`（**通用算法层，不随规则变化**）

**听牌数计算 (shanten.rs)**：

- 标准手牌听牌数：枚举面子+雀头组合，计算最小听牌数
- 七对听牌数：统计对子数
- 取两者最小值

**胡牌判定 (winning.rs)**：

- 标准胡牌：4 面子 + 1 雀头
- 七对：7 对子（含龙七对：七对中有一组 4 张相同，叠加根修饰即为龙七对 = 七对×根 = 8番）
- 清一色：仅 1 花色
- 对对胡：全面子为刻子
- 金钩钓：全部副露 + 单张听牌
- **胡牌方式**：自摸、点炮、杠上花、抢杠胡
  - 杠上花：杠后从牌墙尾补摸的牌正好自摸胡牌
  - 抢杠胡：他人补杠时，补杠的那张牌正好是自己听的牌，可抢杠胡；暗杠不可抢

**预计算表 (counter.rs)**：

- 启动时初始化听牌数表和胡牌判定表
- 使用 `OnceLock` 保证线程安全的一次初始化

#### 1.3 规则层（trait 驱动，可替换）

**文件**：`engine/src/rules/`（**易变规则层，可通过预设切换玩法**）

> 设计原则：所有玩法规则通过 trait 抽象，与通用算法解耦。调整番型/计分/换三张/定缺规则只需修改对应模块，不影响 algo 和 state。

**trait 定义 (mod.rs)**：

```rust
/// fan.rs 输出：番数 + 不受封顶限制的额外底数
pub struct FanResult {
    pub fan: u32,          // 参与封顶 16 的番数（含所有乘法修饰）
    pub extra_bases: u32,  // 不受封顶限制的额外底数（目前只有自摸加底=1）
}

/// 杠类型（用于杠结算）
pub enum KongType { AnKan, MinKan, BuKan }

/// 副露类型（rules 层轻量视图，与 state::MeldType 解耦）
pub enum MeldKind { Pong, ExposedKong, ConcealedKong, AddKong }

/// 副露轻量视图（用于根的统计——杠=4 张同牌计入根）
pub struct MeldView { pub tile: Tile, pub kind: MeldKind }

pub trait RuleSet: Send + Sync {
    fn fan(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult;
    fn score_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;
    fn score_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;
    fn swap_rule(&self) -> &dyn SwapRule;
    fn missing_rule(&self) -> &dyn MissingRule;
    fn name(&self) -> &'static str;
}
pub trait FanRule { fn calculate(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult; }
pub trait ScoringRule {
    fn calculate_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;
    fn calculate_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;
}
pub trait SwapRule {
    fn determine_direction(&self, dice: u8) -> SwapDirection;
    fn select_swap_tiles(&self, hand: &Hand) -> Vec<Tile>;  // AI 选 3 张同花色换出牌
}
pub trait MissingRule {
    fn is_valid_missing(&self, hand: &Hand, suit: Suit) -> bool;
    fn select_missing_suit(&self, hand: &Hand) -> Suit;     // AI 选缺门花色
}
```

**trait 分层关系**：`RuleSet` 是 arena/state 层调用的**顶层聚合接口**，其内部方法委托给下层独立 trait 实现：

| RuleSet 方法 | 委托目标 | 实现文件 |
|-------------|---------|---------|
| `fan(hand, melds, win_type)` | `FanRule.calculate(hand, melds, win_type)` | `rules/fan.rs` |
| `score_hu(result, win_type, base, num_payers)` | `ScoringRule.calculate_hu(result, win_type, base, num_payers)` | `rules/scoring.rs` |
| `score_kong(kong_type, base, num_payers)` | `ScoringRule.calculate_kong(kong_type, base, num_payers)` | `rules/scoring.rs` |
| `swap_rule()` | 返回 `&dyn SwapRule`（`rules/presets/xue_zhan.rs` 持有实例） | `rules/swap.rs` |
| `missing_rule()` | 返回 `&dyn MissingRule`（同上） | `rules/missing.rs` |

**调用约定**：
- arena/state 层只与 `RuleSet` 交互，不直接调用 `FanRule` / `ScoringRule`
- `swap_rule()` / `missing_rule()` 返回的 trait 对象供 `phases/` 层使用
- arena 层负责将 `&[state::Meld]` 转换为 `&[rules::MeldView]`（`MeldType → MeldKind` 一一映射），再传给 `fan()` / `calculate()`

**番型计算 (fan.rs — 实现 `FanRule` trait)**：

> fan.rs 负责**全部规则层面的胡牌修饰**——基础番型、组合相乘、乘法修饰（根/杠上花×2/抢杠胡×2）、自摸加底。输出 `FanResult` 结构体。

- 实现 `传统川麻规则.md` 第六节番型表的全部番型
- **基础番型**：平胡(1)、对对胡(2)、清一色(4)、七对(4)、金钩钓(4)
- **组合番型**（基础番型相乘）：清对(4×2=8)、清七对(4×4=16)、清金钩钓(4×4=16)
  > 注：清金钩钓为按"多个基础番型同时满足时番数相乘"原则推导（清一色×金钩钓），规则文档番型表未单列
- **乘法修饰**（叠加相乘）：根×2、杠上花×2、抢杠胡×2
  - **根**：每组独立的 4 张同牌 ×2，**按组数叠加**（N 组根 → ×2^N）
  - **根的统计范围** = 手牌中四张同牌 + **所有副露杠**（明杠/暗杠/补杠均为 4 张同牌，计入根）
  - 龙七对 = 七对(4) × 根(×2) = 8（七对中含一组 4 张相同牌，该组即根修饰来源）
  - 双龙七对 = 七对(4) × 根(×2) × 根(×2) = 16（含两组 4 张相同牌）
  - 清龙七对 = 清一色(4) × 七对(4) × 根(×2) = 32 → 封顶 16
  > 注：抢杠胡仅在他人**补杠**时可抢；**暗杠不可抢**（采用标准川麻惯例，规则文档未明写）
- **封顶 16 番**：`FanResult.fan` 已封顶（返回 `min(总番数, 16)`）
- **自摸加底**：自摸胡牌时 `FanResult.extra_bases = 1`，否则为 0；不受封顶限制
- **最终得分公式**（fan.rs 输出 `FanResult`，交由 scoring.rs 统一套入）：
  - `每家支付 = 底注 × FanResult.fan + FanResult.extra_bases × 底注`
  - scoring.rs 根据 `num_payers` 计算总计：`总得分 = 每家支付 × num_payers`

**计分规则 (scoring.rs — 实现 `ScoringRule` trait)**：

> scoring.rs 负责所有得分计算：胡牌得分、杠结算、流局结算。fan.rs 只输出番型结果，scoring.rs 统一套入公式。

- **胡牌得分**：`(base × result.fan + result.extra_bases × base) × num_payers`
  - 自摸：`num_payers` = 未胡者数（各家支付，含自摸加底）
  - 点炮：`num_payers` = 1（点炮者独付）
- **杠结算（刮风下雨）**：
  - 暗杠：`2 × base × num_payers`（未胡者各付 2 底）
  - 明杠：`2 × base × 1`（点杠者独付 2 底，须未胡）
  - 补杠：`1 × base × num_payers`（未胡者各付 1 底）
- **流局结算**：查花猪 + 查大叫

**换三张规则 (swap.rs — 实现 `SwapRule` trait)**：

- 根据骰子确定换三张方向（下家/对家/上家）
- AI 策略：选最少花色的 3 张（若同花色不足则选次少）
- 四家同步交换

**定缺规则 (missing.rs — 实现 `MissingRule` trait)**：

- 三门齐全：选最少门
- 两门（天缺）：75% 选空门、25% 保留冲清一色
- 一门（天缺）：在空门中选
- 缺门固定后不可更改

**预设规则集 (presets/xue_zhan.rs)**：

- 将 fan.rs / scoring.rs / swap.rs / missing.rs 的实现组合成 `XueZhanRuleSet`
- 实现 `RuleSet` trait，供 arena 层调用
- 支持通过 `XueZhanConfig`（max_fan、封顶、自摸加底等）参数化

#### 1.4 游戏状态

**文件**：`engine/src/state/player.rs`, `engine/src/state/action.rs`

**PlayerState**：

- 手牌 `Hand`、副露列表 `Vec<Meld>`（碰/明杠/暗杠/补杠）、缺门花色、弃牌序列
- 已胡牌标记、花猪/大叫标记、天缺标记
- 方法：`can_pong()`, `can_kong()`, `can_add_kong()`, `can_win()` 等查询方法；**合法动作生成由 `ActionValidator::generate_legal_actions()` 负责，状态更新由 Board/Game 流程编排层负责**

**ActionCandidate**：

- 动作类型：Discard(打牌)、Pong(碰)、Kong(杠)、Win(胡)、Pass(过)
- 优先级：胡 > 杠 > 碰
- **一炮多响**：同张牌多家可同时胡，按优先级依次检查，多家胡牌时同时结算
- **抢杠胡检测**：补杠时检查其他玩家是否可抢杠胡（暗杠不可抢）

#### 1.5 开局阶段

> **职责边界**：`phases/` 负责**流程编排**（配牌、掷骰子、同步交换等时序控制），
> AI 策略细节（选哪 3 张换、选哪门缺）委托给 `rules/swap.rs` 和 `rules/missing.rs`，
> 通过 `SwapRule` / `MissingRule` trait 解耦。

**文件**：`engine/src/phases/`

**配牌 (deal.rs)**：

- 洗牌（108张）→ 庄家14张、闲家13张 → 剩余牌墙

**换三张 (swap.rs — 流程编排)**：

- 调用 `SwapRule::determine_direction()` 掷骰子定向
- 四家同步执行：从各家收集待换牌 → 交换 → 更新手牌
- AI 选哪 3 张同花色牌：调用 `SwapRule::select_swap_tiles(hand)`

**定缺 (missing.rs — 流程编排)**：

- 公开声明缺门花色
- 验证合法性：调用 `MissingRule::is_valid_missing(hand, suit)`
- AI 选哪门缺：调用 `MissingRule::select_missing_suit(hand)`
- 缺门固定后不可更改

#### 1.6 对局模拟

**文件**：`engine/src/arena/board.rs`, `engine/src/arena/game.rs`

**Board**：

- 牌墙管理：正序摸牌、杠补牌取尾段
- 当前轮次、当前玩家、庄家
- 四家 PlayerState

**Game 完整流程**：

1. 配牌 → 换三张 → 定缺
2. 庄家出牌 → 逆时针轮流
3. 每轮：摸牌 → 检查自摸/暗杠/补杠 → 出牌 → 检查他人胡/杠/碰
4. 杠后补摸 → 继续出牌；补摸牌若自摸则为杠上花
5. 胡牌后：标记已胡，其余继续（血战到底）；一炮多响时多家同时胡
6. 抢杠胡：补杠时其他玩家可抢杠胡（暗杠不可抢）
7. 结束条件：三家胡牌 或 牌墙摸完
8. 流局结算：查花猪 + 查大叫
9. 杠即时结算（刮风下雨）
10. 已胡玩家不再参与摸打出牌，但仍计入花猪/大叫结算

#### 1.7 PyO3 绑定

**文件**：`engine/src/lib.rs`, `engine/src/py_helper.rs`

- `#[pymodule] fn mjai_engine(py, m)` 入口
- 注册子模块：tile、algo、rules、state、arena、observation
- 暴露规则系统 Python 包装（RuleSet / FanCalculator / ScoringCalculator / SwapRuleCalculator / MissingRuleCalculator / XueZhanRuleSet），其中 ScoringCalculator 暴露 `calculate_hu` 和 `calculate_kong` 两个方法
- 使用 `sys.modules` 注册支持 `from mjai_engine.rules import XueZhanRuleSet`

#### 1.8 特征编码

**文件**：`engine/src/observation/encoder.rs`

将游戏状态编码为观测张量 `[C, 27]`（C 为通道数），展平为 `Vec<f32>` 总长度 = C × 27（无截断、无填充）。

| # | 通道组 | 内容 | 通道数 | 说明 |
|---|------|------|-------|------|
| 1 | 手牌计数标记 | 0/1/2/3/4 张标记 | 5 | 本家；5 通道分别标记每种牌恰有 0\~4 张，可完整还原 13-14 张手牌分布 |
| 2 | 四家副露 | 各家碰/明杠/暗杠/补杠 | 4×4 = 16 | 每家 4 通道（4 种副露类型），每种副露的 tile 位置标 1 |
| 3 | 四家弃牌河 | 各家归一化弃牌计数 | 4 | 每家 1 通道，27 维计数归一化到 [0,1]（count/4）；**保留 27 种完整编码**，缺门阶段会打出缺门牌，不能压缩缺门花色 |
| 4 | 四家缺门 | 各家缺门花色标记 | 4 | 每家 1 通道，缺门花色对应的 9 个位置标记为 1 |
| 5 | 剩余牌 | 各牌剩余张数 | 1 | `(4 - 暴露牌数) / 4`；**已胡玩家手牌不计入暴露**（已胡玩家不再碰/杠/胡）；副露按杠型计 3 或 4、弃牌计 1 |
| 6 | 轮次标量 | 全局回合比例 | 1 | `turn / 80` 广播到 27 维（假设单局最多约 80 回合） |
| 7 | 庄家标量 | 庄家位置比例 | 1 | `dealer_position / 3` 广播 |
| 8 | 牌墙剩余 | 牌墙剩余比例 | 1 | `wall.remaining_ratio()` 广播；含 NaN 保护（空墙时返回 0） |
| 9 | 四家已胡标记 | 各家是否已胡 | 4 | 每家 1 通道，已胡→全 27 维 = 1；血战到底状态 |
| 10 | 换三张方向 | 骰子定向 one-hot | 3 | Down/Across/Up 三选一广播；知道谁和谁交换 |
| 11 | 杠上花标记 | 是否刚杠完摸牌 | 1 | `kan_shang_active` 广播；番型修饰判定 |
| 12 | 本家是否听牌 | is_tenpai() | 1 | 广播；`shanten == 0`（通用听牌判定） + 缺门花色已清干净（川麻胡牌前置条件） |
| 13 | 本家听牌数 | waiting_count() | 1 | 广播；`min(听牌种类数, 9) / 9`，没听牌时为 0 |
| 14 | 番型进度 | 清一色/对对胡/七对进度（本家） | 3 | 清一色=最大花色张数/14；对对胡=刻子潜力/4；七对=对子数/7；全部广播 |
| 15 | 根进度 | 本家四张同牌组数 | 1 | 广播；`手牌中 count≥4 的牌种数 / 4`；川麻乘法修饰×2 的显式捷径特征 |
| 16 | 四家天缺标记 | 是否天缺 | 4 | 每家 1 通道广播；encoder **自推断**：初始缺门牌数 == 0 → 天缺；否则从 PlayerState.is_natural_missing 取 |
| 17 | 四家 tsumogiri 占比 | 摸牌即打占总弃牌的比例 | 4 | 每家 1 通道广播；高占比表示该玩家多摸打、手牌灵活 |
| 18 | **四家清缺进度** | **已弃缺门牌数 / 初始缺门牌数** | **4** | **每家 1 通道广播**；encoder **自包含推断**：初始缺门牌数 = 已弃缺门牌数 + 手牌剩余缺门牌数；天缺固定为 1.0 |
| 19 | **四家最近 3 张弃牌）** | **各家最近 3 张弃牌 one-hot** | **4×3 = 12** | **每玩家 3 通道，每张弃牌 27 维 one-hot**；从最新开始倒序；弃牌不足 3 张时剩余通道填 0 |
| | **合计** | | **71** | **展平 = 1917 维** |


#### 1.9 引擎自检

- 3000 局完整对局，0 规则违规
- 四家**正收益率均衡（约 67%-71%）**
  - 定义：单局结束后，该玩家最终得分 > 0 视为"正收益"
  - 血战到底每局最多 3 家胡牌（理论胡牌率 75%），但流局时花猪/大叫会让部分未胡者也有正收益，整体正收益率低于理论胡牌率
  - 四家正收益率接近说明洗牌/发牌无偏倚、无位置优势
- 零和验证（总得分 = 0）
- 杠结算、花猪、大叫计分正确

***

### Phase 2：神经网络模型（阶段二）

**目标**：定义并验证全部神经网络模型，确保前向传播正常。

**文件**：`mjai/models/`（**模型层，可独立替换/调整**）

```
mjai/models/
├── __init__.py      # 统一导出
├── base.py          # ModelBase 抽象接口 + DuelingQHead
├── brain.py         # ResNet 特征提取器（残差块 + 通道注意力）
├── heads.py         # 6 个 Dueling DQN 决策头 + MultiHeadDQN
├── factory.py       # ResNetDQN 主模型 + ModelFactory 工厂
└── grp.py           # GRP 奖励预测器（GRU 架构）
```

#### 2.1 特征提取器 (Brain)

```
Observation [N, C, 27]
    ↓
Conv1d(C → 256, kernel=3, padding=1)
    ↓
Residual Block × 12  (Conv1d→BN→Mish→Conv1d→BN→+)
    ↓
Conv1d(256 → 256, kernel=3)
    ↓
Channel Attention (avg+max pool → shared MLP → sigmoid → scale)
    ↓
Flatten → Linear(256×27 → 1024)
    ↓
Feature Vector [N, 1024]
```

- 激活函数：Mish
- 归一化：BatchNorm1d (eps=1e-3)
- 残差连接：标准 2 层残差块

#### 2.2 决策网络 (Dueling DQN)

**6 个独立决策头**，共享 Brain 特征：

| 头           | 输出维度 | 决策       | 说明                         |
| ----------- | ---- | -------- | -------------------------- |
| SwapHead    | 3    | 换三张选哪门花色 | 万/筒/条，选同花色3张换出             |
| MissingHead | 3    | 定哪门缺     | 万/筒/条，含天缺特例                |
| DiscardHead | 27   | 打哪张牌     | 27种牌选1                     |
| PongHead    | 2    | 是否碰      | 是/否（弃碰可追求清一色/七对等更大番型）      |
| KongHead    | 2    | 是否杠      | 是/否（弃杠可保留手牌灵活性）            |
| WinHead     | 2    | 是否胡      | 是/否（弃胡可追求更大番型，如放弃平胡等自摸清一色） |

- **Dueling 架构**：6 个决策头**共享一个 V(s) 分支**，各自拥有独立 A(s,a) 分支
  - 共享 V(s)：Brain 输出的 1024 维特征经同一个 Value 分支投影为标量，所有头共用
  - 独立 A(s,a)：每个头各自 Advantage 分支（Swap: 1024→3, Missing: 1024→3, Discard: 1024→27, Pong/Kong/Win: 1024→2）
  - Q(s,a) = V(s) + A(s,a) - mean(A(s,a))（每个头内部独立做 mean 归一化）
  - 设计理由：所有决策头面对**同一个状态**，对状态"好坏"的评估应统一；共享 V 分支减少冗余参数、提升训练稳定性
- 动作掩码：通过引擎 `legal_actions()` 屏蔽非法动作
- 换三张和定缺在开局阶段触发，仅执行一次
- **弃碰/弃杠/弃胡的策略空间**：Q 值对比"立即碰/杠/胡的收益" vs "放弃后追求更大番型的期望收益"，由网络自主学习权衡

#### 2.3 奖励预测器 (GRP)

```
状态序列 [T, feature_dim]
    ↓
GRU (input=feature_dim, hidden=64, layers=2)
    ↓
FC(64 → 1) → Linear
    ↓
最终得分预测（底注单位）
```

- 使用 float64 保证数值稳定
- **预测最终得分而非排名**：川麻血战到底为得分制，不是排名制
- 为每轮提供中间学习信号（解决长序列信用分配问题）
- 输出为标量（本家最终得分），通过 MSE 损失训练

#### 2.4 模型验证

- 各网络前向传播维度正确（Brain→[N, 1024]、6 决策头各 [N, action_dim]、GRP→[N, 1]）
- 动作掩码生效（非法动作 Q 值被屏蔽为 -1e9，经 softmax 后概率接近 0）
- GRP 输出为标量得分预测（范围合理，-20 ~ +50），MSE Loss 可正常反传
- ModelFactory 可根据 `config.arch` 切换架构（resnet_dqn / transformer_dqn）

***

### Phase 3：训练系统（阶段三）

**目标**：实现完整自博弈训练循环，模型可以从零学到合理策略。

**文件**：

```
mjai/
├── rewards/                    # ★ 奖励层（可独立调整）
│   ├── __init__.py
│   ├── base.py                 # RewardCalculator 抽象接口
│   ├── immediate.py            # 即时奖励（杠/胡/花猪/大叫）
│   ├── terminal.py             # 终局奖励
│   ├── grp_reward.py           # GRP 辅助奖励
│   └── factory.py              # 奖励工厂（根据 config 创建）
├── opponent_pool.py            # 自博弈对手池管理
├── buffer.py                   # 经验回放缓冲区
├── train.py                    # 训练主脚本
├── engine.py                   # 推理引擎（模型加载 + 动作选择）
├── dataloader.py               # 牌谱数据加载
├── config.py                   # 配置数据类（加载 configs/*.toml）
├── server.py                   # 分布式训练服务端（可选）
├── client.py                   # 自博弈训练客户端（可选）
└── bots/                       # 基线 AI（需引擎 PyO3 绑定可用后实现，见里程碑 M6）
    ├── __init__.py
    ├── base.py                 # BotBase 抽象接口
    ├── random_bot.py           # 随机基线
    └── heuristic_bot.py        # 启发式基线
```

#### 3.1 奖励设计

**即时奖励**：

- 杠结算（刮风下雨，独立于胡牌番数，**仅未胡玩家参与**）：暗杠 +2×未胡者数底、明杠 +2底（点杠者须未胡）、补杠 +1×未胡者数底
- 胡牌：底注 × min(番数, 16)（**线性倍数，封顶16番**；自摸额外+1底加底）
- 自摸：其余未胡者**各家支付**（含加底）；点炮：**点炮者独付**
- 花猪：赔所有未胡者牌型全额
- 大叫：赔已听未胡者最大番

**番型激励梯度**：

- 奖励与番数线性正相关（底注×min(番数,16)），驱动模型追求高番型而非"有胡就胡"
- 示例：平胡(1番)得1底 vs 清一色(4番)得4底 vs 清龙七对(32→封顶16番)得16底
- 封顶后清七对(16番)与清龙七对(32→封顶16番)收益相同，但自摸加底和杠结算仍独立计算
- **弃胡奖励**：若模型选择弃胡并最终获得更大番型，终局奖励中体现额外增益
- **弃碰/弃杠奖励**：同理，若放弃副露后获得清一色/七对等高番型，通过终局得分自然体现

**终局奖励**：

- 最终得分（底注单位）— 高番型胡牌的收益通过得分差异自然体现

**GRP 辅助奖励**：

- 每轮使用 GRP 预测的最终得分作为中间信号

#### 3.2 探索机制

**ε-greedy 策略**：

- 训练初期 ε=0.3（高探索），逐步衰减至 ε=0.05（高利用）
- 衰减方式：线性衰减或指数衰减，按训练步数控制

**Softmax 温度采样**：

- 对于 DiscardHead，可选使用温度采样替代 ε-greedy
- 温度 τ 从 1.0 衰减至 0.1，高温时探索性强

**熵正则化**：

- 在损失函数中加入策略熵项，鼓励多样化动作选择
- `loss += entropy_weight × (-entropy)`，权重逐步减小

#### 3.3 自博弈对手池

**对手池 (Opponent Pool) 设计**：

自博弈中若始终用当前模型自我对弈，容易出现策略崩溃 (Nash Collapse) 或循环克制。对手池通过保留历史模型快照，增加对手多样性，提升策略鲁棒性。

**池结构**：

- 容量：保留最近 N 个模型快照（默认 N=20）
- 快照触发：每隔 `snapshot_interval` 局（如 500 局）生成候选快照
- 池满策略：淘汰池中胜率最低或最旧的快照

**准入门槛（候选快照入池条件）**：

- 候选快照需与当前池中对手进行 `eval_games` 局评估（如 100 局）
- **准入胜率门槛**：候选快照 vs 池中对手平均胜率 ≥ `admission_win_rate`（默认 30%）
  - 胜率过低说明候选太弱，入池只会浪费采样机会
  - 胜率达标才入池，保证池中对手都有一定竞争力
- **特殊情况**：池为空时首个快照直接入池（初始模型）

**淘汰策略**：

- 池满时淘汰优先级：最旧快照 > 胜率最低快照
- 定期（如每 `snapshot_interval` 局）重新评估池中所有快照 vs 当前模型的胜率
- 若某快照对当前模型胜率 < `retire_win_rate`（默认 10%），说明已被远远超越，直接淘汰

**对手采样策略**：

- 自博弈对局中，4 个座位从对手池中随机采样历史模型
- 当前模型始终占至少 1 个座位（保证学习信号）
- 其余 3 个座位从池中采样，采样方式：
  - 均匀采样：各快照等概率（默认）
  - 对抗性采样：优先选择当前模型胜率低的对手，强化弱点修补
- 采样时避免同一快照占多个座位（增加多样性）

**LFSP (League Fictitious Self-Play) 可选增强**：

- 将对手池分层：Main Player（当前模型）、League Players（历史快照）、Main Exploiter（专门针对当前模型弱点的对手）
- 定期评估当前模型 vs 池中所有对手的胜率，监控策略多样性

#### 3.4 经验回放缓冲区

**Replay Buffer 设计**：

- 容量：100,000 局完整对局（约数百万个决策点）
- 存储内容：(观测张量, 动作, 奖励, 下一观测, 是否终止, 决策头类型)
- 采样策略：
  - 初期：按决策头**分层均匀采样**（每个头取相同数量样本），避免高频头压制低频头
  - 后期：分层 + 优先经验回放 (PER)，各头内部按 TD 误差优先采样
- 多决策头共享同一 Buffer，按决策头类型字段索引

**样本不均衡问题处理**：

6 个决策头的样本量差异可达 50-100 倍：

| 决策头 | 每局触发次数 | 相对比例 |
|--------|------------|---------|
| DiscardHead | ~50-60 次 | 100%（基准） |
| PongHead | ~3-5 次 | ~8% |
| KongHead | ~1-3 次 | ~4% |
| WinHead | ~2-4 次 | ~6% |
| SwapHead | 1 次 | ~2% |
| MissingHead | 1 次 | ~2% |

**采用分层采样解决**：训练一个 batch 时，从每个决策头分别抽取固定数量样本（如每个头 64 条，batch size = 6×64 = 384），确保低频头有足够的梯度更新机会。

**牌谱存储设计（参考 MJAI 标准格式，按模型迭代轮次保存）**：

牌谱采用 MJAI 标准格式的 JSON 事件序列（每行一个 JSON 对象），适配川麻特点（万筒条用 `1m-9m`/`1p-9p`/`1s-9s`，无字牌）：

```jsonl
{"type":"start_game","names":["model_v10","model_v10","pool_v8","pool_v9"],"rule_set":"xue_zhan"}
{"type":"deal","actor":0,"pais":["1m","1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]}
{"type":"swap","actor":0,"suit":"p","direction":1}
{"type":"missing","actor":0,"suit":"s"}
{"type":"tsumo","actor":0,"pai":"5m"}
{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}
{"type":"pon","actor":1,"target":0,"pai":"4p","consumed":["4p","4p"]}
{"type":"kan","actor":2,"pai":"7s","kong_type":"ankan","consumed":["7s","7s","7s"]}
{"type":"hora","actor":0,"pai":"3m","win_type":"tsumo","fan":4,"score":17}
{"type":"end_game","final_scores":[17,-5,-3,-9]}
```

**按模型迭代轮次保存**：

```
replays/
├── iter_0010/                    # 第10轮模型迭代的牌谱
│   ├── game_000001.jsonl
│   ├── game_000002.jsonl
│   └── ...
├── iter_0020/                    # 第20轮模型迭代的牌谱
│   ├── game_000001.jsonl
│   └── ...
└── eval/                         # 评估对局牌谱
    └── model_v10_vs_heuristic/
        ├── game_000001.jsonl
        └── ...
```

**存储规则**：

- **训练牌谱**：每轮迭代生成的自博弈对局保存到 `replays/iter_{iteration}/` 目录
- **评估牌谱**：模型对比评估的对局保存到 `replays/eval/{model_version}_{opponent}/`
- **文件格式**：`.jsonl`（JSON Lines，每行一个事件），兼容 MJAI 标准工具链
- **保留策略**：默认保留最近 N 轮迭代牌谱（如 50 轮），旧的可清理或归档
- **训练时**：内存中以 `Vec<GameReplay>` 存储，转为张量后喂给 Replay Buffer

**牌谱用途**：

1. **训练数据**：从牌谱提取 (观测, 动作, 奖励) 元组存入 Replay Buffer
2. **对局回放**：逐步重放 JSON 事件序列，可视化展示
3. **规则验证**：检查引擎自检的对局是否符合规则
4. **模型对比**：对比不同迭代轮次模型的对局牌谱，分析策略演进
5. **互操作性**：MJAI 标准格式便于与其他麻将 AI 项目交换数据

#### 3.5 训练循环

```python
for episode in range(total_episodes):
    # 1. 对手池采样：当前模型占1座，其余从池中随机采样
    opponents = opponent_pool.sample(num_seats=3)
    
    # 2. 自博弈生成数据（使用 ε-greedy 探索）
    replay = self_play(model, opponents=opponents, num_games=games_per_round, epsilon=epsilon)

    # 3. 存入 Replay Buffer
    buffer.add(replay)

    # 4. 采样训练
    batch = buffer.sample(batch_size)
    rewards = calculate_rewards(batch, gamma=0.99, grp_model=grp)

    # 5. 训练模型（使用 target network 计算 MC 回报目标值）
    loss = dqn_loss + grp_loss * grp_weight + entropy_loss * entropy_weight
    loss.backward()
    optimizer.step()

    # 5b. 定期硬更新 target network
    if episode % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    # 6. ε 衰减
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # 7. 定期快照入池 + 评估
    if episode % snapshot_interval == 0:
        opponent_pool.add(model.clone())
    if episode % eval_interval == 0:
        evaluate(model, baseline=bots.heuristic)
```

**损失函数**：

```
loss = dqn_loss + (grp_loss × grp_weight) + (entropy_loss × entropy_weight)
```

- DQN Loss: MSE(预测Q值, MC回报)
  - 使用 **target network** 稳定训练：主网络每 N 步将参数硬更新到 target network，target network 用于计算 MC 回报的目标 Q 值，避免自举偏差导致 Q 值发散
- GRP Loss: MSE(预测得分, 实际得分)
- Entropy Loss: -Σ π(a|s) log π(a|s)，鼓励探索

#### 3.6 分布式架构（可选，单机验证收敛后启用）

```
Training Server (server.py)
  ├── Parameter Storage        # 模型参数存储与分发
  ├── Opponent Pool            # 对手池管理（模型快照分发）
  ├── Replay Buffer            # 牌谱收集
  └── Trainer Loop             # 训练循环

Training Clients (client.py) × N
  ├── fetch_params()           # 拉取最新模型 + 对手池快照
  ├── Self-Play Games          # 并行自博弈（从对手池采样对手）
  └── submit_replay()          # 提交牌谱
```

- **优先实现单机多进程自博弈**：使用 `torch.multiprocessing` 在单机多 GPU/CPU 上并行
- 分布式 Server-Client 架构在单机验证收敛后再启用
- Server 定期广播最新模型参数和对手池快照，Clients 异步提交自博弈数据

#### 3.7 Oracle Guiding（可选增强）

- 训练一个能看到完美信息（对手手牌、牌墙）的 Oracle agent
- Oracle agent 因信息优势变强，作为初期强对手
- 逐步移除 Oracle 的完美信息，过渡为正常 agent
- 加速初期训练收敛

#### 3.8 基线 AI

**文件**：`mjai/bots/`（已在 Phase 3 文件列表中展开）

- **RandomBot**：合法动作中随机选择（最低基线）
- **HeuristicBot**：启发式策略（优先打缺门牌、听牌优先碰杠等）
- 评估对比：训练过程中定期与 RandomBot / HeuristicBot 对局，监控胜率提升

***

### Phase 4：评估与工具（阶段四）

**目标**：完整的评估、回放和可视化能力。

**文件**：`scripts/eval.py`, `scripts/replay.py`

#### 4.1 评估指标

| 指标      | 说明                          |
| ------- | --------------------------- |
| 平均排名    | 1-4名分布                      |
| 胜率（胡牌率） | 和牌局数 / 总局数                  |
| 平均番数    | 胡牌时平均番数                     |
| 放铳率     | 点炮局数 / 总局数                  |
| 杠率      | 杠局数 / 总局数                   |
| 花猪率     | 花猪局数 / 总局数                  |
| 对阵胜率    | vs RandomBot / HeuristicBot |

#### 4.2 对局回放

**文件**：`scripts/replay.py`

- 从 MJAI 格式牌谱文件（`replays/**/*.jsonl`）加载 JSON 事件序列
- 逐步重放事件序列（摸/打/碰/杠/胡），支持前进/后退/跳转
- 显示四家手牌、牌墙、弃牌河、副露、计分
- 兼容 MJAI 标准工具链，可与其他麻将 AI 项目互操作

#### 4.3 训练监控

- Loss 曲线（DQN Loss、GRP Loss）
- 奖励曲线（平均奖励、胜率）
- TensorBoard / WandB 集成

***

## 四、依赖清单

### Rust 根 Cargo.toml (workspace)

```toml
[workspace]
members = ["engine"]
resolver = "2"
```

### Rust (engine/Cargo.toml)

```toml
[lib]
name = "mjai_engine"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
rayon = "1.8"
anyhow = "1.0"
thiserror = "1.0"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
numpy = "0.23"           # PyO3 NumPy 绑定，零拷贝张量传递
```

### Python (pyproject.toml)

```toml
[project]
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "tensorboard>=2.14",
    "tqdm>=4.65",
    "tomli>=2.0",
]
```

***

## 五、构建与运行

### 5.1 构建引擎

```bash
# 开发模式
maturin develop

# Release 模式
maturin develop --release
```

### 5.2 运行测试

```bash
# Rust 测试
cargo test

# Python 测试
pytest tests/python/

# 引擎自检（3000局）
python scripts/eval.py --self-check --games 3000
```

### 5.3 训练

```bash
# 单机多进程训练（优先）
python scripts/train.py --config configs/train.toml

# 分布式（可选，单机验证收敛后启用）：启动 Server + N 个 Client
python -m mjai.server --port 9876 &
python -m mjai.client --server localhost:9876 --num-games 1000 &
```

***

## 六、实施顺序与里程碑

| 里程碑               | 内容                                     | 验证标准                 |
| ----------------- | -------------------------------------- | -------------------- |
| M1: 引擎核心          | tile + algo (听牌/胡牌/抢杠胡) + rules (trait 规则层) | 单元测试全通过              |
| M2: PyO3绑定        | lib.rs + rules 包装 + observation encoder | Python 可 import 引擎模块 |
| M3: 模型定义          | models/ (Brain + 6决策头 + GRP + Factory) | 前向传播维度正确             |
| M4: 游戏流程          | state + phases + arena + rules 集成         | 1局完整对局无报错            |
| M5: 引擎自检          | 3000局零违规、零和、四家正收益率均衡                     | 自检脚本通过               |
| M6: 基线AI          | bots/ (RandomBot + HeuristicBot)          | 对局可运行、胜率合理           |
| M7: 训练循环          | rewards/ + opponent_pool + buffer + train  | 100轮训练 Loss 下降       |
| M8: 评估系统          | eval.py + replay.py                    | 指标统计 + 回放可用          |
| M9: 分布式训练（可选）     | server.py + client.py                  | 多客户端并行训练             |
| M10: Oracle增强（可选） | Oracle Guiding                         | 训练收敛速度提升             |

> **里程碑调整说明**：PyO3 绑定 (M2) 和模型定义 (M3) 提前，与引擎开发并行；分布式训练降级为可选 (M9)；规则 trait 层新增为 M1 组成部分。

***

## 七、假设与决策

### 7.1 关键决策

1. **Rust + PyO3**：参考 Mortal 架构，Rust 保证引擎性能和安全，PyTorch 保证 RL 灵活性
2. **Dueling DQN**：参考 Mortal v4，比纯 Policy Gradient 更稳定
3. **6 个决策头**：川麻独有的换三张、定缺也作为 RL 决策点，加弃/碰/杠/胡共 6 个
4. **27 维特征**：川麻仅 27 种牌（万筒条 1-9）
5. **GRP 预测得分**：川麻为得分制非排名制，GRP 输出标量得分预测
6. **单机优先**：先验证单机多进程训练收敛，再考虑分布式 Server-Client 架构
7. **ε-greedy + 熵正则化**：双探索机制保证自博弈多样性
8. **对手池 (Opponent Pool)**：保留历史模型快照增加对手多样性，防止策略崩溃和循环克制
9. **模块解耦架构**：规则层（Rust trait）、模型层（Python 工厂）、奖励层（Python 策略）三大易变模块通过接口+配置文件驱动，便于独立调整玩法规则、评分规则和模型架构

### 7.2 假设

- 无人类牌谱数据可用于 SL 预训练（如有可后续添加 SL 阶段）
- 初始训练从零开始（纯 RL 自博弈）
- 本地单机或小规模集群训练（非大规模分布式）
- 底注单位统一为 1，可通过参数调整

### 7.3 风险与缓解

| 风险          | 缓解                        |
| ----------- | ------------------------- |
| Rust 学习曲线   | 引擎逻辑独立，可先用 Python 原型验证再移植 |
| 自博弈收敛慢      | Oracle Guiding 加速初期训练     |
| 血战到底长序列信用分配 | GRP 模型提供中间信号              |
| 引擎规则复杂度高    | 严格依据 `传统川麻规则.md`，逐条实现+测试  |

***

## 八、验证步骤

1. **引擎正确性**：3000 局自检零违规 + 零和 + 四家正收益率均衡（67%-71%，定义见 Phase 1.9）
2. **模型正确性**：前向传播维度测试 + 动作掩码测试
3. **训练有效性**：Loss 下降 + 胜率 vs 基线提升
4. **规则完整性**：番型表全覆盖 + 流局结算（花猪/大叫）正确
5. **性能达标**：引擎 ≥ 1000 局/秒（Release 模式）

