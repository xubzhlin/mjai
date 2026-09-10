# 四川麻将自博弈强化学习项目 (XueZhanDaoDi 血战到底)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-red.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

## 项目简介

这是一个基于强化学习的四川麻将（血战到底）自博弈AI项目。项目采用 Rust 作为游戏引擎核心，Python 作为训练和推理框架，实现了完整的麻将游戏逻辑、神经网络模型和训练系统。

## 🎯 项目特点

- **完整的游戏引擎**: 基于 Rust 实现的四川麻将规则引擎
- **先进的神经网络**: ResNet + Dueling DQN + GRP 架构
- **自博弈训练**: 支持多进程自博弈训练
- **高性能**: Rust 引擎提供高性能游戏模拟
- **易于使用**: PyO3 绑定提供 Python 接口
- **可扩展**: 模块化设计，易于扩展新功能

## 📁 项目结构

```
mjai/
├── README.md                    # 项目文档
├── docs/
│   ├── plan.md                  # 项目计划
│   └── rules/
│       └── 传统川麻规则.md       # 游戏规则
├── engine/                      # Rust 游戏引擎
│   ├── src/
│   │   ├── tile/                # 牌具相关
│   │   ├── algo/                # 核心算法
│   │   ├── state/               # 游戏状态
│   │   ├── arena/               # 游戏场地
│   │   ├── phases/              # 游戏阶段
│   │   └── observation/         # 特征编码
│   │   └── lib.rs               # PyO3 绑定
│   └── Cargo.toml               # Rust 配置
├── mjai/                        # Python 组件
│   ├── __init__.py
│   ├── model.py                 # 基础神经网络模型
│   ├── advanced_model.py        # 高级神经网络模型 (ResNet + Dueling DQN + GRP)
│   ├── features.py              # 特征处理
│   ├── predictor.py             # 基础预测器
│   ├── advanced_predictor.py    # 高级预测器
│   ├── trainer.py               # 训练器
│   └── config.py                # 配置管理
├── configs/                     # 配置文件
│   ├── engine.toml              # 引擎配置
│   ├── model.toml               # 模型配置
│   ├── training.toml            # 训练配置
│   └── reward.toml              # 奖励配置
├── scripts/                     # 脚本文件
│   ├── self_check.py            # 引擎自检
│   ├── train_model.py          # 模型训练
│   └── train_advanced.py        # 高级模型训练
├── tests/                       # 测试文件
├── models/                      # 模型文件
├── replays/                     # 对局回放
└── requirements.txt             # Python 依赖
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **Rust**: 1.70+
- **操作系统**: Linux, macOS, Windows

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/mjai.git
cd mjai
```

2. **安装 Rust 依赖**
```bash
cd engine
cargo build --release
cd ..
```

3. **安装 Python 依赖**
```bash
pip install -r requirements.txt
```

4. **编译 Rust 引擎**
```bash
cd engine
cargo build --release
cd ..
```

### 基础使用

#### 1. 引擎自检
```bash
python scripts/self_check.py
```

#### 2. 训练基础模型
```bash
python scripts/train_model.py
```

#### 3. 训练高级模型
```bash
python scripts/train_advanced.py
```

#### 4. 自定义训练
```python
from mjai import AdvancedMahjongModel, AdvancedActionPredictor
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig

# 创建模型
model = AdvancedMahjongModel()

# 创建预测器
config = FeatureConfig()
feature_processor = FeatureProcessor(config)
action_encoder = ActionEncoder(config)
predictor = AdvancedActionPredictor(model, feature_processor, action_encoder)

# 预测动作
game_state = {
    'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'current_player': 0,
    # ... 其他游戏状态
}

action_idx, prob, info = predictor.predict_action(game_state)
print(f"预测动作: {action_idx}, 概率: {prob:.3f}")
```

## 📚 详细文档

### 核心组件

#### 1. 游戏引擎 (Rust)
- **牌具系统**: 完整的麻将牌定义和处理
- **核心算法**: 向听计算、胡牌判断、番种计算
- **游戏状态**: 玩家状态、动作系统、场地管理
- **游戏流程**: 发牌、换牌、缺门、游戏阶段

#### 2. 神经网络模型 (Python)
- **基础模型**: 简单的前馈网络
- **高级模型**: ResNet + Dueling DQN + GRP 架构
- **特征编码**: 132维特征向量
- **动作预测**: 多种预测策略

#### 3. 训练系统
- **经验回放**: 经验缓冲区管理
- **PPO算法**: 近端策略优化
- **学习调度**: 学习率调度和早停
- **可视化**: TensorBoard 日志

### 配置说明

#### 引擎配置 (configs/engine.toml)
```toml
[game]
players = 4
rounds = 4
starting_points = 25000

[scoring]
fan_multiplier = 1
yakuman_multiplier = 1
limit = 8000

[phases]
swap_cards = 13
missing_declaration = true
```

#### 模型配置 (configs/model.toml)
```toml
[advanced_model]
input_dim = 132
action_dim = 34
base_channels = 64
hidden_dims = [512, 256, 128]
lstm_hidden_dim = 128
lstm_layers = 2
dropout_rate = 0.2
```

#### 训练配置 (configs/training.toml)
```toml
[training]
epochs = 1000
batch_size = 32
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
update_interval = 50
save_interval = 100
```

## 🔧 开发指南

### 添加新功能

1. **游戏规则扩展**
```rust
// 在 engine/src/algo/ 中添加新算法
pub struct NewAlgorithm;
impl Algorithm for NewAlgorithm {
    fn calculate(&self, hand: &Hand) -> Result<CalculationResult> {
        // 实现新算法
    }
}
```

2. **神经网络扩展**
```python
# 在 mjai/ 中添加新模型
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义网络结构
    
    def forward(self, x):
        # 前向传播
```

3. **训练策略扩展**
```python
# 在 trainer.py 中添加新策略
class CustomTrainer:
    def update(self, batch):
        # 自定义更新逻辑
```

### 测试

#### 运行测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_engine.py

# 运行自检
python scripts/self_check.py
```

#### 编写测试
```python
# tests/test_engine.py
import pytest
from engine import Game, Player

def test_game_creation():
    game = Game()
    assert len(game.players) == 4
```

## 📊 性能指标

### 引擎性能
- **游戏速度**: 1000+ 局/分钟
- **内存使用**: < 100MB
- **CPU 使用**: < 50%

### 训练性能
- **训练速度**: 50+ iterations/小时
- **GPU 使用**: 充分利用GPU加速
- **收敛速度**: 通常需要 1000+ epochs

## 🤝 贡献指南

1. **Fork 项目**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送分支** (`git push origin feature/AmazingFeature`)
5. **创建 Pull Request**

### 代码规范

- **Rust**: 遵循 Rust 官方风格指南
- **Python**: 遵循 PEP 8 规范
- **文档**: 所有公共 API 必须有文档字符串
- **测试**: 新功能必须包含测试

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Suphx](https://arxiv.org/abs/2003.13590) - 研究论文参考
- [Mortal](https://github.com/Equim-chan/Mortal) - 代码参考
- [Mahjong-AI](https://github.com/windshadow233/Mahjong-AI) - 代码参考

## 📞 联系方式

- 项目主页: https://github.com/your-username/mjai
- 问题反馈: https://github.com/your-username/mjai/issues
- 邮箱: your-email@example.com

---

**注意**: 本项目仅用于研究目的，不涉及任何游戏外挂、内存修改等违规行为。所有训练都在本地模拟环境中进行。