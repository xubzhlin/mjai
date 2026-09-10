# API 文档

本文档详细介绍了四川麻将自博弈强化学习项目的 API 接口。

## 目录

1. [游戏引擎 API](#游戏引擎-api)
2. [神经网络模型 API](#神经网络模型-api)
3. [特征处理 API](#特征处理-api)
4. [预测器 API](#预测器-api)
5. [训练器 API](#训练器-api)
6. [配置管理 API](#配置管理-api)

## 游戏引擎 API

### 核心类

#### `Game`
游戏主类，管理整个游戏流程。

```python
from engine import Game

# 创建游戏
game = Game()

# 运行游戏到结束
game.run_to_end()

# 获取游戏结果
winners = game.get_winners()
scores = game.get_player_scores()
```

**方法:**
- `__init__(config=None)`: 初始化游戏
- `run_to_end()`: 运行游戏到结束
- `get_winners()`: 获取赢家列表
- `get_player_scores()`: 获取玩家分数
- `get_game_state()`: 获取当前游戏状态
- `is_game_over()`: 检查游戏是否结束

#### `Player`
玩家类，管理玩家状态。

```python
from engine import Player

# 创建玩家
player = Player(player_id=0)

# 获取玩家手牌
hand = player.get_hand()

# 获取玩家吃碰杠
melds = player.get_melds()
```

**方法:**
- `__init__(player_id)`: 初始化玩家
- `get_hand()`: 获取手牌
- `get_melds()`: 获取吃碰杠
- `get_discards()`: 获取弃牌
- `add_tile(tile)`: 添加牌到手牌
- `remove_tile(tile)`: 从手牌移除牌
- `add_meld(meld)`: 添加吃碰杠

#### `Board`
游戏场地类，管理游戏场地状态。

```python
from engine import Board

# 创建场地
board = Board()

# 获取剩余牌数
remaining = board.get_remaining_tiles()

# 获取当前玩家
current_player = board.get_current_player()
```

**方法:**
- `__init__()`: 初始化场地
- `get_remaining_tiles()`: 获取剩余牌数
- `get_current_player()`: 获取当前玩家
- `get_wall()`: 获取牌墙
- `deal_tile()`: 发牌
- `swap_cards(cards)`: 换牌

### 核心算法

#### `ShantenCalculator`
向听计算器。

```python
from engine.algo import ShantenCalculator

# 创建计算器
calculator = ShantenCalculator()

# 计算向听数
shanten = calculator.calculate(hand)
```

**方法:**
- `calculate(hand)`: 计算向听数
- `calculate_seven_pairs(hand)`: 计算七对向听数

#### `WinningChecker`
胡牌判断器。

```python
from engine.algo import WinningChecker

# 创建判断器
checker = WinningChecker()

# 检查是否胡牌
is_winning, win_method = checker.check(hand, melds)
```

**方法:**
- `check(hand, melds)`: 检查是否胡牌
- `get_win_methods()`: 获取所有胡牌方法

#### `FanCalculator`
番种计算器。

```python
from engine.algo import FanCalculator

# 创建计算器
calculator = FanCalculator()

# 计算番种
fan_points = calculator.calculate(hand, melds, win_method)
```

**方法:**
- `calculate(hand, melds, win_method)`: 计算番种
- `get_base_fans()`: 获取基础番种
- `get_combo_fans()`: 获取组合番种

## 神经网络模型 API

### 基础模型

#### `MahjongModel`
基础神经网络模型。

```python
from mjai.model import MahjongModel

# 创建模型
model = MahjongModel()

# 预测动作
action_probs, value = model.predict(features)

# 更新模型
loss = model.update(batch_features, batch_actions, batch_rewards, batch_old_probs)
```

**方法:**
- `__init__(policy_net=None, value_net=None, learning_rate=0.001)`: 初始化模型
- `predict(features)`: 预测动作概率和价值
- `update(features, actions, rewards, old_probs)`: 更新模型
- `save(path)`: 保存模型
- `load(path)`: 加载模型
- `get_device()`: 获取计算设备

#### `MahjongNet`
策略网络。

```python
from mjai.model import MahjongNet

# 创建网络
net = MahjongNet(input_dim=132, action_dim=34)

# 前向传播
action_probs = net(features)
```

**方法:**
- `__init__(input_dim=132, hidden_dims=[512, 256, 128], action_dim=34, dropout_rate=0.2)`: 初始化网络
- `forward(x)`: 前向传播
- `predict(features)`: 预测动作概率
- `save(path)`: 保存网络
- `load(path)`: 加载网络

#### `ValueNet`
价值网络。

```python
from mjai.model import ValueNet

# 创建网络
net = ValueNet(input_dim=132)

# 评估状态价值
value = net.evaluate(features)
```

**方法:**
- `__init__(input_dim=132, hidden_dims=[256, 128], dropout_rate=0.2)`: 初始化网络
- `forward(x)`: 前向传播
- `evaluate(features)`: 评估状态价值
- `save(path)`: 保存网络
- `load(path)`: 加载网络

### 高级模型

#### `AdvancedMahjongModel`
高级神经网络模型 - ResNet + Dueling DQN + GRP。

```python
from mjai.advanced_model import AdvancedMahjongModel

# 创建模型
model = AdvancedMahjongModel()

# 预测动作
action, q_value, hidden = model.predict(features)

# 更新模型
stats = model.update(batch)
```

**方法:**
- `__init__(model=None, learning_rate=0.001, gamma=0.99, epsilon=0.1)`: 初始化模型
- `predict(features, hidden=None)`: 预测动作
- `update(batch)`: 更新模型
- `get_q_values(features)`: 获取Q值
- `add_experience(state, action, reward, next_state, done, old_q_value)`: 添加经验
- `sample_batch(batch_size=32)`: 采样批次
- `save(path)`: 保存模型
- `load(path)`: 加载模型

#### `AdvancedMahjongNet`
高级神经网络架构。

```python
from mjai.advanced_model import AdvancedMahjongNet

# 创建网络
net = AdvancedMahjongNet(input_dim=132, action_dim=34)

# 前向传播
action_probs, hidden = net(features, hidden)
```

**方法:**
- `__init__(input_dim=132, action_dim=34, base_channels=64, hidden_dims=[512, 256], lstm_hidden_dim=128, lstm_layers=2, dropout_rate=0.2)`: 初始化网络
- `forward(x, hidden=None)`: 前向传播
- `predict(features, hidden=None)`: 预测动作概率
- `get_q_values(features)`: 获取Q值
- `save(path)`: 保存网络
- `load(path)`: 加载网络

## 特征处理 API

### `FeatureProcessor`
特征处理器。

```python
from mjai.features import FeatureProcessor, FeatureConfig

# 创建配置和处理器
config = FeatureConfig()
processor = FeatureProcessor(config)

# 编码特征
features = processor.encode_features(game_state)

# 归一化特征
normalized_features = processor.normalize_features(features)
```

**方法:**
- `__init__(config)`: 初始化处理器
- `encode_features(game_state)`: 编码游戏状态特征
- `normalize_features(features)`: 归一化特征
- `encode_hand_features(hand)`: 编码手牌特征
- `encode_meld_features(players)`: 编码吃碰杠特征
- `encode_discard_features(players)`: 编码弃牌特征
- `encode_round_features(round_info)`: 编码回合特征
- `encode_player_features(player_info)`: 编码玩家特征

### `ActionEncoder`
动作编码器。

```python
from mjai.features import ActionEncoder, FeatureConfig

# 创建配置和编码器
config = FeatureConfig()
encoder = ActionEncoder(config)

# 获取合法动作
valid_actions = encoder.get_valid_actions(game_state)

# 编码动作
action_idx = encoder.encode_action(action)

# 解码动作
action = encoder.decode_action(action_idx)
```

**方法:**
- `__init__(config)`: 初始化编码器
- `get_valid_actions(game_state)`: 获取合法动作
- `encode_action(action)`: 编码动作
- `decode_action(action_idx)`: 解码动作
- `get_action_space()`: 获取动作空间
- `is_action_valid(action, game_state)`: 检查动作是否合法

### `FeatureConfig`
特征配置。

```python
from mjai.features import FeatureConfig

# 创建配置
config = FeatureConfig()

# 获取配置
input_dim = config.input_dim
action_dim = config.action_dim
hand_dim = config.hand_dim
```

**属性:**
- `input_dim`: 输入特征维度 (132)
- `action_dim`: 动作空间维度 (34)
- `hand_dim`: 手牌特征维度 (135)
- `meld_dim`: 吃碰杠特征维度 (16)
- `round_dim`: 回合特征维度 (4)
- `player_dim`: 玩家特征维度 (4)

## 预测器 API

### 基础预测器

#### `ActionPredictor`
动作预测器。

```python
from mjai.predictor import ActionPredictor, PredictionConfig

# 创建配置和预测器
config = PredictionConfig()
predictor = ActionPredictor(model, feature_processor, action_encoder, config)

# 预测动作
action_idx, prob, info = predictor.predict_action(game_state)

# 使用束搜索
action_idx, prob, info = predictor.predict_with_beam_search(game_state)

# 使用MCTS
action_idx, prob, info = predictor.predict_with_mcts(game_state)
```

**方法:**
- `__init__(model, feature_processor, action_encoder, config=None)`: 初始化预测器
- `predict_action(game_state)`: 预测动作
- `predict_with_beam_search(game_state)`: 使用束搜索预测
- `predict_with_mcts(game_state)`: 使用MCTS预测
- `get_stats()`: 获取统计信息
- `reset_stats()`: 重置统计信息
- `save_stats(path)`: 保存统计信息
- `load_stats(path)`: 加载统计信息

### 高级预测器

#### `AdvancedActionPredictor`
高级动作预测器。

```python
from mjai.advanced_predictor import AdvancedActionPredictor, AdvancedPredictionConfig

# 创建配置和预测器
config = AdvancedPredictionConfig()
predictor = AdvancedActionPredictor(model, feature_processor, action_encoder, config)

# 预测动作
action_idx, prob, info = predictor.predict_action(game_state)

# 重置GRP隐藏状态
predictor.reset_grp_hidden()
```

**方法:**
- `__init__(model, feature_processor, action_encoder, config=None)`: 初始化预测器
- `predict_action(game_state)`: 预测动作
- `predict_with_beam_search(game_state)`: 使用束搜索预测
- `predict_with_mcts(game_state)`: 使用MCTS预测
- `get_stats()`: 获取统计信息
- `reset_stats()`: 重置统计信息
- `reset_grp_hidden()`: 重置GRP隐藏状态
- `save_stats(path)`: 保存统计信息
- `load_stats(path)`: 加载统计信息

### 批量预测器

#### `BatchPredictor`
批量预测器。

```python
from mjai.predictor import BatchPredictor

# 创建预测器
predictor = BatchPredictor(model, feature_processor, action_encoder)

# 批量预测
results = predictor.predict_batch(game_states)
```

**方法:**
- `__init__(model, feature_processor, action_encoder, config=None)`: 初始化预测器
- `predict_batch(game_states)`: 批量预测

#### `AdvancedBatchPredictor`
高级批量预测器。

```python
from mjai.advanced_predictor import AdvancedBatchPredictor

# 创建预测器
predictor = AdvancedBatchPredictor(model, feature_processor, action_encoder)

# 批量预测
results = predictor.predict_batch(game_states)
```

**方法:**
- `__init__(model, feature_processor, action_encoder, config=None)`: 初始化预测器
- `predict_batch(game_states)`: 批量预测

## 训练器 API

### `MahjongTrainer`
训练器。

```python
from mjai.trainer import MahjongTrainer

# 创建训练器
trainer = MahjongTrainer(model, config)

# 训练一个批次
loss = trainer.train_step(batch)

# 训练多个epoch
trainer.train(epochs=10)

# 评估模型
eval_score = trainer.evaluate(eval_states)
```

**方法:**
- `__init__(model, config)`: 初始化训练器
- `train_step(batch)`: 训练一个批次
- `train(epochs)`: 训练多个epoch
- `evaluate(eval_states)`: 评估模型
- `save_model(path)`: 保存模型
- `load_model(path)`: 加载模型
- `get_stats()`: 获取训练统计信息
- `reset_stats()`: 重置统计信息

### 训练配置

#### `TrainingConfig`
训练配置。

```python
from mjai.trainer import TrainingConfig

# 创建配置
config = TrainingConfig(
    epochs=1000,
    batch_size=32,
    learning_rate=0.001,
    gamma=0.99,
    buffer_size=10000,
    update_interval=50,
    save_interval=100,
    eval_interval=50
)
```

**参数:**
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `gamma`: 折扣因子
- `buffer_size`: 缓冲区大小
- `update_interval`: 更新间隔
- `save_interval`: 保存间隔
- `eval_interval`: 评估间隔
- `log_dir`: 日志目录
- `tensorboard_dir`: TensorBoard目录
- `save_dir`: 模型保存目录
- `use_scheduler`: 是否使用学习率调度器
- `early_stopping`: 是否使用早停

## 配置管理 API

### `ConfigManager`
配置管理器。

```python
from mjai.config import ConfigManager

# 创建管理器
manager = ConfigManager()

# 加载配置
engine_config = manager.load_engine_config()
model_config = manager.load_model_config()
training_config = manager.load_training_config()

# 保存配置
manager.save_config(config, path)
```

**方法:**
- `__init__()`: 初始化管理器
- `load_engine_config()`: 加载引擎配置
- `load_model_config()`: 加载模型配置
- `load_training_config()`: 加载训练配置
- `load_reward_config()`: 加载奖励配置
- `save_config(config, path)`: 保存配置
- `get_default_config(type)`: 获取默认配置

### 配置类

#### `EngineConfig`
引擎配置。

```python
from mjai.config import EngineConfig

# 创建配置
config = EngineConfig(
    players=4,
    rounds=4,
    starting_points=25000,
    fan_multiplier=1,
    yakuman_multiplier=1,
    limit=8000,
    swap_cards=13,
    missing_declaration=True
)
```

**参数:**
- `players`: 玩家数量
- `rounds`: 回合数
- `starting_points`: 起始分数
- `fan_multiplier`: 番种倍数
- `yakuman_multiplier`: 役满倍数
- `limit`: 分数限制
- `swap_cards`: 换牌数量
- `missing_declaration`: 是否缺门声明

#### `ModelConfig`
模型配置。

```python
from mjai.config import ModelConfig

# 创建配置
config = ModelConfig(
    input_dim=132,
    action_dim=34,
    hidden_dims=[512, 256, 128],
    dropout_rate=0.2,
    learning_rate=0.001
)
```

**参数:**
- `input_dim`: 输入维度
- `action_dim`: 动作维度
- `hidden_dims`: 隐藏层维度
- `dropout_rate`: Dropout率
- `learning_rate`: 学习率

## 使用示例

### 完整训练流程

```python
import torch
import numpy as np
from mjai.advanced_model import AdvancedMahjongModel
from mjai.advanced_predictor import AdvancedActionPredictor
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig
from mjai.trainer import MahjongTrainer, TrainingConfig

# 1. 创建模型
model = AdvancedMahjongModel()

# 2. 创建特征处理器和动作编码器
config = FeatureConfig()
feature_processor = FeatureProcessor(config)
action_encoder = ActionEncoder(config)

# 3. 创建预测器
predictor = AdvancedActionPredictor(model, feature_processor, action_encoder)

# 4. 创建训练配置
training_config = TrainingConfig(
    epochs=1000,
    batch_size=32,
    learning_rate=0.001,
    gamma=0.99,
    buffer_size=10000,
    update_interval=50,
    save_interval=100,
    eval_interval=50
)

# 5. 创建训练器
trainer = MahjongTrainer(model, training_config)

# 6. 训练循环
for epoch in range(training_config.epochs):
    # 生成训练数据
    game_states = generate_training_data()
    
    # 预测动作
    results = predictor.predict_batch(game_states)
    
    # 更新模型
    for result in results:
        action_idx, prob, info = result
        # 构建训练批次
        batch = prepare_batch(result)
        
        # 训练
        loss = trainer.train_step(batch)
    
    # 评估
    if epoch % training_config.eval_interval == 0:
        eval_score = trainer.evaluate(generate_eval_data())
        print(f"Epoch {epoch}, Loss: {loss}, Eval Score: {eval_score}")
    
    # 保存模型
    if epoch % training_config.save_interval == 0:
        trainer.save_model(f"models/model_epoch_{epoch}.pth")

# 7. 保存最终模型
trainer.save_model("models/final_model.pth")
```

### 自定义游戏流程

```python
from engine import Game, Player
from mjai.advanced_predictor import AdvancedActionPredictor
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig

# 创建游戏
game = Game()

# 创建预测器
config = FeatureConfig()
feature_processor = FeatureProcessor(config)
action_encoder = ActionEncoder(config)

# 加载模型
model = AdvancedMahjongModel.load("models/final_model.pth")
predictor = AdvancedActionPredictor(model, feature_processor, action_encoder)

# 游戏循环
while not game.is_game_over():
    # 获取当前玩家
    current_player = game.get_current_player()
    
    # 获取游戏状态
    game_state = game.get_game_state()
    
    # 预测动作
    action_idx, prob, info = predictor.predict_action(game_state)
    
    # 执行动作
    action = action_encoder.decode_action(action_idx)
    game.execute_action(current_player, action)
    
    # 继续游戏
    game.next_turn()

# 获取结果
winners = game.get_winners()
scores = game.get_player_scores()
print(f"Winners: {winners}, Scores: {scores}")
```

这个API文档提供了项目中所有主要组件的详细接口说明，包括方法、参数、返回值和使用示例。开发者可以根据这些API来使用和扩展项目功能。