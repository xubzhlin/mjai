"""
mjai.models — 模型层（可独立替换/调整）

模块结构：
    base.py    — ModelBase 抽象接口 + DuelingQHead
    brain.py   — ResNet 特征提取器（残差块 + 通道注意力）
    heads.py   — 6 个 Dueling DQN 决策头
    factory.py — ResNetDQN 主模型 + ModelFactory 工厂
    grp.py     — GRP 奖励预测器（GRU 架构）

设计原则：
    调整模型结构 → 修改 brain.py 或 heads.py
    替换模型架构 → 在 ModelFactory 注册新架构，通过 config.arch 切换
"""

from .base import ModelBase, DuelingQHead
from .brain import Brain, ResidualBlock, ChannelAttention, Mish
from .heads import (
    SwapHead, MissingHead, DiscardHead, PongHead, KongHead, WinHead,
    MultiHeadDQN, HEAD_NAMES, HEAD_ACTION_DIMS,
    apply_action_mask, select_action,
)
from .factory import (
    ModelConfig, ResNetDQN, TransformerDQN, ModelFactory, create_model,
)
from .grp import GRP, GRPConfig, GRPTrainer, GRPLoss

__all__ = [
    # Base
    'ModelBase', 'DuelingQHead',
    # Brain
    'Brain', 'ResidualBlock', 'ChannelAttention', 'Mish',
    # Heads
    'SwapHead', 'MissingHead', 'DiscardHead', 'PongHead', 'KongHead', 'WinHead',
    'MultiHeadDQN', 'HEAD_NAMES', 'HEAD_ACTION_DIMS',
    'apply_action_mask', 'select_action',
    # Factory
    'ModelConfig', 'ResNetDQN', 'TransformerDQN', 'ModelFactory', 'create_model',
    # GRP
    'GRP', 'GRPConfig', 'GRPTrainer', 'GRPLoss',
]