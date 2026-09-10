"""
ResNetDQN 完整模型 + ModelFactory 工厂

ResNetDQN: Brain(ResNet 特征提取) + 6 个 Dueling DQN 决策头
ModelFactory: 根据 config.arch 创建对应模型架构
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import ModelBase
from .brain import Brain
from .heads import (
    SwapHead, MissingHead, DiscardHead, PongHead, KongHead, WinHead,
    MultiHeadDQN, HEAD_NAMES, HEAD_ACTION_DIMS,
    apply_action_mask, select_action,
)


@dataclass
class ModelConfig:
    """模型配置类（与 configs/model.toml 对齐）"""
    
    # 架构类型
    arch: str = "resnet_dqn"
    
    # Brain 配置
    input_channels: int = 132
    hidden_channels: int = 256
    num_residual_blocks: int = 12
    seq_length: int = 27
    feature_dim: int = 1024
    
    # 决策头配置
    head_hidden_dim: int = 256
    
    # 正则化
    dropout_rate: float = 0.1
    attention_reduction: int = 4
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


class ResNetDQN(ModelBase):
    """
    ResNet + Dueling DQN 完整模型
    
    架构：
        Observation [N, C, 27]
            ↓
        Brain (ResNet + Channel Attention) → Feature [N, 1024]
            ↓
        ├── SwapHead    → Q values [N, 3]
        ├── MissingHead → Q values [N, 3]
        ├── DiscardHead → Q values [N, 27]
        ├── PongHead    → Q values [N, 2]
        ├── KongHead    → Q values [N, 2]
        └── WinHead     → Q values [N, 2]
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Brain 特征提取器
        self.brain = Brain(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            num_residual_blocks=config.num_residual_blocks,
            seq_length=config.seq_length,
            feature_dim=config.feature_dim,
            dropout_rate=config.dropout_rate,
            attention_reduction=config.attention_reduction,
        )
        
        # 6 个 Dueling DQN 决策头
        self.heads = MultiHeadDQN(
            feature_dim=config.feature_dim,
            hidden_dim=config.head_hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            obs: [N, C, 27] 或 [N, C*27]
            
        Returns:
            {head_name: Q_values [N, action_dim]}
        """
        features = self.brain(obs)
        return self.heads(features)
    
    def get_action(
        self,
        obs: torch.Tensor,
        legal_mask: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
        temperature: float = 1.0,
    ) -> Dict[str, int]:
        """
        选择各决策头的动作
        
        Args:
            obs: 观测张量
            legal_mask: 各头的合法动作掩码（True=合法）
            epsilon: ε-greedy 率
            temperature: Softmax 温度
            
        Returns:
            {head_name: action_index}
        """
        self.eval()
        with torch.no_grad():
            # 确保 batch 维度
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            outputs = self.forward(obs)
            
            actions = {}
            for head_name in HEAD_NAMES:
                q_values = outputs[head_name]  # [1, action_dim]
                
                # 获取对应的 legal_mask
                mask = legal_mask.get(head_name, None)
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.to(self.device)
                    else:
                        mask = torch.BoolTensor(mask).to(self.device)
                
                # 选择动作
                action = select_action(q_values, mask, epsilon, temperature)
                actions[head_name] = int(action.item())
            
            return actions
    
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """提取共享特征（用于 GRP 等辅助任务）"""
        return self.brain(obs)
    
    def get_config(self) -> dict:
        return self.config.to_dict()
    
    def clone_weights(self) -> 'ResNetDQN':
        """克隆权重（用于目标网络更新等）"""
        new_model = ResNetDQN(self.config)
        new_model.load_state_dict(self.state_dict())
        new_model.to(self.device)
        return new_model


class TransformerDQN(ModelBase):
    """
    扩展架构预留：Transformer + DQN
    
    用于未来可扩展的模型架构
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO: 实现 Transformer 架构
        raise NotImplementedError("TransformerDQN 尚未实现")
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
    
    def get_action(self, obs, legal_mask, epsilon=0.0, temperature=1.0):
        raise NotImplementedError()
    
    def get_config(self) -> dict:
        return self.config.to_dict()


class ModelFactory:
    """
    模型工厂
    
    根据 config.arch 创建不同架构的模型：
        - "resnet_dqn"    → ResNetDQN    (默认，参考 Mortal/Suphx)
        - "transformer_dqn" → TransformerDQN (未来扩展)
    """
    
    @staticmethod
    def create(config: ModelConfig) -> ModelBase:
        """根据架构名称创建模型"""
        if config.arch == "resnet_dqn":
            return ResNetDQN(config)
        elif config.arch == "transformer_dqn":
            return TransformerDQN(config)
        else:
            raise ValueError(f"未知模型架构: {config.arch}，支持: resnet_dqn, transformer_dqn")
    
    @staticmethod
    def create_from_dict(config_dict: dict) -> ModelBase:
        """从字典创建模型"""
        config = ModelConfig.from_dict(config_dict)
        return ModelFactory.create(config)
    
    @staticmethod
    def create_default() -> ResNetDQN:
        """创建默认 ResNetDQN"""
        return ResNetDQN(ModelConfig())


def create_model(config_dict: Optional[dict] = None) -> ModelBase:
    """便捷函数：创建模型"""
    if config_dict is None:
        return ModelFactory.create_default()
    return ModelFactory.create_from_dict(config_dict)


def test_resnet_dqn():
    """测试 ResNetDQN 完整前向传播"""
    print("=" * 60)
    print("测试 ResNetDQN 完整模型...")
    
    # 创建默认配置
    config = ModelConfig()
    model = ResNetDQN(config)
    print(f"总参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 4
    obs = torch.randn(batch_size, config.input_channels, config.seq_length)
    outputs = model(obs)
    
    print("\n各决策头输出形状:")
    for name, q in outputs.items():
        expected = HEAD_ACTION_DIMS[name]
        print(f"  {name:10s}: {tuple(q.shape):<20s} → 期望 [N, {expected}]")
        assert q.shape == (batch_size, expected), f"{name} 维度错误"
    
    # 测试 get_action
    legal_mask = {
        'swap': torch.tensor([True, True, True]),
        'missing': torch.tensor([True, True, True]),
        'discard': torch.ones(27, dtype=torch.bool),
        'pong': torch.tensor([True, False]),  # 不允许碰
        'kong': torch.tensor([True, False]),   # 不允许杠
        'win': torch.tensor([True, True]),
    }
    
    actions = model.get_action(obs[0], legal_mask, epsilon=0.0)
    print(f"\nget_action 结果 (无探索): {actions}")
    
    # 验证掩码约束
    assert actions['pong'] == 0, "碰动作掩码未生效"
    assert actions['kong'] == 0, "杠动作掩码未生效"
    
    # 测试 ε-greedy
    actions_random = model.get_action(obs[0], legal_mask, epsilon=1.0)
    print(f"get_action 结果 (ε=1.0): {actions_random}")
    
    # 测试 save/load
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pth")
        model.save(path)
        
        # 加载回新模型
        model2 = ResNetDQN(ModelConfig())
        model2.load(path)
        
        # 验证权重一致
        with torch.no_grad():
            out1 = model(obs)
            out2 = model2(obs)
            for name in outputs:
                assert torch.allclose(out1[name], out2[name], atol=1e-6), \
                    f"{name} 权重加载不一致"
        
        print("save/load ✓")
    
    # 测试 ModelFactory
    model3 = ModelFactory.create_default()
    assert model3.config.arch == "resnet_dqn"
    print("ModelFactory.create_default() ✓")
    
    print("\nResNetDQN 完整测试通过！")
    return model


if __name__ == "__main__":
    model = test_resnet_dqn()