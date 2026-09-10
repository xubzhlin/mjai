"""
6 个 Dueling DQN 决策头

| 头           | 输出维度 | 决策       | 说明                         |
| ----------- | ---- | -------- | -------------------------- |
| SwapHead    | 3    | 换三张选哪门花色 | 万/筒/条                    |
| MissingHead | 3    | 定哪门缺     | 万/筒/条                    |
| DiscardHead | 27   | 打哪张牌     | 27种牌选1                     |
| PongHead    | 2    | 是否碰      | 是/否                      |
| KongHead    | 2    | 是否杠      | 是/否                      |
| WinHead     | 2    | 是否胡      | 是/否                      |

Dueling 架构：分离 V(s) 和 A(s,a)
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .base import DuelingQHead
from .brain import Mish


class SwapHead(DuelingQHead):
    """换三张决策头：选哪门花色（万/筒/条 = 0/1/2）"""
    
    NUM_ACTIONS = 3
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, SwapHead.NUM_ACTIONS, hidden_dim)


class MissingHead(DuelingQHead):
    """定缺决策头：定哪门缺（万/筒/条 = 0/1/2）"""
    
    NUM_ACTIONS = 3
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, MissingHead.NUM_ACTIONS, hidden_dim)


class DiscardHead(DuelingQHead):
    """打牌决策头：27种牌选1"""
    
    NUM_ACTIONS = 27
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, DiscardHead.NUM_ACTIONS, hidden_dim)


class PongHead(DuelingQHead):
    """碰决策头：是/否"""
    
    NUM_ACTIONS = 2
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, PongHead.NUM_ACTIONS, hidden_dim)


class KongHead(DuelingQHead):
    """杠决策头：是/否"""
    
    NUM_ACTIONS = 2
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, KongHead.NUM_ACTIONS, hidden_dim)


class WinHead(DuelingQHead):
    """胡决策头：是/否"""
    
    NUM_ACTIONS = 2
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__(feature_dim, WinHead.NUM_ACTIONS, hidden_dim)


# 决策头名称映射
HEAD_NAMES = ['swap', 'missing', 'discard', 'pong', 'kong', 'win']
HEADS = {
    'swap': SwapHead,
    'missing': MissingHead,
    'discard': DiscardHead,
    'pong': PongHead,
    'kong': KongHead,
    'win': WinHead,
}
HEAD_ACTION_DIMS = {
    'swap': 3,
    'missing': 3,
    'discard': 27,
    'pong': 2,
    'kong': 2,
    'win': 2,
}


class MultiHeadDQN(nn.Module):
    """
    多头 DQN 容器
    共享输入特征，6 个独立决策头各自输出 Q 值
    
    用于训练时的完整前向传播
    """
    
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 6 个独立决策头
        self.heads = nn.ModuleDict({
            'swap': SwapHead(feature_dim, hidden_dim),
            'missing': MissingHead(feature_dim, hidden_dim),
            'discard': DiscardHead(feature_dim, hidden_dim),
            'pong': PongHead(feature_dim, hidden_dim),
            'kong': KongHead(feature_dim, hidden_dim),
            'win': WinHead(feature_dim, hidden_dim),
        })
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 共享特征 [batch_size, feature_dim]
            
        Returns:
            各决策头的 Q 值字典
        """
        return {name: head(features) for name, head in self.heads.items()}
    
    def get_value(self, features: torch.Tensor, head_name: str) -> torch.Tensor:
        """获取指定决策头的 V(s)"""
        return self.heads[head_name].get_value(features)
    
    def get_advantage(self, features: torch.Tensor, head_name: str) -> torch.Tensor:
        """获取指定决策头的 A(s,a)"""
        return self.heads[head_name].get_advantage(features)


class ResNetDQN(nn.Module):
    """
    ResNet + Dueling DQN 完整模型
    Brain 共享特征提取 + 6 个独立决策头
    """
    
    def __init__(self, brain, heads: MultiHeadDQN):
        super().__init__()
        self.brain = brain
        self.heads = heads
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """完整前向传播"""
        features = self.brain(obs)
        return self.heads(features)
    
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """提取共享特征"""
        return self.brain(obs)


def apply_action_mask(
    q_values: torch.Tensor,
    legal_mask: Optional[torch.Tensor],
    invalid_value: float = -1e9
) -> torch.Tensor:
    """
    应用动作掩码
    
    Args:
        q_values: Q 值 [batch_size, action_dim]
        legal_mask: 合法动作掩码 [batch_size, action_dim]，True 表示合法
        invalid_value: 非法动作的 Q 值设置
        
    Returns:
        掩码后的 Q 值
    """
    if legal_mask is None:
        return q_values
    
    # 确保 mask 形状正确
    if legal_mask.dim() == 1:
        legal_mask = legal_mask.unsqueeze(0)  # [1, action_dim]
    
    return torch.where(legal_mask, q_values, torch.full_like(q_values, invalid_value))


def select_action(
    q_values: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    epsilon: float = 0.0,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    选择动作
    
    Args:
        q_values: Q 值 [batch_size, action_dim]
        legal_mask: 合法动作掩码
        epsilon: ε-greedy 探索率
        temperature: Softmax 温度（>0 时使用温度采样）
        
    Returns:
        选择的动作索引 [batch_size]
    """
    batch_size = q_values.size(0)
    action_dim = q_values.size(1)
    
    # 应用动作掩码
    masked_q = apply_action_mask(q_values, legal_mask)
    
    # 如果使用温度采样
    if temperature > 0 and temperature != 1.0:
        probs = F.softmax(masked_q / temperature, dim=-1)
        # 重新归一化（确保合法动作概率和为 1）
        if legal_mask is not None:
            probs = probs * legal_mask.float()
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # ε-greedy
    if epsilon > 0:
        greedy_actions = masked_q.argmax(dim=-1)  # [batch_size]
        
        # 生成随机探索
        random_actions = torch.randint(0, action_dim, (batch_size,), device=q_values.device)
        
        # 如果有合法动作约束随机探索
        if legal_mask is not None:
            # 对每个样本，从合法动作中随机选一个
            legal_actions = []
            for i in range(batch_size):
                mask_i = legal_mask[i]
                legal_indices = torch.where(mask_i)[0]
                if len(legal_indices) > 0:
                    legal_actions.append(legal_indices[torch.randint(len(legal_indices), (1,))])
                else:
                    legal_actions.append(torch.tensor(0, device=q_values.device))
            random_actions = torch.stack(legal_actions).squeeze(-1)
        
        # ε-greedy 选择
        is_exploring = torch.rand(batch_size, device=q_values.device) < epsilon
        return torch.where(is_exploring, random_actions, greedy_actions)
    
    # 纯 greedy
    return masked_q.argmax(dim=-1)


if __name__ == "__main__":
    print("测试决策头...")
    
    feature_dim = 1024
    hidden_dim = 256
    batch_size = 4
    
    # 测试各决策头
    heads = {
        'swap': SwapHead(feature_dim, hidden_dim),
        'missing': MissingHead(feature_dim, hidden_dim),
        'discard': DiscardHead(feature_dim, hidden_dim),
        'pong': PongHead(feature_dim, hidden_dim),
        'kong': KongHead(feature_dim, hidden_dim),
        'win': WinHead(feature_dim, hidden_dim),
    }
    
    features = torch.randn(batch_size, feature_dim)
    
    for name, head in heads.items():
        q_values = head(features)
        expected_dim = HEAD_ACTION_DIMS[name]
        assert q_values.shape == (batch_size, expected_dim), \
            f"{name}: 期望 {(batch_size, expected_dim)}, 实际 {q_values.shape}"
        print(f"  {name}: {q_values.shape} ✓")
    
    # 测试 MultiHeadDQN
    multi_head = MultiHeadDQN(feature_dim, hidden_dim)
    outputs = multi_head(features)
    for name, q in outputs.items():
        expected_dim = HEAD_ACTION_DIMS[name]
        assert q.shape == (batch_size, expected_dim), f"{name} 输出错误"
    
    print(f"MultiHeadDQN 总参数量: {sum(p.numel() for p in multi_head.parameters()):,}")
    print("决策头测试通过！")