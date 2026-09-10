"""
ModelBase 抽象接口
所有神经网络模型的基类，确保模型层可独立替换
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class ModelBase(ABC, nn.Module):
    """
    模型抽象基类
    
    所有模型必须实现的接口：
    - forward: 前向传播，返回各决策头的输出
    - get_action: 根据观测和 legal_mask 选择动作
    - predict: 推理模式，用于自博弈
    - save/load: 模型持久化
    """
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测张量 [N, C, 27] 或 [N, feature_dim]
            
        Returns:
            字典，包含各决策头的输出：
            {
                'swap': Tensor [N, 3],       # 换三张选哪门花色
                'missing': Tensor [N, 3],    # 定哪门缺
                'discard': Tensor [N, 27],   # 打哪张牌
                'pong': Tensor [N, 2],       # 是否碰
                'kong': Tensor [N, 2],       # 是否杠
                'win': Tensor [N, 2],        # 是否胡
            }
        """
        ...
    
    @abstractmethod
    def get_action(
        self,
        obs: torch.Tensor,
        legal_mask: Dict[str, torch.Tensor],
        epsilon: float = 0.0
    ) -> Dict[str, int]:
        """
        根据观测和合法动作掩码选择动作
        
        Args:
            obs: 观测张量
            legal_mask: 各决策头的合法动作掩码
            epsilon: ε-greedy 探索率
            
        Returns:
            各决策头选择的动作索引
            {
                'swap': 0-2 或 None,
                'missing': 0-2 或 None,
                'discard': 0-27,
                'pong': 0-1,
                'kong': 0-1,
                'win': 0-1,
            }
        """
        ...
    
    def predict(self, obs: np.ndarray, legal_mask: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        推理模式：返回各决策头的 Q 值分布
        
        Args:
            obs: 观测数组
            legal_mask: 合法动作掩码（可选）
            
        Returns:
            各决策头的 Q 值数组
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            outputs = self.forward(obs_tensor)
            
            result = {}
            for key, value in outputs.items():
                result[key] = value.squeeze(0).cpu().numpy()
            
            # 如果有合法动作掩码，屏蔽非法动作
            if legal_mask is not None:
                for key in result:
                    if key in legal_mask:
                        mask = legal_mask[key]
                        if isinstance(mask, np.ndarray):
                            result[key] = np.where(mask, result[key], -np.inf)
            
            return result
    
    def save(self, path: str):
        """保存模型"""
        import os
        import json
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # 保存模型状态和配置
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }
        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型权重（不改变配置）"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")
    
    @abstractmethod
    def get_config(self) -> dict:
        """获取模型配置，用于序列化"""
        ...
    
    def clone(self) -> 'ModelBase':
        """克隆模型（深拷贝权重）"""
        import copy
        new_model = copy.deepcopy(self)
        new_model.load_state_dict(self.state_dict())
        return new_model
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def to_device(self, device: Optional[torch.device] = None):
        """移动模型到指定设备"""
        if device is None:
            device = self.device
        self.device = device
        self.to(device)


class DuelingQHead(nn.Module):
    """
    Dueling DQN 决策头
    
    将特征表示分为价值函数 V(s) 和优势函数 A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 价值函数流
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.action_dim = action_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征向量 [batch_size, feature_dim]
            
        Returns:
            Q 值 [batch_size, action_dim]
        """
        V = self.value_stream(x)          # [batch_size, 1]
        A = self.advantage_stream(x)       # [batch_size, action_dim]
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        
        return Q
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """获取状态价值 V(s)"""
        return self.value_stream(x)
    
    def get_advantage(self, x: torch.Tensor) -> torch.Tensor:
        """获取优势函数 A(s,a)"""
        return self.advantage_stream(x)