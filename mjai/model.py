"""
川麻将神经网络模型
基于 PyTorch 实现，输入132维特征向量，输出动作概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os

class MahjongNet(nn.Module):
    """川麻将神经网络模型"""
    
    def __init__(self, 
                 input_dim: int = 132,
                 hidden_dims: List[int] = [512, 256, 128],
                 action_dim: int = 34,
                 dropout_rate: float = 0.2):
        """
        初始化神经网络模型
        
        Args:
            input_dim: 输入特征维度 (132)
            hidden_dims: 隐藏层维度列表
            action_dim: 动作空间维度 (34种可能的动作)
            dropout_rate: Dropout率
        """
        super(MahjongNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率分布
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量 [batch_size, input_dim]
            
        Returns:
            动作概率分布 [batch_size, action_dim]
        """
        return self.network(x)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测动作概率
        
        Args:
            features: 输入特征 [input_dim]
            
        Returns:
            动作概率 [action_dim]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            probabilities = self.forward(x).squeeze(0).numpy()
        return probabilities
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'action_dim': self.action_dim,
            'dropout_rate': self.dropout_rate
        }, path)
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            action_dim=checkpoint['action_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")
        return model

class ValueNet(nn.Module):
    """价值网络，用于评估当前游戏状态的价值"""
    
    def __init__(self, 
                 input_dim: int = 132,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.2):
        """
        初始化价值网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout率
        """
        super(ValueNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层（单个价值值）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征张量 [batch_size, input_dim]
            
        Returns:
            价值评估 [batch_size, 1]
        """
        return self.network(x)
    
    def evaluate(self, features: np.ndarray) -> float:
        """
        评估状态价值
        
        Args:
            features: 输入特征 [input_dim]
            
        Returns:
            价值评估值
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            value = self.forward(x).squeeze(0).item()
        return value
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        }, path)
        print(f"价值网络已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"价值网络已从 {path} 加载")
        return model

class MahjongModel:
    """完整的川麻将模型，包含策略网络和价值网络"""
    
    def __init__(self, 
                 policy_net: Optional[MahjongNet] = None,
                 value_net: Optional[ValueNet] = None,
                 learning_rate: float = 0.001):
        """
        初始化模型
        
        Args:
            policy_net: 策略网络
            value_net: 价值网络
            learning_rate: 学习率
        """
        if policy_net is None:
            policy_net = MahjongNet()
        if value_net is None:
            value_net = ValueNet()
        
        self.policy_net = policy_net
        self.value_net = value_net
        self.learning_rate = learning_rate
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), 
            lr=learning_rate
        )
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测动作概率和价值
        
        Args:
            features: 输入特征 [input_dim]
            
        Returns:
            (动作概率, 价值评估)
        """
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            # 策略网络预测
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(x).squeeze(0).cpu().numpy()
            
            # 价值网络评估
            value = self.value_net(x).squeeze(0).cpu().item()
        
        return action_probs, value
    
    def update(self, 
               features: np.ndarray, 
               actions: np.ndarray, 
               rewards: np.ndarray,
               old_probs: np.ndarray):
        """
        更新模型参数
        
        Args:
            features: 特征数组 [batch_size, input_dim]
            actions: 动作数组 [batch_size]
            rewards: 奖励数组 [batch_size]
            old_probs: 旧的概率数组 [batch_size, action_dim]
        """
        self.policy_net.train()
        self.value_net.train()
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        old_probs_tensor = torch.FloatTensor(old_probs).to(self.device)
        
        # 计算优势函数
        values = self.value_net(features_tensor).squeeze()
        advantages = rewards_tensor - values.detach()
        
        # 策略损失
        new_probs = self.policy_net(features_tensor)
        action_probs = new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        ratio = action_probs / (old_probs_tensor + 1e-8)
        
        # PPO损失
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()
        
        # 价值损失
        value_loss = self.value_loss_fn(values, rewards_tensor)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        return total_loss.item()
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存策略网络
        policy_path = path.replace('.pth', '_policy.pth')
        self.policy_net.save(policy_path)
        
        # 保存价值网络
        value_path = path.replace('.pth', '_value.pth')
        self.value_net.save(value_path)
        
        # 保存配置
        config = {
            'learning_rate': self.learning_rate,
            'policy_path': policy_path,
            'value_path': value_path
        }
        
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        # 加载配置
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载策略网络
        policy_net = MahjongNet.load(config['policy_path'])
        
        # 加载价值网络
        value_net = ValueNet.load(config['value_path'])
        
        # 创建模型
        model = cls(policy_net=policy_net, value_net=value_net)
        model.learning_rate = config['learning_rate']
        
        print(f"模型已从 {path} 加载")
        return model

def create_model_from_config(config_path: str) -> MahjongModel:
    """从配置文件创建模型"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    policy_net = MahjongNet(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        action_dim=config['action_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    value_net = ValueNet(
        input_dim=config['input_dim'],
        hidden_dims=config['value_hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    
    return MahjongModel(policy_net=policy_net, value_net=value_net)

if __name__ == "__main__":
    # 测试模型
    print("测试神经网络模型...")
    
    # 创建模型
    model = MahjongModel()
    
    # 测试输入
    features = np.random.randn(132)
    action_probs, value = model.predict(features)
    
    print(f"动作概率维度: {action_probs.shape}")
    print(f"价值评估: {value}")
    print(f"动作概率和: {action_probs.sum()}")
    
    # 测试更新
    batch_size = 32
    batch_features = np.random.randn(batch_size, 132)
    batch_actions = np.random.randint(0, 34, batch_size)
    batch_rewards = np.random.randn(batch_size)
    batch_old_probs = np.random.rand(batch_size, 34)
    
    loss = model.update(batch_features, batch_actions, batch_rewards, batch_old_probs)
    print(f"训练损失: {loss}")
    
    # 保存模型
    model.save("e:/ai/mjai/models/test_model.pth")
    
    print("模型测试完成！")