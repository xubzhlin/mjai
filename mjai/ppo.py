"""
PPO (Proximal Policy Optimization) 算法实现
用于策略梯度强化学习训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
import copy

from mjai.advanced_model import AdvancedMahjongNet


@dataclass
class PPOConfig:
    """PPO配置"""
    # 网络配置
    input_dim: int = 132
    action_dim: int = 34
    
    # PPO超参数
    clip_ratio: float = 0.2          # PPO裁剪比率
    ppo_epochs: int = 4             # PPO更新轮数
    batch_size: int = 64            # 批次大小
    
    # 优化器配置
    learning_rate: float = 3e-4      # 学习率
    beta1: float = 0.9               # Adam beta1
    beta2: float = 0.999             # Adam beta2
    weight_decay: float = 1e-5       # 权重衰减
    
    # 损失权重
    value_loss_coef: float = 0.5     # 价值损失系数
    entropy_coef: float = 0.01      # 熵正则化系数
    max_grad_norm: float = 0.5       # 梯度裁剪
    
    # GAE配置
    gamma: float = 0.99             # 折扣因子
    gae_lambda: float = 0.95        # GAE lambda
    
    # 学习率调度
    use_lr_schedule: bool = True
    lr_schedule_type: str = 'linear'  # linear, cosine, exponential
    min_lr: float = 1e-5
    
    # 探索
    entropy_target: float = 2.5     # 目标熵值
    use_entropy_adjust: bool = False


class ActorCriticNet(nn.Module):
    """Actor-Critic 网络"""
    
    def __init__(self, 
                 input_dim: int = 132,
                 action_dim: int = 34,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.2):
        super(ActorCriticNet, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Actor (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            (动作概率, 价值估计)
        """
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        
        return action_probs, value
    
    def act(self, x: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择动作
        
        Args:
            x: 输入特征 [batch_size, input_dim] (batch_size=1)
            
        Returns:
            (动作, 对数概率, 价值, 熵)
        """
        action_probs, value = self.forward(x)
        
        # 创建分布并采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算对数概率和熵
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, value.squeeze(0), entropy
    
    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            actions: 动作 [batch_size]
            
        Returns:
            (对数概率, 价值, 熵)
        """
        action_probs, values = self.forward(x)
        
        # 创建分布
        dist = Categorical(action_probs)
        
        # 计算对数概率
        log_probs = dist.log_prob(actions)
        
        # 计算熵
        entropies = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropies


class PPO:
    """PPO算法实现"""
    
    def __init__(self, config: PPOConfig):
        """
        初始化PPO算法
        
        Args:
            config: PPO配置
        """
        self.config = config
        
        # 创建网络
        self.policy_net = ActorCriticNet(
            input_dim=config.input_dim,
            action_dim=config.action_dim
        )
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        if config.use_lr_schedule:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=0.95
            )
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        
        # 状态
        self.iteration = 0
        self.stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'learning_rate': config.learning_rate
        }
    
    def select_action(self, state: np.ndarray) -> Dict:
        """
        选择动作
        
        Args:
            state: 状态数组 [input_dim]
            
        Returns:
            动作信息字典
        """
        self.policy_net.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value, entropy = self.policy_net.act(x)
            
            return {
                'action': action,
                'log_prob': log_prob.item(),
                'value': value.item(),
                'entropy': entropy.item()
            }
    
    def compute_gae(self, 
                   rewards: np.ndarray, 
                   values: np.ndarray, 
                   next_values: np.ndarray, 
                   dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            next_values: 下一状态价值序列
            dones: 结束标志
            
        Returns:
            (优势函数, 回报)
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            # TD误差
            delta = rewards[t] + self.config.gamma * next_values[t] * (1 - dones[t]) - values[t]
            
            # GAE累积
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
        
        # 计算回报
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, 
               states: np.ndarray,
               actions: np.ndarray,
               old_log_probs: np.ndarray,
               advantages: np.ndarray,
               returns: np.ndarray,
               values: np.ndarray) -> Dict:
        """
        更新策略网络
        
        Args:
            states: 状态 [batch_size, input_dim]
            actions: 动作 [batch_size]
            old_log_probs: 旧对数概率 [batch_size]
            advantages: 优势函数 [batch_size]
            returns: 回报 [batch_size]
            values: 旧价值估计 [batch_size]
            
        Returns:
            更新统计信息
        """
        self.policy_net.train()
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新循环
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        
        for epoch in range(self.config.ppo_epochs):
            # 生成随机索引
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.config.batch_size):
                # 获取批次索引
                batch_idx = indices[start_idx:start_idx + self.config.batch_size]
                
                # 获取批次数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_values = values[batch_idx]
                
                # 前向传播
                new_log_probs, new_values, entropy = self.policy_net.evaluate(
                    batch_states, batch_actions
                )
                
                # 计算重要性采样比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                
                # 熵损失（鼓励探索）
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # 计算近似KL散度
                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                
                # 计算裁剪比例
                clip_fraction = (ratio.abs() > self.config.clip_ratio).float().mean().item()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    self.config.max_grad_norm
                )
                
                # 更新参数
                self.optimizer.step()
                
                # 更新累积损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
        
        # 平均损失
        num_batches = (len(states) + self.config.batch_size - 1) // self.config.batch_size
        num_updates = num_batches * self.config.ppo_epochs
        
        self.stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'total_loss': (total_policy_loss + 
                          self.config.value_loss_coef * total_value_loss + 
                          self.config.entropy_coef * total_entropy) / num_updates,
            'approx_kl': total_approx_kl / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 更新学习率
        if self.config.use_lr_schedule:
            self.scheduler.step()
        
        self.iteration += 1
        
        return self.stats
    
    def get_stats(self) -> Dict:
        """获取训练统计信息"""
        return self.stats.copy()
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'iteration': self.iteration,
            'stats': self.stats
        }, path)
        
        print(f"PPO模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.stats = checkpoint.get('stats', {})
        
        print(f"PPO模型已从 {path} 加载")
    
    def get_model(self) -> ActorCriticNet:
        """获取策略网络"""
        return self.policy_net
    
    def set_model(self, model: ActorCriticNet):
        """设置策略网络"""
        self.policy_net = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )


class RolloutBuffer:
    """Rollout缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取数据"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones)
        )
    
    def __len__(self):
        return len(self.states)


def create_ppo_from_config(config_dict: Dict) -> PPO:
    """
    从配置字典创建PPO
    
    Args:
        config_dict: 配置字典
        
    Returns:
        PPO实例
    """
    config = PPOConfig(
        input_dim=config_dict.get('input_dim', 132),
        action_dim=config_dict.get('action_dim', 34),
        clip_ratio=config_dict.get('clip_ratio', 0.2),
        ppo_epochs=config_dict.get('ppo_epochs', 4),
        batch_size=config_dict.get('batch_size', 64),
        learning_rate=config_dict.get('learning_rate', 3e-4),
        value_loss_coef=config_dict.get('value_loss_coef', 0.5),
        entropy_coef=config_dict.get('entropy_coef', 0.01),
        max_grad_norm=config_dict.get('max_grad_norm', 0.5),
        gamma=config_dict.get('gamma', 0.99),
        gae_lambda=config_dict.get('gae_lambda', 0.95)
    )
    
    return PPO(config)


if __name__ == "__main__":
    # 测试PPO算法
    print("测试PPO算法...")
    
    # 创建配置
    config = PPOConfig(
        input_dim=132,
        action_dim=34,
        clip_ratio=0.2,
        ppo_epochs=4,
        batch_size=64,
        learning_rate=3e-4
    )
    
    # 创建PPO
    ppo = PPO(config)
    
    # 测试选择动作
    state = np.random.randn(132)
    action_info = ppo.select_action(state)
    print(f"选择动作: {action_info}")
    
    # 测试更新
    batch_size = 128
    states = np.random.randn(batch_size, 132)
    actions = np.random.randint(0, 34, batch_size)
    old_log_probs = np.random.randn(batch_size)
    advantages = np.random.randn(batch_size)
    returns = np.random.randn(batch_size)
    values = np.random.randn(batch_size)
    
    stats = ppo.update(states, actions, old_log_probs, advantages, returns, values)
    print(f"更新统计: {stats}")
    
    # 测试保存和加载
    ppo.save("test_ppo.pth")
    ppo.load("test_ppo.pth")
    
    import os
    if os.path.exists("test_ppo.pth"):
        os.remove("test_ppo.pth")
    
    print("PPO算法测试完成！")