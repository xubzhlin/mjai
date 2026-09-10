"""
川麻将高级神经网络模型
实现 ResNet + Dueling DQN + GRP 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 短连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet架构用于川麻将特征提取"""
    
    def __init__(self, 
                 input_dim: int = 132,
                 base_channels: int = 64,
                 num_blocks: List[int] = [2, 2, 2],
                 dropout_rate: float = 0.2):
        super(ResNet, self).__init__()
        
        self.input_dim = input_dim
        self.base_channels = base_channels
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [batch_size, input_dim]
        x = x.unsqueeze(1)  # 添加通道维度: [batch_size, 1, input_dim]
        
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 展平: [batch_size, base_channels * 4]
        x = x.view(x.size(0), -1)
        
        return x


class DuelingDQN(nn.Module):
    """Dueling DQN架构"""
    
    def __init__(self, 
                 feature_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.2):
        super(DuelingDQN, self).__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 共享层
        self.shared_layers = nn.ModuleList()
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            self.shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 共享层
        features = x
        for layer in self.shared_layers:
            features = layer(features)
        
        # 计算价值和优势
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling组合
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class GRP(nn.Module):
    """Gated Recurrent Processor (门控循环处理器)"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2):
        super(GRP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 门控处理
        gate = self.gate(lstm_out)
        gated_output = gate * lstm_out
        
        # 输出处理
        output = self.output_layer(gated_output)
        
        return output, hidden


class AdvancedMahjongNet(nn.Module):
    """高级川麻将神经网络 - ResNet + Dueling DQN + GRP"""
    
    def __init__(self, 
                 input_dim: int = 132,
                 action_dim: int = 34,
                 base_channels: int = 64,
                 hidden_dims: List[int] = [512, 256],
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 dropout_rate: float = 0.2):
        super(AdvancedMahjongNet, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.base_channels = base_channels
        self.hidden_dims = hidden_dims
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        
        # ResNet特征提取器
        self.resnet = ResNet(
            input_dim=input_dim,
            base_channels=base_channels,
            num_blocks=[2, 2, 2],
            dropout_rate=dropout_rate
        )
        
        # ResNet输出维度
        resnet_output_dim = base_channels * 4
        
        # Dueling DQN
        self.dueling_dqn = DuelingDQN(
            feature_dim=resnet_output_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # GRP (门控循环处理器)
        self.grp = GRP(
            input_dim=resnet_output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout_rate=dropout_rate
        )
        
        # 最终决策层
        self.final_layer = nn.Sequential(
            nn.Linear(resnet_output_dim + lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # ResNet特征提取
        resnet_features = self.resnet(x)
        
        # Dueling DQN Q值计算
        q_values = self.dueling_dqn(resnet_features)
        
        # GRP处理
        grp_output, hidden = self.grp(resnet_features.unsqueeze(1), hidden)
        grp_output = grp_output.squeeze(1)
        
        # 组合特征
        combined_features = torch.cat([resnet_features, grp_output], dim=1)
        
        # 最终决策
        action_probs = self.final_layer(combined_features)
        
        return action_probs, hidden
    
    def predict(self, features: np.ndarray, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[np.ndarray, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        预测动作概率和价值
        
        Args:
            features: 输入特征 [input_dim]
            hidden: GRP隐藏状态
            
        Returns:
            (动作概率, Q值, 新的隐藏状态)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            action_probs, hidden = self.forward(x, hidden)
            q_values = self.dueling_dqn(self.resnet(x))
            
            action_probs = action_probs.squeeze(0).cpu().numpy()
            q_value = q_values.max().item()
        
        return action_probs, q_value, hidden
    
    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """获取Q值"""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            q_values = self.dueling_dqn(self.resnet(x))
            return q_values.squeeze(0).cpu().numpy()
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'base_channels': self.base_channels,
            'hidden_dims': self.hidden_dims,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'dropout_rate': self.dropout_rate
        }, path)
        print(f"高级模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            input_dim=checkpoint['input_dim'],
            action_dim=checkpoint['action_dim'],
            base_channels=checkpoint['base_channels'],
            hidden_dims=checkpoint['hidden_dims'],
            lstm_hidden_dim=checkpoint['lstm_hidden_dim'],
            lstm_layers=checkpoint['lstm_layers'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"高级模型已从 {path} 加载")
        return model


class AdvancedMahjongModel:
    """完整的川麻将高级模型"""
    
    def __init__(self, 
                 model: Optional[AdvancedMahjongNet] = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        """
        初始化高级模型
        
        Args:
            model: 高级神经网络模型
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
        """
        if model is None:
            model = AdvancedMahjongNet()
        
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9
        )
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 经验回放缓冲区
        self.buffer = []
        self.buffer_size = 10000
    
    def predict(self, features: np.ndarray, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[int, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        预测动作并选择
        
        Args:
            features: 输入特征 [input_dim]
            hidden: GRP隐藏状态
            
        Returns:
            (选择的动作, Q值, 新的隐藏状态)
        """
        # 获取动作概率和Q值
        action_probs, q_value, new_hidden = self.model.predict(features, hidden)
        
        # epsilon-greedy选择
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.model.action_dim)
        else:
            action = np.argmax(action_probs)
        
        return action, q_value, new_hidden
    
    def update(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        更新模型
        
        Args:
            batch: (states, actions, rewards, next_states, dones, old_q_values)
            
        Returns:
            训练统计信息
        """
        states, actions, rewards, next_states, dones, old_q_values = batch
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_q_values = torch.FloatTensor(old_q_values).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.model.dueling_dqn(self.model.resnet(states))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.model.dueling_dqn(self.model.resnet(next_states))
            target_q_values = rewards + (self.gamma * next_q_values.max(dim=1)[0] * (~dones))
        
        # 计算损失
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(q_values, target_q_values)
        
        # 更新参数
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step()
        
        # 更新探索率
        self.epsilon = max(0.01, self.epsilon * 0.9995)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, old_q_value: float):
        """添加经验到缓冲区"""
        experience = (state, action, reward, next_state, done, old_q_value)
        self.buffer.append(experience)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample_batch(self, batch_size: int = 32) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """从缓冲区采样批次"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in batch]
        
        states, actions, rewards, next_states, dones, old_q_values = zip(*experiences)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(old_q_values)
        )
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # 保存额外信息
        config = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'buffer_size': self.buffer_size,
            'buffer_length': len(self.buffer)
        }
        
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"高级模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        # 加载配置
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载模型
        model = AdvancedMahjongNet.load(path)
        
        # 创建高级模型
        advanced_model = cls(
            model=model,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            epsilon=config['epsilon']
        )
        
        # 恢复缓冲区大小
        advanced_model.buffer_size = config['buffer_size']
        
        print(f"高级模型已从 {path} 加载")
        return advanced_model


def create_advanced_model_from_config(config_path: str) -> AdvancedMahjongModel:
    """从配置文件创建高级模型"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = AdvancedMahjongNet(
        input_dim=config['input_dim'],
        action_dim=config['action_dim'],
        base_channels=config['base_channels'],
        hidden_dims=config['hidden_dims'],
        lstm_hidden_dim=config['lstm_hidden_dim'],
        lstm_layers=config['lstm_layers'],
        dropout_rate=config['dropout_rate']
    )
    
    return AdvancedMahjongModel(
        model=model,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon']
    )


if __name__ == "__main__":
    # 测试高级模型
    print("测试高级神经网络模型...")
    
    # 创建模型
    model = AdvancedMahjongModel()
    
    # 测试输入
    features = np.random.randn(132)
    action, q_value, hidden = model.predict(features)
    
    print(f"动作: {action}")
    print(f"Q值: {q_value}")
    print(f"隐藏状态形状: {[h.shape for h in hidden]}")
    
    # 测试更新
    batch_size = 32
    batch_states = np.random.randn(batch_size, 132)
    batch_actions = np.random.randint(0, 34, batch_size)
    batch_rewards = np.random.randn(batch_size)
    batch_next_states = np.random.randn(batch_size, 132)
    batch_dones = np.random.randint(0, 2, batch_size).astype(bool)
    batch_old_q_values = np.random.randn(batch_size)
    
    # 添加经验
    for i in range(batch_size):
        model.add_experience(
            batch_states[i], 
            batch_actions[i], 
            batch_rewards[i], 
            batch_next_states[i], 
            batch_dones[i], 
            batch_old_q_values[i]
        )
    
    # 更新模型
    stats = model.update(model.sample_batch(batch_size))
    print(f"训练统计: {stats}")
    
    # 保存模型
    model.save("e:/ai/mjai/models/advanced_model.pth")
    
    print("高级模型测试完成！")