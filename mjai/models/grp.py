"""
GRP (Game Result Predictor) 奖励预测器
基于 GRU 网络架构，预测川麻将血战到底的最终得分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
import json
import os
from dataclasses import dataclass

@dataclass
class GRPConfig:
    """GRP 配置类"""
    input_dim: int = 132  # 输入特征维度
    hidden_dim: int = 64  # GRU 隐藏层维度
    num_layers: int = 2   # GRU 层数
    dropout_rate: float = 0.1  # Dropout 率
    use_bidirectional: bool = True  # 是否使用双向 GRU
    output_dim: int = 1   # 输出维度（预测得分）
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GRPConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_bidirectional': self.use_bidirectional,
            'output_dim': self.output_dim,
        }

class GRP(nn.Module):
    """
    GRP (Game Result Predictor) 网络
    使用 GRU 架构预测川麻将血战到底的最终得分
    """
    
    def __init__(self, config: GRPConfig):
        super(GRP, self).__init__()
        
        self.config = config
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            bidirectional=config.use_bidirectional,
            batch_first=True
        )
        
        # 计算双向 GRU 的输出维度
        gru_output_dim = config.hidden_dim * 2 if config.use_bidirectional else config.hidden_dim
        
        # 全连接层
        self.fc1 = nn.Linear(gru_output_dim, gru_output_dim // 2)
        self.bn1 = nn.BatchNorm1d(gru_output_dim // 2)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        
        self.fc2 = nn.Linear(gru_output_dim // 2, gru_output_dim // 4)
        self.bn2 = nn.BatchNorm1d(gru_output_dim // 4)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        
        # 输出层
        self.fc3 = nn.Linear(gru_output_dim // 4, config.output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # GRU 权重初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 全连接层权重初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            hidden: 初始隐藏状态 [num_layers * num_directions, batch_size, hidden_dim]
            
        Returns:
            output: 输出序列 [batch_size, seq_len, output_dim]
            hidden: 最终隐藏状态 [num_layers * num_directions, batch_size, hidden_dim]
        """
        # GRU 前向传播
        gru_output, hidden = self.gru(x, hidden)
        
        # 取最后一个时间步的输出
        last_output = gru_output[:, -1, :]  # [batch_size, gru_output_dim]
        
        # 全连接层前向传播
        fc1_output = self.fc1(last_output)
        fc1_output = self.bn1(fc1_output)
        fc1_output = F.relu(fc1_output)
        fc1_output = self.dropout1(fc1_output)
        
        fc2_output = self.fc2(fc1_output)
        fc2_output = self.bn2(fc2_output)
        fc2_output = F.relu(fc2_output)
        fc2_output = self.dropout2(fc2_output)
        
        # 最终输出（使用 sigmoid 激活函数，确保得分在合理范围内）
        output = torch.sigmoid(self.fc3(fc2_output)) * 100  # 假设最大得分为 100
        
        return output, hidden
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        预测单个序列的最终得分
        
        Args:
            sequence: 输入序列 [seq_len, input_dim]
            
        Returns:
            预测得分
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            x = torch.FloatTensor(sequence).unsqueeze(0)  # [1, seq_len, input_dim]
            
            # 前向传播
            output, _ = self.forward(x)
            
            # 返回预测得分
            return output.squeeze(0).item()
    
    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        批量预测多个序列的最终得分
        
        Args:
            sequences: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            预测得分数组 [batch_size]
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            x = torch.FloatTensor(sequences)  # [batch_size, seq_len, input_dim]
            
            # 前向传播
            output, _ = self.forward(x)
            
            # 返回预测得分
            return output.squeeze(-1).numpy()
    
    def get_hidden_state(self, sequence: np.ndarray) -> np.ndarray:
        """
        获取序列的隐藏状态
        
        Args:
            sequence: 输入序列 [seq_len, input_dim]
            
        Returns:
            隐藏状态 [num_layers * num_directions, 1, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            x = torch.FloatTensor(sequence).unsqueeze(0)  # [1, seq_len, input_dim]
            
            # 获取隐藏状态
            _, hidden = self.forward(x)
            
            return hidden.numpy()
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, path)
        print(f"GRP 模型已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GRP':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # 创建模型
        config = GRPConfig.from_dict(checkpoint['config'])
        model = cls(config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"GRP 模型已从 {path} 加载")
        return model
    
    def reset_parameters(self):
        """重置模型参数"""
        self._initialize_weights()

class GRPLoss(nn.Module):
    """
    GRP 损失函数
    使用 MSE 损失函数，同时考虑预测误差的时序一致性
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        """
        初始化损失函数
        
        Args:
            alpha: MSE 损失的权重
            beta: 时序一致性的权重
        """
        super(GRPLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                sequence_predictions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 最终预测得分 [batch_size, 1]
            targets: 实际得分 [batch_size, 1]
            sequence_predictions: 序列预测得分 [batch_size, seq_len, 1]
            
        Returns:
            总损失
        """
        # MSE 损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 时序一致性损失（如果提供了序列预测）
        temporal_loss = torch.tensor(0.0)
        if sequence_predictions is not None:
            # 计算相邻时间步预测的差异
            diff = torch.abs(sequence_predictions[:, 1:, :] - sequence_predictions[:, :-1, :])
            temporal_loss = torch.mean(diff)
        
        # 总损失
        total_loss = self.alpha * mse_loss + self.beta * temporal_loss
        
        return total_loss

class GRPTrainer:
    """
    GRP 训练器
    负责 GRP 模型的训练和评估
    """
    
    def __init__(self, model: GRP, learning_rate: float = 0.001, 
                 alpha: float = 1.0, beta: float = 0.1):
        """
        初始化训练器
        
        Args:
            model: GRP 模型
            learning_rate: 学习率
            alpha: MSE 损失的权重
            beta: 时序一致性的权重
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = GRPLoss(alpha=alpha, beta=beta)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    def train_step(self, sequences: torch.Tensor, targets: torch.Tensor,
                   sequence_sequences: Optional[torch.Tensor] = None) -> float:
        """
        训练一步
        
        Args:
            sequences: 输入序列 [batch_size, seq_len, input_dim]
            targets: 目标得分 [batch_size, 1]
            sequence_sequences: 序列输入 [batch_size, seq_len, seq_len, input_dim]
            
        Returns:
            损失值
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predictions, _ = self.model(sequences)
        
        # 计算损失
        if sequence_sequences is not None:
            # 计算序列预测
            seq_predictions = []
            for i in range(sequence_sequences.size(1)):
                seq_input = sequence_sequences[:, i, :, :]  # [batch_size, seq_len, input_dim]
                seq_pred, _ = self.model(seq_input)
                seq_predictions.append(seq_pred)
            sequence_predictions = torch.stack(seq_predictions, dim=1)  # [batch_size, seq_len, 1]
        else:
            sequence_predictions = None
        
        loss = self.criterion(predictions, targets, sequence_predictions)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, sequences: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        评估模型
        
        Args:
            sequences: 输入序列 [batch_size, seq_len, input_dim]
            targets: 目标得分 [batch_size, 1]
            
        Returns:
            (平均损失, 平均绝对误差)
        """
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            predictions, _ = self.model(sequences)
            
            # 计算损失
            loss = self.criterion(predictions, targets)
            
            # 计算绝对误差
            mae = torch.abs(predictions - targets).mean()
        
        return loss.item(), mae.item()
    
    def predict(self, sequences: torch.Tensor) -> np.ndarray:
        """
        预测得分
        
        Args:
            sequences: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            预测得分 [batch_size]
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions, _ = self.model(sequences)
            return predictions.squeeze(-1).cpu().numpy()
    
    def save(self, path: str):
        """保存训练器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        model_path = path.replace('.pth', '_model.pth')
        self.model.save(model_path)
        
        # 保存训练器状态
        trainer_state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_path': model_path,
        }
        
        torch.save(trainer_state, path)
        print(f"训练器已保存到: {path}")
    
    def load(self, path: str):
        """加载训练器"""
        trainer_state = torch.load(path, map_location='cpu')
        
        # 加载模型
        self.model.load(trainer_state['model_path'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(trainer_state['scheduler_state_dict'])
        
        print(f"训练器已从 {path} 加载")

def create_grp_from_config(config_path: str) -> GRP:
    """从配置文件创建 GRP 模型"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = GRPConfig.from_dict(config_dict)
    return GRP(config)

def test_grp():
    """测试 GRP 模型"""
    print("测试 GRP 奖励预测器...")
    
    # 创建配置
    config = GRPConfig(
        input_dim=132,
        hidden_dim=64,
        num_layers=2,
        dropout_rate=0.1,
        use_bidirectional=True,
        output_dim=1
    )
    
    # 创建模型
    model = GRP(config)
    
    # 测试输入
    batch_size = 32
    seq_len = 10
    sequences = np.random.randn(batch_size, seq_len, config.input_dim)
    targets = np.random.rand(batch_size, 1) * 100  # 模拟得分 0-100
    
    # 测试预测
    predictions = model.predict_batch(sequences)
    print(f"预测得分维度: {predictions.shape}")
    print(f"预测得分范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # 测试训练器
    trainer = GRPTrainer(model, learning_rate=0.001)
    
    # 转换为张量
    sequences_tensor = torch.FloatTensor(sequences).to(trainer.device)
    targets_tensor = torch.FloatTensor(targets).to(trainer.device)
    
    # 训练一步
    loss = trainer.train_step(sequences_tensor, targets_tensor)
    print(f"训练损失: {loss:.4f}")
    
    # 评估
    eval_loss, mae = trainer.evaluate(sequences_tensor, targets_tensor)
    print(f"评估损失: {eval_loss:.4f}, MAE: {mae:.4f}")
    
    # 保存模型
    model.save("e:/ai/mjai/models/test_grp.pth")
    
    print("GRP 模型测试完成！")

if __name__ == "__main__":
    test_grp()