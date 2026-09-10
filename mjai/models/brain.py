"""
Brain 特征提取器
基于 ResNet 架构，包含残差块和通道注意力机制

架构：
    Observation [N, C, 27]
        ↓
    Conv1d(C → 256, kernel=3, padding=1)
        ↓
    Residual Block × 12  (Conv1d→BN→Mish→Conv1d→BN→+)
        ↓
    Conv1d(256 → 256, kernel=3)
        ↓
    Channel Attention (avg+max pool → shared MLP → sigmoid → scale)
        ↓
    Flatten → Linear(256×27 → 1024)
        ↓
    Feature Vector [N, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Mish(nn.Module):
    """Mish 激活函数"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class ResidualBlock(nn.Module):
    """
    标准 2 层残差块
    
    结构:
        x → Conv → BN → Mish → Conv → BN → + (skip connection)
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-3)
        self.mish1 = Mish()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-3)
        
        self.mish = Mish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # 残差连接
        out = self.mish(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    
    结构:
        x → avg pool → shared MLP → + 
          → max pool → shared MLP → sigmoid → scale
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        hidden_channels = max(channels // reduction, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            Mish(),
            nn.Conv1d(hidden_channels, channels, 1),
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            [batch_size, channels, seq_len]  (与输入同形状，已被注意力缩放)
        """
        # 平均池化分支
        avg_out = self.avg_pool(x)          # [batch_size, channels, 1]
        avg_out = self.mlp(avg_out)         # [batch_size, channels, 1]
        
        # 最大池化分支
        max_out = self.max_pool(x)          # [batch_size, channels, 1]
        max_out = self.mlp(max_out)         # [batch_size, channels, 1]
        
        # 融合并通过 sigmoid
        out = avg_out + max_out             # [batch_size, channels, 1]
        attention = self.sigmoid(out)       # [batch_size, channels, 1]
        
        # 广播并缩放
        return x * attention


class Brain(nn.Module):
    """
    ResNet 特征提取器
    
    将观测张量 [N, C, 27] 转换为特征向量 [N, feature_dim]
    """
    
    def __init__(
        self,
        input_channels: int = 132,    # 输入通道数（观测特征维度）
        hidden_channels: int = 256,   # 隐藏通道数
        num_residual_blocks: int = 12, # 残差块数量
        seq_length: int = 27,          # 序列长度（牌种类）
        feature_dim: int = 1024,       # 输出特征维度
        dropout_rate: float = 0.1,     # Dropout 率
        attention_reduction: int = 4,  # 通道注意力缩减率
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_residual_blocks = num_residual_blocks
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        
        # 输入层：将 C 通道投影到 hidden_channels
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels, eps=1e-3)
        self.input_mish = Mish()
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, dropout_rate=dropout_rate)
            for _ in range(num_residual_blocks)
        ])
        
        # 输出卷积
        self.output_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3)
        self.output_bn = nn.BatchNorm1d(hidden_channels, eps=1e-3)
        self.output_mish = Mish()
        
        # 通道注意力
        self.channel_attention = ChannelAttention(hidden_channels, reduction=attention_reduction)
        
        # 展平 + 全连接
        self.flatten = nn.Flatten()
        flattened_dim = hidden_channels * (seq_length - 2)  # kernel_size=3, 无 padding
        self.fc1 = nn.Linear(flattened_dim, feature_dim)
        self.fc_bn = nn.BatchNorm1d(feature_dim, eps=1e-3)
        self.fc_mish = Mish()
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 观测张量 [batch_size, input_channels, seq_length] 或 [batch_size, input_channels * seq_length]
            
        Returns:
            特征向量 [batch_size, feature_dim]
        """
        # 如果输入是展平的，重塑为 [N, C, 27]
        if x.dim() == 2 and x.size(1) == self.input_channels * self.seq_length:
            x = x.view(-1, self.input_channels, self.seq_length)
        
        # 输入层
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_mish(x)
        
        # 残差块
        x = self.residual_blocks(x)
        
        # 输出卷积
        x = self.output_conv(x)
        x = self.output_bn(x)
        x = self.output_mish(x)
        
        # 通道注意力
        x = self.channel_attention(x)
        
        # 展平 + 全连接
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = self.fc_mish(x)
        x = self.fc_dropout(x)
        
        return x
    
    def get_config(self) -> dict:
        """获取配置字典"""
        return {
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'num_residual_blocks': self.num_residual_blocks,
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'dropout_rate': 0.1,
            'attention_reduction': 4,
        }
    
    def count_params(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def reset_parameters(self):
        """重置参数"""
        self._initialize_weights()


def create_brain(config: dict) -> Brain:
    """从配置创建 Brain"""
    return Brain(**config)


if __name__ == "__main__":
    # 测试 Brain
    print("测试 Brain 特征提取器...")
    
    brain = Brain(
        input_channels=132,
        hidden_channels=256,
        num_residual_blocks=12,
        seq_length=27,
        feature_dim=1024,
    )
    
    print(f"参数量: {brain.count_params():,}")
    
    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 132, 27)
    output = brain(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (batch_size, 1024), f"输出维度错误: {output.shape}"
    
    # 测试展平输入
    x_flat = torch.randn(batch_size, 132 * 27)
    output_flat = brain(x_flat)
    print(f"展平输入形状: {x_flat.shape}")
    print(f"展平输出形状: {output_flat.shape}")
    assert torch.allclose(output, output_flat, atol=1e-5), "展平输入处理有误"
    
    print("Brain 测试通过！")