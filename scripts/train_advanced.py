#!/usr/bin/env python3
"""
川麻将高级模型训练脚本
使用 ResNet + Dueling DQN + GRP 架构进行训练
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjai.advanced_model import AdvancedMahjongModel, create_advanced_model_from_config
from mjai.advanced_predictor import AdvancedActionPredictor, AdvancedPredictionConfig
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig
from mjai.trainer import MahjongTrainer, TrainingConfig
from mjai.config import ConfigManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    """高级训练管道"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config_manager = ConfigManager()
        
        # 加载配置
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建特征处理器和动作编码器
        self.feature_processor = FeatureProcessor(FeatureConfig())
        self.action_encoder = ActionEncoder(FeatureConfig())
        
        # 创建预测器
        self.predictor = AdvancedActionPredictor(
            self.model,
            self.feature_processor,
            self.action_encoder,
            AdvancedPredictionConfig()
        )
        
        # 创建训练器
        self.trainer = MahjongTrainer(self.model, self.config.training)
        
        # 训练统计
        self.stats = {
            'total_epochs': 0,
            'total_steps': 0,
            'best_eval_score': -float('inf'),
            'training_time': 0,
            'model_size': 0
        }
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("训练管道初始化完成")
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'model': {
                'input_dim': 132,
                'action_dim': 34,
                'base_channels': 64,
                'hidden_dims': [512, 256, 128],
                'lstm_hidden_dim': 128,
                'lstm_layers': 2,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon': 0.1
            },
            'training': {
                'epochs': 1000,
                'batch_size': 32,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'buffer_size': 10000,
                'update_interval': 50,
                'save_interval': 100,
                'eval_interval': 50,
                'log_dir': str(self.output_dir / "logs"),
                'tensorboard_dir': str(self.output_dir / "tensorboard"),
                'save_dir': str(self.output_dir / "models"),
                'use_scheduler': True,
                'early_stopping': True,
                'patience': 50
            },
            'output_dir': 'output'
        }
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config.model
        
        model = AdvancedMahjongModel(
            learning_rate=model_config['learning_rate'],
            gamma=model_config['gamma'],
            epsilon=model_config['epsilon']
        )
        
        logger.info(f"模型创建完成，参数数量: {self._count_parameters(model)}")
        return model
    
    def _count_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def generate_training_data(self, num_games: int = 100):
        """生成训练数据"""
        logger.info(f"生成 {num_games} 局训练数据...")
        
        training_data = []
        
        for game_id in tqdm(range(num_games), desc="生成训练数据"):
            # 模拟游戏
            game_states = self._simulate_game()
            
            # 为每个状态生成训练样本
            for state in game_states:
                # 获取特征
                features = self.feature_processor.encode_features(state)
                normalized_features = self.feature_processor.normalize_features(features)
                
                # 预测动作
                action_idx, prob, info = self.predictor.predict_action(state)
                
                # 获取Q值
                q_values = self.model.get_q_values(normalized_features)
                
                # 计算奖励（简化版）
                reward = self._calculate_reward(state, action_idx)
                
                # 添加到训练数据
                training_data.append({
                    'state': normalized_features,
                    'action': action_idx,
                    'reward': reward,
                    'q_values': q_values,
                    'prob': prob,
                    'info': info
                })
        
        logger.info(f"生成 {len(training_data)} 个训练样本")
        return training_data
    
    def _simulate_game(self):
        """模拟一局游戏"""
        # 简化的游戏模拟
        game_states = []
        
        # 初始化游戏状态
        state = {
            'hand': list(range(13)),  # 简化的手牌
            'players': [
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []}
            ],
            'round_info': {
                'honba': 0,
                'round_wind': 0,
                'player_wind': 0,
                'remaining_tiles': 70
            },
            'current_player': 0,
            'game_over': False
        }
        
        # 模拟游戏进程
        for turn in range(50):  # 最多50回合
            # 添加当前状态
            game_states.append(state.copy())
            
            # 预测动作
            action_idx, prob, info = self.predictor.predict_action(state)
            
            # 执行动作（简化）
            if action_idx == 0:  # 弃牌
                state['hand'] = state['hand'][1:]  # 移除第一张牌
            elif action_idx == 1:  # 胡牌
                state['game_over'] = True
                state['winner'] = state['current_player']
                break
            
            # 更新游戏状态
            state['current_player'] = (state['current_player'] + 1) % 4
            state['round_info']['remaining_tiles'] -= 1
            
            # 检查游戏结束
            if state['round_info']['remaining_tiles'] <= 0:
                state['game_over'] = True
                break
        
        return game_states
    
    def _calculate_reward(self, state: dict, action: int) -> float:
        """计算奖励"""
        # 简化的奖励函数
        if state.get('game_over', False):
            if state.get('winner') == 0:  # 玩家0获胜
                return 10.0
            else:
                return -10.0
        else:
            # 基于动作的奖励
            if action == 1:  # 胡牌动作
                return 1.0
            elif action == 0:  # 弃牌动作
                return -0.1
            else:
                return 0.0
    
    def train(self, num_games: int = 100):
        """训练模型"""
        logger.info("开始训练...")
        start_time = time.time()
        
        # 创建输出目录
        os.makedirs(self.config.training['save_dir'], exist_ok=True)
        os.makedirs(self.config.training['log_dir'], exist_ok=True)
        os.makedirs(self.config.training['tensorboard_dir'], exist_ok=True)
        
        # 训练循环
        for epoch in range(self.config.training['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config.training['epochs']}")
            
            # 生成训练数据
            training_data = self.generate_training_data(num_games)
            
            # 训练
            epoch_losses = []
            for i, sample in enumerate(training_data):
                # 构建批次
                batch = self._create_batch([sample])
                
                # 训练
                loss = self.trainer.train_step(batch)
                epoch_losses.append(loss)
                
                # 更新统计
                self.stats['total_steps'] += 1
                
                # 更新探索率
                self.model.epsilon = max(0.01, self.model.epsilon * 0.9995)
            
            # 计算平均损失
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
            
            # 评估
            if epoch % self.config.training['eval_interval'] == 0:
                eval_score = self.evaluate(num_games=10)
                logger.info(f"Epoch {epoch + 1} 评估分数: {eval_score:.4f}")
                
                # 保存最佳模型
                if eval_score > self.stats['best_eval_score']:
                    self.stats['best_eval_score'] = eval_score
                    self.save_model(f"best_model_epoch_{epoch}.pth")
            
            # 保存模型
            if epoch % self.config.training['save_interval'] == 0:
                self.save_model(f"model_epoch_{epoch}.pth")
            
            # 更新统计
            self.stats['total_epochs'] = epoch + 1
            
            # 早停检查
            if self.config.training['early_stopping']:
                if self._should_stop_early():
                    logger.info("早停触发，训练结束")
                    break
        
        # 训练完成
        training_time = time.time() - start_time
        self.stats['training_time'] = training_time
        
        logger.info(f"训练完成！总时间: {training_time:.2f}秒")
        logger.info(f"最佳评估分数: {self.stats['best_eval_score']:.4f}")
        
        # 保存最终模型
        self.save_model("final_model.pth")
        
        # 保存训练统计
        self.save_stats()
        
        return self.stats
    
    def _create_batch(self, samples: list):
        """创建训练批次"""
        states = np.array([s['state'] for s in samples])
        actions = np.array([s['action'] for s in samples])
        rewards = np.array([s['reward'] for s in samples])
        q_values = np.array([s['q_values'] for s in samples])
        
        return (states, actions, rewards, q_values)
    
    def evaluate(self, num_games: int = 10):
        """评估模型"""
        logger.info(f"评估模型，使用 {num_games} 局游戏...")
        
        total_score = 0.0
        
        for game_id in tqdm(range(num_games), desc="评估"):
            # 模拟游戏
            score = self._simulate_game_for_evaluation()
            total_score += score
        
        avg_score = total_score / num_games
        logger.info(f"平均评估分数: {avg_score:.4f}")
        
        return avg_score
    
    def _simulate_game_for_evaluation(self):
        """为评估模拟游戏"""
        state = {
            'hand': list(range(13)),
            'players': [
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []}
            ],
            'round_info': {
                'honba': 0,
                'round_wind': 0,
                'player_wind': 0,
                'remaining_tiles': 70
            },
            'current_player': 0,
            'game_over': False
        }
        
        score = 0.0
        
        for turn in range(50):
            # 预测动作
            action_idx, prob, info = self.predictor.predict_action(state)
            
            # 执行动作
            if action_idx == 1:  # 胡牌
                score += 10.0
                break
            elif action_idx == 0:  # 弃牌
                score -= 0.1
            
            # 更新状态
            state['current_player'] = (state['current_player'] + 1) % 4
            state['round_info']['remaining_tiles'] -= 1
            
            if state['round_info']['remaining_tiles'] <= 0:
                break
        
        return score
    
    def _should_stop_early(self):
        """检查是否应该早停"""
        # 简化的早停逻辑
        return False
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.config.training['save_dir'], filename)
        self.trainer.save_model(filepath)
        
        # 计算模型大小
        model_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        self.stats['model_size'] = model_size
        
        logger.info(f"模型已保存到: {filepath} (大小: {model_size:.2f}MB)")
    
    def save_stats(self):
        """保存训练统计"""
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"训练统计已保存到: {stats_path}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.trainer.load_model(filepath)
        logger.info(f"模型已从 {filepath} 加载")
    
    def export_model(self, format: str = 'torch'):
        """导出模型"""
        if format == 'torch':
            # 导出为PyTorch格式
            export_path = os.path.join(self.output_dir, "exported_model.pth")
            self.trainer.save_model(export_path)
            logger.info(f"模型已导出到: {export_path}")
        
        elif format == 'onnx':
            # 导出为ONNX格式
            try:
                self._export_to_onnx()
            except Exception as e:
                logger.error(f"ONNX导出失败: {e}")
        
        else:
            logger.error(f"不支持的导出格式: {format}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='川麻将高级模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--games_per_epoch', type=int, default=100, help='每局游戏数')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 创建训练管道
    pipeline = AdvancedTrainingPipeline(args.config)
    
    # 更新配置
    pipeline.config.training['epochs'] = args.epochs
    pipeline.config.training['batch_size'] = args.batch_size
    pipeline.config['output_dir'] = args.output_dir
    
    # 恢复训练
    if args.resume:
        pipeline.load_model(args.resume)
        logger.info(f"从 {args.resume} 恢复训练")
    
    # 开始训练
    try:
        stats = pipeline.train(num_games=args.games_per_epoch)
        
        # 导出模型
        pipeline.export_model('torch')
        
        logger.info("训练完成！")
        return stats
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()