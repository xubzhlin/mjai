"""
完整训练脚本
整合自博弈、PPO训练、评估等功能
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjai.advanced_model import AdvancedMahjongModel
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig
from mjai.buffer import PrioritizedReplayBuffer, Experience
from mjai.reward import RewardConfig
from mjai.selfplay import SelfPlayEngine, RolloutBuffer
from mjai.ppo import PPO, PPOConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练轮数
    total_iterations: int = 1000
    
    # 自博弈配置
    games_per_iteration: int = 10
    steps_per_update: int = 2048
    
    # PPO配置
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 学习率
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    lr_schedule: str = 'linear'  # linear, cosine
    
    # 奖励配置
    win_reward: float = 10.0
    lose_penalty: float = -5.0
    draw_penalty: float = -1.0
    discard_penalty: float = -0.1
    
    # 网络配置
    input_dim: int = 132
    action_dim: int = 34
    hidden_dims: List[int] = None
    
    # 保存和评估
    save_interval: int = 100
    eval_interval: int = 50
    eval_games: int = 10
    
    # 探索
    epsilon_start: float = 0.3
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    
    # 早停
    use_early_stopping: bool = False
    patience: int = 50
    min_delta: float = 0.01
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


class Trainer:
    """完整训练器"""
    
    def __init__(self, config: TrainingConfig, output_dir: str = 'output'):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.model_dir = self.output_dir / 'models'
        self.log_dir = self.output_dir / 'logs'
        self.eval_dir = self.output_dir / 'eval'
        
        for d in [self.model_dir, self.log_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 训练状态
        self.current_iteration = 0
        self.best_eval_score = -float('inf')
        self.early_stopping_counter = 0
        self.epsilon = config.epsilon_start
        
        # 训练历史
        self.history = []
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'total_time': 0,
            'total_games': 0,
            'total_steps': 0
        }
    
    def _init_components(self):
        """初始化所有组件"""
        logger.info("初始化训练组件...")
        
        # 特征处理器
        self.feature_config = FeatureConfig()
        self.feature_processor = FeatureProcessor(self.feature_config)
        self.action_encoder = ActionEncoder(self.feature_config)
        
        # 奖励配置
        reward_config = RewardConfig(
            win_reward=self.config.win_reward,
            lose_penalty=self.config.lose_penalty,
            draw_penalty=self.config.draw_penalty,
            discard_penalty=self.config.discard_penalty
        )
        
        # 自博弈引擎
        self.selfplay_engine = SelfPlayEngine(
            feature_processor=self.feature_processor,
            action_encoder=self.action_encoder,
            reward_config=reward_config
        )
        
        # 经验回放缓冲器
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=0.6,
            beta=0.4
        )
        
        # Rollout缓冲区
        self.rollout_buffer = RolloutBuffer()
        
        # PPO配置
        ppo_config = PPOConfig(
            input_dim=self.config.input_dim,
            action_dim=self.config.action_dim,
            clip_ratio=self.config.clip_ratio,
            ppo_epochs=self.config.ppo_epochs,
            batch_size=self.config.ppo_batch_size,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # PPO算法
        self.ppo = PPO(ppo_config)
        
        logger.info("训练组件初始化完成")
    
    def train(self):
        """开始训练"""
        logger.info(f"开始训练，共 {self.config.total_iterations} 轮")
        
        try:
            for iteration in range(self.current_iteration, self.config.total_iterations):
                self.current_iteration = iteration + 1
                
                # 1. 收集经验
                rollout_data = self._collect_rollout()
                
                # 2. 处理rollout数据
                self._process_rollout(rollout_data)
                
                # 3. PPO更新
                ppo_stats = self._update_ppo()
                
                # 4. 评估
                if (iteration + 1) % self.config.eval_interval == 0:
                    eval_score = self._evaluate()
                    
                    # 检查早停
                    if self.config.use_early_stopping:
                        if eval_score > self.best_eval_score + self.config.min_delta:
                            self.best_eval_score = eval_score
                            self.early_stopping_counter = 0
                            self._save_checkpoint('best_model.pth')
                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.config.patience:
                                logger.info("早停触发")
                                break
                
                # 5. 定期保存
                if (iteration + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(f'checkpoint_iter_{iteration + 1}.pth')
                
                # 6. 更新探索率
                self.epsilon = max(self.config.epsilon_end, 
                                  self.epsilon * self.config.epsilon_decay)
                
                # 7. 记录历史
                self._record_history(ppo_stats, rollout_data)
                
                # 8. 打印日志
                self._log_iteration(iteration, ppo_stats)
                
        except KeyboardInterrupt:
            logger.info("训练被中断")
            self._save_checkpoint('interrupted_model.pth')
        
        # 最终保存
        self._save_checkpoint('final_model.pth')
        self._save_training_history()
        
        logger.info(f"训练完成！总用时: {self.stats['total_time']:.1f}秒")
        
        return self.history
    
    def _collect_rollout(self) -> Dict:
        """收集rollout数据"""
        logger.info(f"第 {self.current_iteration} 轮：收集经验...")
        
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        
        num_games = self.config.games_per_iteration
        total_steps = 0
        
        for game_id in range(num_games):
            # 生成一局游戏
            game_record = self.selfplay_engine.generate_game()
            
            # 收集经验
            for exp in game_record.experiences:
                # 使用PPO网络获取对数概率和价值
                with torch.no_grad():
                    x = torch.FloatTensor(exp.state).unsqueeze(0).to(self.ppo.device)
                    action_probs, value = self.ppo.policy_net(x)
                    
                    # 创建分布
                    dist = torch.distributions.Categorical(action_probs)
                    log_prob = dist.log_prob(torch.LongTensor([exp.action]).to(self.ppo.device)).item()
                
                states_list.append(exp.state)
                actions_list.append(exp.action)
                log_probs_list.append(log_prob)
                rewards_list.append(exp.reward)
                values_list.append(value.item())
                dones_list.append(exp.done)
                
                total_steps += 1
        
        logger.info(f"收集了 {num_games} 局游戏，共 {total_steps} 步")
        
        # 添加到经验回放缓冲器
        for exp in self.selfplay_engine.generate_batch(1):
            self.replay_buffer.add(exp)
        
        return {
            'states': np.array(states_list),
            'actions': np.array(actions_list),
            'log_probs': np.array(log_probs_list),
            'rewards': np.array(rewards_list),
            'values': np.array(values_list),
            'dones': np.array(dones_list),
            'num_games': num_games,
            'total_steps': total_steps
        }
    
    def _process_rollout(self, rollout_data: Dict):
        """处理rollout数据"""
        # 计算下一状态的价值
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        # 添加虚拟的下一个价值
        next_values = np.append(values[1:], 0)
        
        # 计算GAE和回报
        advantages, returns = self.ppo.compute_gae(
            rollout_data['rewards'],
            values,
            next_values,
            dones
        )
        
        # 存储处理后的数据用于更新
        self.last_batch = {
            'states': rollout_data['states'],
            'actions': rollout_data['actions'],
            'old_log_probs': rollout_data['log_probs'],
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
    
    def _update_ppo(self) -> Dict:
        """PPO更新"""
        logger.info("PPO更新...")
        
        # 执行更新
        stats = self.ppo.update(
            self.last_batch['states'],
            self.last_batch['actions'],
            self.last_batch['old_log_probs'],
            self.last_batch['advantages'],
            self.last_batch['returns'],
            self.last_batch['values']
        )
        
        logger.info(f"PPO更新完成 - policy_loss: {stats['policy_loss']:.4f}, "
                   f"value_loss: {stats['value_loss']:.4f}, entropy: {stats['entropy']:.4f}")
        
        return stats
    
    def _evaluate(self) -> float:
        """评估模型"""
        logger.info(f"第 {self.current_iteration} 轮：评估模型...")
        
        total_reward = 0.0
        win_count = 0
        
        for _ in range(self.config.eval_games):
            # 使用当前PPO网络进行游戏
            game_record = self.selfplay_engine.generate_game()
            
            # 计算平均奖励
            if game_record.experiences:
                avg_reward = np.mean([exp.reward for exp in game_record.experiences])
                total_reward += avg_reward
            
            # 统计胜利
            if game_record.winner is not None:
                win_count += 1
        
        # 计算评估分数
        avg_eval_score = total_reward / self.config.eval_games
        win_rate = win_count / self.config.eval_games
        
        logger.info(f"评估完成 - 平均奖励: {avg_eval_score:.4f}, 胜率: {win_rate:.2%}")
        
        # 保存评估结果
        eval_result = {
            'iteration': self.current_iteration,
            'avg_score': avg_eval_score,
            'win_rate': win_rate,
            'timestamp': time.time()
        }
        
        eval_path = self.eval_dir / f'eval_iter_{self.current_iteration}.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_result, f, indent=2)
        
        return avg_eval_score
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        filepath = self.model_dir / filename
        
        # 保存PPO模型
        self.ppo.save(str(filepath))
        
        # 保存训练状态
        checkpoint = {
            'iteration': self.current_iteration,
            'epsilon': self.epsilon,
            'best_eval_score': self.best_eval_score,
            'early_stopping_counter': self.early_stopping_counter,
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        state_path = self.model_dir / f'{filename.replace(".pth", "")}_state.json'
        with open(state_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"检查点已保存: {filename}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        # 加载PPO模型
        self.ppo.load(filepath)
        
        # 加载训练状态
        state_path = filepath.replace('.pth', '_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.current_iteration = checkpoint['iteration']
            self.epsilon = checkpoint['epsilon']
            self.best_eval_score = checkpoint['best_eval_score']
            self.early_stopping_counter = checkpoint['early_stopping_counter']
            
            logger.info(f"检查点已加载: {filepath}, 从第 {self.current_iteration} 轮继续")
    
    def _record_history(self, ppo_stats: Dict, rollout_data: Dict):
        """记录训练历史"""
        entry = {
            'iteration': self.current_iteration,
            'timestamp': time.time(),
            'ppo_stats': ppo_stats,
            'rollout_stats': {
                'num_games': rollout_data['num_games'],
                'total_steps': rollout_data['total_steps'],
                'avg_reward': np.mean(rollout_data['rewards'])
            },
            'epsilon': self.epsilon,
            'learning_rate': ppo_stats.get('learning_rate', self.config.learning_rate)
        }
        
        self.history.append(entry)
        
        # 更新统计
        self.stats['total_games'] += rollout_data['num_games']
        self.stats['total_steps'] += rollout_data['total_steps']
        self.stats['total_time'] = time.time() - self.stats['start_time']
    
    def _log_iteration(self, iteration: int, ppo_stats: Dict):
        """打印迭代日志"""
        elapsed_time = time.time() - self.stats['start_time']
        
        logger.info(
            f"迭代 {iteration + 1}/{self.config.total_iterations} | "
            f"时间: {elapsed_time:.1f}s | "
            f"Policy Loss: {ppo_stats['policy_loss']:.4f} | "
            f"Value Loss: {ppo_stats['value_loss']:.4f} | "
            f"Entropy: {ppo_stats['entropy']:.4f} | "
            f"Epsilon: {self.epsilon:.4f}"
        )
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 保存最终统计
        stats_path = self.log_dir / 'final_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info("训练历史已保存")
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        stats = self.stats.copy()
        
        # 添加PPO统计
        stats['ppo'] = self.ppo.get_stats()
        
        # 添加自博弈统计
        stats['selfplay'] = self.selfplay_engine.get_stats()
        
        # 添加缓冲器统计
        stats['replay_buffer'] = {
            'size': len(self.replay_buffer),
            'capacity': self.replay_buffer.capacity
        }
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='川麻将RL训练脚本')
    
    # 训练参数
    parser.add_argument('--iterations', type=int, default=1000,
                       help='训练轮数')
    parser.add_argument('--games_per_iteration', type=int, default=10,
                       help='每轮游戏数')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                       help='PPO更新轮数')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                       help='PPO裁剪比率')
    
    # 输出和恢复
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='保存间隔')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='评估间隔')
    
    # 探索和奖励
    parser.add_argument('--epsilon_start', type=float, default=0.3,
                       help='初始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='最终探索率')
    parser.add_argument('--win_reward', type=float, default=10.0,
                       help='胜利奖励')
    parser.add_argument('--lose_penalty', type=float, default=-5.0,
                       help='失败惩罚')
    
    # 早停
    parser.add_argument('--early_stopping', action='store_true',
                       help='启用早停')
    parser.add_argument('--patience', type=int, default=50,
                       help='早停耐心值')
    
    args = parser.parse_args()
    
    # 创建训练配置
    config = TrainingConfig(
        total_iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        learning_rate=args.lr,
        ppo_batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_ratio=args.clip_ratio,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        win_reward=args.win_reward,
        lose_penalty=args.lose_penalty,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        use_early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    # 创建训练器
    trainer = Trainer(config, output_dir=args.output_dir)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    logger.info("=" * 60)
    logger.info("川麻将RL训练开始")
    logger.info(f"配置: {config.__dict__}")
    logger.info("=" * 60)
    
    # 训练
    history = trainer.train()
    
    # 打印最终统计
    stats = trainer.get_stats()
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"总游戏数: {stats['total_games']}")
    logger.info(f"总步数: {stats['total_steps']}")
    logger.info(f"总时间: {stats['total_time']:.1f}秒")
    logger.info("=" * 60)
    
    return history


if __name__ == "__main__":
    main()