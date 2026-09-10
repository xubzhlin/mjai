"""
川麻将模型训练器
使用PPO算法训练神经网络模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .model import MahjongModel
from .features import FeatureProcessor, ActionEncoder, FeatureConfig
from .predictor import ActionPredictor, PredictionConfig

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    clip_grad_norm: float = 0.5
    update_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    buffer_size: int = 10000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # PPO参数
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # 优化器参数
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    
    # 学习率调度器
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'
    scheduler_params: Dict = None
    
    # 数据增强
    data_augmentation: bool = True
    noise_std: float = 0.01
    
    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # 日志
    log_dir: str = 'logs'
    tensorboard_dir: str = 'tensorboard'
    save_dir: str = 'models'

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, buffer_size: int):
        """
        初始化经验回放缓冲区
        
        Args:
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.position = 0
        self.size = 0
    
    def add(self, observation: np.ndarray, action: int, reward: float, 
            log_prob: float, value: float, done: bool):
        """
        添加经验样本
        
        Args:
            observation: 观察值
            action: 动作
            reward: 奖励
            log_prob: 动作对数概率
            value: 价值估计
            done: 是否结束
        """
        if self.size < self.buffer_size:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(done)
            self.size += 1
        else:
            # 覆盖最老的经验
            self.observations[self.position] = observation
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.log_probs[self.position] = log_prob
            self.values[self.position] = value
            self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray, np.ndarray, np.ndarray]:
        """
        采样经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (observations, actions, rewards, log_probs, values, dones)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        observations = np.array([self.observations[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        log_probs = np.array([self.log_probs[i] for i in indices])
        values = np.array([self.values[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        
        return observations, actions, rewards, log_probs, values, dones
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float = 0.95):
        """
        计算优势函数和回报
        
        Args:
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        advantages = np.zeros(self.size)
        returns = np.zeros(self.size)
        
        # 计算GAE
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0 if self.dones[t] else self.values[t]
                next_non_terminal = 1 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        # 计算回报
        returns = advantages + self.values
        
        self.advantages = advantages
        self.returns = returns
    
    def __len__(self):
        return self.size

class MahjongTrainer:
    """川麻将模型训练器"""
    
    def __init__(self, 
                 model: MahjongModel,
                 feature_processor: FeatureProcessor,
                 action_encoder: ActionEncoder,
                 config: Optional[TrainingConfig] = None):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            feature_processor: 特征处理器
            action_encoder: 动作编码器
            config: 训练配置
        """
        self.model = model
        self.feature_processor = feature_processor
        self.action_encoder = action_encoder
        self.config = config or TrainingConfig()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(self.config.buffer_size)
        
        # 预测器
        self.predictor = ActionPredictor(
            model, feature_processor, action_encoder,
            PredictionConfig(
                epsilon=self.config.epsilon,
                epsilon_decay=self.config.epsilon_decay,
                epsilon_min=self.config.epsilon_min
            )
        )
        
        # 优化器
        self._setup_optimizer()
        
        # 学习率调度器
        self._setup_scheduler()
        
        # 日志记录器
        self._setup_logger()
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'learning_rate': 0.0,
            'episode_rewards': [],
            'episode_lengths': [],
            'best_score': -float('inf'),
            'patience_counter': 0
        }
        
        # 早停相关
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _setup_optimizer(self):
        """设置优化器"""
        if self.config.optimizer == 'adam':
            self.policy_optimizer = optim.Adam(
                self.model.policy_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            self.value_optimizer = optim.Adam(
                self.model.value_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'rmsprop':
            self.policy_optimizer = optim.RMSprop(
                self.model.policy_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            self.value_optimizer = optim.RMSprop(
                self.model.value_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if self.config.use_scheduler:
            if self.config.scheduler_type == 'cosine':
                self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.policy_optimizer,
                    T_max=self.config.epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
                self.value_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.value_optimizer,
                    T_max=self.config.epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
            elif self.config.scheduler_type == 'step':
                self.policy_scheduler = optim.lr_scheduler.StepLR(
                    self.policy_optimizer,
                    step_size=10,
                    gamma=0.1
                )
                self.value_scheduler = optim.lr_scheduler.StepLR(
                    self.value_optimizer,
                    step_size=10,
                    gamma=0.1
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        else:
            self.policy_scheduler = None
            self.value_scheduler = None
    
    def _setup_logger(self):
        """设置日志记录器"""
        # 创建日志目录
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_dir, exist_ok=True)
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # TensorBoard记录器
        self.writer = SummaryWriter(self.config.tensorboard_dir)
        
        # 文件日志记录器
        self.log_file = open(os.path.join(self.config.log_dir, 'training.log'), 'w')
    
    def collect_experience(self, env, num_episodes: int):
        """
        收集经验数据
        
        Args:
            env: 游戏环境
            num_episodes: 收集的回合数
        """
        print(f"开始收集经验数据，{num_episodes}个回合...")
        
        for episode in tqdm(range(num_episodes), desc="收集经验"):
            game_state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 预测动作
                action_idx, action_prob, info = self.predictor.predict_action(game_state)
                
                # 获取特征
                features = self.feature_processor.encode_features(game_state)
                normalized_features = self.feature_processor.normalize_features(features)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action_idx)
                
                # 存储经验
                self.buffer.add(
                    observation=normalized_features,
                    action=action_idx,
                    reward=reward,
                    log_prob=np.log(action_prob + 1e-8),
                    value=info['value'],
                    done=done
                )
                
                episode_reward += reward
                episode_length += 1
                
                # 更新游戏状态
                game_state = next_state
                
                if done:
                    break
            
            # 更新统计信息
            self.stats['total_episodes'] += 1
            self.stats['episode_rewards'].append(episode_reward)
            self.stats['episode_lengths'].append(episode_length)
            
            # 记录日志
            if episode % 100 == 0:
                avg_reward = np.mean(self.stats['episode_rewards'][-100:])
                avg_length = np.mean(self.stats['episode_lengths'][-100:])
                print(f"回合 {episode}: 奖励={avg_reward:.2f}, 长度={avg_length:.2f}")
                self.writer.add_scalar('Episode/Reward', avg_reward, episode)
                self.writer.add_scalar('Episode/Length', avg_length, episode)
        
        print(f"经验收集完成，共收集 {len(self.buffer)} 个样本")
    
    def train_step(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                     np.ndarray, np.ndarray, np.ndarray]) -> Dict:
        """
        训练一个批次
        
        Args:
            batch: 批次数据
            
        Returns:
            训练损失信息
        """
        observations, actions, rewards, log_probs, values, dones = batch
        
        # 获取批次索引
        batch_size = len(observations)
        indices = list(range(batch_size))
        
        # 转换为张量
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)
        
        # 计算优势函数和回报
        self.buffer.compute_advantages_and_returns(self.config.gamma)
        
        # 获取当前批次的advantages和returns
        batch_advantages = torch.FloatTensor([self.buffer.advantages[i] for i in indices]).to(self.device)
        batch_returns = torch.FloatTensor([self.buffer.returns[i] for i in indices]).to(self.device)
        
        # 归一化优势函数
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        
        # PPO训练
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        for _ in range(self.config.ppo_epochs):
            # 前向传播
            action_probs = self.model.policy_net(observations)
            values = self.model.value_net(observations).squeeze()
            
            # 计算新的对数概率
            new_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 
                               1 + self.config.ppo_clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.MSELoss()(values, batch_returns)
            
            # 熵损失
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            
            # 总损失
            loss = policy_loss + self.config.value_loss_coef * value_loss - \
                   self.config.entropy_coef * entropy
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 
                                          self.config.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.value_net.parameters(), 
                                          self.config.clip_grad_norm)
            
            # 更新参数
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.item()
        
        # 平均损失
        num_updates = self.config.ppo_epochs
        avg_loss = total_loss / num_updates
        avg_policy_loss = policy_loss_sum / num_updates
        avg_value_loss = value_loss_sum / num_updates
        avg_entropy = entropy_sum / num_updates
        
        # 更新统计信息
        self.stats['total_loss'] += avg_loss
        self.stats['policy_loss'] += avg_policy_loss
        self.stats['value_loss'] += avg_value_loss
        self.stats['entropy'] += avg_entropy
        self.stats['total_steps'] += 1
        
        # 记录日志
        self.writer.add_scalar('Loss/Total', avg_loss, self.stats['total_steps'])
        self.writer.add_scalar('Loss/Policy', avg_policy_loss, self.stats['total_steps'])
        self.writer.add_scalar('Loss/Value', avg_value_loss, self.stats['total_steps'])
        self.writer.add_scalar('Loss/Entropy', avg_entropy, self.stats['total_steps'])
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
    
    def train(self, env, num_episodes: int):
        """
        训练模型
        
        Args:
            env: 游戏环境
            num_episodes: 总回合数
        """
        print("开始训练模型...")
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="训练"):
            # 收集经验
            self.collect_experience(env, 1)
            
            # 训练
            if len(self.buffer) >= self.config.batch_size:
                batch = self.buffer.sample(self.config.batch_size)
                loss_info = self.train_step(batch)
                
                # 更新学习率
                if self.policy_scheduler:
                    self.policy_scheduler.step()
                    self.value_scheduler.step()
                    self.stats['learning_rate'] = self.policy_scheduler.get_last_lr()[0]
                
                # 记录日志
                if episode % self.config.log_interval == 0:
                    self._log_training_info(episode, loss_info, start_time)
                
                # 保存模型
                if episode % self.config.save_interval == 0:
                    self.save_model(f'model_episode_{episode}.pth')
                
                # 评估模型
                if episode % self.config.eval_interval == 0:
                    eval_score = self.evaluate(env)
                    self.writer.add_scalar('Eval/Score', eval_score, episode)
                    
                    # 早停检查
                    if self.config.early_stopping:
                        if eval_score > self.best_score + self.config.early_stopping_min_delta:
                            self.best_score = eval_score
                            self.patience_counter = 0
                            self.save_model('best_model.pth')
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.config.early_stopping_patience:
                                print(f"早停触发，在回合 {episode} 停止训练")
                                break
        
        # 保存最终模型
        self.save_model('final_model.pth')
        
        # 关闭日志记录器
        self.writer.close()
        self.log_file.close()
        
        print("训练完成！")
    
    def evaluate(self, env, num_episodes: int = 100) -> float:
        """
        评估模型
        
        Args:
            env: 游戏环境
            num_episodes: 评估回合数
            
        Returns:
            平均奖励
        """
        print(f"开始评估模型，{num_episodes}个回合...")
        
        total_reward = 0
        
        with torch.no_grad():
            for episode in range(num_episodes):
                game_state = env.reset()
                episode_reward = 0
                
                while True:
                    # 使用确定性策略
                    action_idx, _, _ = self.predictor.predict_action(game_state)
                    
                    next_state, reward, done, _ = env.step(action_idx)
                    episode_reward += reward
                    
                    game_state = next_state
                    
                    if done:
                        break
                
                total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        print(f"评估完成，平均奖励: {avg_reward:.2f}")
        
        return avg_reward
    
    def _log_training_info(self, episode: int, loss_info: Dict, start_time: float):
        """记录训练信息"""
        elapsed_time = time.time() - start_time
        
        log_info = {
            'episode': episode,
            'total_steps': self.stats['total_steps'],
            'total_episodes': self.stats['total_episodes'],
            'loss': loss_info['loss'],
            'policy_loss': loss_info['policy_loss'],
            'value_loss': loss_info['value_loss'],
            'entropy': loss_info['entropy'],
            'learning_rate': self.stats['learning_rate'],
            'epsilon': self.predictor.epsilon,
            'avg_reward': np.mean(self.stats['episode_rewards'][-100:]) if self.stats['episode_rewards'] else 0,
            'avg_length': np.mean(self.stats['episode_lengths'][-100:]) if self.stats['episode_lengths'] else 0,
            'elapsed_time': elapsed_time
        }
        
        # 打印日志
        print(f"回合 {episode}: 损失={loss_info['loss']:.4f}, "
              f"策略损失={loss_info['policy_loss']:.4f}, "
              f"价值损失={loss_info['value_loss']:.4f}, "
              f"熵={loss_info['entropy']:.4f}, "
              f"平均奖励={log_info['avg_reward']:.2f}")
        
        # 写入日志文件
        self.log_file.write(f"{json.dumps(log_info)}\n")
        self.log_file.flush()
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.config.save_dir, filename)
        self.model.save(model_path)
        
        # 保存统计信息
        stats_path = os.path.join(self.config.save_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_model(self, filename: str):
        """加载模型"""
        model_path = os.path.join(self.config.save_dir, filename)
        self.model = MahjongModel.load(model_path)
        
        # 加载统计信息
        stats_path = os.path.join(self.config.save_dir, 'stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.stats['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 长度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.stats['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        # 损失曲线
        plt.subplot(2, 2, 3)
        plt.plot([self.stats['total_loss'] / max(1, self.stats['total_steps'])] * len(self.stats['episode_rewards']))
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        # 学习率曲线
        plt.subplot(2, 2, 4)
        plt.plot([self.stats['learning_rate']] * len(self.stats['episode_rewards']))
        plt.title('Learning Rate')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_curves.png'))
        plt.close()

if __name__ == "__main__":
    # 测试训练器
    print("测试训练器...")
    
    # 创建模型
    from mjai.model import MahjongModel
    model = MahjongModel()
    
    # 创建特征处理器和动作编码器
    from mjai.features import FeatureProcessor, ActionEncoder
    config = FeatureConfig()
    feature_processor = FeatureProcessor(config)
    action_encoder = ActionEncoder(config)
    
    # 创建训练器
    trainer = MahjongTrainer(model, feature_processor, action_encoder)
    
    # 创建模拟环境
    class MockEnv:
        def __init__(self):
            self.reset_count = 0
        
        def reset(self):
            self.reset_count += 1
            return {
                'hand': list(range(13)),
                'players': [{'melds': [], 'discards': []} for _ in range(4)],
                'round_info': {'honba': 0, 'round_wind': 0, 'player_wind': 0, 'remaining_tiles': 70},
                'current_player': 0
            }
        
        def step(self, action):
            # 模拟环境交互
            reward = np.random.randn() * 0.1
            done = np.random.random() < 0.1
            next_state = self.reset()
            return next_state, reward, done, {}
    
    env = MockEnv()
    
    # 训练几个回合
    trainer.train(env, 10)
    
    print("训练器测试完成！")