"""
自博弈引擎
实现多进程自博弈训练数据生成
"""

import numpy as np
import random
import time
import threading
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy
import json
import os

from mjai.buffer import Experience, PrioritizedReplayBuffer, BufferManager
from mjai.reward import RewardFunction, RewardConfig, RewardTracker
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig


@dataclass
class GameRecord:
    """游戏记录"""
    game_id: int = 0
    winner: Optional[int] = None
    scores: List[float] = field(default_factory=list)
    steps: int = 0
    experiences: List[Experience] = field(default_factory=list)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'game_id': self.game_id,
            'winner': self.winner,
            'scores': self.scores,
            'steps': self.steps,
            'num_experiences': len(self.experiences),
            'timestamp': self.timestamp
        }


class SimulatedGame:
    """简化的模拟游戏"""
    
    def __init__(self, game_id: int = 0):
        """
        初始化模拟游戏
        
        Args:
            game_id: 游戏ID
        """
        self.game_id = game_id
        self.current_player = 0
        self.round = 0
        self.max_rounds = 50
        
        # 游戏状态
        self.state = self._init_state()
        self.done = False
        self.winner = None
        self.scores = [0.0, 0.0, 0.0, 0.0]
        
        # 历史
        self.history = []
        self.steps = 0
    
    def _init_state(self) -> Dict:
        """初始化游戏状态"""
        # 简化的初始化
        hand_size = 13
        hand = list(range(hand_size))  # 简化的手牌
        
        state = {
            'hand': hand,
            'players': [
                {'melds': [], 'discards': [], 'points': 25000},
                {'melds': [], 'discards': [], 'points': 25000},
                {'melds': [], 'discards': [], 'points': 25000},
                {'melds': [], 'discards': [], 'points': 25000}
            ],
            'round_info': {
                'honba': 0,
                'round_wind': 0,
                'player_wind': self.current_player,
                'remaining_tiles': 70,
                'phase': 'playing'
            },
            'current_player': self.current_player,
            'game_over': False,
            'winner': None
        }
        
        return state
    
    def reset(self):
        """重置游戏"""
        self.current_player = 0
        self.round = 0
        self.state = self._init_state()
        self.done = False
        self.winner = None
        self.scores = [0.0, 0.0, 0.0, 0.0]
        self.history = []
        self.steps = 0
    
    def get_state(self, player_id: int = 0) -> Dict:
        """
        获取玩家视角的游戏状态
        
        Args:
            player_id: 玩家ID
            
        Returns:
            游戏状态字典
        """
        return deepcopy(self.state)
    
    def take_action(self, player_id: int, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        Args:
            player_id: 执行动作的玩家
            action: 动作索引
            
        Returns:
            (下一状态, 奖励, 是否结束, 额外信息)
        """
        info = {}
        reward = 0.0
        
        # 只有当前玩家可以行动
        if player_id != self.current_player:
            reward = -1.0  # 非法行动
            info['error'] = 'Not current player'
            return self.state, reward, self.done, info
        
        # 执行动作
        if action == 0:  # 弃牌
            # 简化：移除第一张牌
            if self.state['hand']:
                discarded_tile = self.state['hand'].pop(0)
                reward = -0.1
                info['action'] = 'discard'
                info['tile'] = discarded_tile
        
        elif action == 1:  # 胡牌
            # 简化：假设胡牌成功
            self.done = True
            self.state['game_over'] = True
            self.state['winner'] = player_id
            self.winner = player_id
            
            # 计算分数（简化）
            self.scores[player_id] = 10.0
            for i in range(4):
                if i != player_id:
                    self.scores[i] = -3.33
            
            reward = 10.0
            info['action'] = 'win'
            info['fan'] = 5  # 简化的番种
        
        elif action in [2, 3, 4]:  # 吃碰杠
            # 简化的副露
            reward = 0.1
            info['action'] = ['eat', 'peng', 'gang'][action - 2]
        
        elif action == 5:  # 过
            reward = -0.05
            info['action'] = 'pass'
        
        else:
            reward = -1.0  # 非法动作
            info['error'] = 'Invalid action'
        
        # 更新游戏进度
        self.steps += 1
        self.state['round_info']['remaining_tiles'] -= 1
        self.state['round_info']['player_wind'] = self.current_player
        
        # 切换玩家（如果未结束）
        if not self.done:
            self.current_player = (self.current_player + 1) % 4
            self.state['current_player'] = self.current_player
            
            # 检查游戏结束条件
            if self.state['round_info']['remaining_tiles'] <= 0:
                self.done = True
                self.state['game_over'] = True
                self.winner = None  # 流局
                reward = -1.0
        
        # 记录历史
        self.history.append({
            'player': player_id,
            'action': action,
            'reward': reward,
            'done': self.done
        })
        
        return self.state, reward, self.done, info
    
    def is_done(self) -> bool:
        """检查游戏是否结束"""
        return self.done or self.steps >= self.max_rounds * 4


class SelfPlayEngine:
    """自博弈引擎"""
    
    def __init__(self, 
                 feature_processor: Optional[FeatureProcessor] = None,
                 action_encoder: Optional[ActionEncoder] = None,
                 reward_config: Optional[RewardConfig] = None,
                 num_workers: int = 1):
        """
        初始化自博弈引擎
        
        Args:
            feature_processor: 特征处理器
            action_encoder: 动作编码器
            reward_config: 奖励配置
            num_workers: 工作进程数
        """
        # 特征处理器
        if feature_processor is None:
            self.feature_processor = FeatureProcessor(FeatureConfig())
        else:
            self.feature_processor = feature_processor
        
        # 动作编码器
        if action_encoder is None:
            self.action_encoder = ActionEncoder(FeatureConfig())
        else:
            self.action_encoder = action_encoder
        
        # 奖励函数
        if reward_config is None:
            self.reward_config = RewardConfig()
        else:
            self.reward_config = reward_config
        
        self.reward_function = RewardFunction(self.reward_config)
        self.reward_tracker = RewardTracker()
        
        # 工作配置
        self.num_workers = num_workers
        self.game_counter = 0
        
        # 统计信息
        self.stats = {
            'total_games': 0,
            'total_steps': 0,
            'win_counts': [0, 0, 0, 0],
            'draw_count': 0,
            'avg_episode_reward': 0.0,
            'avg_steps_per_game': 0
        }
    
    def generate_game(self, model_predictor=None) -> GameRecord:
        """
        生成一局游戏的数据
        
        Args:
            model_predictor: 模型预测器（可选）
            
        Returns:
            游戏记录
        """
        # 创建游戏
        game = SimulatedGame(game_id=self.game_counter)
        record = GameRecord(
            game_id=self.game_counter,
            timestamp=time.time()
        )
        
        self.game_counter += 1
        self.reward_tracker.start_episode()
        
        # 存储每个玩家的GRP隐藏状态
        player_hidden_states = {i: None for i in range(4)}
        
        # 游戏循环
        while not game.is_done():
            player_id = game.current_player
            state = game.get_state(player_id)
            
            # 编码特征
            features = self.feature_processor.encode_features(state)
            normalized_features = self.feature_processor.normalize_features(features)
            
            # 预测或随机选择动作
            if model_predictor is not None:
                action, prob, info = model_predictor.predict(state, player_hidden_states[player_id])
                if 'hidden' in info:
                    player_hidden_states[player_id] = info['hidden']
            else:
                # 随机选择合法动作
                valid_actions = self.action_encoder.get_valid_actions(state)
                action = random.choice(valid_actions) if valid_actions else 0
                prob = 1.0 / max(len(valid_actions), 1)
                info = {}
            
            # 执行动作
            next_state, reward, done, action_info = game.take_action(player_id, action)
            
            # 计算奖励（如果有奖励函数）
            reward = self.reward_function.calculate_reward(
                state, action, next_state, done
            )
            
            # 创建经验
            next_features = self.feature_processor.encode_features(next_state)
            next_normalized = self.feature_processor.normalize_features(next_features)
            
            exp = Experience(
                state=normalized_features,
                action=action,
                reward=reward,
                next_state=next_normalized,
                done=done,
                timestamp=time.time(),
                game_info={
                    'game_id': game.game_id,
                    'player_id': player_id,
                    'step': record.steps,
                    'action_info': action_info,
                    'prob': prob
                }
            )
            
            # 记录经验
            record.experiences.append(exp)
            record.steps += 1
            
            # 记录奖励
            self.reward_tracker.record_step({
                'total': reward,
                'action': action,
                'player': player_id
            })
        
        # 游戏结束，处理最终奖励
        record.winner = game.winner
        record.scores = game.scores
        
        # 更新统计
        self._update_stats(record)
        
        self.reward_tracker.finish_episode()
        
        return record
    
    def generate_batch(self, num_games: int, model_predictor=None) -> List[Experience]:
        """
        批量生成游戏数据
        
        Args:
            num_games: 游戏数量
            model_predictor: 模型预测器
            
        Returns:
            所有经验列表
        """
        all_experiences = []
        
        for _ in range(num_games):
            record = self.generate_game(model_predictor)
            all_experiences.extend(record.experiences)
        
        return all_experiences
    
    def generate_to_buffer(self, 
                          num_games: int, 
                          buffer: PrioritizedReplayBuffer,
                          model_predictor=None):
        """
        生成数据并直接添加到缓冲器
        
        Args:
            num_games: 游戏数量
            buffer: 经验回放缓冲器
            model_predictor: 模型预测器
        """
        for _ in range(num_games):
            record = self.generate_game(model_predictor)
            for exp in record.experiences:
                buffer.add(exp)
    
    def _update_stats(self, record: GameRecord):
        """更新统计信息"""
        self.stats['total_games'] += 1
        self.stats['total_steps'] += record.steps
        
        if record.winner is not None:
            self.stats['win_counts'][record.winner] += 1
        else:
            self.stats['draw_count'] += 1
        
        # 更新平均值
        if self.stats['total_games'] > 0:
            self.stats['avg_steps_per_game'] = (
                self.stats['total_steps'] / self.stats['total_games']
            )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加胜率统计
        total_wins = sum(stats['win_counts'])
        if stats['total_games'] > 0:
            stats['win_rate'] = total_wins / stats['total_games']
            stats['draw_rate'] = stats['draw_count'] / stats['total_games']
            
            # 计算每个玩家的胜率
            stats['player_win_rates'] = [
                count / stats['total_games'] for count in stats['win_counts']
            ]
        
        # 添加奖励统计
        reward_stats = self.reward_tracker.get_stats()
        stats['reward'] = reward_stats
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_games': 0,
            'total_steps': 0,
            'win_counts': [0, 0, 0, 0],
            'draw_count': 0,
            'avg_episode_reward': 0.0,
            'avg_steps_per_game': 0
        }
        self.game_counter = 0

class ParallelSelfPlayEngine:
    """并行自博弈引擎"""
    
    def __init__(self, 
                 num_workers: int = 4,
                 feature_processor: Optional[FeatureProcessor] = None,
                 action_encoder: Optional[ActionEncoder] = None,
                 reward_config: Optional[RewardConfig] = None):
        """
        初始化并行自博弈引擎
        
        Args:
            num_workers: 工作进程数
            feature_processor: 特征处理器
            action_encoder: 动作编码器
            reward_config: 奖励配置
        """
        self.num_workers = num_workers
        
        # 创建多个自博弈引擎
        self.engines = [
            SelfPlayEngine(feature_processor, action_encoder, reward_config)
            for _ in range(num_workers)
        ]
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 全局统计
        self.global_stats = {
            'total_games': 0,
            'total_steps': 0,
            'start_time': time.time()
        }
    
    def generate_games_parallel(self, 
                              num_games_per_worker: int,
                              model_predictor=None,
                              buffer: Optional[PrioritizedReplayBuffer] = None) -> List[Experience]:
        """
        并行生成游戏数据
        
        Args:
            num_games_per_worker: 每个工作进程生成的游戏数
            model_predictor: 模型预测器
            buffer: 经验回放缓冲器
            
        Returns:
            所有经验列表
        """
        all_experiences = []
        threads = []
        
        def worker(engine, games, predictor, exp_list, buf):
            local_experiences = []
            for _ in range(games):
                record = engine.generate_game(predictor)
                
                # 收集经验
                local_experiences.extend(record.experiences)
                
                # 如果有缓冲器，添加进去
                if buf is not None:
                    for exp in record.experiences:
                        buf.add(exp)
            
            # 线程安全地合并结果
            with self.lock:
                exp_list.extend(local_experiences)
        
        # 创建工作线程
        for engine in self.engines:
            thread = threading.Thread(
                target=worker,
                args=(engine, num_games_per_worker, model_predictor, all_experiences, buffer)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 更新全局统计
        total_games = num_games_per_worker * self.num_workers
        self.global_stats['total_games'] += total_games
        
        return all_experiences
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.global_stats.copy()
        
        # 合并所有工作引擎的统计
        stats['worker_stats'] = [engine.get_stats() for engine in self.engines]
        
        # 计算运行时间
        stats['run_time'] = time.time() - stats['start_time']
        
        return stats


class ModelPredictorWrapper:
    """模型预测器包装类"""
    
    def __init__(self, model, epsilon: float = 0.1):
        """
        初始化预测器包装
        
        Args:
            model: 神经网络模型
            epsilon: 探索率
        """
        self.model = model
        self.epsilon = epsilon
        self.stats = {
            'total_predictions': 0,
            'exploration_count': 0,
            'exploitation_count': 0
        }
    
    def predict(self, state: Dict, hidden=None) -> Tuple[int, float, Dict]:
        """
        预测动作
        
        Args:
            state: 游戏状态
            hidden: GRP隐藏状态
            
        Returns:
            (动作索引, 概率, 额外信息)
        """
        self.stats['total_predictions'] += 1
        
        # 使用模型预测
        if hasattr(self.model, 'predict'):
            if hidden is not None:
                action, q_value, new_hidden = self.model.predict(state, hidden)
            else:
                action, q_value, new_hidden = self.model.predict(state)
        
        # Epsilon-greedy探索
        if random.random() < self.epsilon:
            action = random.randint(0, 33)
            self.stats['exploration_count'] += 1
        else:
            self.stats['exploitation_count'] += 1
        
        return action, 1.0 / 34, {'hidden': new_hidden}
    
    def update_epsilon(self, decay: float = 0.9995):
        """更新探索率"""
        self.epsilon = max(0.01, self.epsilon * decay)


def create_selfplay_engine(config: Dict) -> SelfPlayEngine:
    """
    创建自博弈引擎的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        自博弈引擎实例
    """
    # 创建特征处理器
    feature_config = FeatureConfig()
    feature_processor = FeatureProcessor(feature_config)
    action_encoder = ActionEncoder(feature_config)
    
    # 创建奖励配置
    reward_config = RewardConfig(
        win_reward=config.get('win_reward', 10.0),
        lose_penalty=config.get('lose_penalty', -5.0),
        draw_penalty=config.get('draw_penalty', -1.0),
        discard_penalty=config.get('discard_penalty', -0.1),
        progress_reward=config.get('progress_reward', 0.05),
        hand_quality_reward=config.get('hand_quality_reward', 0.2)
    )
    
    # 创建引擎
    engine = SelfPlayEngine(
        feature_processor=feature_processor,
        action_encoder=action_encoder,
        reward_config=reward_config,
        num_workers=config.get('num_workers', 1)
    )
    
    return engine


if __name__ == "__main__":
    # 测试自博弈引擎
    print("测试自博弈引擎...")
    
    # 创建自博弈引擎
    engine = SelfPlayEngine()
    
    # 生成一些游戏数据
    print("生成游戏数据...")
    experiences = engine.generate_batch(num_games=10)
    
    print(f"生成了 {len(experiences)} 个经验")
    
    # 获取统计信息
    stats = engine.get_stats()
    print(f"统计信息: {json.dumps(stats, indent=2)}")
    
    print("自博弈引擎测试完成！")