"""
经验回放缓冲器
实现各种类型的经验回放策略
"""

import numpy as np
import random
import torch
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import heapq
from dataclasses import dataclass
import threading
import time

@dataclass
class Experience:
    """经验数据结构"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0  # 用于优先级回放
    weight: float = 1.0    # 用于重要性采样
    timestamp: float = 0.0  # 时间戳
    game_info: Dict = None  # 游戏相关信息
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.game_info is None:
            self.game_info = {}

class ReplayBuffer:
    """基础经验回放缓冲器"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        初始化回放缓冲器
        
        Args:
            capacity: 缓冲器容量
            alpha: 优先级指数，0表示完全随机，1表示完全优先级
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priority_sum = 0.0
        self.max_priority = 1.0
        
        # 用于线程安全
        self.lock = threading.Lock()
    
    def add(self, experience: Experience):
        """
        添加经验到缓冲器
        
        Args:
            experience: 经验数据
        """
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            # 更新优先级
            experience.priority = self.max_priority
            self.priority_sum += experience.priority
            
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        随机采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            采样到的经验列表
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            
            return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲器"""
        with self.lock:
            self.buffer.clear()
            self.position = 0
            self.priority_sum = 0.0
            self.max_priority = 1.0

class PrioritizedReplayBuffer(ReplayBuffer):
    """优先级经验回放缓冲器"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先级回放缓冲器
        
        Args:
            capacity: 缓冲器容量
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        super().__init__(capacity, alpha)
        self.beta = beta
        self.beta_increment = 0.001
        self.beta_max = 1.0
    
    def add(self, experience: Experience):
        """添加经验并更新优先级"""
        with self.lock:
            super().add(experience)
            # 更新优先级和总和
            self.priority_sum += experience.priority
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        优先级采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (采样到的经验列表, 索引, 权重)
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return [], [], []
            
            # 计算采样概率
            priorities = np.array([exp.priority for exp in self.buffer])
            probabilities = priorities / self.priority_sum
            
            # 采样
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            experiences = [self.buffer[i] for i in indices]
            
            # 计算重要性采样权重
            self.beta = min(self.beta + self.beta_increment, self.beta_max)
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights = weights / weights.max()
            
            return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 要更新的经验索引
            priorities: 新的优先级
        """
        with self.lock:
            for idx, priority in zip(indices, priorities):
                if idx < len(self.buffer):
                    old_priority = self.buffer[idx].priority
                    self.buffer[idx].priority = max(priority, 0.01)  # 最小优先级
                    self.priority_sum += self.buffer[idx].priority - old_priority
                    self.max_priority = max(self.max_priority, self.buffer[idx].priority)

class EpisodeReplayBuffer:
    """整局游戏经验回放缓冲器"""
    
    def __init__(self, capacity: int):
        """
        初始化整局游戏回放缓冲器
        
        Args:
            capacity: 缓冲器容量（存储的局数）
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
    
    def add_step(self, experience: Experience):
        """
        添加单步经验到当前局
        
        Args:
            experience: 单步经验
        """
        self.current_episode.append(experience)
    
    def finish_episode(self, final_reward: float = 0.0):
        """
        完成当前局，添加到缓冲器
        
        Args:
            final_reward: 最终奖励
        """
        if self.current_episode:
            # 为当前局的所有经验添加最终奖励
            for exp in self.current_episode:
                exp.reward += final_reward * 0.1  # 衰减因子
            
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def sample_episode(self) -> Optional[List[Experience]]:
        """
        采样一整局游戏
        
        Returns:
            一局游戏的经验列表
        """
        if not self.episodes:
            return None
        
        return random.choice(self.episodes)
    
    def sample_batch(self, batch_size: int) -> List[List[Experience]]:
        """
        采样多局游戏
        
        Args:
            batch_size: 批次大小
            
        Returns:
            多局游戏的经验列表
        """
        if len(self.episodes) < batch_size:
            return []
        
        episodes = random.sample(list(self.episodes), batch_size)
        return episodes
    
    def __len__(self):
        return len(self.episodes)

class HERReplayBuffer(ReplayBuffer):
    """基于目标的经验回放缓冲器"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, k: int = 4):
        """
        初始化HER缓冲器
        
        Args:
            capacity: 缓冲器容量
            alpha: 优先级指数
            k: 替换目标数量
        """
        super().__init__(capacity, alpha)
        self.k = k
    
    def add_episode(self, episode: List[Experience], goal: np.ndarray):
        """
        添加一整局游戏到缓冲器
        
        Args:
            episode: 一局游戏的经验列表
            goal: 目标状态
        """
        # 为每个经验创建变体
        for i, exp in enumerate(episode):
            if not exp.done:
                # 创建HER经验
                her_exp = Experience(
                    state=exp.state.copy(),
                    action=exp.action,
                    reward=self._compute_goal_reward(exp.next_state, goal),
                    next_state=exp.next_state.copy(),
                    done=exp.done,
                    priority=exp.priority,
                    weight=exp.weight,
                    timestamp=exp.timestamp,
                    game_info=exp.game_info.copy()
                )
                self.add(her_exp)
    
    def _compute_goal_reward(self, state: np.ndarray, goal: np.ndarray) -> float:
        """
        计算目标奖励
        
        Args:
            state: 当前状态
            goal: 目标状态
            
        Returns:
            奖励值
        """
        # 简化的目标奖励计算
        distance = np.linalg.norm(state - goal)
        return max(0, 1.0 - distance)

class NStepReplayBuffer:
    """N步经验回放缓冲器"""
    
    def __init__(self, capacity: int, n_steps: int = 1, gamma: float = 0.99):
        """
        初始化N步回放缓冲器
        
        Args:
            capacity: 缓冲器容量
            n_steps: N步大小
            gamma: 折扣因子
        """
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = []
        self.position = 0
        self.temporal_buffer = deque(maxlen=n_steps)
        
        # 用于线程安全
        self.lock = threading.Lock()
    
    def add(self, experience: Experience):
        """
        添加经验到缓冲器
        
        Args:
            experience: 经验数据
        """
        with self.lock:
            # 添加到临时缓冲器
            self.temporal_buffer.append(experience)
            
            # 如果有足够的步数，创建N步经验
            if len(self.temporal_buffer) >= self.n_steps:
                # 计算累积奖励
                cumulative_reward = 0.0
                for i, exp in enumerate(self.temporal_buffer):
                    cumulative_reward += exp.reward * (self.gamma ** i)
                
                # 创建N步经验
                n_step_exp = Experience(
                    state=self.temporal_buffer[0].state,
                    action=self.temporal_buffer[0].action,
                    reward=cumulative_reward,
                    next_state=experience.next_state,
                    done=experience.done,
                    priority=experience.priority,
                    weight=experience.weight,
                    timestamp=experience.timestamp,
                    game_info=experience.game_info.copy()
                )
                
                # 添加到主缓冲器
                if len(self.buffer) < self.capacity:
                    self.buffer.append(n_step_exp)
                else:
                    self.buffer[self.position] = n_step_exp
                    self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            采样到的经验列表
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            
            return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ProportionalReplayBuffer(ReplayBuffer):
    """比例经验回放缓冲器"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        初始化比例回放缓冲器
        
        Args:
            capacity: 缓冲器容量
            alpha: 优先级指数
        """
        super().__init__(capacity, alpha)
        self.sum_tree = SumTree(capacity)
    
    def add(self, experience: Experience):
        """添加经验并更新求和树"""
        with self.lock:
            idx = self.position
            self.sum_tree[idx] = experience.priority ** self.alpha
            super().add(experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        比例采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (采样到的经验列表, 索引, 权重)
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return [], [], []
            
            # 获取采样范围
            priorities = []
            for _ in range(batch_size):
                a = 0
                b = self.sum_tree.total() / batch_size
                s = random.uniform(a, b)
                idx = self.sum_tree.get(s)
                priorities.append(self.sum_tree[idx])
            
            # 转换为概率
            priorities = np.array(priorities)
            probabilities = priorities / self.sum_tree.total()
            
            # 采样
            indices = [self.sum_tree.get(random.uniform(0, self.sum_tree.total() / batch_size)) 
                      for _ in range(batch_size)]
            experiences = [self.buffer[i] for i in indices]
            
            # 计算重要性采样权重
            weights = (len(self.buffer) * probabilities) ** (-self.beta)
            weights = weights / weights.max()
            
            return experiences, indices, weights

class SumTree:
    """求和树数据结构"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity)
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """传播变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """检索索引"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """获取总和"""
        return self.tree[0]
    
    def add(self, p, data):
        """添加数据"""
        idx = self.n_entries + self.capacity - 1
        self.data[self.n_entries] = data
        self.update(idx, p)
        self.n_entries += 1
        
        if self.n_entries >= self.capacity:
            self.n_entries = 0
    
    def update(self, idx, p):
        """更新优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        """获取索引"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx

class BufferManager:
    """缓冲器管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化缓冲器管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.buffers = {}
        
        # 创建各种缓冲器
        self._create_buffers()
    
    def _create_buffers(self):
        """创建各种缓冲器"""
        # 基础回放缓冲器
        if 'replay_buffer' in self.config:
            self.buffers['replay'] = ReplayBuffer(
                capacity=self.config['replay_buffer']['capacity'],
                alpha=self.config['replay_buffer'].get('alpha', 0.6)
            )
        
        # 优先级回放缓冲器
        if 'prioritized_buffer' in self.config:
            self.buffers['prioritized'] = PrioritizedReplayBuffer(
                capacity=self.config['prioritized_buffer']['capacity'],
                alpha=self.config['prioritized_buffer'].get('alpha', 0.6),
                beta=self.config['prioritized_buffer'].get('beta', 0.4)
            )
        
        # 整局游戏缓冲器
        if 'episode_buffer' in self.config:
            self.buffers['episode'] = EpisodeReplayBuffer(
                capacity=self.config['episode_buffer']['capacity']
            )
        
        # N步回放缓冲器
        if 'nstep_buffer' in self.config:
            self.buffers['nstep'] = NStepReplayBuffer(
                capacity=self.config['nstep_buffer']['capacity'],
                n_steps=self.config['nstep_buffer'].get('n_steps', 1),
                gamma=self.config['nstep_buffer'].get('gamma', 0.99)
            )
    
    def add_experience(self, buffer_type: str, experience: Experience):
        """
        添加经验到指定缓冲器
        
        Args:
            buffer_type: 缓冲器类型
            experience: 经验数据
        """
        if buffer_type in self.buffers:
            self.buffers[buffer_type].add(experience)
    
    def sample_batch(self, buffer_type: str, batch_size: int):
        """
        从指定缓冲器采样批次
        
        Args:
            buffer_type: 缓冲器类型
            batch_size: 批次大小
            
        Returns:
            采样结果
        """
        if buffer_type not in self.buffers:
            return None
        
        buffer = self.buffers[buffer_type]
        
        if isinstance(buffer, PrioritizedReplayBuffer):
            return buffer.sample(batch_size)
        elif isinstance(buffer, EpisodeReplayBuffer):
            return buffer.sample_episode()
        else:
            return buffer.sample(batch_size)
    
    def get_buffer_stats(self) -> Dict:
        """
        获取所有缓冲器的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        for buffer_type, buffer in self.buffers.items():
            stats[buffer_type] = {
                'size': len(buffer),
                'capacity': getattr(buffer, 'capacity', None),
                'type': type(buffer).__name__
            }
        
        return stats
    
    def clear_all(self):
        """清空所有缓冲器"""
        for buffer in self.buffers.values():
            buffer.clear()

# 工厂函数
def create_buffer(buffer_type: str, config: Dict) -> ReplayBuffer:
    """
    创建缓冲器的工厂函数
    
    Args:
        buffer_type: 缓冲器类型
        config: 配置字典
        
    Returns:
        缓冲器实例
    """
    if buffer_type == 'replay':
        return ReplayBuffer(
            capacity=config.get('capacity', 10000),
            alpha=config.get('alpha', 0.6)
        )
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            capacity=config.get('capacity', 10000),
            alpha=config.get('alpha', 0.6),
            beta=config.get('beta', 0.4)
        )
    elif buffer_type == 'episode':
        return EpisodeReplayBuffer(
            capacity=config.get('capacity', 1000)
        )
    elif buffer_type == 'nstep':
        return NStepReplayBuffer(
            capacity=config.get('capacity', 10000),
            n_steps=config.get('n_steps', 1),
            gamma=config.get('gamma', 0.99)
        )
    elif buffer_type == 'her':
        return HERReplayBuffer(
            capacity=config.get('capacity', 10000),
            alpha=config.get('alpha', 0.6),
            k=config.get('k', 4)
        )
    else:
        raise ValueError(f"未知的缓冲器类型: {buffer_type}")

if __name__ == "__main__":
    # 测试缓冲器
    print("测试经验回放缓冲器...")
    
    # 创建缓冲器
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    
    # 添加经验
    for i in range(100):
        exp = Experience(
            state=np.random.randn(132),
            action=i % 34,
            reward=np.random.randn(),
            next_state=np.random.randn(132),
            done=False
        )
        buffer.add(exp)
    
    # 采样批次
    batch = buffer.sample(32)
    print(f"采样到 {len(batch)} 个经验")
    
    # 获取统计信息
    stats = buffer.get_buffer_stats()
    print(f"缓冲器统计: {stats}")
    
    print("缓冲器测试完成！")