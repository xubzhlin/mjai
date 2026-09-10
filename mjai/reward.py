"""
奖励系统
实现各种奖励函数和奖励策略
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

class RewardType(Enum):
    """奖励类型枚举"""
    IMMEDIATE = "immediate"      # 即时奖励
    DELAYED = "delayed"          # 延迟奖励
    SHAPED = "shaped"           # 形状奖励
    GOAL = "goal"               # 目标奖励
    PENALTY = "penalty"         # 惩罚奖励
    BONUS = "bonus"             # 奖励加成
    CUMULATIVE = "cumulative"   # 累积奖励

@dataclass
class RewardConfig:
    """奖励配置"""
    # 基础奖励权重
    win_reward: float = 10.0
    lose_penalty: float = -5.0
    draw_penalty: float = -1.0
    
    # 动作奖励权重
    discard_penalty: float = -0.1
    action_reward: float = 0.1
    illegal_action_penalty: float = -1.0
    
    # 游戏进程奖励
    progress_reward: float = 0.05
    tile_efficiency_reward: float = 0.1
    hand_quality_reward: float = 0.2
    
    # 风险奖励权重
    risk_penalty: float = -0.2
    risk_reward: float = 0.3
    
    # 番种奖励权重
    basic_fan_reward: float = 1.0
    special_fan_reward: float = 2.0
    yakuman_reward: float = 5.0
    
    # 时间衰减
    time_decay: float = 0.99
    max_steps: int = 100
    
    # 奖励裁剪
    reward_clip: float = 10.0
    
    # 归一化
    normalize_rewards: bool = True
    
    # 奖励缩放
    reward_scale: float = 1.0

class RewardFunction:
    """基础奖励函数"""
    
    def __init__(self, config: RewardConfig):
        """
        初始化奖励函数
        
        Args:
            config: 奖励配置
        """
        self.config = config
        self.step_count = 0
        self.game_history = []
    
    def calculate_reward(self, 
                       state: Dict, 
                       action: int, 
                       next_state: Dict, 
                       done: bool) -> float:
        """
        计算奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            done: 游戏是否结束
            
        Returns:
            奖励值
        """
        # 重置步数
        if done:
            self.step_count = 0
        
        # 计算各种奖励
        rewards = []
        
        # 1. 游戏结束奖励
        if done:
            game_reward = self._calculate_game_end_reward(state, next_state)
            rewards.append(('game_end', game_reward))
        
        # 2. 动作奖励
        action_reward = self._calculate_action_reward(state, action, next_state)
        rewards.append(('action', action_reward))
        
        # 3. 进度奖励
        progress_reward = self._calculate_progress_reward(state, next_state)
        rewards.append(('progress', progress_reward))
        
        # 4. 手牌质量奖励
        hand_reward = self._calculate_hand_quality_reward(state, next_state)
        rewards.append(('hand_quality', hand_reward))
        
        # 5. 风险奖励
        risk_reward = self._calculate_risk_reward(state, action, next_state)
        rewards.append(('risk', risk_reward))
        
        # 6. 番种奖励
        fan_reward = self._calculate_fan_reward(state, next_state)
        rewards.append(('fan', fan_reward))
        
        # 7. 时间衰减
        time_decay = self._calculate_time_decay()
        
        # 计算总奖励
        total_reward = sum(reward for _, reward in rewards) * time_decay
        
        # 应用奖励缩放和裁剪
        total_reward *= self.config.reward_scale
        total_reward = np.clip(total_reward, -self.config.reward_clip, self.config.reward_clip)
        
        # 记录奖励历史
        self.game_history.append({
            'step': self.step_count,
            'action': action,
            'rewards': rewards,
            'total_reward': total_reward,
            'done': done
        })
        
        self.step_count += 1
        
        return total_reward
    
    def _calculate_game_end_reward(self, state: Dict, next_state: Dict) -> float:
        """计算游戏结束奖励"""
        if next_state.get('winner') is not None:
            if next_state['winner'] == 0:  # 玩家0获胜
                return self.config.win_reward
            else:  # 玩家0失败
                return self.config.lose_penalty
        elif next_state.get('game_over', False):
            # 流局
            return self.config.draw_penalty
        return 0.0
    
    def _calculate_action_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """计算动作奖励"""
        action_str = self._action_to_string(action)
        
        if action_str == 'win':
            return self.config.win_reward
        elif action_str.startswith('discard_'):
            return self.config.discard_penalty
        elif action_str in ['eat', 'peng', 'gang']:
            return self.config.action_reward
        elif action_str == 'pass':
            return self.config.illegal_action_penalty
        
        return 0.0
    
    def _calculate_progress_reward(self, state: Dict, next_state: Dict) -> float:
        """计算进度奖励"""
        current_tiles = state.get('round_info', {}).get('remaining_tiles', 0)
        next_tiles = next_state.get('round_info', {}).get('remaining_tiles', 0)
        
        # 基于剩余牌数的进度奖励
        if current_tiles > next_tiles:
            progress = (current_tiles - next_tiles) / 70  # 初始剩余牌数
            return self.config.progress_reward * progress
        
        return 0.0
    
    def _calculate_hand_quality_reward(self, state: Dict, next_state: Dict) -> float:
        """计算手牌质量奖励"""
        current_hand = state.get('hand', [])
        next_hand = next_state.get('hand', [])
        
        # 计算向听数改进
        current_shanten = self._calculate_shanten(current_hand)
        next_shanten = self._calculate_shanten(next_hand)
        
        if next_shanten < current_shanten:
            improvement = current_shanten - next_shanten
            return self.config.hand_quality_reward * improvement
        
        return 0.0
    
    def _calculate_risk_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """计算风险奖励"""
        action_str = self._action_to_string(action)
        
        # 计算风险分数
        risk_score = 0.0
        
        # 弃牌风险
        if action_str.startswith('discard_'):
            discarded_tile = int(action_str.split('_')[1])
            risk_score = self._calculate_discard_risk(state, discarded_tile)
        
        # 根据风险分数给予奖励或惩罚
        if risk_score > 0.5:  # 高风险
            return self.config.risk_penalty * risk_score
        elif risk_score < 0.3:  # 低风险
            return self.config.risk_reward * (1 - risk_score)
        
        return 0.0
    
    def _calculate_fan_reward(self, state: Dict, next_state: Dict) -> float:
        """计算番种奖励"""
        # 简化的番种计算
        current_fans = self._estimate_fans(state.get('hand', []))
        next_fans = self._estimate_fans(next_state.get('hand', []))
        
        if next_fans > current_fans:
            fan_diff = next_fans - current_fans
            
            # 根据番种类型给予不同奖励
            if fan_diff >= 13:  # 役满
                return self.config.yakuman_reward
            elif fan_diff >= 8:  # 特殊番种
                return self.config.special_fan_reward
            else:  # 基础番种
                return self.config.basic_fan_reward * fan_diff
        
        return 0.0
    
    def _calculate_time_decay(self) -> float:
        """计算时间衰减"""
        return self.config.time_decay ** min(self.step_count, self.config.max_steps)
    
    def _action_to_string(self, action: int) -> str:
        """将动作索引转换为字符串"""
        # 简化的动作转换
        if action == 0:
            return 'discard_0'
        elif action == 1:
            return 'win'
        elif action == 2:
            return 'eat'
        elif action == 3:
            return 'peng'
        elif action == 4:
            return 'gang'
        else:
            return 'pass'
    
    def _calculate_shanten(self, hand: List[int]) -> int:
        """计算向听数（简化版）"""
        if not hand:
            return 13
        
        # 简化的向听数计算
        unique_tiles = len(set(hand))
        return max(0, 13 - unique_tiles)
    
    def _calculate_discard_risk(self, state: Dict, tile: int) -> float:
        """计算弃牌风险"""
        hand = state.get('hand', [])
        
        # 计算弃牌后可能影响的对子数
        pairs = sum(1 for t in hand if t == tile)
        
        # 计算可能影响的有效牌数
        effective_tiles = sum(1 for t in hand if abs(t - tile) <= 2)
        
        # 风险分数 (0-1)
        risk_score = (pairs + effective_tiles / 10) / len(hand)
        return min(risk_score, 1.0)
    
    def _estimate_fans(self, hand: List[int]) -> int:
        """估算番种数（简化版）"""
        if not hand:
            return 0
        
        # 简化的番种估算
        unique_suits = len(set(t // 9 for t in hand))
        
        # 基础番种
        fans = 0
        
        # 清一色
        if unique_suits == 1:
            fans += 8
        
        # 混一色
        if unique_suits <= 2:
            fans += 5
        
        # 对对胡
        if self._is_all_pairs(hand):
            fans += 4
        
        # 平胡
        if self._is_standard_win(hand):
            fans += 2
        
        return fans
    
    def _is_all_pairs(self, hand: List[int]) -> bool:
        """检查是否是对对胡"""
        from collections import Counter
        counts = Counter(hand)
        return all(count >= 2 for count in counts.values())
    
    def _is_standard_win(self, hand: List[int]) -> bool:
        """检查是否是标准胡牌"""
        # 简化的标准胡牌检查
        return len(hand) == 13  # 基本检查

class ShapedRewardFunction(RewardFunction):
    """形状奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.potential_function = PotentialFunction()
    
    def calculate_reward(self, 
                       state: Dict, 
                       action: int, 
                       next_state: Dict, 
                       done: bool) -> float:
        """
        使用势函数计算奖励
        """
        # 计算势差
        current_potential = self.potential_function.calculate(state)
        next_potential = self.potential_function.calculate(next_state)
        
        # 基础奖励
        base_reward = super().calculate_reward(state, action, next_state, done)
        
        # 形状奖励
        shaped_reward = next_potential - current_potential
        
        # 组合奖励
        total_reward = base_reward + shaped_reward
        
        return total_reward

class PotentialFunction:
    """势函数"""
    
    def calculate(self, state: Dict) -> float:
        """
        计算状态的势值
        
        Args:
            state: 游戏状态
            
        Returns:
            势值
        """
        hand = state.get('hand', [])
        
        # 基于向听数的势值
        shanten = self._calculate_shanten(hand)
        
        # 基于手牌质量的势值
        hand_quality = self._calculate_hand_quality(hand)
        
        # 基于游戏进度的势值
        progress = self._calculate_progress(state)
        
        # 组合势值
        potential = -shanten + hand_quality + progress
        
        return potential
    
    def _calculate_shanten(self, hand: List[int]) -> int:
        """计算向听数"""
        if not hand:
            return 13
        
        # 简化的向听数计算
        unique_tiles = len(set(hand))
        return max(0, 13 - unique_tiles)
    
    def _calculate_hand_quality(self, hand: List[int]) -> float:
        """计算手牌质量"""
        if not hand:
            return 0.0
        
        # 基于牌型的质量评估
        unique_suits = len(set(t // 9 for t in hand))
        pairs = sum(1 for t in hand if hand.count(t) >= 2)
        
        quality = 0.0
        quality += (3 - unique_suits) * 0.1  # 花色集中度
        quality += pairs * 0.2  # 对子数量
        
        return min(quality, 1.0)
    
    def _calculate_progress(self, state: Dict) -> float:
        """计算游戏进度"""
        remaining_tiles = state.get('round_info', {}).get('remaining_tiles', 0)
        return (70 - remaining_tiles) / 70  # 0-1之间的进度值

class MultiObjectiveRewardFunction:
    """多目标奖励函数"""
    
    def __init__(self, config: RewardConfig):
        """
        初始化多目标奖励函数
        
        Args:
            config: 奖励配置
        """
        self.config = config
        self.objectives = {
            'win': 1.0,
            'progress': 0.5,
            'efficiency': 0.3,
            'safety': 0.2
        }
        
        self.objective_functions = {
            'win': WinObjectiveFunction(),
            'progress': ProgressObjectiveFunction(),
            'efficiency': EfficiencyObjectiveFunction(),
            'safety': SafetyObjectiveFunction()
        }
    
    def calculate_reward(self, 
                       state: Dict, 
                       action: int, 
                       next_state: Dict, 
                       done: bool) -> float:
        """
        计算多目标奖励
        """
        total_reward = 0.0
        
        for objective_name, weight in self.objectives.items():
            objective_func = self.objective_functions[objective_name]
            reward = objective_func.calculate_reward(state, action, next_state, done)
            total_reward += reward * weight
        
        return total_reward

class WinObjectiveFunction:
    """获胜目标函数"""
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, done: bool) -> float:
        """计算获胜奖励"""
        if done and next_state.get('winner') == 0:
            return 10.0
        elif done and next_state.get('winner') is not None:
            return -5.0
        return 0.0

class ProgressObjectiveFunction:
    """进度目标函数"""
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, done: bool) -> float:
        """计算进度奖励"""
        current_tiles = state.get('round_info', {}).get('remaining_tiles', 0)
        next_tiles = next_state.get('round_info', {}).get('remaining_tiles', 0)
        
        if current_tiles > next_tiles:
            return 0.1 * (current_tiles - next_tiles) / 70
        
        return 0.0

class EfficiencyObjectiveFunction:
    """效率目标函数"""
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, done: bool) -> float:
        """计算效率奖励"""
        # 简化的效率计算
        if action == 0:  # 弃牌
            return -0.05
        elif action in [2, 3, 4]:  # 吃碰杠
            return 0.1
        
        return 0.0

class SafetyObjectiveFunction:
    """安全目标函数"""
    
    def calculate_reward(self, state: Dict, action: int, next_state: Dict, done: bool) -> float:
        """计算安全奖励"""
        # 简化的安全计算
        if action == 0:  # 弃牌
            hand = state.get('hand', [])
            discarded_tile = int(action)  # 简化处理
            risk = self._calculate_discard_risk(hand, discarded_tile)
            return -risk * 0.2
        
        return 0.0
    
    def _calculate_discard_risk(self, hand: List[int], tile: int) -> float:
        """计算弃牌风险"""
        if not hand:
            return 0.0
        
        pairs = sum(1 for t in hand if t == tile)
        return min(pairs / len(hand), 1.0)

class RewardNormalizer:
    """奖励归一化器"""
    
    def __init__(self, config: RewardConfig):
        """
        初始化奖励归一化器
        
        Args:
            config: 奖励配置
        """
        self.config = config
        self.reward_history = []
        self.mean = 0.0
        self.std = 1.0
        self.update_count = 0
    
    def normalize(self, reward: float) -> float:
        """
        归一化奖励
        
        Args:
            reward: 原始奖励
            
        Returns:
            归一化后的奖励
        """
        if not self.config.normalize_rewards:
            return reward
        
        # 更新统计信息
        self.reward_history.append(reward)
        self.update_count += 1
        
        if self.update_count % 100 == 0:  # 每100步更新一次
            self.mean = np.mean(self.reward_history[-1000:])  # 使用最近1000个样本
            self.std = np.std(self.reward_history[-1000:])
            self.std = max(self.std, 1e-8)  # 避免除零
        
        # 归一化
        normalized_reward = (reward - self.mean) / self.std
        
        return normalized_reward
    
    def reset(self):
        """重置归一化器"""
        self.reward_history.clear()
        self.mean = 0.0
        self.std = 1.0
        self.update_count = 0

class RewardTracker:
    """奖励跟踪器"""
    
    def __init__(self):
        """初始化奖励跟踪器"""
        self.episode_rewards = []
        self.step_rewards = []
        self.objective_rewards = {}
        self.stats = {}
    
    def start_episode(self):
        """开始新的一局"""
        self.current_episode_rewards = []
        self.current_step_rewards = []
    
    def record_step(self, reward: Dict):
        """
        记录单步奖励
        
        Args:
            reward: 奖励字典
        """
        self.current_step_rewards.append(reward)
        self.current_episode_rewards.append(reward)
    
    def finish_episode(self):
        """结束当前局"""
        total_reward = sum(r.get('total', 0) for r in self.current_episode_rewards)
        self.episode_rewards.append(total_reward)
        
        # 记录步数奖励
        for step_reward in self.current_step_rewards:
            self.step_rewards.append(step_reward)
        
        # 更新统计信息
        self._update_stats()
    
    def _update_stats(self):
        """更新统计信息"""
        if self.episode_rewards:
            self.stats = {
                'total_episodes': len(self.episode_rewards),
                'avg_episode_reward': np.mean(self.episode_rewards),
                'std_episode_reward': np.std(self.episode_rewards),
                'max_episode_reward': np.max(self.episode_rewards),
                'min_episode_reward': np.min(self.episode_rewards),
                'total_steps': len(self.step_rewards),
                'avg_step_reward': np.mean([r.get('total', 0) for r in self.step_rewards]),
                'std_step_reward': np.std([r.get('total', 0) for r in self.step_rewards])
            }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def get_recent_rewards(self, n: int = 100) -> List[float]:
        """获取最近的奖励"""
        return self.episode_rewards[-n:]

# 工厂函数
def create_reward_function(reward_type: str, config: RewardConfig) -> RewardFunction:
    """
    创建奖励函数的工厂函数
    
    Args:
        reward_type: 奖励函数类型
        config: 奖励配置
        
    Returns:
        奖励函数实例
    """
    if reward_type == 'basic':
        return RewardFunction(config)
    elif reward_type == 'shaped':
        return ShapedRewardFunction(config)
    elif reward_type == 'multiobjective':
        return MultiObjectiveRewardFunction(config)
    else:
        raise ValueError(f"未知的奖励函数类型: {reward_type}")

if __name__ == "__main__":
    # 测试奖励系统
    print("测试奖励系统...")
    
    # 创建奖励配置
    config = RewardConfig(
        win_reward=10.0,
        lose_penalty=-5.0,
        discard_penalty=-0.1,
        progress_reward=0.05,
        hand_quality_reward=0.2
    )
    
    # 创建奖励函数
    reward_func = RewardFunction(config)
    
    # 测试状态
    state = {
        'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'round_info': {'remaining_tiles': 70},
        'current_player': 0
    }
    
    next_state = {
        'hand': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'round_info': {'remaining_tiles': 69},
        'current_player': 1
    }
    
    # 计算奖励
    reward = reward_func.calculate_reward(state, 0, next_state, False)
    print(f"计算奖励: {reward}")
    
    # 测试形状奖励
    shaped_reward_func = ShapedRewardFunction(config)
    shaped_reward = shaped_reward_func.calculate_reward(state, 0, next_state, False)
    print(f"形状奖励: {shaped_reward}")
    
    # 测试多目标奖励
    multiobjective_func = MultiObjectiveRewardFunction(config)
    multi_reward = multiobjective_func.calculate_reward(state, 0, next_state, False)
    print(f"多目标奖励: {multi_reward}")
    
    print("奖励系统测试完成！")