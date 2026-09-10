"""
川麻将动作预测器
基于神经网络模型进行动作选择
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import random
import math
from dataclasses import dataclass
from .model import MahjongModel
from .features import FeatureProcessor, ActionEncoder, FeatureConfig

@dataclass
class PredictionConfig:
    """预测配置"""
    epsilon: float = 0.1  # 探索率
    epsilon_decay: float = 0.995  # 探索率衰减
    epsilon_min: float = 0.01  # 最小探索率
    temperature: float = 1.0  # 温度参数
    top_k: int = 5  # Top-K动作选择
    beam_width: int = 3  # 束搜索宽度
    max_search_depth: int = 3  # 最大搜索深度
    use_value_network: bool = True  # 是否使用价值网络
    use_mcts: bool = False  # 是否使用蒙特卡洛树搜索
    mcts_simulations: int = 100  # MCTS模拟次数
    ucb_c: float = 2.0  # UCB常数

class ActionPredictor:
    """动作预测器"""
    
    def __init__(self, 
                 model: MahjongModel,
                 feature_processor: FeatureProcessor,
                 action_encoder: ActionEncoder,
                 config: Optional[PredictionConfig] = None):
        """
        初始化动作预测器
        
        Args:
            model: 神经网络模型
            feature_processor: 特征处理器
            action_encoder: 动作编码器
            config: 预测配置
        """
        self.model = model
        self.feature_processor = feature_processor
        self.action_encoder = action_encoder
        self.config = config or PredictionConfig()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 探索率
        self.epsilon = self.config.epsilon
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'exploration_count': 0,
            'exploitation_count': 0,
            'action_entropy': 0.0
        }
    
    def predict_action(self, game_state: Dict) -> Tuple[int, float, Dict]:
        """
        预测动作
        
        Args:
            game_state: 游戏状态
            
        Returns:
            (动作索引, 动作概率, 预测信息)
        """
        self.stats['total_predictions'] += 1
        
        # 获取合法动作
        valid_actions = self.action_encoder.get_valid_actions(game_state)
        
        if not valid_actions:
            return -1, 0.0, {'error': 'No valid actions'}
        
        # 编码特征
        features = self.feature_processor.encode_features(game_state)
        normalized_features = self.feature_processor.normalize_features(features)
        
        # 预测动作概率和价值
        action_probs, value = self.model.predict(normalized_features)
        
        # 过滤非法动作
        valid_probs = []
        valid_action_indices = []
        
        for action_idx in valid_actions:
            if action_idx < len(action_probs):
                valid_probs.append(action_probs[action_idx])
                valid_action_indices.append(action_idx)
        
        if not valid_probs:
            return -1, 0.0, {'error': 'No valid probabilities'}
        
        # 归一化概率
        valid_probs = np.array(valid_probs)
        valid_probs = valid_probs / valid_probs.sum()
        
        # 根据策略选择动作
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = random.choice(valid_action_indices)
            self.stats['exploration_count'] += 1
        else:
            # 利用：根据概率选择动作
            action_idx = np.random.choice(valid_action_indices, p=valid_probs)
            self.stats['exploitation_count'] += 1
        
        # 获取动作概率
        action_prob = valid_probs[valid_action_indices.index(action_idx)]
        
        # 计算动作熵
        entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-8))
        self.stats['action_entropy'] = entropy
        
        # 更新探索率
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
        
        # 构建预测信息
        prediction_info = {
            'value': value,
            'entropy': entropy,
            'epsilon': self.epsilon,
            'valid_actions_count': len(valid_actions),
            'action_prob': action_prob,
            'top_actions': self._get_top_actions(valid_action_indices, valid_probs),
            'game_state': {
                'hand': game_state.get('hand', []),
                'current_player': game_state.get('current_player', 0)
            }
        }
        
        return action_idx, action_prob, prediction_info
    
    def predict_with_beam_search(self, game_state: Dict) -> Tuple[int, float, Dict]:
        """
        使用束搜索预测动作
        
        Args:
            game_state: 游戏状态
            
        Returns:
            (动作索引, 动作概率, 预测信息)
        """
        # 获取合法动作
        valid_actions = self.action_encoder.get_valid_actions(game_state)
        
        if not valid_actions:
            return -1, 0.0, {'error': 'No valid actions'}
        
        # 编码当前特征
        current_features = self.feature_processor.encode_features(game_state)
        normalized_features = self.feature_processor.normalize_features(current_features)
        
        # 预测当前状态价值
        _, current_value = self.model.predict(normalized_features)
        
        # 束搜索
        beam = [(action_idx, current_value, [action_idx]) for action_idx in valid_actions]
        
        for depth in range(self.config.max_search_depth):
            new_beam = []
            
            for action_idx, value, action_sequence in beam:
                # 模拟动作后的状态
                new_state = self._simulate_action(game_state, action_idx)
                if new_state is None:
                    continue
                
                # 预测新状态价值
                new_features = self.feature_processor.encode_features(new_state)
                normalized_new_features = self.feature_processor.normalize_features(new_features)
                _, new_value = self.model.predict(normalized_new_features)
                
                # 计算累计价值
                cumulative_value = value + new_value * (0.99 ** (depth + 1))
                
                new_beam.append((action_idx, cumulative_value, action_sequence + [action_idx]))
            
            # 选择Top-K
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:self.config.beam_width]
        
        # 选择最佳动作序列的第一个动作
        best_action_idx = beam[0][2][0]
        
        # 计算动作概率
        action_probs = [1.0 / len(beam)] * len(beam)
        
        prediction_info = {
            'value': beam[0][1],
            'beam_width': self.config.beam_width,
            'search_depth': self.config.max_search_depth,
            'best_sequence': beam[0][2],
            'beam_values': [b[1] for b in beam]
        }
        
        return best_action_idx, action_probs[0], prediction_info
    
    def predict_with_mcts(self, game_state: Dict) -> Tuple[int, float, Dict]:
        """
        使用蒙特卡洛树搜索预测动作
        
        Args:
            game_state: 游戏状态
            
        Returns:
            (动作索引, 动作概率, 预测信息)
        """
        # MCTS节点
        class MCTSNode:
            def __init__(self, state, parent=None, action=None):
                self.state = state
                self.parent = parent
                self.action = action
                self.children = []
                self.visits = 0
                self.value = 0.0
                self.ucb_score = 0.0
            
            def update(self, reward):
                self.visits += 1
                self.value += (reward - self.value) / self.visits
                self.ucb_score = self.value + self.config.ucb_c * math.sqrt(
                    math.log(self.parent.visits) / self.visits
                ) if self.parent else float('inf')
        
        # 初始化根节点
        root = MCTSNode(game_state)
        
        # MCTS模拟
        for _ in range(self.config.mcts_simulations):
            node = root
            path = [node]
            
            # 选择
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score)
                path.append(node)
            
            # 扩展
            if node.visits > 0:
                valid_actions = self.action_encoder.get_valid_actions(node.state)
                for action_idx in valid_actions:
                    new_state = self._simulate_action(node.state, action_idx)
                    if new_state is not None:
                        child = MCTSNode(new_state, node, action_idx)
                        node.children.append(child)
                        path.append(child)
                
                if node.children:
                    node = random.choice(node.children)
                    path.append(node)
            
            # 模拟
            reward = self._simulate_game(node.state)
            
            # 回溯
            for node in path:
                node.update(reward)
        
        # 选择最佳动作
        action_probs = []
        for child in root.children:
            prob = child.visits / root.visits
            action_probs.append((child.action, prob))
        
        if not action_probs:
            return -1, 0.0, {'error': 'No MCTS actions found'}
        
        # 选择访问次数最多的动作
        best_action_idx = max(action_probs, key=lambda x: x[1])[0]
        
        prediction_info = {
            'mcts_simulations': self.config.mcts_simulations,
            'total_visits': root.visits,
            'action_visits': [(child.action, child.visits) for child in root.children],
            'best_value': max(child.value for child in root.children)
        }
        
        return best_action_idx, action_probs[0][1], prediction_info
    
    def _get_top_actions(self, valid_actions: List[int], valid_probs: np.ndarray) -> List[Dict]:
        """获取Top-K动作"""
        if len(valid_actions) <= self.config.top_k:
            return [
                {'action': action_idx, 'prob': prob}
                for action_idx, prob in zip(valid_actions, valid_probs)
            ]
        
        # 获取Top-K
        top_indices = np.argsort(valid_probs)[-self.config.top_k:][::-1]
        return [
            {'action': valid_actions[i], 'prob': valid_probs[i]}
            for i in top_indices
        ]
    
    def _simulate_action(self, game_state: Dict, action_idx: int) -> Optional[Dict]:
        """
        模拟动作执行
        
        Args:
            game_state: 游戏状态
            action_idx: 动作索引
            
        Returns:
            模拟后的游戏状态
        """
        # 简化的动作模拟
        action = self.action_encoder.decode_action(action_idx)
        
        new_state = game_state.copy()
        
        if action.startswith('discard_'):
            tile = int(action.split('_')[1])
            hand = new_state.get('hand', [])
            if tile in hand:
                hand.remove(tile)
                new_state['hand'] = hand
        
        elif action.startswith('eat_'):
            # 简化的吃牌模拟
            pass
        
        elif action.startswith('peng_'):
            # 简化的碰牌模拟
            pass
        
        elif action.startswith('gang_'):
            # 简化的杠牌模拟
            pass
        
        elif action == 'win':
            # 胡牌
            new_state['game_over'] = True
            new_state['winner'] = new_state.get('current_player', 0)
        
        elif action == 'draw':
            # 流局
            new_state['game_over'] = True
        
        return new_state
    
    def _simulate_game(self, game_state: Dict) -> float:
        """
        模拟游戏直到结束
        
        Args:
            game_state: 游戏状态
            
        Returns:
            奖励值
        """
        # 简化的游戏模拟
        if game_state.get('game_over', False):
            if 'winner' in game_state:
                return 1.0 if game_state['winner'] == 0 else -1.0
            else:
                return 0.0
        
        # 随机模拟几步
        for _ in range(10):
            valid_actions = self.action_encoder.get_valid_actions(game_state)
            if not valid_actions:
                break
            
            action_idx = random.choice(valid_actions)
            game_state = self._simulate_action(game_state, action_idx)
            
            if game_state.get('game_over', False):
                break
        
        return 0.0  # 默认奖励
    
    def get_stats(self) -> Dict:
        """获取预测统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_predictions': 0,
            'exploration_count': 0,
            'exploitation_count': 0,
            'action_entropy': 0.0
        }
    
    def save_stats(self, path: str):
        """保存统计信息"""
        import json
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_stats(self, path: str):
        """加载统计信息"""
        import json
        with open(path, 'r') as f:
            self.stats = json.load(f)

class BatchPredictor:
    """批量预测器"""
    
    def __init__(self, 
                 model: MahjongModel,
                 feature_processor: FeatureProcessor,
                 action_encoder: ActionEncoder,
                 config: Optional[PredictionConfig] = None):
        """
        初始化批量预测器
        
        Args:
            model: 神经网络模型
            feature_processor: 特征处理器
            action_encoder: 动作编码器
            config: 预测配置
        """
        self.model = model
        self.feature_processor = feature_processor
        self.action_encoder = action_encoder
        self.config = config or PredictionConfig()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict_batch(self, game_states: List[Dict]) -> List[Tuple[int, float, Dict]]:
        """
        批量预测动作
        
        Args:
            game_states: 游戏状态列表
            
        Returns:
            预测结果列表
        """
        # 批量编码特征
        features = []
        for state in game_states:
            feature = self.feature_processor.encode_features(state)
            normalized_feature = self.feature_processor.normalize_features(feature)
            features.append(normalized_feature)
        
        # 批量预测
        features_tensor = torch.FloatTensor(np.array(features)).to(self.device)
        
        with torch.no_grad():
            action_probs_batch, values_batch = self.model.policy_net(features_tensor), \
                                             self.model.value_net(features_tensor)
        
        results = []
        for i, (state, action_probs, value) in enumerate(zip(game_states, action_probs_batch, values_batch)):
            # 获取合法动作
            valid_actions = self.action_encoder.get_valid_actions(state)
            
            # 过滤非法动作
            valid_probs = []
            valid_action_indices = []
            
            for action_idx in valid_actions:
                if action_idx < len(action_probs):
                    valid_probs.append(action_probs[action_idx].item())
                    valid_action_indices.append(action_idx)
            
            if not valid_probs:
                results.append((-1, 0.0, {'error': 'No valid actions'}))
                continue
            
            # 归一化概率
            valid_probs = np.array(valid_probs)
            valid_probs = valid_probs / valid_probs.sum()
            
            # 选择动作
            action_idx = np.random.choice(valid_action_indices, p=valid_probs)
            action_prob = valid_probs[valid_action_indices.index(action_idx)]
            
            # 构建预测信息
            prediction_info = {
                'value': value.item(),
                'valid_actions_count': len(valid_actions),
                'action_prob': action_prob,
                'game_state': {
                    'hand': state.get('hand', []),
                    'current_player': state.get('current_player', 0)
                }
            }
            
            results.append((action_idx, action_prob, prediction_info))
        
        return results

if __name__ == "__main__":
    # 测试动作预测器
    print("测试动作预测器...")
    
    # 创建模型
    from mjai.model import MahjongModel
    model = MahjongModel()
    
    # 创建特征处理器和动作编码器
    from mjai.features import FeatureProcessor, ActionEncoder
    config = FeatureConfig()
    feature_processor = FeatureProcessor(config)
    action_encoder = ActionEncoder(config)
    
    # 创建预测器
    predictor = ActionPredictor(model, feature_processor, action_encoder)
    
    # 创建测试游戏状态
    game_state = {
        'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
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
        'current_player': 0
    }
    
    # 预测动作
    action_idx, action_prob, info = predictor.predict_action(game_state)
    print(f"预测动作: {action_idx}, 概率: {action_prob:.3f}")
    print(f"预测信息: {info}")
    
    # 获取统计信息
    stats = predictor.get_stats()
    print(f"统计信息: {stats}")
    
    print("动作预测器测试完成！")