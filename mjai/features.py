"""
川麻将特征处理模块
将游戏状态转换为神经网络输入特征
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class FeatureConfig:
    """特征配置"""
    feature_dim: int = 132
    hand_feature_dim: int = 135  # 5 × 27 = 135维
    meld_feature_dim: int = 16    # 4 × 4 = 16维
    discard_feature_dim: int = 108  # 4 × 27 = 108维
    round_feature_dim: int = 4     # 4维
    player_feature_dim: int = 4     # 4维
    # 总计: 135 + 16 + 108 + 4 + 4 = 267维，但实际使用132维
    
    # 特征归一化参数
    hand_max_count: int = 4
    meld_max_count: int = 4
    discard_max_count: int = 4
    
    # 动作空间
    action_dim: int = 34
    tile_types: List[int] = None
    
    def __post_init__(self):
        if self.tile_types is None:
            # 定义牌型：万1-9，筒1-9，条1-9，东、南、西、北、中、发、白
            self.tile_types = list(range(27))

class FeatureProcessor:
    """特征处理器"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征处理器
        
        Args:
            config: 特征配置
        """
        self.config = config or FeatureConfig()
        
        # 创建牌型到索引的映射
        self.tile_to_idx = {tile: idx for idx, tile in enumerate(self.config.tile_types)}
        self.idx_to_tile = {idx: tile for tile, idx in self.tile_to_idx.items()}
        
        # 特归一化参数
        self.hand_scale = 1.0 / self.config.hand_max_count
        self.meld_scale = 1.0 / self.config.meld_max_count
        self.discard_scale = 1.0 / self.config.discard_max_count
        
        # 统计信息
        self.feature_stats = {
            'mean': np.zeros(self.config.feature_dim),
            'std': np.ones(self.config.feature_dim),
            'min': np.zeros(self.config.feature_dim),
            'max': np.ones(self.config.feature_dim)
        }
    
    def encode_hand_features(self, hand: List[int]) -> np.ndarray:
        """
        编码手牌特征
        
        Args:
            hand: 手牌列表
            
        Returns:
            手牌特征向量 [135维]
        """
        features = np.zeros(self.config.hand_feature_dim)
        
        # 统计每种牌的数量
        tile_counts = np.zeros(27)
        for tile in hand:
            if tile < 27:
                tile_counts[tile] += 1
        
        # 编码为5个玩家的手牌特征（这里只编码当前玩家）
        for i in range(5):  # 5个玩家位置
            for j, tile_count in enumerate(tile_counts):
                idx = i * 27 + j
                features[idx] = tile_count * self.hand_scale
        
        return features
    
    def encode_meld_features(self, players: List[Dict]) -> np.ndarray:
        """
        编码吃碰杠特征
        
        Args:
            players: 玩家状态列表
            
        Returns:
            吃碰杠特征向量 [16维]
        """
        features = np.zeros(self.config.meld_feature_dim)
        
        for i, player in enumerate(players[:4]):  # 4个玩家
            melds = player.get('melds', [])
            for j in range(4):  # 最多4个吃碰杠
                if j < len(melds):
                    meld_type = melds[j].get('type', 0)
                    features[i * 4 + j] = meld_type * self.meld_scale
        
        return features
    
    def encode_discard_features(self, players: List[Dict]) -> np.ndarray:
        """
        编码弃牌特征
        
        Args:
            players: 玩家状态列表
            
        Returns:
            弃牌特征向量 [108维]
        """
        features = np.zeros(self.config.discard_feature_dim)
        
        for i, player in enumerate(players[:4]):  # 4个玩家
            discards = player.get('discards', [])
            tile_counts = np.zeros(27)
            
            for tile in discards[-27:]:  # 只考虑最近27张弃牌
                if tile < 27:
                    tile_counts[tile] += 1
            
            for j, tile_count in enumerate(tile_counts):
                idx = i * 27 + j
                features[idx] = tile_count * self.discard_scale
        
        return features
    
    def encode_round_features(self, round_info: Dict) -> np.ndarray:
        """
        编译回合特征
        
        Args:
            round_info: 回合信息
            
        Returns:
            回合特征向量 [4维]
        """
        features = np.zeros(self.config.round_feature_dim)
        
        # 本场数
        honba = round_info.get('honba', 0)
        features[0] = honba / 10.0  # 归一化到0-1
        
        # 场风
        round_wind = round_info.get('round_wind', 0)  # 0=东, 1=南, 2=西, 3=北
        features[1] = round_wind / 3.0
        
        # 自风
        player_wind = round_info.get('player_wind', 0)
        features[2] = player_wind / 3.0
        
        # 剩余牌数
        remaining_tiles = round_info.get('remaining_tiles', 70)
        features[3] = remaining_tiles / 70.0
        
        return features
    
    def encode_player_features(self, player_info: Dict) -> np.ndarray:
        """
        编码玩家特征
        
        Args:
            player_info: 玩家信息
            
        Returns:
            玩家特征向量 [4维]
        """
        features = np.zeros(self.config.player_feature_dim)
        
        # 分数
        score = player_info.get('score', 0)
        features[0] = score / 100000.0  # 归一化
        
        # 立直状态
        riichi = player_info.get('riichi', False)
        features[1] = 1.0 if riichi else 0.0
        
        # 天胡状态
        tenhou = player_info.get('tenhou', False)
        features[2] = 1.0 if tenhou else 0.0
        
        # 地胡状态
        chihou = player_info.get('chihou', False)
        features[3] = 1.0 if chihou else 0.0
        
        return features
    
    def encode_features(self, game_state: Dict) -> np.ndarray:
        """
        编码完整游戏状态
        
        Args:
            game_state: 游戏状态字典
            
        Returns:
            完整特征向量 [132维]
        """
        # 提取各个组件
        hand = game_state.get('hand', [])
        players = game_state.get('players', [])
        round_info = game_state.get('round_info', {})
        current_player = game_state.get('current_player', 0)
        
        # 编码各个特征
        hand_features = self.encode_hand_features(hand)
        meld_features = self.encode_meld_features(players)
        discard_features = self.encode_discard_features(players)
        round_features = self.encode_round_features(round_info)
        
        # 当前玩家特征
        current_player_info = players[current_player] if current_player < len(players) else {}
        player_features = self.encode_player_features(current_player_info)
        
        # 组合特征（选择132维）
        # 这里我们选择最重要的特征组合
        features = np.zeros(132)
        
        # 手牌特征 (5 × 27 = 135维，但我们只取前132维中的前108维)
        features[:108] = hand_features[:108]
        
        # 吃碰杠特征 (16维)
        features[108:124] = meld_features[:16]
        
        # 回合特征 (4维)
        features[124:128] = round_features[:4]
        
        # 玩家特征 (4维)
        features[128:132] = player_features[:4]
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征
        
        Args:
            features: 原始特征
            
        Returns:
            归一化后的特征
        """
        # 使用Z-score归一化
        normalized = (features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8)
        
        # 限制在合理范围内
        normalized = np.clip(normalized, -3, 3)
        
        return normalized
    
    def denormalize_features(self, normalized_features: np.ndarray) -> np.ndarray:
        """
        反归一化特征
        
        Args:
            normalized_features: 归一化后的特征
            
        Returns:
            原始特征
        """
        return normalized_features * self.feature_stats['std'] + self.feature_stats['mean']
    
    def update_stats(self, features: np.ndarray):
        """
        更新特征统计信息
        
        Args:
            features: 特征向量
        """
        # 使用移动平均更新统计信息
        alpha = 0.001
        self.feature_stats['mean'] = (1 - alpha) * self.feature_stats['mean'] + alpha * features
        self.feature_stats['std'] = np.sqrt(
            (1 - alpha) * self.feature_stats['std']**2 + alpha * (features - self.feature_stats['mean'])**2
        )
        
        # 更新最小最大值
        self.feature_stats['min'] = np.minimum(self.feature_stats['min'], features)
        self.feature_stats['max'] = np.maximum(self.feature_stats['max'], features)
    
    def save_stats(self, path: str):
        """保存特征统计信息"""
        with open(path, 'w') as f:
            json.dump({
                'feature_stats': {
                    'mean': self.feature_stats['mean'].tolist(),
                    'std': self.feature_stats['std'].tolist(),
                    'min': self.feature_stats['min'].tolist(),
                    'max': self.feature_stats['max'].tolist()
                },
                'config': {
                    'feature_dim': self.config.feature_dim,
                    'hand_feature_dim': self.config.hand_feature_dim,
                    'meld_feature_dim': self.config.meld_feature_dim,
                    'discard_feature_dim': self.config.discard_feature_dim,
                    'round_feature_dim': self.config.round_feature_dim,
                    'player_feature_dim': self.config.player_feature_dim,
                    'action_dim': self.config.action_dim
                }
            }, f, indent=2)
    
    def load_stats(self, path: str):
        """加载特征统计信息"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.feature_stats = data['feature_stats']
        self.config.feature_dim = data['config']['feature_dim']
        
        # 转换numpy数组
        for key in self.feature_stats:
            self.feature_stats[key] = np.array(self.feature_stats[key])

class ActionEncoder:
    """动作编码器"""
    
    def __init__(self, config: FeatureConfig):
        """
        初始化动作编码器
        
        Args:
            config: 特征配置
        """
        self.config = config
        self.action_to_idx = {}
        self.idx_to_action = {}
        
        # 定义动作空间
        self._build_action_space()
    
    def _build_action_space(self):
        """构建动作空间"""
        # 打牌动作 (27种)
        for tile in range(27):
            action = f'discard_{tile}'
            idx = len(self.action_to_idx)
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action
        
        # 吃动作 (9种 × 3种吃法)
        for tile in range(9):  # 只考虑序数牌
            for eat_type in ['eat_left', 'eat_center', 'eat_right']:
                action = f'eat_{tile}_{eat_type}'
                idx = len(self.action_to_idx)
                self.action_to_idx[action] = idx
                self.idx_to_action[idx] = action
        
        # 碰动作 (9种)
        for tile in range(9):
            action = f'peng_{tile}'
            idx = len(self.action_to_idx)
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action
        
        # 杠动作 (9种)
        for tile in range(9):
            action = f'gang_{tile}'
            idx = len(self.action_to_idx)
            self.action_to_idx[action] = idx
            self.idx_to_action[idx] = action
        
        # 胡牌动作 (1种)
        action = 'win'
        idx = len(self.action_to_idx)
        self.action_to_idx[action] = idx
        self.idx_to_action[idx] = action
        
        # 流局动作 (1种)
        action = 'draw'
        idx = len(self.action_to_idx)
        self.action_to_idx[action] = idx
        self.idx_to_action[idx] = action
    
    def encode_action(self, action: str) -> int:
        """
        编码动作
        
        Args:
            action: 动作字符串
            
        Returns:
            动作索引
        """
        return self.action_to_idx.get(action, -1)
    
    def decode_action(self, action_idx: int) -> str:
        """
        解码动作
        
        Args:
            action_idx: 动作索引
            
        Returns:
            动作字符串
        """
        return self.idx_to_action.get(action_idx, 'unknown')
    
    def get_valid_actions(self, game_state: Dict) -> List[int]:
        """
        获取合法动作
        
        Args:
            game_state: 游戏状态
            
        Returns:
            合法动作索引列表
        """
        valid_actions = []
        
        # 检查所有可能的动作
        for action_str, action_idx in self.action_to_idx.items():
            if self._is_valid_action(action_str, game_state):
                valid_actions.append(action_idx)
        
        return valid_actions
    
    def _is_valid_action(self, action: str, game_state: Dict) -> bool:
        """
        检查动作是否合法
        
        Args:
            action: 动作字符串
            game_state: 游戏状态
            
        Returns:
            是否合法
        """
        # 简化的动作合法性检查
        if action.startswith('discard_'):
            tile = int(action.split('_')[1])
            hand = game_state.get('hand', [])
            return tile in hand
        
        elif action.startswith('eat_'):
            # 简化的吃牌检查
            return True
        
        elif action.startswith('peng_'):
            # 简化的碰牌检查
            return True
        
        elif action.startswith('gang_'):
            # 简化的杠牌检查
            return True
        
        elif action == 'win':
            # 简化的胡牌检查
            return True
        
        elif action == 'draw':
            # 简化的流局检查
            return True
        
        return False

if __name__ == "__main__":
    # 测试特征处理
    print("测试特征处理模块...")
    
    # 创建特征处理器
    config = FeatureConfig()
    processor = FeatureProcessor(config)
    
    # 创建测试游戏状态
    game_state = {
        'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 13张牌
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
    
    # 编码特征
    features = processor.encode_features(game_state)
    print(f"特征维度: {features.shape}")
    print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # 归一化特征
    normalized_features = processor.normalize_features(features)
    print(f"归一化特征范围: [{normalized_features.min():.3f}, {normalized_features.max():.3f}]")
    
    # 创建动作编码器
    action_encoder = ActionEncoder(config)
    print(f"动作空间大小: {len(action_encoder.action_to_idx)}")
    
    # 测试动作编码
    test_action = 'discard_0'
    action_idx = action_encoder.encode_action(test_action)
    decoded_action = action_encoder.decode_action(action_idx)
    print(f"动作编码: {test_action} -> {action_idx} -> {decoded_action}")
    
    # 获取合法动作
    valid_actions = action_encoder.get_valid_actions(game_state)
    print(f"合法动作数量: {len(valid_actions)}")
    
    print("特征处理模块测试完成！")