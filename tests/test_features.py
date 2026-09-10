"""
特征处理单元测试
"""

import unittest
import numpy as np
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig

class TestFeatureConfig(unittest.TestCase):
    """特征配置测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = FeatureConfig()
    
    def test_config_attributes(self):
        """测试配置属性"""
        self.assertEqual(self.config.input_dim, 132)
        self.assertEqual(self.config.action_dim, 34)
        self.assertEqual(self.config.hand_dim, 135)
        self.assertEqual(self.config.meld_dim, 16)
        self.assertEqual(self.config.round_dim, 4)
        self.assertEqual(self.config.player_dim, 4)

class TestFeatureProcessor(unittest.TestCase):
    """特征处理器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = FeatureConfig()
        self.processor = FeatureProcessor(self.config)
    
    def test_encode_features(self):
        """测试特征编码"""
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
        
        features = self.processor.encode_features(game_state)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 132)
    
    def test_normalize_features(self):
        """测试特征归一化"""
        features = np.random.randn(132)
        normalized_features = self.processor.normalize_features(features)
        self.assertIsInstance(normalized_features, np.ndarray)
        self.assertEqual(len(normalized_features), 132)
        
        # 检查归一化后的范围
        self.assertTrue(np.all(normalized_features >= -3))
        self.assertTrue(np.all(normalized_features <= 3))
    
    def test_encode_hand_features(self):
        """测试手牌特征编码"""
        hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        hand_features = self.processor.encode_hand_features(hand)
        self.assertIsInstance(hand_features, np.ndarray)
        self.assertEqual(len(hand_features), 135)
    
    def test_encode_meld_features(self):
        """测试吃碰杠特征编码"""
        players = [
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []}
        ]
        meld_features = self.processor.encode_meld_features(players)
        self.assertIsInstance(meld_features, np.ndarray)
        self.assertEqual(len(meld_features), 16)
    
    def test_encode_discard_features(self):
        """测试弃牌特征编码"""
        players = [
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []},
            {'melds': [], 'discards': []}
        ]
        discard_features = self.processor.encode_discard_features(players)
        self.assertIsInstance(discard_features, np.ndarray)
        self.assertEqual(len(discard_features), 16)
    
    def test_encode_round_features(self):
        """测试回合特征编码"""
        round_info = {
            'honba': 0,
            'round_wind': 0,
            'player_wind': 0,
            'remaining_tiles': 70
        }
        round_features = self.processor.encode_round_features(round_info)
        self.assertIsInstance(round_features, np.ndarray)
        self.assertEqual(len(round_features), 4)
    
    def test_encode_player_features(self):
        """测试玩家特征编码"""
        player_info = {
            'points': 25000,
            'rank': 0,
            'is_dealer': True
        }
        player_features = self.processor.encode_player_features(player_info)
        self.assertIsInstance(player_features, np.ndarray)
        self.assertEqual(len(player_features), 4)

class TestActionEncoder(unittest.TestCase):
    """动作编码器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.config = FeatureConfig()
        self.encoder = ActionEncoder(self.config)
    
    def test_get_valid_actions(self):
        """测试获取合法动作"""
        game_state = {
            'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'current_player': 0,
            'players': [
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []}
            ]
        }
        
        valid_actions = self.encoder.get_valid_actions(game_state)
        self.assertIsInstance(valid_actions, list)
        self.assertGreater(len(valid_actions), 0)
    
    def test_encode_decode_action(self):
        """测试动作编码和解码"""
        # 测试弃牌动作
        action = "discard_0"
        action_idx = self.encoder.encode_action(action)
        decoded_action = self.encoder.decode_action(action_idx)
        self.assertEqual(action, decoded_action)
        
        # 测试胡牌动作
        action = "win"
        action_idx = self.encoder.encode_action(action)
        decoded_action = self.encoder.decode_action(action_idx)
        self.assertEqual(action, decoded_action)
    
    def test_get_action_space(self):
        """测试获取动作空间"""
        action_space = self.encoder.get_action_space()
        self.assertIsInstance(action_space, list)
        self.assertEqual(len(action_space), 34)
    
    def test_is_action_valid(self):
        """测试检查动作合法性"""
        game_state = {
            'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'current_player': 0,
            'players': [
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []},
                {'melds': [], 'discards': []}
            ]
        }
        
        # 测试合法动作
        valid_action = "discard_0"
        self.assertTrue(self.encoder.is_action_valid(valid_action, game_state))
        
        # 测试非法动作
        invalid_action = "discard_99"  # 99不在手牌中
        self.assertFalse(self.encoder.is_action_valid(invalid_action, game_state))

if __name__ == '__main__':
    unittest.main()