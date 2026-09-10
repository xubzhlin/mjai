"""
牌具模块单元测试
"""

import unittest
import numpy as np
from engine.src.tile import Tile, Hand
from engine.src.tile.tile import TileType, TileNumber

class TestTile(unittest.TestCase):
    """牌具测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.tile = Tile(TileType.MAN, TileNumber.ONE)
        self.hand = Hand()
    
    def test_tile_creation(self):
        """测试牌的创建"""
        tile = Tile(TileType.MAN, TileNumber.ONE)
        self.assertEqual(tile.tile_type, TileType.MAN)
        self.assertEqual(tile.tile_number, TileNumber.ONE)
        self.assertEqual(tile.value, 0)  # MAN_1 = 0
    
    def test_tile_comparison(self):
        """测试牌的比较"""
        tile1 = Tile(TileType.MAN, TileNumber.ONE)
        tile2 = Tile(TileType.MAN, TileNumber.ONE)
        tile3 = Tile(TileType.MAN, TileNumber.TWO)
        
        self.assertEqual(tile1, tile2)
        self.assertNotEqual(tile1, tile3)
    
    def test_hand_creation(self):
        """测试手牌的创建"""
        hand = Hand()
        self.assertEqual(len(hand.tiles), 0)
        self.assertEqual(hand.counts, [0] * 27)
    
    def test_hand_add_tile(self):
        """测试添加牌到手牌"""
        self.hand.add_tile(self.tile)
        self.assertEqual(len(self.hand.tiles), 1)
        self.assertEqual(self.hand.counts[0], 1)
    
    def test_hand_remove_tile(self):
        """测试从手牌移除牌"""
        self.hand.add_tile(self.tile)
        removed_tile = self.hand.remove_tile(self.tile)
        self.assertEqual(removed_tile, self.tile)
        self.assertEqual(len(self.hand.tiles), 0)
        self.assertEqual(self.hand.counts[0], 0)
    
    def test_hand_count_tiles(self):
        """测试统计牌的数量"""
        self.hand.add_tile(self.tile)
        self.hand.add_tile(self.tile)
        count = self.hand.count_tiles(self.tile)
        self.assertEqual(count, 2)
    
    def test_hand_is_complete(self):
        """测试检查是否完成"""
        # 添加13张相同的牌
        for _ in range(13):
            self.hand.add_tile(self.tile)
        
        # 这里应该根据实际的完成逻辑来测试
        # 目前只是一个简单的框架测试
        self.assertEqual(len(self.hand.tiles), 13)
    
    def test_hand_clear(self):
        """测试清空手牌"""
        self.hand.add_tile(self.tile)
        self.hand.clear()
        self.assertEqual(len(self.hand.tiles), 0)
        self.assertEqual(self.hand.counts, [0] * 27)
    
    def test_hand_get_suits(self):
        """测试获取花色"""
        # 添加不同花色的牌
        man_tile = Tile(TileType.MAN, TileNumber.ONE)
        pin_tile = Tile(TileType.PIN, TileNumber.ONE)
        sou_tile = Tile(TileType.SOU, TileNumber.ONE)
        
        self.hand.add_tile(man_tile)
        self.hand.add_tile(pin_tile)
        self.hand.add_tile(sou_tile)
        
        suits = self.hand.get_suits()
        self.assertIn(TileType.MAN, suits)
        self.assertIn(TileType.PIN, suits)
        self.assertIn(TileType.SOU, suits)

class TestTileNumber(unittest.TestCase):
    """牌号测试类"""
    
    def test_tile_number_values(self):
        """测试牌号的值"""
        self.assertEqual(TileNumber.ONE.value, 1)
        self.assertEqual(TileNumber.NINE.value, 9)
        self.assertEqual(TileNumber.TWO.value, 2)

if __name__ == '__main__':
    unittest.main()