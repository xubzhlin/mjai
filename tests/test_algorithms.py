"""
核心算法单元测试
"""

import unittest
import numpy as np
from engine.src.algo import ShantenCalculator, WinningChecker, FanCalculator
from engine.src.tile import Tile, TileType, TileNumber

class TestShantenCalculator(unittest.TestCase):
    """向听计算器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.calculator = ShantenCalculator()
        
        # 创建测试手牌
        self.normal_hand = [
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.THREE),
            Tile(TileType.SOU, TileNumber.ONE),
            Tile(TileType.SOU, TileNumber.TWO),
            Tile(TileType.SOU, TileNumber.THREE),
            Tile(TileType.MAN, TileNumber.FOUR),
            Tile(TileType.PIN, TileNumber.FOUR),
            Tile(TileType.SOU, TileNumber.FOUR),
            Tile(TileType.MAN, TileNumber.FIVE)
        ]
        
        self.seven_pairs_hand = [
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.THREE),
            Tile(TileType.MAN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.FOUR),
            Tile(TileType.PIN, TileNumber.FOUR)
        ]
    
    def test_calculate_normal_hand(self):
        """测试普通手牌的向听计算"""
        shanten = self.calculator.calculate(self.normal_hand)
        self.assertIsInstance(shanten, int)
        self.assertGreaterEqual(shanten, 0)
    
    def test_calculate_seven_pairs(self):
        """测试七对向听计算"""
        shanten = self.calculator.calculate_seven_pairs(self.seven_pairs_hand)
        self.assertIsInstance(shanten, int)
        self.assertGreaterEqual(shanten, 0)
    
    def test_calculate_empty_hand(self):
        """测试空手牌"""
        shanten = self.calculator.calculate([])
        self.assertIsInstance(shanten, int)
        self.assertGreaterEqual(shanten, 0)
    
    def test_calculate_single_tile(self):
        """测试单张牌"""
        shanten = self.calculator.calculate([Tile(TileType.MAN, TileNumber.ONE)])
        self.assertIsInstance(shanten, int)
        self.assertGreaterEqual(shanten, 0)

class TestWinningChecker(unittest.TestCase):
    """胡牌判断器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.checker = WinningChecker()
        
        # 创建测试手牌
        self.winning_hand = [
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.THREE),
            Tile(TileType.SOU, TileNumber.ONE),
            Tile(TileType.SOU, TileNumber.TWO),
            Tile(TileType.SOU, TileNumber.THREE),
            Tile(TileType.MAN, TileNumber.FOUR),
            Tile(TileType.PIN, TileNumber.FOUR),
            Tile(TileType.SOU, TileNumber.FOUR),
            Tile(TileType.MAN, TileNumber.FIVE)
        ]
        
        self.non_winning_hand = [
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.FOUR),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.FOUR),
            Tile(TileType.SOU, TileNumber.ONE),
            Tile(TileType.SOU, TileNumber.TWO),
            Tile(TileType.SOU, TileNumber.FOUR),
            Tile(TileType.MAN, TileNumber.FIVE),
            Tile(TileType.PIN, TileNumber.FIVE),
            Tile(TileType.SOU, TileNumber.FIVE),
            Tile(TileType.MAN, TileNumber.SIX)
        ]
    
    def test_check_winning_hand(self):
        """测试胡牌手牌"""
        is_winning, win_method = self.checker.check(self.winning_hand, [])
        self.assertIsInstance(is_winning, bool)
        self.assertIsInstance(win_method, str)
    
    def test_check_non_winning_hand(self):
        """测试非胡牌手牌"""
        is_winning, win_method = self.checker.check(self.non_winning_hand, [])
        self.assertIsInstance(is_winning, bool)
    
    def test_get_win_methods(self):
        """测试获取胡牌方法"""
        win_methods = self.checker.get_win_methods()
        self.assertIsInstance(win_methods, list)
        self.assertGreater(len(win_methods), 0)

class TestFanCalculator(unittest.TestCase):
    """番种计算器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.calculator = FanCalculator()
        
        # 创建测试手牌
        self.winning_hand = [
            Tile(TileType.MAN, TileNumber.ONE),
            Tile(TileType.MAN, TileNumber.TWO),
            Tile(TileType.MAN, TileNumber.THREE),
            Tile(TileType.PIN, TileNumber.ONE),
            Tile(TileType.PIN, TileNumber.TWO),
            Tile(TileType.PIN, TileNumber.THREE),
            Tile(TileType.SOU, TileNumber.ONE),
            Tile(TileType.SOU, TileNumber.TWO),
            Tile(TileType.SOU, TileNumber.THREE),
            Tile(TileType.MAN, TileNumber.FOUR),
            Tile(TileType.PIN, TileNumber.FOUR),
            Tile(TileType.SOU, TileNumber.FOUR),
            Tile(TileType.MAN, TileNumber.FIVE)
        ]
        
        self.melds = []
    
    def test_calculate_fan_points(self):
        """测试番种计算"""
        fan_points = self.calculator.calculate(self.winning_hand, self.melds, "standard_win")
        self.assertIsInstance(fan_points, int)
        self.assertGreaterEqual(fan_points, 0)
    
    def test_get_base_fans(self):
        """测试获取基础番种"""
        base_fans = self.calculator.get_base_fans()
        self.assertIsInstance(base_fans, list)
        self.assertGreater(len(base_fans), 0)
    
    def test_get_combo_fans(self):
        """测试获取组合番种"""
        combo_fans = self.calculator.get_combo_fans()
        self.assertIsInstance(combo_fans, list)
        self.assertGreater(len(combo_fans), 0)

if __name__ == '__main__':
    unittest.main()