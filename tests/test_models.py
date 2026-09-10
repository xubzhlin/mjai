"""
神经网络模型单元测试
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from mjai.model import MahjongModel, MahjongNet, ValueNet
from mjai.advanced_model import AdvancedMahjongModel, AdvancedMahjongNet

class TestMahjongNet(unittest.TestCase):
    """基础神经网络模型测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.net = MahjongNet(input_dim=132, action_dim=34)
        self.model = MahjongModel()
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 32
        input_dim = 132
        
        # 创建输入张量
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        output = self.net(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, 34))
        
        # 检查输出是否为概率分布
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6))
    
    def test_predict(self):
        """测试预测"""
        features = np.random.randn(132)
        probabilities = self.net.predict(features)
        
        # 检查输出形状
        self.assertEqual(len(probabilities), 34)
        
        # 检查是否为概率分布
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 创建测试输入
        test_input = torch.randn(1, 132)
        
        # 获取原始输出
        original_output = self.net(test_input)
        
        # 保存模型
        self.net.save("test_model.pth")
        
        # 创建新模型并加载
        new_net = MahjongNet.load("test_model.pth")
        
        # 测试加载后的模型
        new_output = new_net(test_input)
        
        # 检查输出是否相同
        self.assertTrue(torch.allclose(original_output, new_output, atol=1e-6))
        
        # 清理测试文件
        import os
        os.remove("test_model.pth")

class TestValueNet(unittest.TestCase):
    """价值网络测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.net = ValueNet(input_dim=132)
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 32
        input_dim = 132
        
        # 创建输入张量
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        output = self.net(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_evaluate(self):
        """测试评估"""
        features = np.random.randn(132)
        value = self.net.evaluate(features)
        
        # 检查输出类型
        self.assertIsInstance(value, float)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 创建测试输入
        test_input = torch.randn(1, 132)
        
        # 获取原始输出
        original_output = self.net(test_input)
        
        # 保存模型
        self.net.save("test_value_model.pth")
        
        # 创建新模型并加载
        new_net = ValueNet.load("test_value_model.pth")
        
        # 测试加载后的模型
        new_output = new_net(test_input)
        
        # 检查输出是否相同
        self.assertTrue(torch.allclose(original_output, new_output, atol=1e-6))
        
        # 清理测试文件
        import os
        os.remove("test_value_model.pth")

class TestAdvancedMahjongNet(unittest.TestCase):
    """高级神经网络模型测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.net = AdvancedMahjongNet(input_dim=132, action_dim=34)
        self.model = AdvancedMahjongModel()
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 32
        input_dim = 132
        
        # 创建输入张量
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        action_probs, hidden = self.net(x)
        
        # 检查输出形状
        self.assertEqual(action_probs.shape, (batch_size, 34))
        self.assertEqual(len(hidden), 2)  # (h_n, c_n)
        
        # 检查输出是否为概率分布
        self.assertTrue(torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6))
    
    def test_predict(self):
        """测试预测"""
        features = np.random.randn(132)
        action_probs, q_value, hidden = self.model.predict(features)
        
        # 检查输出形状
        self.assertEqual(len(action_probs), 34)
        self.assertIsInstance(q_value, float)
        self.assertEqual(len(hidden), 2)
        
        # 检查是否为概率分布
        self.assertAlmostEqual(np.sum(action_probs), 1.0, places=6)
    
    def test_get_q_values(self):
        """测试获取Q值"""
        features = np.random.randn(132)
        q_values = self.model.get_q_values(features)
        
        # 检查输出形状
        self.assertEqual(len(q_values), 34)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 创建测试输入
        test_input = torch.randn(1, 132)
        
        # 获取原始输出
        original_output, original_hidden = self.net(test_input)
        
        # 保存模型
        self.net.save("test_advanced_model.pth")
        
        # 创建新模型并加载
        new_net = AdvancedMahjongNet.load("test_advanced_model.pth")
        
        # 测试加载后的模型
        new_output, new_hidden = new_net(test_input)
        
        # 检查输出是否相同
        self.assertTrue(torch.allclose(original_output, new_output, atol=1e-6))
        
        # 清理测试文件
        import os
        os.remove("test_advanced_model.pth")

class TestAdvancedMahjongModel(unittest.TestCase):
    """高级神经网络模型测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.model = AdvancedMahjongModel()
    
    def test_predict(self):
        """测试预测"""
        features = np.random.randn(132)
        action, q_value, hidden = self.model.predict(features)
        
        # 检查输出类型
        self.assertIsInstance(action, int)
        self.assertIsInstance(q_value, float)
        self.assertEqual(len(hidden), 2)
        
        # 检查动作范围
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 34)
    
    def test_update(self):
        """测试模型更新"""
        # 创建测试批次
        batch_size = 32
        states = np.random.randn(batch_size, 132)
        actions = np.random.randint(0, 34, batch_size)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, 132)
        dones = np.random.randint(0, 2, batch_size).astype(bool)
        old_q_values = np.random.randn(batch_size)
        
        # 构建批次
        batch = (states, actions, rewards, next_states, dones, old_q_values)
        
        # 更新模型
        stats = self.model.update(batch)
        
        # 检查返回的统计信息
        self.assertIsInstance(stats, dict)
        self.assertIn('loss', stats)
        self.assertIn('epsilon', stats)
        self.assertIn('learning_rate', stats)
    
    def test_add_experience_and_sample(self):
        """测试经验添加和采样"""
        # 添加经验
        state = np.random.randn(132)
        action = 0
        reward = 1.0
        next_state = np.random.randn(132)
        done = False
        old_q_value = 0.5
        
        self.model.add_experience(state, action, reward, next_state, done, old_q_value)
        
        # 采样批次
        batch = self.model.sample_batch(batch_size=16)
        
        # 检查批次
        self.assertIsNotNone(batch)
        states, actions, rewards, next_states, dones, old_q_values = batch
        
        self.assertEqual(len(states), 16)
        self.assertEqual(len(actions), 16)
        self.assertEqual(len(rewards), 16)
        self.assertEqual(len(next_states), 16)
        self.assertEqual(len(dones), 16)
        self.assertEqual(len(old_q_values), 16)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 训练模型
        batch_size = 32
        states = np.random.randn(batch_size, 132)
        actions = np.random.randint(0, 34, batch_size)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, 132)
        dones = np.random.randint(0, 2, batch_size).astype(bool)
        old_q_values = np.random.randn(batch_size)
        
        batch = (states, actions, rewards, next_states, dones, old_q_values)
        self.model.update(batch)
        
        # 保存模型
        self.model.save("test_advanced_model_complete.pth")
        
        # 创建新模型并加载
        new_model = AdvancedMahjongModel.load("test_advanced_model_complete.pth")
        
        # 测试加载后的模型
        test_features = np.random.randn(132)
        action, q_value, hidden = new_model.predict(test_features)
        
        # 检查输出类型
        self.assertIsInstance(action, int)
        self.assertIsInstance(q_value, float)
        
        # 清理测试文件
        import os
        os.remove("test_advanced_model_complete.pth")

if __name__ == '__main__':
    unittest.main()