"""
测试配置文件
"""

import unittest
import os
import tempfile
import json
from mjai.config import ConfigManager, EngineConfig, ModelConfig, TrainingConfig

class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_engine_config(self):
        """测试加载引擎配置"""
        config = self.config_manager.load_engine_config()
        self.assertIsInstance(config, EngineConfig)
        self.assertEqual(config.players, 4)
        self.assertEqual(config.rounds, 4)
        self.assertEqual(config.starting_points, 25000)
    
    def test_load_model_config(self):
        """测试加载模型配置"""
        config = self.config_manager.load_model_config()
        self.assertIsInstance(config, ModelConfig)
        self.assertEqual(config.input_dim, 132)
        self.assertEqual(config.action_dim, 34)
        self.assertEqual(config.hidden_dims, [512, 256, 128])
    
    def test_load_training_config(self):
        """测试加载训练配置"""
        config = self.config_manager.load_training_config()
        self.assertIsInstance(config, TrainingConfig)
        self.assertEqual(config.epochs, 1000)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
    
    def test_save_config(self):
        """测试保存配置"""
        config = EngineConfig(
            players=4,
            rounds=4,
            starting_points=25000
        )
        
        config_path = os.path.join(self.temp_dir, "test_engine_config.toml")
        self.config_manager.save_config(config, config_path)
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(config_path))
        
        # 检查文件内容
        with open(config_path, 'r') as f:
            content = f.read()
            self.assertIn('players = 4', content)
            self.assertIn('rounds = 4', content)
            self.assertIn('starting_points = 25000', content)
    
    def test_get_default_config(self):
        """测试获取默认配置"""
        engine_config = self.config_manager.get_default_config('engine')
        self.assertIsInstance(engine_config, EngineConfig)
        
        model_config = self.config_manager.get_default_config('model')
        self.assertIsInstance(model_config, ModelConfig)
        
        training_config = self.config_manager.get_default_config('training')
        self.assertIsInstance(training_config, TrainingConfig)

class TestEngineConfig(unittest.TestCase):
    """引擎配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EngineConfig()
        
        self.assertEqual(config.players, 4)
        self.assertEqual(config.rounds, 4)
        self.assertEqual(config.starting_points, 25000)
        self.assertEqual(config.fan_multiplier, 1)
        self.assertEqual(config.yakuman_multiplier, 1)
        self.assertEqual(config.limit, 8000)
        self.assertEqual(config.swap_cards, 13)
        self.assertEqual(config.missing_declaration, True)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = EngineConfig(
            players=3,
            rounds=2,
            starting_points=30000,
            fan_multiplier=2,
            yakuman_multiplier=3,
            limit=12000,
            swap_cards=11,
            missing_declaration=False
        )
        
        self.assertEqual(config.players, 3)
        self.assertEqual(config.rounds, 2)
        self.assertEqual(config.starting_points, 30000)
        self.assertEqual(config.fan_multiplier, 2)
        self.assertEqual(config.yakuman_multiplier, 3)
        self.assertEqual(config.limit, 12000)
        self.assertEqual(config.swap_cards, 11)
        self.assertEqual(config.missing_declaration, False)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的玩家数量
        with self.assertRaises(ValueError):
            config = EngineConfig(players=1)
        
        # 测试无效的回合数
        with self.assertRaises(ValueError):
            config = EngineConfig(rounds=0)
        
        # 测试无效的起始分数
        with self.assertRaises(ValueError):
            config = EngineConfig(starting_points=0)

class TestModelConfig(unittest.TestCase):
    """模型配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        
        self.assertEqual(config.input_dim, 132)
        self.assertEqual(config.action_dim, 34)
        self.assertEqual(config.hidden_dims, [512, 256, 128])
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertEqual(config.learning_rate, 0.001)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ModelConfig(
            input_dim=100,
            action_dim=30,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            learning_rate=0.002
        )
        
        self.assertEqual(config.input_dim, 100)
        self.assertEqual(config.action_dim, 30)
        self.assertEqual(config.hidden_dims, [256, 128, 64])
        self.assertEqual(config.dropout_rate, 0.3)
        self.assertEqual(config.learning_rate, 0.002)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的输入维度
        with self.assertRaises(ValueError):
            config = ModelConfig(input_dim=0)
        
        # 测试无效的动作维度
        with self.assertRaises(ValueError):
            config = ModelConfig(action_dim=0)
        
        # 测试无效的隐藏层维度
        with self.assertRaises(ValueError):
            config = ModelConfig(hidden_dims=[])
        
        # 测试无效的Dropout率
        with self.assertRaises(ValueError):
            config = ModelConfig(dropout_rate=1.5)

class TestTrainingConfig(unittest.TestCase):
    """训练配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        self.assertEqual(config.epochs, 1000)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.buffer_size, 10000)
        self.assertEqual(config.update_interval, 50)
        self.assertEqual(config.save_interval, 100)
        self.assertEqual(config.eval_interval, 50)
        self.assertEqual(config.use_scheduler, True)
        self.assertEqual(config.early_stopping, True)
        self.assertEqual(config.patience, 50)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            epochs=500,
            batch_size=64,
            learning_rate=0.002,
            gamma=0.95,
            buffer_size=5000,
            update_interval=25,
            save_interval=50,
            eval_interval=25,
            use_scheduler=False,
            early_stopping=False,
            patience=30
        )
        
        self.assertEqual(config.epochs, 500)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 0.002)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.buffer_size, 5000)
        self.assertEqual(config.update_interval, 25)
        self.assertEqual(config.save_interval, 50)
        self.assertEqual(config.eval_interval, 25)
        self.assertEqual(config.use_scheduler, False)
        self.assertEqual(config.early_stopping, False)
        self.assertEqual(config.patience, 30)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的轮数
        with self.assertRaises(ValueError):
            config = TrainingConfig(epochs=0)
        
        # 测试无效的批次大小
        with self.assertRaises(ValueError):
            config = TrainingConfig(batch_size=0)
        
        # 测试无效的学习率
        with self.assertRaises(ValueError):
            config = TrainingConfig(learning_rate=0)
        
        # 测试无效的折扣因子
        with self.assertRaises(ValueError):
            config = TrainingConfig(gamma=2.0)
        
        # 测试无效的缓冲区大小
        with self.assertRaises(ValueError):
            config = TrainingConfig(buffer_size=0)

class TestConfigIntegration(unittest.TestCase):
    """配置集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_config_workflow(self):
        """测试完整的配置工作流程"""
        # 创建配置
        engine_config = EngineConfig(
            players=4,
            rounds=4,
            starting_points=25000
        )
        
        model_config = ModelConfig(
            input_dim=132,
            action_dim=34,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2
        )
        
        training_config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001
        )
        
        # 保存配置
        engine_path = os.path.join(self.temp_dir, "engine.toml")
        model_path = os.path.join(self.temp_dir, "model.toml")
        training_path = os.path.join(self.temp_dir, "training.toml")
        
        self.config_manager.save_config(engine_config, engine_path)
        self.config_manager.save_config(model_config, model_path)
        self.config_manager.save_config(training_config, training_path)
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(engine_path))
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(training_path))
        
        # 加载配置
        loaded_engine_config = self.config_manager.load_engine_config()
        loaded_model_config = self.config_manager.load_model_config()
        loaded_training_config = self.config_manager.load_training_config()
        
        # 验证配置
        self.assertEqual(loaded_engine_config.players, 4)
        self.assertEqual(loaded_engine_config.rounds, 4)
        self.assertEqual(loaded_engine_config.starting_points, 25000)
        
        self.assertEqual(loaded_model_config.input_dim, 132)
        self.assertEqual(loaded_model_config.action_dim, 34)
        self.assertEqual(loaded_model_config.hidden_dims, [512, 256, 128])
        self.assertEqual(loaded_model_config.dropout_rate, 0.2)
        
        self.assertEqual(loaded_training_config.epochs, 100)
        self.assertEqual(loaded_training_config.batch_size, 32)
        self.assertEqual(loaded_training_config.learning_rate, 0.001)
    
    def test_config_file_format(self):
        """测试配置文件格式"""
        # 创建配置
        config = EngineConfig(
            players=4,
            rounds=4,
            starting_points=25000
        )
        
        # 保存配置
        config_path = os.path.join(self.temp_dir, "test_config.toml")
        self.config_manager.save_config(config, config_path)
        
        # 检查文件格式
        with open(config_path, 'r') as f:
            content = f.read()
            
            # 检查TOML格式
            self.assertIn('[game]', content)
            self.assertIn('players = 4', content)
            self.assertIn('rounds = 4', content)
            self.assertIn('starting_points = 25000', content)
            
            # 检查没有JSON格式
            self.assertNotIn('{', content)
            self.assertNotIn('}', content)
    
    def test_config_error_handling(self):
        """测试配置错误处理"""
        # 测试加载不存在的配置文件
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent.toml")
        
        # 测试加载无效的配置文件
        invalid_config_path = os.path.join(self.temp_dir, "invalid.toml")
        with open(invalid_config_path, 'w') as f:
            f.write("invalid = config")
        
        with self.assertRaises(Exception):
            self.config_manager.load_config(invalid_config_path)

if __name__ == '__main__':
    unittest.main()