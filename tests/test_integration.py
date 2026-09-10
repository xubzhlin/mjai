"""
集成测试
测试整个系统的协同工作
"""

import unittest
import numpy as np
import tempfile
import os
from mjai.advanced_model import AdvancedMahjongModel
from mjai.advanced_predictor import AdvancedActionPredictor, AdvancedPredictionConfig
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig
from mjai.trainer import MahjongTrainer, TrainingConfig

class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建配置
        self.feature_config = FeatureConfig()
        self.prediction_config = AdvancedPredictionConfig()
        self.training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            gamma=0.99,
            buffer_size=1000,
            update_interval=10,
            save_interval=20,
            eval_interval=10,
            use_scheduler=False,
            early_stopping=False
        )
        
        # 创建组件
        self.model = AdvancedMahjongModel()
        self.feature_processor = FeatureProcessor(self.feature_config)
        self.action_encoder = ActionEncoder(self.feature_config)
        self.predictor = AdvancedActionPredictor(
            self.model,
            self.feature_processor,
            self.action_encoder,
            self.prediction_config
        )
        self.trainer = MahjongTrainer(self.model, self.training_config)
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 1. 创建游戏状态
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
        
        # 2. 预测动作
        action_idx, prob, info = self.predictor.predict_action(game_state)
        
        # 检查预测结果
        self.assertIsInstance(action_idx, int)
        self.assertGreaterEqual(action_idx, 0)
        self.assertLess(action_idx, 34)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIsInstance(info, dict)
        
        # 3. 生成训练数据
        training_data = []
        for _ in range(10):
            # 模拟游戏状态
            state = game_state.copy()
            
            # 预测动作
            action_idx, prob, info = self.predictor.predict_action(state)
            
            # 计算奖励
            reward = np.random.randn()
            
            # 获取特征
            features = self.feature_processor.encode_features(state)
            normalized_features = self.feature_processor.normalize_features(features)
            
            # 获取Q值
            q_values = self.model.get_q_values(normalized_features)
            
            # 添加到训练数据
            training_data.append({
                'state': normalized_features,
                'action': action_idx,
                'reward': reward,
                'q_values': q_values,
                'prob': prob,
                'info': info
            })
        
        # 4. 训练模型
        for sample in training_data:
            batch = self._create_batch([sample])
            loss = self.trainer.train_step(batch)
            
            # 检查训练损失
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0.0)
        
        # 5. 评估模型
        eval_score = self.trainer.evaluate(num_games=5)
        self.assertIsInstance(eval_score, float)
        
        # 6. 保存模型
        model_path = os.path.join(self.temp_dir, "test_model.pth")
        self.trainer.save_model(model_path)
        
        # 检查模型文件是否存在
        self.assertTrue(os.path.exists(model_path))
        
        # 7. 加载模型
        new_trainer = MahjongTrainer(self.model, self.training_config)
        new_trainer.load_model(model_path)
        
        # 8. 测试加载后的模型
        test_features = np.random.randn(132)
        action, q_value, hidden = self.model.predict(test_features)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(q_value, float)
    
    def test_batch_prediction(self):
        """测试批量预测"""
        # 创建多个游戏状态
        game_states = []
        for i in range(5):
            state = {
                'hand': [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10, i+11, i+12],
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
                'current_player': i % 4
            }
            game_states.append(state)
        
        # 批量预测
        results = []
        for state in game_states:
            action_idx, prob, info = self.predictor.predict_action(state)
            results.append((action_idx, prob, info))
        
        # 检查结果
        self.assertEqual(len(results), 5)
        for action_idx, prob, info in results:
            self.assertIsInstance(action_idx, int)
            self.assertGreaterEqual(action_idx, 0)
            self.assertLess(action_idx, 34)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_training_pipeline(self):
        """测试训练管道"""
        # 创建训练数据
        training_data = []
        for i in range(20):
            state = {
                'hand': [i % 13, (i+1) % 13, (i+2) % 13, (i+3) % 13, (i+4) % 13, 
                        (i+5) % 13, (i+6) % 13, (i+7) % 13, (i+8) % 13, 
                        (i+9) % 13, (i+10) % 13, (i+11) % 13, (i+12) % 13],
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
                'current_player': i % 4
            }
            
            # 预测动作
            action_idx, prob, info = self.predictor.predict_action(state)
            
            # 计算奖励
            reward = np.random.randn()
            
            # 获取特征
            features = self.feature_processor.encode_features(state)
            normalized_features = self.feature_processor.normalize_features(features)
            
            # 获取Q值
            q_values = self.model.get_q_values(normalized_features)
            
            # 添加到训练数据
            training_data.append({
                'state': normalized_features,
                'action': action_idx,
                'reward': reward,
                'q_values': q_values,
                'prob': prob,
                'info': info
            })
        
        # 训练模型
        epoch_losses = []
        for sample in training_data:
            batch = self._create_batch([sample])
            loss = self.trainer.train_step(batch)
            epoch_losses.append(loss)
        
        # 检查训练结果
        self.assertIsInstance(epoch_losses, list)
        self.assertEqual(len(epoch_losses), 20)
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        self.assertIsInstance(avg_loss, float)
        self.assertGreaterEqual(avg_loss, 0.0)
        
        # 评估模型
        eval_score = self.trainer.evaluate(num_games=5)
        self.assertIsInstance(eval_score, float)
        
        # 保存模型
        model_path = os.path.join(self.temp_dir, "pipeline_model.pth")
        self.trainer.save_model(model_path)
        
        # 检查模型文件是否存在
        self.assertTrue(os.path.exists(model_path))
    
    def test_model_save_load_cycle(self):
        """测试模型保存和加载循环"""
        # 训练模型
        batch_size = 16
        states = np.random.randn(batch_size, 132)
        actions = np.random.randint(0, 34, batch_size)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, 132)
        dones = np.random.randint(0, 2, batch_size).astype(bool)
        old_q_values = np.random.randn(batch_size)
        
        batch = (states, actions, rewards, next_states, dones, old_q_values)
        self.model.update(batch)
        
        # 保存模型
        save_path = os.path.join(self.temp_dir, "save_load_test.pth")
        self.model.save(save_path)
        
        # 创建新模型并加载
        new_model = AdvancedMahjongModel.load(save_path)
        
        # 测试加载后的模型
        test_features = np.random.randn(132)
        action, q_value, hidden = new_model.predict(test_features)
        
        # 检查输出
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 34)
        self.assertIsInstance(q_value, float)
        self.assertEqual(len(hidden), 2)
        
        # 测试更新功能
        new_batch = self.model.sample_batch(batch_size=8)
        if new_batch is not None:
            stats = new_model.update(new_batch)
            self.assertIsInstance(stats, dict)
            self.assertIn('loss', stats)
    
    def _create_batch(self, samples):
        """创建训练批次"""
        states = np.array([s['state'] for s in samples])
        actions = np.array([s['action'] for s in samples])
        rewards = np.array([s['reward'] for s in samples])
        q_values = np.array([s['q_values'] for s in samples])
        
        return (states, actions, rewards, q_values)

class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.model = AdvancedMahjongModel()
        self.feature_processor = FeatureProcessor(FeatureConfig())
        self.action_encoder = ActionEncoder(FeatureConfig())
        self.predictor = AdvancedActionPredictor(
            self.model,
            self.feature_processor,
            self.action_encoder
        )
    
    def test_prediction_speed(self):
        """测试预测速度"""
        import time
        
        # 创建测试状态
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
        
        # 测试预测速度
        num_predictions = 100
        start_time = time.time()
        
        for _ in range(num_predictions):
            action_idx, prob, info = self.predictor.predict_action(game_state)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_predictions
        
        print(f"平均预测时间: {avg_time:.4f}秒")
        self.assertLess(avg_time, 0.1)  # 预测时间应小于0.1秒
    
    def test_batch_prediction_speed(self):
        """测试批量预测速度"""
        import time
        
        # 创建测试状态
        game_states = []
        for i in range(10):
            state = {
                'hand': [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10, i+11, i+12],
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
                'current_player': i % 4
            }
            game_states.append(state)
        
        # 测试批量预测速度
        num_batches = 100
        start_time = time.time()
        
        for _ in range(num_batches):
            results = []
            for state in game_states:
                action_idx, prob, info = self.predictor.predict_action(state)
                results.append((action_idx, prob, info))
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_batches
        
        print(f"平均批量预测时间: {avg_time:.4f}秒")
        self.assertLess(avg_time, 1.0)  # 批量预测时间应小于1秒

if __name__ == '__main__':
    unittest.main()