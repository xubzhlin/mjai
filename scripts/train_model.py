#!/usr/bin/env python3
"""
模型训练脚本
测试整个神经网络训练系统
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mjai.model import MahjongModel
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig
from mjai.predictor import ActionPredictor, PredictionConfig
from mjai.trainer import MahjongTrainer, TrainingConfig

class SimpleMahjongEnv:
    """简单的麻将环境，用于测试训练系统"""
    
    def __init__(self):
        self.reset_count = 0
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """重置环境"""
        self.reset_count += 1
        self.step_count = 0
        
        # 生成随机手牌
        hand = np.random.choice(27, 13, replace=False).tolist()
        
        # 生成随机弃牌
        discards = []
        for _ in range(20):
            discards.append(np.random.choice(27))
        
        return {
            'hand': hand,
            'players': [
                {
                    'melds': [],
                    'discards': discards[:5],
                    'score': 25000,
                    'riichi': False,
                    'tenhou': False,
                    'chihou': False
                },
                {
                    'melds': [],
                    'discards': discards[5:10],
                    'score': 25000,
                    'riichi': False,
                    'tenhou': False,
                    'chihou': False
                },
                {
                    'melds': [],
                    'discards': discards[10:15],
                    'score': 25000,
                    'riichi': False,
                    'tenhou': False,
                    'chihou': False
                },
                {
                    'melds': [],
                    'discards': discards[15:20],
                    'score': 25000,
                    'riichi': False,
                    'tenhou': False,
                    'chihou': False
                }
            ],
            'round_info': {
                'honba': 0,
                'round_wind': np.random.choice(4),
                'player_wind': 0,
                'remaining_tiles': 70,
                'kyoku': 1,
                'game_count': 1
            },
            'current_player': 0,
            'game_over': False
        }
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        # 模拟奖励
        reward = np.random.randn() * 0.1
        
        # 模拟游戏结束
        done = self.step_count >= self.max_steps or np.random.random() < 0.1
        
        if done:
            # 游戏结束，生成最终状态
            next_state = self.reset()
            next_state['game_over'] = True
            next_state['winner'] = np.random.choice(4)
            
            # 设置奖励
            if next_state['winner'] == 0:
                reward = 1.0
            else:
                reward = -1.0
        else:
            # 继续游戏
            next_state = self.reset()
        
        return next_state, reward, done, {}

def main():
    """主函数"""
    print("开始模型训练测试...")
    
    # 创建输出目录
    output_dir = Path("e:/ai/mjai/training_output")
    output_dir.mkdir(exist_ok=True)
    
    # 创建模型
    print("1. 创建神经网络模型...")
    model = MahjongModel()
    
    # 创建特征处理器和动作编码器
    print("2. 创建特征处理器和动作编码器...")
    feature_config = FeatureConfig()
    feature_processor = FeatureProcessor(feature_config)
    action_encoder = ActionEncoder(feature_config)
    
    # 创建训练配置
    print("3. 创建训练配置...")
    training_config = TrainingConfig(
        epochs=5,  # 减少训练轮数用于测试
        batch_size=16,
        learning_rate=0.001,
        buffer_size=1000,
        update_interval=50,
        save_interval=100,
        eval_interval=50,
        log_dir=str(output_dir / "logs"),
        tensorboard_dir=str(output_dir / "tensorboard"),
        save_dir=str(output_dir / "models"),
        use_scheduler=False,  # 测试时禁用调度器
        early_stopping=False  # 测试时禁用早停
    )
    
    # 创建训练器
    print("4. 创建训练器...")
    trainer = MahjongTrainer(model, feature_processor, action_encoder, training_config)
    
    # 创建环境
    print("5. 创建训练环境...")
    env = SimpleMahjongEnv()
    
    # 测试预测器
    print("6. 测试预测器...")
    test_state = env.reset()
    action_idx, action_prob, info = trainer.predictor.predict_action(test_state)
    print(f"   预测动作: {action_idx}, 概率: {action_prob:.3f}")
    print(f"   预测信息: {info}")
    
    # 测试经验收集
    print("7. 测试经验收集...")
    trainer.collect_experience(env, 10)
    print(f"   收集经验数量: {len(trainer.buffer)}")
    
    # 测试训练
    print("8. 测试模型训练...")
    if len(trainer.buffer) >= training_config.batch_size:
        batch = trainer.buffer.sample(training_config.batch_size)
        loss_info = trainer.train_step(batch)
        print(f"   训练损失: {loss_info['loss']:.4f}")
        print(f"   策略损失: {loss_info['policy_loss']:.4f}")
        print(f"   价值损失: {loss_info['value_loss']:.4f}")
        print(f"   熵: {loss_info['entropy']:.4f}")
    
    # 测试评估
    print("9. 测试模型评估...")
    eval_score = trainer.evaluate(env, num_episodes=5)
    print(f"   评估分数: {eval_score:.3f}")
    
    # 保存模型
    print("10. 保存模型...")
    trainer.save_model("test_model.pth")
    
    # 保存统计信息
    print("11. 保存统计信息...")
    stats = trainer.stats
    with open(output_dir / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n训练测试完成！")
    print(f"输出目录: {output_dir}")
    print(f"模型文件: {output_dir / 'models' / 'test_model.pth'}")
    print(f"统计信息: {output_dir / 'training_stats.json'}")

if __name__ == "__main__":
    main()