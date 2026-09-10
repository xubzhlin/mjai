"""
评估系统
实现模型评估、对战、基准测试等功能
"""

import numpy as np
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import random
from collections import defaultdict

from mjai.ppo import PPO
from mjai.selfplay import SelfPlayEngine, SimulatedGame, GameRecord, ModelPredictorWrapper
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig


@dataclass
class EvalConfig:
    """评估配置"""
    num_games: int = 100
    num_players: int = 4
    
    # 评估类型
    eval_type: str = 'selfplay'  # selfplay, benchmark, compare
    
    # 对手配置
    opponent_type: str = 'random'  # random, greedy, pretrained
    opponent_model_path: Optional[str] = None
    
    # 评估指标
    metrics: List[str] = None  # win_rate, avg_reward, fan_count, score
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['win_rate', 'avg_reward', 'score']


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: EvalConfig, model_path: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
            model_path: 模型路径
        """
        self.config = config
        
        # 特征处理器
        feature_config = FeatureConfig()
        self.feature_processor = FeatureProcessor(feature_config)
        self.action_encoder = ActionEncoder(feature_config)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.ppo = PPO.load(model_path)
        else:
            from mjai.ppo import PPOConfig
            self.ppo = PPO(PPOConfig())
        
        # 自博弈引擎
        self.selfplay_engine = SelfPlayEngine(
            feature_processor=self.feature_processor,
            action_encoder=self.action_encoder
        )
        
        # 评估结果
        self.results = {}
        self.game_records = []
        
        # 统计
        self.stats = {
            'eval_count': 0,
            'start_time': time.time()
        }
    
    def evaluate_selfplay(self, num_games: Optional[int] = None) -> Dict:
        """
        自博弈评估
        
        Args:
            num_games: 评估游戏数
            
        Returns:
            评估结果
        """
        n_games = num_games or self.config.num_games
        
        print(f"开始自博弈评估，共 {n_games} 局...")
        
        results = {
            'num_games': n_games,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'avg_reward': 0.0,
            'avg_steps': 0,
            'scores': [0.0, 0.0, 0.0, 0.0],
            'game_lengths': [],
            'timestamps': []
        }
        
        for i in range(n_games):
            # 生成游戏
            game_record = self.selfplay_engine.generate_game()
            
            # 记录结果
            results['game_lengths'].append(game_record.steps)
            results['scores'] = [a + b for a, b in zip(results['scores'], game_record.scores)]
            
            # 判断胜负
            if game_record.winner == 0:
                results['wins'] += 1
            elif game_record.winner is not None:
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            # 计算平均奖励
            if game_record.experiences:
                avg_reward = np.mean([exp.reward for exp in game_record.experiences])
                results['avg_reward'] += avg_reward
            
            results['timestamps'].append(time.time())
            
            # 保存游戏记录
            self.game_records.append(game_record.to_dict())
        
        # 计算汇总指标
        results['win_rate'] = results['wins'] / n_games
        results['draw_rate'] = results['draws'] / n_games
        results['loss_rate'] = results['losses'] / n_games
        results['avg_reward'] /= n_games
        results['avg_steps'] = np.mean(results['game_lengths'])
        results['scores'] = [s / n_games for s in results['scores']]
        
        # 计算标准差
        if len(results['game_lengths']) > 0:
            results['std_steps'] = np.std(results['game_lengths'])
        
        self.results = results
        self.stats['eval_count'] += 1
        
        print(f"评估完成！")
        print(f"  胜率: {results['win_rate']:.2%}")
        print(f"  平均奖励: {results['avg_reward']:.4f}")
        print(f"  平均步数: {results['avg_steps']:.1f}")
        
        return results
    
    def evaluate_vs_opponent(self, 
                            opponent_model_path: Optional[str] = None,
                            num_games: Optional[int] = None) -> Dict:
        """
        与对手对战评估
        
        Args:
            opponent_model_path: 对手模型路径
            num_games: 评估游戏数
            
        Returns:
            评估结果
        """
        n_games = num_games or self.config.num_games
        
        print(f"开始对战评估，共 {n_games} 局...")
        
        results = {
            'num_games': n_games,
            'agent_wins': 0,
            'opponent_wins': 0,
            'draws': 0,
            'win_rate': 0.0,
            'opponent_type': self.config.opponent_type
        }
        
        # 设置对手
        for i in range(n_games):
            # 根据对手类型选择动作
            winner = self._play_game_vs_opponent(opponent_model_path)
            
            if winner == 0:  # 智能体获胜
                results['agent_wins'] += 1
            elif winner is not None:  # 对手获胜
                results['opponent_wins'] += 1
            else:  # 流局
                results['draws'] += 1
        
        # 计算胜率
        results['win_rate'] = results['agent_wins'] / n_games
        results['opponent_win_rate'] = results['opponent_wins'] / n_games
        
        print(f"对战评估完成！")
        print(f"  智能体胜率: {results['win_rate']:.2%}")
        print(f"  对手胜率: {results['opponent_win_rate']:.2%}")
        
        return results
    
    def _play_game_vs_opponent(self, opponent_model_path: Optional[str]) -> Optional[int]:
        """
        与对手玩一局游戏
        
        Args:
            opponent_model_path: 对手模型路径
            
        Returns:
            赢家ID (0表示智能体, None表示流局)
        """
        game = SimulatedGame()
        
        # 加载对手模型（如果有）
        opponent_ppo = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            from mjai.ppo import PPO
            opponent_ppo = PPO.load(opponent_model_path)
        
        # 游戏循环
        max_steps = 200
        step = 0
        
        while not game.is_done() and step < max_steps:
            player_id = game.current_player
            state = game.get_state(player_id)
            
            # 根据玩家选择动作策略
            if player_id == 0:
                # 智能体使用PPO网络
                action = self._select_action_ppo(state)
            else:
                # 对手根据类型选择
                action = self._select_opponent_action(
                    state, player_id, opponent_ppo
                )
            
            # 执行动作
            next_state, reward, done, info = game.take_action(player_id, action)
            
            step += 1
        
        return game.winner
    
    def _select_action_ppo(self, state: Dict) -> int:
        """使用PPO网络选择动作"""
        self.ppo.policy_net.eval()
        
        with torch.no_grad():
            features = self.feature_processor.encode_features(state)
            normalized = self.feature_processor.normalize_features(features)
            
            x = torch.FloatTensor(normalized).unsqueeze(0).to(self.ppo.device)
            action_probs, value = self.ppo.policy_net(x)
            
            # 选择概率最高的动作
            action = action_probs.argmax(dim=-1).item()
            
            return action
    
    def _select_opponent_action(self, 
                               state: Dict, 
                               player_id: int,
                               opponent_ppo: Optional[PPO]) -> int:
        """选择对手动作"""
        opponent_type = self.config.opponent_type
        
        if opponent_type == 'random':
            # 随机动作
            valid_actions = self.action_encoder.get_valid_actions(state)
            if valid_actions:
                return random.choice(valid_actions)
            return 0
        
        elif opponent_type == 'greedy':
            # 贪心策略（简化）
            return random.randint(0, 33)
        
        elif opponent_type == 'pretrained' and opponent_ppo is not None:
            # 使用预训练模型
            opponent_ppo.policy_net.eval()
            
            with torch.no_grad():
                features = self.feature_processor.encode_features(state)
                normalized = self.feature_processor.normalize_features(features)
                
                x = torch.FloatTensor(normalized).unsqueeze(0).to(opponent_ppo.device)
                action_probs, _ = opponent_ppo.policy_net(x)
                
                return action_probs.argmax(dim=-1).item()
        
        return random.randint(0, 33)
    
    def benchmark(self, model_paths: List[str], num_games: int = 50) -> Dict:
        """
        基准测试 - 多个模型互相对战
        
        Args:
            model_paths: 模型路径列表
            num_games: 每对模型的对战局数
            
        Returns:
            基准测试结果
        """
        print(f"开始基准测试，共 {len(model_paths)} 个模型...")
        
        # 加载所有模型
        models = []
        for path in model_paths:
            try:
                ppo = PPO.load(path)
                models.append({'path': path, 'ppo': ppo, 'wins': 0, 'total': 0})
                print(f"  已加载: {path}")
            except Exception as e:
                print(f"  加载失败: {path} - {e}")
        
        # 两两对战
        results = {
            'num_models': len(models),
            'num_games_per_pair': num_games,
            'pairwise_results': [],
            'overall_results': []
        }
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                # 对战
                wins_i, wins_j, draws = self._play_match(
                    models[i]['ppo'], models[j]['ppo'], num_games
                )
                
                models[i]['wins'] += wins_i
                models[i]['total'] += num_games
                models[j]['wins'] += wins_j
                models[j]['total'] += num_games
                
                results['pairwise_results'].append({
                    'model_a': models[i]['path'],
                    'model_b': models[j]['path'],
                    'wins_a': wins_i,
                    'wins_b': wins_j,
                    'draws': draws,
                    'win_rate_a': wins_i / num_games,
                    'win_rate_b': wins_j / num_games
                })
        
        # 计算总体结果
        for model_info in models:
            results['overall_results'].append({
                'model': model_info['path'],
                'total_games': model_info['total'],
                'total_wins': model_info['wins'],
                'win_rate': model_info['wins'] / max(model_info['total'], 1)
            })
        
        # 按胜率排序
        results['overall_results'].sort(key=lambda x: x['win_rate'], reverse=True)
        
        print("基准测试完成！")
        print("\n排名:")
        for rank, result in enumerate(results['overall_results'], 1):
            print(f"  {rank}. {result['model']}: 胜率 {result['win_rate']:.2%}")
        
        return results
    
    def _play_match(self, 
                   ppo_a: PPO, 
                   ppo_b: PPO, 
                   num_games: int) -> Tuple[int, int, int]:
        """
        两个模型对战
        
        Args:
            ppo_a: 模型A
            ppo_b: 模型B
            num_games: 对战局数
            
        Returns:
            (模型A获胜数, 模型B获胜数, 流局数)
        """
        wins_a = 0
        wins_b = 0
        draws = 0
        
        for game_id in range(num_games):
            game = SimulatedGame()
            max_steps = 200
            
            # 决定哪个模型控制哪个玩家
            swap = (game_id % 2 == 1)
            
            while not game.is_done() and game.steps < max_steps:
                player_id = game.current_player
                state = game.get_state(player_id)
                
                # 选择模型
                if player_id in [0, 2]:
                    ppo = ppo_a if not swap else ppo_b
                else:
                    ppo = ppo_b if not swap else ppo_a
                
                # 选择动作
                action = self._select_action_from_ppo(state, ppo)
                
                # 执行动作
                next_state, reward, done, info = game.take_action(player_id, action)
            
            # 判断胜负
            if game.winner is not None:
                # 根据swap修正赢家
                actual_winner = game.winner
                if swap:
                    # swap后，原来的玩家0变成了玩家1
                    pass
                
                if actual_winner in [0, 2]:
                    wins_a += 1
                else:
                    wins_b += 1
            else:
                draws += 1
        
        return wins_a, wins_b, draws
    
    def _select_action_from_ppo(self, state: Dict, ppo: PPO) -> int:
        """从PPO选择动作"""
        import torch
        
        ppo.policy_net.eval()
        
        with torch.no_grad():
            features = self.feature_processor.encode_features(state)
            normalized = self.feature_processor.normalize_features(features)
            
            x = torch.FloatTensor(normalized).unsqueeze(0).to(ppo.device)
            action_probs, value = ppo.policy_net(x)
            
            # 根据温度采样
            action = torch.distributions.Categorical(action_probs).sample().item()
            
            return action
    
    def evaluate_random(self, num_games: int = 100) -> Dict:
        """
        随机策略评估（基线）
        
        Args:
            num_games: 评估游戏数
            
        Returns:
            评估结果
        """
        print(f"开始随机策略评估，共 {num_games} 局...")
        
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(num_games):
            winner = self._play_game_random()
            
            if winner == 0:
                wins += 1
            elif winner is not None:
                losses += 1
            else:
                draws += 1
        
        results = {
            'num_games': num_games,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'draw_rate': draws / num_games,
            'eval_type': 'random_baseline'
        }
        
        print(f"随机策略评估完成！")
        print(f"  胜率: {results['win_rate']:.2%}")
        
        return results
    
    def _play_game_random(self) -> Optional[int]:
        """玩一局随机策略游戏"""
        game = SimulatedGame()
        max_steps = 200
        
        while not game.is_done() and game.steps < max_steps:
            player_id = game.current_player
            state = game.get_state(player_id)
            
            # 随机选择动作
            valid_actions = self.action_encoder.get_valid_actions(state)
            action = random.choice(valid_actions) if valid_actions else 0
            
            # 执行动作
            game.take_action(player_id, action)
        
        return game.winner
    
    def save_results(self, output_dir: str = 'output/eval'):
        """保存评估结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存最终结果
        result_path = output_path / f"eval_result_{int(time.time())}.json"
        with open(result_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存游戏记录
        if self.game_records:
            records_path = output_path / f"game_records_{int(time.time())}.json"
            with open(records_path, 'w') as f:
                json.dump(self.game_records, f, indent=2)
        
        print(f"评估结果已保存到: {result_path}")
    
    def get_comparison_report(self, 
                             model_path_a: str, 
                             model_path_b: str, 
                             num_games: int = 100) -> Dict:
        """
        获取两个模型的对战报告
        
        Args:
            model_path_a: 模型A路径
            model_path_b: 模型B路径
            num_games: 对战局数
            
        Returns:
            对比报告
        """
        print(f"生成对战报告: {model_path_a} vs {model_path_b}")
        
        # 加载模型
        ppo_a = PPO.load(model_path_a)
        ppo_b = PPO.load(model_path_b)
        
        # 对战
        wins_a, wins_b, draws = self._play_match(ppo_a, ppo_b, num_games)
        
        # 生成报告
        report = {
            'model_a': model_path_a,
            'model_b': model_path_b,
            'num_games': num_games,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'draws': draws,
            'win_rate_a': wins_a / num_games,
            'win_rate_b': wins_b / num_games,
            'win_rate_diff': (wins_a - wins_b) / num_games,
            'timestamp': time.time()
        }
        
        print(f"  模型A胜率: {report['win_rate_a']:.2%}")
        print(f"  模型B胜率: {report['win_rate_b']:.2%}")
        
        return report


def main():
    """主函数 - 命令行评估"""
    import argparse
    
    parser = argparse.ArgumentParser(description='川麻将模型评估')
    
    parser.add_argument('--model', type=str, help='要评估的模型路径')
    parser.add_argument('--eval_type', type=str, default='selfplay',
                       choices=['selfplay', 'opponent', 'benchmark', 'random'],
                       help='评估类型')
    parser.add_argument('--opponent', type=str, help='对手模型路径')
    parser.add_argument('--models', type=str, nargs='+', help='基准测试的模型列表')
    parser.add_argument('--num_games', type=int, default=100, help='评估局数')
    parser.add_argument('--output', type=str, default='output/eval', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvalConfig(
        num_games=args.num_games,
        eval_type=args.eval_type
    )
    
    # 创建评估器
    evaluator = Evaluator(config, args.model)
    
    # 执行评估
    if args.eval_type == 'selfplay':
        results = evaluator.evaluate_selfplay()
    elif args.eval_type == 'opponent':
        results = evaluator.evaluate_vs_opponent(args.opponent)
    elif args.eval_type == 'benchmark' and args.models:
        results = evaluator.benchmark(args.models, args.num_games)
    elif args.eval_type == 'random':
        results = evaluator.evaluate_random()
    else:
        print("请提供模型路径")
        return
    
    # 保存结果
    evaluator.save_results(args.output)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()