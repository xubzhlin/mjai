"""
回放系统
实现游戏回放、分析、可视化等功能
"""

import json
import time
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random

from mjai.selfplay import GameRecord, SimulatedGame
from mjai.features import FeatureProcessor, ActionEncoder, FeatureConfig


@dataclass
class ReplayFrame:
    """回放帧"""
    frame_id: int = 0
    timestamp: float = 0.0
    player: int = 0
    action: int = 0
    action_str: str = ""
    reward: float = 0.0
    state: Dict = None
    next_state: Dict = None
    done: bool = False
    info: Dict = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'player': self.player,
            'action': self.action,
            'action_str': self.action_str,
            'reward': self.reward,
            'done': self.done,
            'info': self.info
        }


class GameReplay:
    """游戏回放"""
    
    def __init__(self, game_id: int = 0):
        """
        初始化游戏回放
        
        Args:
            game_id: 游戏ID
        """
        self.game_id = game_id
        self.frames: List[ReplayFrame] = []
        self.start_time = time.time()
        self.end_time = None
        
        # 游戏结果
        self.winner = None
        self.scores = [0.0, 0.0, 0.0, 0.0]
        self.total_rewards = [0.0, 0.0, 0.0, 0.0]
        
        # 当前播放位置
        self.current_frame = 0
    
    def add_frame(self, frame: ReplayFrame):
        """添加回放帧"""
        frame.frame_id = len(self.frames)
        frame.timestamp = time.time() - self.start_time
        self.frames.append(frame)
        
        # 累积奖励
        if frame.player < 4:
            self.total_rewards[frame.player] += frame.reward
    
    def set_result(self, winner: Optional[int], scores: List[float]):
        """设置游戏结果"""
        self.winner = winner
        self.scores = scores
        self.end_time = time.time()
    
    def get_frame(self, frame_idx: int) -> Optional[ReplayFrame]:
        """获取指定帧"""
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None
    
    def get_next_frame(self) -> Optional[ReplayFrame]:
        """获取下一帧"""
        if self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            self.current_frame += 1
            return frame
        return None
    
    def reset(self):
        """重置播放位置"""
        self.current_frame = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'game_id': self.game_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': (self.end_time - self.start_time) if self.end_time else None,
            'num_frames': len(self.frames),
            'winner': self.winner,
            'scores': self.scores,
            'total_rewards': self.total_rewards,
            'frames': [frame.to_dict() for frame in self.frames]
        }
    
    def load_from_dict(self, data: Dict):
        """从字典加载"""
        self.game_id = data.get('game_id', 0)
        self.start_time = data.get('start_time', time.time())
        self.end_time = data.get('end_time')
        self.winner = data.get('winner')
        self.scores = data.get('scores', [0.0, 0.0, 0.0, 0.0])
        self.total_rewards = data.get('total_rewards', [0.0, 0.0, 0.0, 0.0])
        
        for frame_data in data.get('frames', []):
            frame = ReplayFrame(
                frame_id=frame_data.get('frame_id', 0),
                timestamp=frame_data.get('timestamp', 0),
                player=frame_data.get('player', 0),
                action=frame_data.get('action', 0),
                action_str=frame_data.get('action_str', ''),
                reward=frame_data.get('reward', 0.0),
                done=frame_data.get('done', False),
                info=frame_data.get('info', {})
            )
            self.frames.append(frame)
    
    def __len__(self):
        return len(self.frames)


class ReplayRecorder:
    """回放录制器"""
    
    def __init__(self):
        """初始化录制器"""
        self.current_replay: Optional[GameReplay] = None
        self.replays: List[GameReplay] = []
        self.record_start_time = None
    
    def start_recording(self, game_id: int = 0):
        """
        开始录制
        
        Args:
            game_id: 游戏ID
        """
        self.current_replay = GameReplay(game_id=game_id)
        self.record_start_time = time.time()
    
    def record_action(self, 
                     player: int, 
                     action: int, 
                     action_str: str,
                     reward: float,
                     state: Dict,
                     next_state: Dict,
                     done: bool,
                     info: Dict):
        """
        记录动作
        
        Args:
            player: 玩家ID
            action: 动作索引
            action_str: 动作字符串
            reward: 奖励
            state: 当前状态
            next_state: 下一状态
            done: 是否结束
            info: 额外信息
        """
        if self.current_replay is None:
            return
        
        frame = ReplayFrame(
            player=player,
            action=action,
            action_str=action_str,
            reward=reward,
            state=state,
            next_state=next_state,
            done=done,
            info=info
        )
        
        self.current_replay.add_frame(frame)
    
    def stop_recording(self, winner: Optional[int], scores: List[float]):
        """
        停止录制
        
        Args:
            winner: 赢家ID
            scores: 分数列表
        """
        if self.current_replay is None:
            return
        
        self.current_replay.set_result(winner, scores)
        self.replays.append(self.current_replay)
        self.current_replay = None
    
    def get_current_replay(self) -> Optional[GameReplay]:
        """获取当前回放"""
        return self.current_replay
    
    def get_all_replays(self) -> List[GameReplay]:
        """获取所有回放"""
        return self.replays
    
    def save_replay(self, replay: GameReplay, output_dir: str = 'output/replays'):
        """
        保存单个回放
        
        Args:
            replay: 回放对象
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"replay_game_{replay.game_id}_{int(time.time())}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(replay.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_all_replays(self, output_dir: str = 'output/replays'):
        """
        保存所有回放
        
        Args:
            output_dir: 输出目录
        """
        for replay in self.replays:
            self.save_replay(replay, output_dir)


class ReplayPlayer:
    """回放播放器"""
    
    def __init__(self, replay: Optional[GameReplay] = None):
        """
        初始化回放播放器
        
        Args:
            replay: 回放对象
        """
        self.replay = replay
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 1.0  # 播放速度倍数
    
    def load_replay(self, replay: GameReplay):
        """加载回放"""
        self.replay = replay
        self.current_frame_idx = 0
    
    def load_from_file(self, filepath: str):
        """
        从文件加载回放
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        replay = GameReplay()
        replay.load_from_dict(data)
        self.load_replay(replay)
    
    def play(self) -> Optional[ReplayFrame]:
        """
        播放一帧
        
        Returns:
            当前帧
        """
        if self.replay is None:
            return None
        
        if self.current_frame_idx < len(self.replay.frames):
            frame = self.replay.frames[self.current_frame_idx]
            self.current_frame_idx += 1
            return frame
        
        return None
    
    def seek(self, frame_idx: int) -> Optional[ReplayFrame]:
        """
        跳转到指定帧
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            目标帧
        """
        if self.replay is None:
            return None
        
        if 0 <= frame_idx < len(self.replay.frames):
            self.current_frame_idx = frame_idx
            return self.replay.frames[frame_idx]
        
        return None
    
    def next_frame(self) -> Optional[ReplayFrame]:
        """播放下一帧"""
        return self.play()
    
    def prev_frame(self) -> Optional[ReplayFrame]:
        """播放上一帧"""
        if self.replay is not None and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            return self.replay.frames[self.current_frame_idx]
        return None
    
    def jump_to_end(self):
        """跳转到末尾"""
        if self.replay is not None:
            self.current_frame_idx = len(self.replay.frames)
    
    def jump_to_start(self):
        """跳转到开头"""
        self.current_frame_idx = 0
    
    def get_progress(self) -> Tuple[int, int]:
        """获取播放进度"""
        if self.replay is None:
            return (0, 0)
        return (self.current_frame_idx, len(self.replay.frames))
    
    def get_summary(self) -> Dict:
        """获取回放摘要"""
        if self.replay is None:
            return {}
        
        return {
            'game_id': self.replay.game_id,
            'duration': self.replay.end_time - self.replay.start_time if self.replay.end_time else 0,
            'num_frames': len(self.replay.frames),
            'winner': self.replay.winner,
            'scores': self.replay.scores,
            'total_rewards': self.replay.total_rewards,
            'progress': self.get_progress()
        }


class ReplayAnalyzer:
    """回放分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.analysis_results = {}
    
    def analyze_replay(self, replay: GameReplay) -> Dict:
        """
        分析单个回放
        
        Args:
            replay: 回放对象
            
        Returns:
            分析结果
        """
        analysis = {
            'game_id': replay.game_id,
            'num_frames': len(replay.frames),
            'winner': replay.winner,
            'scores': replay.scores,
            'player_stats': self._analyze_players(replay),
            'action_distribution': self._analyze_actions(replay),
            'reward_analysis': self._analyze_rewards(replay),
            'game_flow': self._analyze_game_flow(replay),
            'recommendations': []
        }
        
        # 生成建议
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.analysis_results[replay.game_id] = analysis
        return analysis
    
    def analyze_batch(self, replays: List[GameReplay]) -> Dict:
        """
        批量分析回放
        
        Args:
            replays: 回放列表
            
        Returns:
            批量分析结果
        """
        batch_results = {
            'num_games': len(replays),
            'games': [],
            'overall_stats': {},
            'trends': {},
            'improvement_suggestions': []
        }
        
        # 分析每个游戏
        for replay in replays:
            analysis = self.analyze_replay(replay)
            batch_results['games'].append(analysis)
        
        # 计算总体统计
        batch_results['overall_stats'] = self._compute_overall_stats(batch_results['games'])
        
        # 分析趋势
        batch_results['trends'] = self._analyze_trends(batch_results['games'])
        
        # 生成改进建议
        batch_results['improvement_suggestions'] = self._generate_batch_suggestions(batch_results)
        
        return batch_results
    
    def _analyze_players(self, replay: GameReplay) -> Dict:
        """分析玩家表现"""
        player_stats = {}
        
        for player_id in range(4):
            player_frames = [f for f in replay.frames if f.player == player_id]
            
            if not player_frames:
                continue
            
            # 计算统计
            total_actions = len(player_frames)
            total_reward = sum(f.reward for f in player_frames)
            avg_reward = total_reward / max(total_actions, 1)
            
            # 动作类型分布
            action_types = defaultdict(int)
            for frame in player_frames:
                action_type = self._get_action_type(frame.action)
                action_types[action_type] += 1
            
            player_stats[player_id] = {
                'total_actions': total_actions,
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'action_distribution': dict(action_types)
            }
        
        return player_stats
    
    def _analyze_actions(self, replay: GameReplay) -> Dict:
        """分析动作分布"""
        action_counts = defaultdict(int)
        action_rewards = defaultdict(list)
        
        for frame in replay.frames:
            action_type = self._get_action_type(frame.action)
            action_counts[action_type] += 1
            action_rewards[action_type].append(frame.reward)
        
        # 计算每种动作的统计
        analysis = {}
        for action_type, count in action_counts.items():
            rewards = action_rewards[action_type]
            analysis[action_type] = {
                'count': count,
                'percentage': count / max(len(replay.frames), 1),
                'avg_reward': np.mean(rewards) if rewards else 0,
                'std_reward': np.std(rewards) if rewards else 0
            }
        
        return analysis
    
    def _analyze_rewards(self, replay: GameReplay) -> Dict:
        """分析奖励"""
        rewards = [f.reward for f in replay.frames]
        
        return {
            'total': sum(rewards),
            'avg': np.mean(rewards) if rewards else 0,
            'std': np.std(rewards) if rewards else 0,
            'min': min(rewards) if rewards else 0,
            'max': max(rewards) if rewards else 0,
            'cumulative': np.cumsum(rewards).tolist() if rewards else []
        }
    
    def _analyze_game_flow(self, replay: GameReplay) -> Dict:
        """分析游戏流程"""
        if not replay.frames:
            return {}
        
        # 每10帧为一个阶段
        stage_size = max(1, len(replay.frames) // 10)
        stages = []
        
        for i in range(0, len(replay.frames), stage_size):
            stage_frames = replay.frames[i:i + stage_size]
            stage_rewards = [f.reward for f in stage_frames]
            
            stages.append({
                'start_frame': i,
                'end_frame': min(i + stage_size, len(replay.frames)),
                'avg_reward': np.mean(stage_rewards) if stage_rewards else 0,
                'cumulative_reward': sum(stage_rewards)
            })
        
        return {
            'num_stages': len(stages),
            'stages': stages
        }
    
    def _compute_overall_stats(self, game_analyses: List[Dict]) -> Dict:
        """计算总体统计"""
        if not game_analyses:
            return {}
        
        # 汇总数据
        total_games = len(game_analyses)
        win_count = sum(1 for a in game_analyses if a['winner'] == 0)
        
        # 收集所有奖励
        all_rewards = []
        for analysis in game_analyses:
            for player_id, stats in analysis.get('player_stats', {}).items():
                all_rewards.append(stats['avg_reward'])
        
        return {
            'total_games': total_games,
            'win_rate': win_count / total_games,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'std_reward': np.std(all_rewards) if all_rewards else 0,
            'num_players': 4
        }
    
    def _analyze_trends(self, game_analyses: List[Dict]) -> Dict:
        """分析趋势"""
        trends = {
            'reward_trend': [],
            'win_rate_trend': [],
            'action_distribution_trend': defaultdict(list)
        }
        
        # 滑动窗口分析
        window_size = max(1, len(game_analyses) // 10)
        
        for i in range(0, len(game_analyses), window_size):
            window = game_analyses[i:i + window_size]
            
            # 计算窗口内的指标
            rewards = []
            win_count = 0
            
            for analysis in window:
                for player_id, stats in analysis.get('player_stats', {}).items():
                    rewards.append(stats['avg_reward'])
                if analysis['winner'] == 0:
                    win_count += 1
            
            trends['reward_trend'].append(np.mean(rewards) if rewards else 0)
            trends['win_rate_trend'].append(win_count / len(window))
        
        return trends
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查胜率
        if analysis['winner'] != 0:
            recommendations.append("模型未获胜，需要改进策略")
        
        # 检查动作分布
        action_dist = analysis.get('action_distribution', {})
        if 'discard' in action_dist:
            discard_pct = action_dist['discard'].get('percentage', 0)
            if discard_pct > 0.8:
                recommendations.append("弃牌比例过高，需要更多进攻性动作")
        
        # 检查奖励
        reward_stats = analysis.get('reward_analysis', {})
        if reward_stats.get('avg', 0) < -0.5:
            recommendations.append("平均奖励过低，考虑调整奖励函数")
        
        return recommendations
    
    def _generate_batch_suggestions(self, batch_results: Dict) -> List[str]:
        """生成批量改进建议"""
        suggestions = []
        
        overall = batch_results['overall_stats']
        win_rate = overall.get('win_rate', 0)
        
        if win_rate < 0.3:
            suggestions.append("整体胜率较低 (<30%)，需要加强训练")
        elif win_rate < 0.5:
            suggestions.append("胜率有提升空间，可以尝试调整超参数")
        
        # 分析趋势
        trends = batch_results.get('trends', {})
        reward_trend = trends.get('reward_trend', [])
        
        if len(reward_trend) >= 2:
            if reward_trend[-1] < reward_trend[0]:
                suggestions.append("奖励呈下降趋势，检查是否需要调整学习率")
        
        return suggestions
    
    def _get_action_type(self, action: int) -> str:
        """获取动作类型字符串"""
        if action == 0:
            return 'discard'
        elif action == 1:
            return 'win'
        elif action == 2:
            return 'eat'
        elif action == 3:
            return 'peng'
        elif action == 4:
            return 'gang'
        elif action == 5:
            return 'pass'
        else:
            return 'other'


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='川麻将回放分析工具')
    
    parser.add_argument('--replay', type=str, help='回放文件路径')
    parser.add_argument('--dir', type=str, help='回放目录路径')
    parser.add_argument('--output', type=str, default='output/analysis', help='分析结果输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ReplayAnalyzer()
    
    if args.replay and os.path.exists(args.replay):
        # 分析单个回放
        with open(args.replay, 'r') as f:
            data = json.load(f)
        
        replay = GameReplay()
        replay.load_from_dict(data)
        
        result = analyzer.analyze_replay(replay)
        
        print(f"\n回放分析结果 - 游戏 {result['game_id']}")
        print(f"  帧数: {result['num_frames']}")
        print(f"  赢家: {result['winner']}")
        print(f"  建议: {result['recommendations']}")
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        result_path = output_path / f"analysis_{result['game_id']}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存到: {result_path}")
    
    elif args.dir and os.path.exists(args.dir):
        # 分析目录下所有回放
        replay_files = list(Path(args.dir).glob('*.json'))
        replays = []
        
        for filepath in replay_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            replay = GameReplay()
            replay.load_from_dict(data)
            replays.append(replay)
        
        print(f"找到 {len(replays)} 个回放文件")
        
        # 批量分析
        batch_result = analyzer.analyze_batch(replays)
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        batch_path = output_path / f"batch_analysis_{int(time.time())}.json"
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n批量分析完成！")
        print(f"  总游戏数: {batch_result['num_games']}")
        print(f"  胜率: {batch_result['overall_stats'].get('win_rate', 0):.2%}")
        print(f"  建议: {batch_result['improvement_suggestions']}")
        print(f"\n分析结果已保存到: {batch_path}")
    
    else:
        print("请提供回放文件路径或目录路径")


if __name__ == "__main__":
    main()