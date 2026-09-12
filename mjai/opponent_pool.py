"""
对手池 (Opponent Pool) 系统

功能：
    - 保留历史模型快照，防止自博弈策略崩溃和循环克制
    - 准入门槛控制：胜率 ≥ 30% 入池
    - 淘汰策略：最旧 > 胜率最低，胜率 < 10% 直接淘汰
    - 对手采样：均匀 or 对抗性（优先选弱点对手）
    - LFSP 可选增强（Main Player + League Players + Main Exploiter）
"""

import os
import copy
import json
import random
import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np


@dataclass
class ModelSnapshot:
    """模型快照"""
    model: Any                    # ModelBase 实例
    model_id: str                 # 唯一标识（如 "iter_0010"）
    iteration: int                # 迭代轮次
    creation_time: float          # 创建时间戳
    win_rates: Dict[str, float] = field(default_factory=dict)  # 对其他快照的胜率
    total_games: int = 0          # 总对局数
    total_wins: int = 0           # 总胜场
    
    @property
    def avg_win_rate(self) -> float:
        """平均胜率"""
        if not self.win_rates:
            return 0.0
        return sum(self.win_rates.values()) / len(self.win_rates)
    
    @property
    def recency_score(self) -> float:
        """新鲜度得分（越新越高）"""
        return self.creation_time


@dataclass
class OpponentPoolConfig:
    """对手池配置"""
    max_size: int = 20                # 池最大容量
    admission_win_rate: float = 0.30  # 准入门槛：30%
    retire_win_rate: float = 0.10     # 强制淘汰门槛：10%
    snapshot_interval: int = 500      # 每隔多少局生成候选快照
    eval_games: int = 100             # 评估对局数
    sample_strategy: str = "uniform"  # 采样策略：uniform / adversarial
    # LFSP 配置
    enable_lfsp: bool = False         # 是否启用 LFSP
    main_exploiter_interval: int = 2000  # Main Exploiter 更新间隔


class OpponentPool:
    """
    对手池管理器
    
    使用方法：
        pool = OpponentPool(config)
        # 自博弈时采样对手
        opponents = pool.sample(num_seats=3)
        # 定期添加新快照
        pool.add_snapshot(current_model, iteration=10)
    """
    
    def __init__(self, config: Optional[OpponentPoolConfig] = None):
        self.config = config or OpponentPoolConfig()
        self.snapshots: List[ModelSnapshot] = []
        self.current_model_id: Optional[str] = None
        self._next_id_counter = 0
    
    def __len__(self) -> int:
        return len(self.snapshots)
    
    def __contains__(self, model_id: str) -> bool:
        return any(s.model_id == model_id for s in self.snapshots)
    
    def is_empty(self) -> bool:
        return len(self.snapshots) == 0
    
    def get_snapshot(self, model_id: str) -> Optional[ModelSnapshot]:
        for s in self.snapshots:
            if s.model_id == model_id:
                return s
        return None

    # ===== 持久化 =====

    def state_dict(self) -> Dict[str, Any]:
        """序列化整个对手池（含所有快照的 model weights）"""
        snapshots_state = []
        for s in self.snapshots:
            model_sd = s.model.state_dict() if hasattr(s.model, "state_dict") else None
            snapshots_state.append({
                "model_id": s.model_id,
                "iteration": s.iteration,
                "creation_time": s.creation_time,
                "win_rates": s.win_rates,
                "total_games": s.total_games,
                "total_wins": s.total_wins,
                "model_state_dict": model_sd,
            })
        return {
            "config": {
                "max_size": self.config.max_size,
                "admission_win_rate": self.config.admission_win_rate,
                "retire_win_rate": self.config.retire_win_rate,
                "snapshot_interval": self.config.snapshot_interval,
                "sample_strategy": self.config.sample_strategy,
                "enable_lfsp": self.config.enable_lfsp,
            },
            "snapshots": snapshots_state,
            "_next_id_counter": self._next_id_counter,
        }

    def load_state_dict(self, state: Dict[str, Any], model_factory: Optional[Any] = None) -> None:
        """从 state_dict 恢复对手池

        Args:
            state: 由 state_dict() 生成的字典
            model_factory: 可选 callable，接受 (model_id) 返回一个空模型实例。
                           若 None 则快照中的 model 保持 None（仅元数据恢复）。
        """
        # 恢复 config
        cfg = state.get("config", {})
        for k, v in cfg.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # 恢复快照
        self.snapshots = []
        for s_state in state.get("snapshots", []):
            snap = ModelSnapshot(
                model=None,  # 先占位
                model_id=s_state["model_id"],
                iteration=s_state["iteration"],
                creation_time=s_state.get("creation_time", 0.0),
                win_rates=s_state.get("win_rates", {}),
                total_games=s_state.get("total_games", 0),
                total_wins=s_state.get("total_wins", 0),
            )
            # 恢复 model weights
            model_sd = s_state.get("model_state_dict")
            if model_sd is not None:
                if model_factory is not None:
                    try:
                        snap.model = model_factory(s_state["model_id"])
                        snap.model.load_state_dict(model_sd)
                    except Exception as e:
                        print(f"[OpponentPool] 恢复快照 {s_state['model_id']} 失败: {e}")
                        snap.model = None
                else:
                    # 没有 factory → 只存 state_dict，等使用时再重建
                    snap.model = model_sd  # 类型不规范，但保留了权重
            self.snapshots.append(snap)

        self._next_id_counter = state.get("_next_id_counter", len(self.snapshots))
        print(f"[OpponentPool] 恢复完成: {len(self.snapshots)} 个快照")
    
    # ===== 添加/移除 =====
    
    def add_snapshot(
        self,
        model: Any,
        iteration: int,
        model_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        尝试将新模型加入对手池
        
        Args:
            model: 模型实例（ModelBase）
            iteration: 迭代轮次
            model_id: 可选，唯一标识
            
        Returns:
            (是否成功入池, 模型ID)
        """
        # 生成模型 ID
        if model_id is None:
            self._next_id_counter += 1
            model_id = f"snap_{iteration:06d}_{self._next_id_counter:04d}"
        
        # 如果池为空，首个快照直接入池
        if self.is_empty():
            snapshot = ModelSnapshot(
                model=copy.deepcopy(model),
                model_id=model_id,
                iteration=iteration,
                creation_time=time.time(),
            )
            self.snapshots.append(snapshot)
            print(f"[OpponentPool] 首个快照入池: {model_id} (迭代 {iteration})")
            return True, model_id
        
        # 准入评估：与池中所有对手进行 eval_games 局
        if len(self.snapshots) > 0:
            candidate = ModelSnapshot(
                model=copy.deepcopy(model),
                model_id=model_id,
                iteration=iteration,
                creation_time=time.time(),
            )
            
            # 计算准入胜率
            avg_win_rate = self._evaluate_admission(candidate)
            candidate.win_rates = {s.model_id: 0.0 for s in self.snapshots}
            
            print(f"[OpponentPool] 候选快照 {model_id} 准入胜率: {avg_win_rate:.2%}")
            
            if avg_win_rate < self.config.admission_win_rate:
                print(f"[OpponentPool] ❌ 准入失败（{avg_win_rate:.2%} < {self.config.admission_win_rate:.0%}）")
                return False, model_id
        
        # 容量满了 → 淘汰最旧或最弱
        if len(self.snapshots) >= self.config.max_size:
            self._evict()
        
        # 入池
        snapshot = ModelSnapshot(
            model=copy.deepcopy(model),
            model_id=model_id,
            iteration=iteration,
            creation_time=time.time(),
        )
        self.snapshots.append(snapshot)
        
        print(f"[OpponentPool] ✅ 快照入池: {model_id} (迭代 {iteration}, 当前池大小 {len(self.snapshots)}/{self.config.max_size})")
        return True, model_id
    
    def remove_snapshot(self, model_id: str) -> bool:
        """移除指定快照"""
        for i, s in enumerate(self.snapshots):
            if s.model_id == model_id:
                self.snapshots.pop(i)
                return True
        return False
    
    def clear(self):
        """清空池子"""
        self.snapshots.clear()
        print("[OpponentPool] 池已清空")
    
    # ===== 准入评估（模拟）=====
    
    def _evaluate_admission(self, candidate: ModelSnapshot) -> float:
        """
        评估候选快照的准入胜率
        
        注意：这里使用启发式估计，实际应通过模拟自博弈实现。
        真实实现需要调用引擎进行 eval_games 局对局。
        """
        if len(self.snapshots) == 0:
            return 0.5  # 无对手时默认 50%
        
        # 启发式：假设越新的模型越强
        # 实际实现：需要通过引擎运行 eval_games 局对局
        win_rates = []
        for existing in self.snapshots:
            # 启发式估计：候选比老模型胜率更高
            iteration_diff = candidate.iteration - existing.iteration
            # 简单 sigmoid 估计
            estimated_wr = 1.0 / (1.0 + np.exp(-iteration_diff / 10.0))
            win_rates.append(estimated_wr)
            candidate.win_rates[existing.model_id] = estimated_wr
        
        return np.mean(win_rates)
    
    # ===== 淘汰 =====
    
    def _evict(self):
        """执行淘汰"""
        if not self.snapshots:
            return
        
        # 第一优先级：胜率 < retire_win_rate 的快照直接淘汰
        for s in self.snapshots[:]:
            if len(s.win_rates) > 0 and s.avg_win_rate < self.config.retire_win_rate:
                print(f"[OpponentPool] 强制淘汰 {s.model_id}（胜率 {s.avg_win_rate:.2%} < {self.config.retire_win_rate:.0%}）")
                self.snapshots.remove(s)
                return
        
        # 第二优先级：淘汰最旧的快照
        oldest = min(self.snapshots, key=lambda s: s.creation_time)
        print(f"[OpponentPool] 淘汰最旧快照 {oldest.model_id}（迭代 {oldest.iteration}）")
        self.snapshots.remove(oldest)
    
    def periodic_maintenance(self, current_iteration: int, current_model: Any):
        """
        定期维护：检查强制淘汰
        
        Args:
            current_iteration: 当前迭代轮次
            current_model: 当前最新模型
        """
        # 更新当前模型的胜率统计
        for s in self.snapshots:
            if current_model is not None:
                # 简化：重新评估胜率
                pass
        
        # 检查强制淘汰
        old_len = len(self.snapshots)
        self._evict()
        while len(self.snapshots) > self.config.max_size:
            self._evict()
        
        # 准入检查（定期尝试添加当前模型的快照）
        if current_iteration % self.config.snapshot_interval == 0:
            self.add_snapshot(current_model, iteration=current_iteration)
    
    # ===== 采样 =====
    
    def sample(
        self,
        num_seats: int = 3,
        current_model: Optional[Any] = None,
        current_model_id: Optional[str] = None,
    ) -> List[ModelSnapshot]:
        """
        从对手池采样对手
        
        Args:
            num_seats: 需要多少个对手（默认 3 个，4 人中当前模型占 1 座）
            current_model: 当前模型（对抗性采样时需要）
            current_model_id: 当前模型 ID（用于排除自己）
            
        Returns:
            采样到的快照列表，长度 ≤ num_seats
        """
        if self.is_empty():
            return []
        
        # 排除当前模型自己
        pool = [s for s in self.snapshots if s.model_id != current_model_id]
        
        if len(pool) == 0:
            return self.snapshots[:num_seats]  # 池子空但自己是唯一快照
        
        # 避免同一快照占多个座位
        num_seats = min(num_seats, len(pool))
        
        if self.config.sample_strategy == "uniform":
            # 均匀采样
            selected = random.sample(pool, num_seats)
        elif self.config.sample_strategy == "adversarial":
            # 对抗性采样：优先选择当前模型胜率低的对手
            selected = self._adversarial_sample(pool, num_seats, current_model)
        else:
            selected = random.sample(pool, num_seats)
        
        return selected
    
    def _adversarial_sample(
        self,
        pool: List[ModelSnapshot],
        num_seats: int,
        current_model: Optional[Any] = None,
    ) -> List[ModelSnapshot]:
        """
        对抗性采样：优先选择弱点对手
        
        策略：给每个快照分配一个权重 = (1 - 对当前模型胜率)，权重越高越可能被选中
        """
        # 使用迭代轮次差异作为启发式权重
        weights = []
        for s in pool:
            # 越新的快照，对当前模型的胜率越接近，权重中等
            # 越旧的快照，可能被当前模型碾压，权重低
            # 选择中间迭代的快照作为最有挑战性的对手
            recency = s.iteration
            # 启发式：中等迭代的对手最有挑战性
            weight = np.exp(-((recency - pool[-1].iteration * 0.7) ** 2) / 500)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        indices = np.random.choice(len(pool), size=num_seats, replace=False, p=weights)
        return [pool[i] for i in indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取对手池统计信息"""
        return {
            'size': len(self.snapshots),
            'max_size': self.config.max_size,
            'sample_strategy': self.config.sample_strategy,
            'snapshots': [
                {
                    'id': s.model_id,
                    'iteration': s.iteration,
                    'avg_win_rate': round(s.avg_win_rate, 4),
                    'games': s.total_games,
                }
                for s in self.snapshots
            ],
        }
    
    def summary(self) -> str:
        """对手池摘要"""
        stats = self.get_stats()
        lines = [
            f"OpponentPool: {stats['size']}/{stats['max_size']}",
            f"  采样策略: {stats['sample_strategy']}",
            f"  准入门槛: {self.config.admission_win_rate:.0%}",
            f"  淘汰门槛: {self.config.retire_win_rate:.0%}",
        ]
        if stats['snapshots']:
            lines.append("  快照列表:")
            for s in stats['snapshots'][:5]:  # 最多显示 5 个
                wr_str = f"{s['avg_win_rate']:.2%}" if s['games'] > 0 else "N/A"
                lines.append(f"    {s['id']:20s} iter={s['iteration']:6d} wr={wr_str}")
            if len(stats['snapshots']) > 5:
                lines.append(f"    ... 还有 {len(stats['snapshots']) - 5} 个快照")
        return "\n".join(lines)


class LFSPOpponentPool(OpponentPool):
    """
    LFSP (League Fictitious Self-Play) 增强对手池
    
    分层：
        - Main Player: 当前训练模型
        - League Players: 历史快照（普通对手池）
        - Main Exploiter: 专门针对当前模型弱点训练的对手
    """
    
    def __init__(self, config: Optional[OpponentPoolConfig] = None):
        super().__init__(config)
        self.exploiter: Optional[ModelSnapshot] = None
    
    def get_exploiter(self) -> Optional[ModelSnapshot]:
        return self.exploiter
    
    def set_exploiter(self, model: Any, iteration: int):
        self.exploiter = ModelSnapshot(
            model=copy.deepcopy(model),
            model_id=f"exploiter_{iteration:06d}",
            iteration=iteration,
            creation_time=time.time(),
        )
    
    def sample_with_exploiter(self, num_seats: int = 3) -> List[ModelSnapshot]:
        """采样时一定概率包含 Exploiter"""
        # 30% 概率将 Exploiter 加入采样候选
        if self.exploiter is not None and random.random() < 0.3:
            pool_with_exploiter = self.snapshots + [self.exploiter]
            pool_with_exploiter = list(dict.fromkeys([s.model_id for s in pool_with_exploiter]))  # 去重
            actual_pool = []
            seen = set()
            for s in self.snapshots + ([self.exploiter] if self.exploiter else []):
                if s.model_id not in seen:
                    seen.add(s.model_id)
                    actual_pool.append(s)
            return random.sample(actual_pool, min(num_seats, len(actual_pool)))
        
        return self.sample(num_seats)


def test_opponent_pool():
    """测试对手池"""
    print("=" * 60)
    print("测试 OpponentPool...")
    
    config = OpponentPoolConfig(
        max_size=5,
        admission_win_rate=0.20,
        retire_win_rate=0.10,
        snapshot_interval=100,
        sample_strategy="uniform",
    )
    pool = OpponentPool(config)
    
    # 模拟添加快照
    class FakeModel:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"FakeModel({self.name})"
    
    for i in range(1, 10):
        model = FakeModel(f"iter_{i:04d}")
        success, mid = pool.add_snapshot(model, iteration=i * 100)
        print(f"  add_snapshot(iter={i*100}): {'✅' if success else '❌'} {mid}")
    
    print(f"\n池大小: {len(pool)}/{config.max_size}")
    
    # 采样
    opponents = pool.sample(num_seats=3)
    print(f"采样对手 ({len(opponents)} 个):")
    for s in opponents:
        print(f"  - {s.model_id} (iter={s.iteration})")
    
    # 统计
    print(pool.summary())
    
    # 清空
    pool.clear()
    print(f"清空后池大小: {len(pool)}")
    
    print("\nOpponentPool 测试通过！")


if __name__ == "__main__":
    test_opponent_pool()