"""
Baseline Bot 抽象基类

所有 Bot 必须实现统一的接口，便于评估对比
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BotBase(ABC):
    """
    Bot 抽象基类
    
    所有 Bot （RandomBot, HeuristicBot, RL 模型包装器等）
    必须实现此接口，用于统一评估
    """
    
    def __init__(self, name: str = "BotBase"):
        self.name = name
        self.games_played = 0
        self.games_won = 0
    
    @abstractmethod
    def select_action(
        self,
        state: Any,
        legal_actions: Dict[str, Any],
        phase: str = "discard",
    ) -> Any:
        """
        选择动作
        
        Args:
            state: 当前游戏状态（引擎 Board 的 Python 表示）
            legal_actions: 合法动作集合
            phase: 当前阶段 → "swap" / "missing" / "discard" / "pong" / "kong" / "win"
            
        Returns:
            选中的动作（具体类型取决于 phase）
        """
        ...
    
    def record_result(self, won: bool):
        """记录对局结果"""
        self.games_played += 1
        if won:
            self.games_won += 1
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.games_played == 0:
            return 0.0
        return self.games_won / self.games_played
    
    def reset_stats(self):
        self.games_played = 0
        self.games_won = 0
    
    def __repr__(self):
        return f"{self.name}(wr={self.win_rate:.1%}, games={self.games_played})"
