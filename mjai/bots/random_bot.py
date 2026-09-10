"""
RandomBot — 最低基线 AI

在所有合法动作中均匀随机选择
"""

import random
from typing import Dict, Any

from .base import BotBase


class RandomBot(BotBase):
    """
    随机策略 Bot
    
    策略：在 legal_actions 中均匀随机选择一个合法动作
    用途：最低基线，所有其他 AI 都应显著优于它
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(name="RandomBot")
        self.rng = random.Random(seed)
    
    def select_action(
        self,
        state: Any,
        legal_actions: Dict[str, Any],
        phase: str = "discard",
    ) -> Any:
        """
        从合法动作中随机选择
        
        Args:
            state: 游戏状态（不使用）
            legal_actions: 各阶段的合法动作列表
            phase: 当前阶段
            
        Returns:
            选中的动作
        """
        # 获取当前阶段的合法动作
        if phase in legal_actions:
            actions = legal_actions[phase]
        elif "discard" in legal_actions:
            # 默认尝试 discard
            actions = legal_actions["discard"]
        else:
            # 兜底：取第一个有动作的阶段
            actions = None
            for key, value in legal_actions.items():
                if value:
                    phase = key
                    actions = value
                    break
        
        if not actions:
            # 没有合法动作 → Pass
            return {"action": "pass", "phase": phase}
        
        # 随机选一个
        selected = self.rng.choice(actions)
        return {"action": selected, "phase": phase}
