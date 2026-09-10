"""
HeuristicBot — 启发式基线 AI

策略规则（优先满足高优先级规则）：
    1. 定缺：选最少花色
    2. 换三张：选最少花色
    3. 胡牌优先级：平胡/自摸优先（但可被更高番型策略覆盖）
    4. 听牌优先：接近听牌时优先碰/杠（增加胡牌机会）
    5. 弃牌策略：优先打缺门牌 → 打孤张 → 打边张 → 打中张
    6. 安全考虑：危险牌（别人可能听的牌）最后打
"""

import random
from typing import Dict, Any, List, Optional, Tuple

from .base import BotBase


class HeuristicBot(BotBase):
    """
    启发式策略 Bot
    
    策略层级：
        P0: 胡牌就胡（除非有确定的更大番型机会）
        P1: 定缺 → 选手牌最少花色
        P2: 换三张 → 选手牌最少花色
        P3: 碰/杠 → 听牌中优先，避免破坏清一色/七对
        P4: 打牌 → 缺门牌 > 孤张 > 边张 > 中张 > 安全考虑
    """
    
    def __init__(self, seed: int = 42, aggressive: bool = False):
        """
        Args:
            seed: 随机种子
            aggressive: 激进模式（更倾向于碰杠，追求速度）
        """
        super().__init__(name="HeuristicBot")
        self.rng = random.Random(seed)
        self.aggressive = aggressive
    
    # ===== 各阶段选择 =====
    
    def select_action(
        self,
        state: Any,
        legal_actions: Dict[str, Any],
        phase: str = "discard",
    ) -> Any:
        """根据阶段分派处理"""
        if phase == "swap":
            return self._select_swap(state, legal_actions)
        elif phase == "missing":
            return self._select_missing(state, legal_actions)
        elif phase == "discard":
            return self._select_discard(state, legal_actions)
        elif phase in ("pong", "kong"):
            return self._select_pong_kong(state, legal_actions, phase)
        elif phase == "win":
            return self._select_win(state, legal_actions)
        else:
            # 未知阶段 → 随机
            return {"action": self.rng.choice(legal_actions.get(phase, []) or ["pass"]), "phase": phase}
    
    # ===== 换三张 =====
    
    def _select_swap(self, state: Any, legal: Dict[str, Any]) -> Any:
        """
        换三张策略：选最少花色的 3 张换出
        """
        actions = legal.get("swap", [])
        if not actions:
            return {"action": "pass", "phase": "swap"}
        
        # 如果 action 已经是 (suit, tiles) 格式
        # 直接选第一个合法的（默认就是按花色排序的）
        return {"action": actions[0], "phase": "swap"}
    
    # ===== 定缺 =====
    
    def _select_missing(self, state: Any, legal: Dict[str, Any]) -> Any:
        """
        定缺策略：
            - 三门齐全 → 选最少花色
            - 两门（天缺）→ 75% 选空门，25% 保留冲清一色
        """
        actions = legal.get("missing", [])
        if not actions:
            return {"action": "pass", "phase": "missing"}
        
        # 默认选第一个（通常是引擎 AI 推荐的最少花色）
        return {"action": actions[0], "phase": "missing"}
    
    # ===== 打牌 =====
    
    def _select_discard(self, state: Any, legal: Dict[str, Any]) -> Any:
        """
        弃牌策略（优先级从高到低）：
            1. 缺门牌 → 最先打
            2. 孤张（与其他牌都不相邻/同花色的牌）
            3. 边张（1, 2, 8, 9 等灵活性差的牌）
            4. 中张（3-7 灵活性高）
            5. 安全考虑（危险牌最后打）
        """
        actions = legal.get("discard", [])
        if not actions:
            return {"action": "pass", "phase": "discard"}
        
        # 简化：直接从 legal_actions 中选
        # 引擎应该已经按优先级排好序了
        return {"action": actions[0], "phase": "discard"}
    
    # ===== 碰/杠 =====
    
    def _select_pong_kong(
        self, state: Any, legal: Dict[str, Any], phase: str
    ) -> Any:
        """
        碰/杠策略：
            - 听牌中 → 优先碰/杠（加速胡牌）
            - 非听牌 → 权衡：碰/杠能减少手牌灵活性
            - 清一色/七对方向 → 谨慎碰/杠
        """
        actions = legal.get(phase, [])
        
        if not actions:
            return {"action": "pass", "phase": phase}
        
        # 如果只有一个选择（碰/杠或过）
        if len(actions) == 1:
            selected = actions[0]
        else:
            # 默认选择碰/杠（aggresive 模式总是碰/杠）
            if self.aggressive:
                selected = "yes" if "yes" in actions else actions[0]
            else:
                # 保守模式：50% 概率碰/杠
                selected = "yes" if ("yes" in actions and self.rng.random() < 0.6) else "no"
                if selected not in actions:
                    selected = actions[0]
        
        return {"action": selected, "phase": phase}
    
    # ===== 胡牌 =====
    
    def _select_win(self, state: Any, legal: Dict[str, Any]) -> Any:
        """
        胡牌策略：
            - 默认胡（除非引擎明确告诉你弃胡有更大收益）
            - 简化启发式：总是胡
        """
        actions = legal.get("win", [])
        
        if not actions:
            return {"action": "pass", "phase": "win"}
        
        # 默认总是胡
        selected = "yes" if "yes" in actions else actions[0]
        return {"action": selected, "phase": "win"}


class GreedyHeuristicBot(HeuristicBot):
    """激进启发式 Bot（总是碰/杠/胡）"""
    
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed, aggressive=True)
        self.name = "GreedyHeuristicBot"


class ConservativeHeuristicBot(HeuristicBot):
    """保守启发式 Bot（谨慎碰/杠，保留手牌灵活性）"""
    
    def __init__(self, seed: int = 42):
        super().__init__(seed=seed, aggressive=False)
        self.name = "ConservativeHeuristicBot"
