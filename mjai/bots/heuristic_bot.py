"""
HeuristicBot — 启发式基线 AI（真实策略）

核心启发式：
    1. 解码 encoder 特征向量 → 手牌 tile counts[27] + 弃牌河
    2. 花色分布（万=0-8, 筒=9-17, 条=18-26）
    3. 弃牌评分：缺门牌→最低分最先打；孤张→次低；边张→中张
    4. 碰/杠：有机会 70% 响应（保守），aggressive 模式 100%
    5. 胡：总是胡
"""

from __future__ import annotations

import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .base import BotBase


# 花色分区
SUIT_MAN = list(range(0, 9))    # 万 0-8
SUIT_TONG = list(range(9, 18))  # 筒 9-17
SUIT_TIAO = list(range(18, 27)) # 条 18-26
SUITS = [SUIT_MAN, SUIT_TONG, SUIT_TIAO]
SUIT_NAMES = ["MAN", "TONG", "TIAO"]


def decode_hand_from_obs(obs_flat: np.ndarray) -> np.ndarray:
    """
    从 encoder 输出解码手牌 tile count

    encoder 通道：
        ch0 = 0张标记（hand 中没有的牌 = 1.0）
        ch1 = 1张
        ch2 = 2张
        ch3 = 3张
        ch4 = 4张以上
    """
    obs_2d = obs_flat.reshape(71, 27)
    counts = np.zeros(27, dtype=np.int32)
    for ch in range(5):  # ch0-ch4
        for tile_idx in range(27):
            val = obs_2d[ch, tile_idx]
            if val > 0.5:
                if ch == 0:
                    counts[tile_idx] = 0
                elif ch == 1:
                    counts[tile_idx] = max(counts[tile_idx], 1)
                elif ch == 2:
                    counts[tile_idx] = max(counts[tile_idx], 2)
                elif ch == 3:
                    counts[tile_idx] = max(counts[tile_idx], 3)
                elif ch == 4:
                    counts[tile_idx] = max(counts[tile_idx], 4)
    return counts


def decode_discard_river(obs_flat: np.ndarray) -> np.ndarray:
    """从 encoder 解码弃牌河（CH_DISCARD_0 = channel 21）"""
    obs_2d = obs_flat.reshape(71, 27)
    discards = np.zeros(27, dtype=np.int32)
    # channel 21-24 = 四家弃牌（简化：叠加）
    for ch in range(21, 25):
        discards += (obs_2d[ch] > 0.5).astype(np.int32)
    return discards


def tile_suit(tile_idx: int) -> int:
    """0=万, 1=筒, 2=条"""
    if tile_idx < 9: return 0
    if tile_idx < 18: return 1
    return 2


def tile_rank(tile_idx: int) -> int:
    """1-9"""
    return (tile_idx % 9) + 1


class HeuristicBot(BotBase):
    """
    启发式策略 Bot

    弃牌评分（越低越先打）：
        - 缺门牌: -100
        - 孤张（花色唯一/无相邻）: 10-20
        - 边张（rank 1,2,8,9）: 30-40
        - 中张（rank 3-7）: 50-60
        - 对子/刻子: 1000（尽量保留）
        - 危险牌（弃牌河已出现多）: +20 惩罚
    """

    def __init__(self, seed: int = 42, aggressive: bool = False):
        super().__init__(name="HeuristicBot")
        self.rng = random.Random(seed)
        self.aggressive = aggressive

    def select_action(
        self,
        state: Any,
        legal_actions: Dict[str, Any],
        phase: str = "discard",
    ) -> Any:
        if phase == "discard":
            return self._select_discard(state, legal_actions)
        elif phase in ("pong", "kong"):
            return self._select_pong_kong(state, legal_actions)
        elif phase == "win":
            return {"action": "yes", "phase": "win"}
        elif phase == "swap":
            return {"action": "random", "phase": "swap"}
        elif phase == "missing":
            return {"action": "random", "phase": "missing"}
        return {"action": "pass", "phase": phase}

    def _select_discard(self, obs_flat, legal: Dict) -> Dict:
        """
        用真实启发式选弃牌

        Args:
            obs_flat: encoder 输出 [1917]（state 传入）
            legal: {"discard": [tile_idx, ...]}
        """
        discard_list = legal.get("discard", [])
        if not discard_list:
            return {"action": "pass", "phase": "discard"}

        # 解码手牌 + 弃牌河
        if obs_flat is not None:
            obs = np.asarray(obs_flat, dtype=np.float32)
            hand_counts = decode_hand_from_obs(obs)
            river_counts = decode_discard_river(obs)
        else:
            hand_counts = np.zeros(27, dtype=np.int32)
            river_counts = np.zeros(27, dtype=np.int32)

        # 花色分布
        suit_counts = [sum(hand_counts[s]) for s in SUITS]
        missing_suit_idx = suit_counts.index(min(suit_counts))
        has_two_suits = sum(1 for c in suit_counts if c > 0) <= 2

        scored = []
        for tile_idx in discard_list:
            score = self._tile_discard_score(
                tile_idx, hand_counts, river_counts,
                missing_suit_idx, has_two_suits,
            )
            scored.append((score, tile_idx))

        # 最低分 = 最先打
        scored.sort(key=lambda x: x[0])

        # 前 3 选一个（加点随机性）
        top_n = min(3, len(scored))
        chosen = self.rng.choice([t for _, t in scored[:top_n]])

        return {"action": chosen, "phase": "discard"}

    def _tile_discard_score(
        self,
        tile_idx: int,
        hand_counts: np.ndarray,
        river_counts: np.ndarray,
        missing_suit_idx: int,
        has_two_suits: bool,
    ) -> float:
        """计算单张牌的弃牌分数（越低越该打）"""
        suit = tile_suit(tile_idx)
        rank = tile_rank(tile_idx)
        count = int(hand_counts[tile_idx])

        # 缺门牌 → 最低分（最先打）
        if suit == missing_suit_idx and hand_counts[tile_idx] > 0:
            return -100 + self.rng.random() * 5

        # 对子/刻子 → 高分（尽量保留）
        if count >= 2:
            return 1000

        # 孤张检测
        neighbors = []
        if rank > 1:
            neighbors.append(tile_idx - 1)
        if rank < 9:
            neighbors.append(tile_idx + 1)
        # 同花色跨 rank 相邻（万1-9的万9不连筒1）
        has_neighbor = any(
            hand_counts[n] > 0 and tile_suit(n) == suit
            for n in neighbors
        )

        if not has_neighbor and count == 1:
            # 孤张
            base = 10 + self.rng.random() * 10
        elif rank in (1, 2, 8, 9):
            # 边张
            base = 30 + self.rng.random() * 10
        else:
            # 中张（3-7）
            base = 50 + self.rng.random() * 15

        # 安全奖励：弃牌河已出现 2+ 次的牌更安全
        river = int(river_counts[tile_idx])
        if river >= 2:
            base -= 15  # 安全牌加分（更愿意打）
        elif river >= 1:
            base -= 5

        return base

    def _select_pong_kong(self, obs_flat, legal: Dict) -> Dict:
        """碰/杠：有机会就响应（保守 60%，aggressive 100%）"""
        can_respond = legal.get("can_pong", False) or legal.get("can_kong", False)
        if can_respond:
            prob = 1.0 if self.aggressive else 0.7
            if self.rng.random() < prob:
                return {"action": "yes", "phase": "pong_kong"}
        return {"action": "no", "phase": "pong_kong"}

    def _select_win(self, obs_flat, legal: Dict) -> Dict:
        """总是胡"""
        return {"action": "yes", "phase": "win"}
