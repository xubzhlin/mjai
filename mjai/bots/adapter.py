"""
BotEngineAdapter — 把 BotBase（启发式等基线 AI）包装成 Engine 兼容接口

让 HeuristicBot / RandomBot 等能直接塞进 selfplay.py 的 engine 管道，
无需改 selfplay 或 OpponentPool 的任何逻辑。

核心职责:
  1. head_type (Engine) → phase (Bot) 翻译
  2. Bot 返回的 dict {"action": ..., "phase": ...} → Engine 期望的 (int, legal_mask)
  3. 复制 Engine._build_legal_mask 的 mask 生成逻辑（让回放 buffer 正确记录）

使用:
    from mjai.bots import HeuristicBot
    from mjai.bots.adapter import BotEngineAdapter
    eng = BotEngineAdapter(HeuristicBot(seed=42))
    action_idx, mask = eng.select_action(obs_flat, legal, "discard", epsilon=0.0)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple

from .base import BotBase


# head_type → Bot phase 翻译表（和 HEAD_CONFIG 的 key 完全对齐）
_HEAD_TO_PHASE = {
    "discard": "discard",
    "kong":    "kong",
    "pong":    "pong",
    "win":     "win",
    "swap":    "swap",
    "missing": "missing",
}

# 被动响应 phase 别名（Bot 侧统一叫 pong/kong，Rust 侧可能叫 pong_kong）
_PHASE_ALIAS = {
    "pong": "pong",
    "kong": "kong",
    "pong_kong": "kong",   # fallback 兼容
    "ron": "win",          # Rust 侧 ron → Bot 侧 win
    "tsumo": "win",
}


class BotEngineAdapter:
    """
    适配器：BotBase → Engine 接口

    只实现 select_action（selfplay 唯一调用的方法）。
    epsilon 参数被忽略 —— Bot 自带内置随机性（如 HeuristicBot 的 top-3 洗牌）。
    """

    def __init__(self, bot: BotBase):
        self.bot = bot

    def __repr__(self) -> str:
        return f"BotEngineAdapter({self.bot.name})"

    # ──────────────────────────────────────────────
    # Engine 兼容 API
    # ──────────────────────────────────────────────

    def select_action(
        self,
        obs_flat,
        legal: Dict,
        head_type: str = "discard",
        epsilon: float = 0.0,
    ) -> Tuple[int, np.ndarray]:
        """
        对齐 Engine.select_action 签名

        Args:
            obs_flat:  encoder 展平输出 [1917]
            legal:     Game.get_legal_actions() 的返回 dict
            head_type: "discard" / "kong" / "pong" / "win" / "swap" / "missing"
            epsilon:   忽略（Bot 自带随机性）

        Returns:
            (action_idx, legal_mask)  和 Engine 格式完全一致
        """
        legal_mask = self._build_legal_mask(legal, head_type)

        # 1. head_type → phase
        phase = _HEAD_TO_PHASE.get(head_type, head_type)

        # 2. 调 Bot
        try:
            bot_result = self.bot.select_action(obs_flat, legal, phase)
        except Exception:
            return self._fallback_action(legal_mask)

        # 3. Bot dict → (action_idx, legal_mask)
        action_idx = self._translate_bot_action(bot_result, head_type, legal_mask)
        return int(action_idx), legal_mask

    # ──────────────────────────────────────────────
    # 翻译层
    # ──────────────────────────────────────────────

    def _translate_bot_action(
        self,
        bot_result: Any,
        head_type: str,
        legal_mask: np.ndarray,
    ) -> int:
        """把 Bot 的 {"action": X, "phase": Y} dict 翻译成整数 action_idx"""
        if not isinstance(bot_result, dict):
            return self._fallback_action(legal_mask)

        action = bot_result.get("action", "pass")

        # discard (output_dim=27): Bot 直接返回 tile_idx
        if head_type == "discard":
            if isinstance(action, (int, np.integer)) and 0 <= action < 27:
                return int(action)
            # 非整数 → 从合法 mask 里随机挑
            return self._fallback_action(legal_mask)

        # swap (output_dim=27): Bot 返回 "random" → 随机挑 0-26
        if head_type == "swap":
            return self._fallback_action(legal_mask)

        # missing (output_dim=3): Bot 返回 "random" → 随机挑 0-2
        if head_type == "missing":
            return self._fallback_action(legal_mask)

        # kong / pong / win (output_dim=2): yes→1, no→0
        if head_type in ("kong", "pong", "win"):
            if action in ("yes", 1, True):
                return 1
            return 0

        return self._fallback_action(legal_mask)

    def _fallback_action(self, legal_mask: np.ndarray) -> int:
        """从 legal_mask 里随机挑一个合法 index"""
        legal_indices = np.where(legal_mask > 0)[0]
        if len(legal_indices) == 0:
            return 0
        return int(np.random.choice(legal_indices))

    # ──────────────────────────────────────────────
    # legal_mask 构造（和 Engine._build_legal_mask 逻辑一致）
    # ──────────────────────────────────────────────

    _HEAD_OUTPUT_DIM = {
        "swap": 27, "missing": 3, "discard": 27,
        "pong": 2, "kong": 2, "win": 2,
    }

    @classmethod
    def _build_legal_mask(cls, legal: Dict, head_type: str) -> np.ndarray:
        """复制 Engine._build_legal_mask 逻辑，确保回放 buffer 的 mask 正确"""
        output_dim = cls._HEAD_OUTPUT_DIM.get(head_type, 27)
        mask = np.zeros(output_dim, dtype=np.float32)

        if head_type == "discard":
            for tile_idx in legal.get("discard", []):
                if 0 <= tile_idx < output_dim:
                    mask[tile_idx] = 1.0

        elif head_type == "kong":
            kong_list = legal.get("kong", [])
            if kong_list:
                mask[1] = 1.0
            mask[0] = 1.0  # pass 永远合法

        elif head_type in ("pong", "win"):
            mask[0] = 1.0  # pass
            can_flag = legal.get(f"can_{head_type}", False)
            if can_flag:
                mask[1] = 1.0  # respond

        elif head_type in ("swap", "missing"):
            mask[:] = 1.0  # 开局所有花色/牌都合法

        return mask
