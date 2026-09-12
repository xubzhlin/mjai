"""
Engine — 模型推理 + 动作选择 + ε-greedy 探索

把 ResNetDQN 模型包装成可和 mjai_engine.arena.Game 对接的推理器。
核心职责:
  1. 把 legal_actions dict → legal_mask tensor
  2. 把 obs_flat → model 输入 [1, 71, 27]
  3. 调用 model.forward → 取对应 head 的 Q 值
  4. legal_mask 屏蔽非法动作 → argmax 选最优
  5. ε-greedy: 以 ε 概率从 legal_actions 里随机选
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from mjai.models import ResNetDQN, ModelConfig, MultiHeadDQN


# 6 个决策头的配置
HEAD_CONFIG = {
    "swap":    {"output_dim": 27, "action_space": "discrete", "tile_based": True},
    "missing": {"output_dim": 3,  "action_space": "discrete", "tile_based": False},
    "discard": {"output_dim": 27, "action_space": "discrete", "tile_based": True},
    "pong":    {"output_dim": 2,  "action_space": "discrete", "tile_based": False},
    "kong":    {"output_dim": 2,  "action_space": "discrete", "tile_based": False},
    "win":     {"output_dim": 2,  "action_space": "discrete", "tile_based": False},
}


class Engine:
    """
    模型推理引擎

    Args:
        model:  ResNetDQN 实例（已加载权重，eval 模式）
        device: 'cpu' 或 'cuda'
    """
    def __init__(self, model: ResNetDQN, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    # ──────────────────────────────────────────────
    # 核心 API
    # ──────────────────────────────────────────────

    def select_action(
        self,
        obs_flat: Union[List[float], np.ndarray],
        legal: Dict,
        head_type: str = "discard",
        epsilon: float = 0.0,
    ) -> Tuple[int, np.ndarray]:
        """
        选一个动作 + 返回完整的 legal_mask（用于 ReplayBuffer 记录）

        Args:
            obs_flat:  Encoder 展平输出 [1917]
            legal:      get_legal_actions() 返回的 dict
            head_type:  "discard" / "kong" / "swap" / ...
            epsilon:    ε-greedy 探索率

        Returns:
            (action_idx, legal_mask)
              action_idx: 对应 head output_dim 空间里的 index
              legal_mask: shape [output_dim], 1=合法 0=非法
        """
        head_info = HEAD_CONFIG[head_type]
        output_dim = head_info["output_dim"]

        legal_mask = self._build_legal_mask(legal, head_type, output_dim)

        # ε-greedy
        if np.random.random() < epsilon:
            legal_indices = np.where(legal_mask > 0)[0]
            if len(legal_indices) == 0:
                # 兜底：mask 全 0 → 选 0（调用方应保证 legal 非空）
                return 0, legal_mask
            action_idx = int(np.random.choice(legal_indices))
            return action_idx, legal_mask

        # 模型推理
        obs = torch.FloatTensor(np.asarray(obs_flat, dtype=np.float32))
        obs = obs.reshape(1, 71, 27).to(self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            q_dict = self.model.forward(obs)
        if was_training:
            self.model.train()

        q_values = q_dict[head_type][0].cpu().numpy()  # [output_dim]

        # 非法动作屏蔽（-inf，argmax 不会选到）
        masked_q = np.where(legal_mask > 0, q_values, -np.inf)

        if np.all(legal_mask == 0):
            # 全部非法 → 选 0
            action_idx = 0
        else:
            action_idx = int(np.argmax(masked_q))

        return action_idx, legal_mask

    def select_actions_multi_head(
        self,
        obs_flat: Union[List[float], np.ndarray],
        legal: Dict,
        epsilon: float = 0.0,
    ) -> Dict[str, Tuple[int, np.ndarray]]:
        """
        一次 forward 产生所有 6 个头的动作选择（效率更高）
        返回 {head_type: (action_idx, legal_mask)}
        """
        obs = torch.FloatTensor(np.asarray(obs_flat, dtype=np.float32))
        obs = obs.reshape(1, 71, 27).to(self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            q_dict = self.model.forward(obs)
        if was_training:
            self.model.train()

        result = {}
        for head_type in HEAD_CONFIG:
            info = HEAD_CONFIG[head_type]
            output_dim = info["output_dim"]
            legal_mask = self._build_legal_mask(legal, head_type, output_dim)

            if np.random.random() < epsilon:
                legal_indices = np.where(legal_mask > 0)[0]
                if len(legal_indices) == 0:
                    action_idx = 0
                else:
                    action_idx = int(np.random.choice(legal_indices))
            else:
                q_values = q_dict[head_type][0].cpu().numpy()
                masked_q = np.where(legal_mask > 0, q_values, -np.inf)
                if np.all(legal_mask == 0):
                    action_idx = 0
                else:
                    action_idx = int(np.argmax(masked_q))

            result[head_type] = (action_idx, legal_mask)

        return result

    # ──────────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_legal_mask(legal: Dict, head_type: str, output_dim: int) -> np.ndarray:
        """
        把引擎 legal dict → shape [output_dim] 的 one-hot mask

        引擎 legal dict 格式 (Game.get_legal_actions()):
          {
            "discard": [tile_idx, ...],   # tile_idx 0..26
            "kong":    [tile_idx, ...],   # tile_idx 0..26
            "can_tsumo": bool,
            "can_kan_shang": bool,
          }

        但我们的 Dueling heads 输出是标准 DQN 动作空间：
          - discard head: [27] = 27 种可选牌
          - kong head:    [2]  = {否, 是}
          - pong head:    [2]  = {否, 是}
          - win head:     [2]  = {否, 是}
          - swap head:    [3]  = {万, 筒, 条}
          - missing head: [3]  = {万, 筒, 条}
        """
        mask = np.zeros(output_dim, dtype=np.float32)

        if head_type == "discard":
            for tile_idx in legal.get("discard", []):
                if 0 <= tile_idx < output_dim:
                    mask[tile_idx] = 1.0

        elif head_type == "kong":
            # 引擎返回 kong 列表（可杠的 tile），映射到 [0,1]
            kong_list = legal.get("kong", [])
            if kong_list:
                mask[1] = 1.0  # 是
            mask[0] = 1.0  # 否 永远合法（可以选择不杠）

        elif head_type in ("pong", "win"):
            mask[0] = 1.0
            can_flag = legal.get(f"can_{head_type}", False)
            if can_flag:
                mask[1] = 1.0

        elif head_type in ("swap", "missing"):
            # Swap/Missing 开局时所有 3 个花色都合法
            mask[:] = 1.0

        return mask
