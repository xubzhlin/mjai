"""
ReplayBuffer — DQN 经验回放（6 头分层采样）

特点：
  - 每个决策头一个独立子 buffer，避免高频头 (discard) 淹没低频头 (swap/missing)
  - add(transition) 自动路由到对应 head buffer
  - sample(batch_size, per_head) 每头均匀取 per_head 条，合计 batch_size
  - 容量上限（全局 + 每头独立），FIFO 淘汰
  - to_tensors() 打包成 torch.Tensor 字典，trainer 可直接消费
"""
from __future__ import annotations

import numpy as np
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from mjai.selfplay import Transition


# 6 个决策头
HEAD_TYPES = ("swap", "missing", "discard", "pong", "kong", "win")


@dataclass
class Batch:
    """Trainer 消费的 batch 结构"""
    obs: np.ndarray              # [B, 1917]
    legal_mask: np.ndarray       # [B, max_action_dim]  不同头 output_dim 不同 → 用 padding
    action_idx: np.ndarray       # [B]
    reward: np.ndarray           # [B]
    done: np.ndarray             # [B] bool
    head_type: np.ndarray        # [B] 字符串（用 index 编码: 0=swap ... 5=win）
    head_dim: np.ndarray         # [B] 每个样本对应 head 的 output_dim
    next_obs: np.ndarray         # [B, 1917]  下一状态（DQN 需要）

    @property
    def size(self) -> int:
        return len(self.obs)


class ReplayBuffer:
    """
    分层经验回放 buffer

    Args:
        capacity:       全局总容量上限（所有头合计）
        per_head_cap:   每头独立容量上限；None 则自动按 capacity // 6
        obs_dim:        Encoder 输出维度（固定 1917）
    """

    def __init__(
        self,
        capacity: int = 100_000,
        per_head_cap: Optional[int] = None,
        obs_dim: int = 1917,
    ):
        self.capacity = capacity
        self.per_head_cap = per_head_cap or (capacity // len(HEAD_TYPES))
        self.obs_dim = obs_dim

        # 按 head_type 分桶
        self.buffers: Dict[str, Deque[Transition]] = {
            h: deque(maxlen=self.per_head_cap) for h in HEAD_TYPES
        }
        self._size = 0

    # ──────────────────────────────────────────
    # 写入
    # ──────────────────────────────────────────

    def add(self, transition: Transition) -> None:
        """添加单条 transition（自动路由到对应 head buffer）"""
        head = transition.head_type
        if head not in self.buffers:
            # 未知 head → 跳过（防御性）
            return

        # 维护全局 size
        if self._size >= self.capacity:
            # 全局满了 → 从最大的 buffer 淘汰一条（近似 FIFO）
            biggest = max(self.buffers.values(), key=len)
            if biggest:
                biggest.popleft()
        else:
            self._size += 1

        self.buffers[head].append(transition)

    def add_many(self, transitions: List[Transition]) -> None:
        """批量添加"""
        for t in transitions:
            self.add(t)

    def add_replay(self, replay) -> None:
        """添加整局 GameReplay"""
        self.add_many(replay.transitions)

    # ──────────────────────────────────────────
    # 采样
    # ──────────────────────────────────────────

    # Head 分组：核心决策、前序决策、被动响应
    _CORE_HEADS = ("discard",)
    _PRE_HEADS = ("swap", "missing")
    _PASSIVE_HEADS = ("pong", "kong", "win")

    def sample(self, batch_size: int, per_head: Optional[int] = None,
               passive_max_ratio: float = 0.3) -> Optional[Batch]:
        """
        分层采样 + 被动头配额限制

        分配策略（batch_size=128 示例）:
          discard (核心)      : batch_size // 3 = 42 → 多余配额回灌 (最终 ≈ 74)
          swap/missing (前序)  : 各 batch_size // 6 = 21
          pong/kong/win (被动): 合计 ≤ discard × passive_max_ratio (≈ 12)

        理由: 被动头样本多样性低（大量 pass 决策），过多采样会稀释
        核心 discard 头的有效梯度，导致后期 wr 退化。

        Args:
            batch_size:         目标总 batch 大小
            per_head:           兼容旧 API（忽略，用配额策略）
            passive_max_ratio:  被动头合计配额 / discard 配额的上限比

        Returns:
            Batch 对象，或 None（buffer 数据不足）
        """
        # ── Step 1: 计算每头配额 ──
        core_count = len(self._CORE_HEADS)   # 1
        pre_count = len(self._PRE_HEADS)     # 2

        core_quota = batch_size // 3                         # discard 基线配额
        pre_each_quota = batch_size // (6 * pre_count) * 2   # swap 和 missing 各拿
        passive_total_quota = max(1, int(core_quota * passive_max_ratio))  # 被动合计上限
        passive_each_quota = max(1, passive_total_quota // len(self._PASSIVE_HEADS))

        # 收集所有配额
        quotas: Dict[str, int] = {}
        for h in self._CORE_HEADS:
            quotas[h] = core_quota
        for h in self._PRE_HEADS:
            quotas[h] = pre_each_quota
        for h in self._PASSIVE_HEADS:
            quotas[h] = passive_each_quota

        # ── Step 2: 按配额采样，缺额回灌给核心头 ──
        sampled: List[Transition] = []
        collected: Dict[str, int] = {}

        for head in HEAD_TYPES:
            buf = self.buffers[head]
            quota = quotas.get(head, 0)
            n = min(quota, len(buf))
            if n > 0:
                sampled.extend(random.sample(buf, n))
            collected[head] = n

        # ── Step 3: 若总采样 < batch_size，用核心头 buffer 补齐 ──
        shortage = batch_size - len(sampled)
        if shortage > 0:
            # 优先从 discard 补（核心决策）
            for head in ("discard", "swap", "missing"):
                if shortage <= 0:
                    break
                buf = self.buffers[head]
                remaining = len(buf) - collected.get(head, 0)
                fill = min(remaining, shortage)
                if fill > 0:
                    sampled.extend(random.sample(buf, fill))
                    shortage -= fill
                    collected[head] = collected.get(head, 0) + fill

        if not sampled:
            return None

        return self._to_batch(sampled)

    def sample_uniform(self, batch_size: int) -> Optional[Batch]:
        """全 buffer 均匀采样（不分头）"""
        all_trans = []
        for buf in self.buffers.values():
            all_trans.extend(buf)
        if len(all_trans) < batch_size:
            return None
        return self._to_batch(random.sample(all_trans, batch_size))

    # ──────────────────────────────────────────
    # 查询
    # ──────────────────────────────────────────

    @property
    def size(self) -> int:
        """全局 transition 数"""
        return sum(len(b) for b in self.buffers.values())

    def head_sizes(self) -> Dict[str, int]:
        return {h: len(self.buffers[h]) for h in HEAD_TYPES}

    def summary(self) -> str:
        """人类可读的 buffer 状态"""
        parts = [f"  {h:10s}: {len(self.buffers[h]):6d}/{self.per_head_cap}"
                 for h in HEAD_TYPES]
        return (f"ReplayBuffer (total={self.size}/{self.capacity}):\n"
                + "\n".join(parts))

    # ──────────────────────────────────────────
    # 内部
    # ──────────────────────────────────────────

    def _to_batch(self, transitions: List[Transition]) -> Batch:
        """把 Transition 列表 → Batch"""
        from mjai.engine import HEAD_CONFIG

        B = len(transitions)
        obs = np.zeros((B, self.obs_dim), dtype=np.float32)
        next_obs = np.zeros((B, self.obs_dim), dtype=np.float32)
        action_idx = np.zeros(B, dtype=np.int64)
        reward = np.zeros(B, dtype=np.float32)
        done = np.zeros(B, dtype=np.float32)
        head_idx = np.zeros(B, dtype=np.int64)
        head_dim = np.zeros(B, dtype=np.int64)

        # legal_mask 长度不统一 → padding 到 max_dim
        max_dim = max(
            HEAD_CONFIG[t.head_type]["output_dim"] for t in transitions
        )
        legal_mask = np.zeros((B, max_dim), dtype=np.float32)

        head_to_idx = {h: i for i, h in enumerate(HEAD_TYPES)}

        for i, t in enumerate(transitions):
            obs[i] = t.obs
            # 真 next_obs（如果 transition 存了），否则 fallback 同 obs
            if hasattr(t, "next_obs") and t.next_obs is not None and len(t.next_obs) == self.obs_dim:
                next_obs[i] = t.next_obs
            else:
                next_obs[i] = t.obs
            action_idx[i] = t.action_idx
            reward[i] = t.reward
            done[i] = float(t.done)
            head_idx[i] = head_to_idx.get(t.head_type, 0)

            out_dim = HEAD_CONFIG[t.head_type]["output_dim"]
            head_dim[i] = out_dim
            legal_mask[i, :out_dim] = t.legal_mask

        return Batch(
            obs=obs,
            legal_mask=legal_mask,
            action_idx=action_idx,
            reward=reward,
            done=done,
            head_type=head_idx,
            head_dim=head_dim,
            next_obs=next_obs,
        )
