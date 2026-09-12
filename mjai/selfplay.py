"""
SelfPlay — 自博弈单局采样

把 mjai_engine.arena.Game + mjai.ObservationEncoder + mjai.engine.Engine
串起来，跑完完整一局并产出 ReplayBuffer 可消费的 transition 列表。

设计要点（来自经验 828636 / 1522732）:
  1. 复用引擎既有 step 函数，不自写回合推进
  2. 每步记录 (obs, head_type, action_idx, legal_mask)，终局后回填 reward
  3. ε-greedy 在 Engine 侧实现，SelfPlay 只负责环境循环
  4. 设置 max_turns 硬约束（防止死循环）
  5. GameReplay / Transition 可序列化为 JSON 牌谱（不含 obs tensor，仅复盘用）
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import mjai_engine  # noqa: F401  (side-effect: registers submodules)


# ────────────────────────────────────────
# 牌面工具
# ────────────────────────────────────────

_TILE_SUITS = {0: "m", 1: "p", 2: "s"}  # 万 / 筒 / 条


def tile_idx_to_str(idx: int) -> str:
    """tile index (0-26) → 人类可读牌面字符串, e.g. 5 → '6m'"""
    if not (0 <= idx <= 26):
        return "?"
    num = idx % 9 + 1
    suit = _TILE_SUITS[idx // 9]
    return f"{num}{suit}"


_HEAD_CN = {
    "swap": "换牌",
    "missing": "定缺",
    "discard": "弃牌",
    "pong": "碰",
    "kong": "杠",
    "win": "胡",
}


def head_action_str(head_type: str, action_idx: int) -> str:
    """把 (head_type, action_idx) 翻译成中文动作描述"""
    cn = _HEAD_CN.get(head_type, head_type)
    if head_type in ("discard", "swap"):
        return f"{cn} {tile_idx_to_str(action_idx)}"
    if head_type == "missing":
        suit_name = {0: "万", 1: "筒", 2: "条"}.get(action_idx, "?")
        return f"缺 {suit_name}"
    if head_type in ("pong", "kong", "win"):
        return f"{'响应' if action_idx == 1 else '放过'} {cn}"
    return f"{cn}[{action_idx}]"

# ────────────────────────────────────────
# Engine 多态：单个 Engine 或 Dict[int, Engine]
# ────────────────────────────────────────

def _get_engine(engine, pid: int):
    """把 engine 参数统一为 per-pid 查询

    engine 可以是:
      - None                 → 全部玩家随机
      - Engine 实例          → 所有玩家共用
      - Dict[int, Engine]    → 每个玩家独立引擎，dict.get(pid, None) 查
    """
    if engine is None:
        return None
    if isinstance(engine, dict):
        return engine.get(pid, None)
    # 单个 Engine 实例
    return engine
from mjai_engine.arena import Game
from mjai_engine.observation import ObservationEncoder

from mjai.engine import Engine, HEAD_CONFIG


# ────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────

@dataclass
class Transition:
    """单条 DQN 训练样本 + 牌谱元数据"""
    obs: np.ndarray           # [1917]  当前状态
    next_obs: np.ndarray      # [1917]  执行动作后的下一状态（真 DQN Bellman）
    head_type: str             # "discard"/"kong"/...
    action_idx: int            # 对应 head 输出空间
    legal_mask: np.ndarray     # [output_dim]
    reward: float = 0.0        # reward shaping + 终局回填
    done: bool = False
    player_id: int = 0
    turn: int = 0              # 在本局中的 step 序号（用于 reward shaping 时序）
    # ── 牌谱元数据（仅序列化输出，训练不依赖） ──
    action_str: str = ""       # e.g. "弃牌 5m" / "响应 胡"
    phase: str = ""            # "swap" / "missing" / "playing" / "passive_win" ...

    def to_dict(self, include_tensors: bool = False) -> Dict:
        """序列化为 JSON-friendly dict

        Args:
            include_tensors: True 时把 obs/next_obs 也存下来（训练回放用）；
                            默认 False — 牌谱复盘不需要，存下来也巨大
        """
        d: Dict = {
            "player_id": int(self.player_id),
            "turn": round(float(self.turn), 4),   # 防 0.30000000000000004
            "head_type": self.head_type,
            "action_idx": int(self.action_idx),
            "action_str": self.action_str,
            "phase": self.phase,
            "reward": round(float(self.reward), 4),
            "done": bool(self.done),
        }
        if include_tensors:
            d["obs"] = self.obs.tolist()
            d["next_obs"] = self.next_obs.tolist()
        return d


@dataclass
class GameReplay:
    """单局完整 replay

    除 DQN 训练用的 transitions 外，额外保存一局的元信息，便于复盘分析。
    可以序列化为 JSON（默认不含 obs tensor，避免体积膨胀）。
    """
    transitions: List[Transition] = field(default_factory=list)
    final_scores: List[int] = field(default_factory=list)
    steps: int = 0
    is_over: bool = False
    # ── 牌谱级元数据 ──
    game_id: int = 0                      # 训练脚本侧注入唯一 ID
    iteration: int = 0                    # 所在训练迭代
    seed: Optional[int] = None            # 随机种子（复现用，未记录则 None）

    # ────────────────────────────────────────
    # 序列化
    # ────────────────────────────────────────

    def to_dict(self, include_tensors: bool = False) -> Dict:
        return {
            "game_id": self.game_id,
            "iteration": self.iteration,
            "steps": self.steps,
            "is_over": self.is_over,
            "final_scores": [int(s) for s in self.final_scores],
            "seed": self.seed,
            "num_transitions": len(self.transitions),
            "transitions": [
                t.to_dict(include_tensors=include_tensors)
                for t in self.transitions
            ],
        }

    def save(self, path: str, include_tensors: bool = False) -> str:
        """保存为 JSON 文件，返回实际写入路径"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_tensors=include_tensors),
                      f, indent=2, ensure_ascii=False)
        return path


# ────────────────────────────────────────
# 核心函数
# ────────────────────────────────────────

def play_one_game(
    engine: Optional[Engine] = None,
    epsilon: float = 0.0,
    max_turns: int = 300,
    num_players: int = 4,
    use_passive_manual: bool = True,
    use_manual_swap_missing: bool = True,
) -> GameReplay:
    """
    跑一局自博弈

    Args:
        engine:    模型推理引擎；None → 用随机策略
        epsilon:   ε-greedy 探索率（仅当 engine 非 None 时生效）
        max_turns: 单局最大步数（安全上限，防止死循环）
        num_players: 玩家数（固定 4）
        use_passive_manual: True → Rust 弃牌后暂停，让 Python 决策碰/杠/胡
        use_manual_swap_missing: True → Python 手动控制 swap/missing phase
                                   False → 用 Rust 默认 AI（向后兼容）

    Returns:
        GameReplay 实例，含所有 step 样本 + 终局得分
    """
    game = Game(num_players)
    if use_passive_manual:
        game.set_passive_manual(True)
    game.start_game_bare()  # phase = Swap
    encoder = ObservationEncoder()
    replay = GameReplay()
    turn = 0

    # ── Swap Phase ──
    if use_manual_swap_missing:
        _run_swap_phase(game, encoder, replay, engine, epsilon, turn)
    else:
        game.process_swap_phase()  # Rust 默认 AI → phase = Missing

    # ── Missing Phase ──
    if use_manual_swap_missing:
        _run_missing_phase(game, encoder, replay, engine, epsilon, turn)
    else:
        game.process_missing_phase()  # Rust 默认 AI → phase = Playing

    # ── Playing Phase ──
    while not game.is_over() and turn < max_turns:
        pid = game.current_player()

        # 1. 记录当前观测
        obs_flat = np.array(encoder.encode(game.board, pid), dtype=np.float32)

        # 2. 获取合法动作
        legal = game.get_legal_actions(pid)

        # 3. 选择动作 — per-pid engine 路由
        _pid_eng = _get_engine(engine, pid)
        head_type, action_idx, tile_idx = _select_action_wrapper(
            _pid_eng, legal, epsilon, obs_flat=obs_flat
        )

        # 4. 执行主动动作
        action_type, tile_index = _map_to_engine_action(head_type, action_idx, legal)
        game.step_with_action(pid, action_type, tile_index)

        # 4b. 被动响应处理（manual 模式下 step_with_action 暂停在弃牌后）
        if use_passive_manual and not game.is_over():
            _handle_passive_responses(
                game, encoder, replay, engine, epsilon, turn
            )

        # 5. 编码 next_obs（下一玩家视角）
        is_done = game.is_over()
        if is_done:
            next_obs_flat = np.zeros(1917, dtype=np.float32)
        else:
            next_pid = game.current_player()
            next_obs_flat = np.array(
                encoder.encode(game.board, next_pid), dtype=np.float32
            )

        # 6. 记录 transition
        output_dim = HEAD_CONFIG[head_type]["output_dim"]
        legal_mask = Engine._build_legal_mask(legal, head_type, output_dim)

        # Reward shaping: 杠 +2, 缺门清理 +0.1 per tile（终局 reward 后续回填）
        shape_reward = 0.0
        if head_type == "kong":
            shape_reward = 2.0

        replay.transitions.append(Transition(
            obs=obs_flat,
            next_obs=next_obs_flat,
            head_type=head_type,
            action_idx=action_idx,
            legal_mask=legal_mask,
            reward=shape_reward,
            done=is_done,
            player_id=pid,
            turn=turn,
            action_str=head_action_str(head_type, action_idx),
            phase="playing",
        ))

        turn += 1

    # 6. 终局回填 reward
    replay.final_scores = [game.board.player_score(i) for i in range(num_players)]
    replay.is_over = game.is_over()
    replay.steps = turn

    _assign_terminal_rewards(replay)

    return replay


def play_n_games(
    n: int,
    engine: Optional[Engine] = None,
    epsilon: float = 0.0,
    max_turns: int = 300,
    verbose: bool = True,
) -> List[GameReplay]:
    """批量跑 n 局自博弈，带进度可观测"""
    replays = []
    t0 = time.time()

    for i in range(n):
        r = play_one_game(engine=engine, epsilon=epsilon, max_turns=max_turns)
        replays.append(r)
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{n}] rate={rate:.1f} games/s  "
                  f"avg_transitions={np.mean([len(x.transitions) for x in replays[-10:]]):.0f}  "
                  f"over={sum(1 for x in replays[-10:] if x.is_over)}/10")

    return replays


# ────────────────────────────────────────
# 内部辅助
# ────────────────────────────────────────

# ────────────────────────────────────────
# Swap / Missing Phase 手动控制
# ────────────────────────────────────────

# Suit index → tile index 范围
_SUIT_RANGES = {
    0: (0, 9),    # Man  : tile 0-8
    1: (9, 18),   # Pin  : tile 9-17
    2: (18, 27),  # Sou  : tile 18-26
}


def _suit_of_tile(tile_idx: int) -> int:
    """tile index → suit index (0/1/2)"""
    return tile_idx // 9


def _tiles_in_suit(hand_tiles: List[int], suit: int) -> List[int]:
    """从 hand tiles 列表里选某花色的所有 tile indices"""
    lo, hi = _SUIT_RANGES[suit]
    return [t for t in hand_tiles if lo <= t < hi]


def _build_swap_legal_mask(hand_tiles: List[int]) -> np.ndarray:
    """
    Swap head legal_mask [27]: tile 属于某花色 ≥3 张时才合法
    （需要从该花色换 3 张同花色的牌）
    """
    suit_ok = [len(_tiles_in_suit(hand_tiles, s)) >= 3 for s in range(3)]
    mask = np.zeros(27, dtype=np.float32)
    if not any(suit_ok):
        # 没有任何花色 ≥3 张 → 全部合法（Rust 会用默认 AI）
        mask[:] = 1.0
        return mask
    for t in hand_tiles:
        suit = _suit_of_tile(t)
        if suit_ok[suit]:
            mask[t] = 1.0
    return mask


def _build_missing_legal_mask() -> np.ndarray:
    """Missing head legal_mask [3]: 花色不可能都 0 张，全部合法"""
    return np.ones(3, dtype=np.float32)


def _reward_swap(
    game: Game, pid: int, shanten_before: int,
) -> Tuple[float, np.ndarray]:
    """
    Swap reward shaping：
      reward = (shanten_before - shanten_after_swap) × 0.3

    Returns:
        (reward, next_obs_after_swap)
    """
    sh_after = game.player_shanten(pid)
    delta = shanten_before - sh_after  # 正数=改善
    reward = float(delta) * 0.3
    return reward, sh_after


def _count_missing_leftover(game: Game, pid: int) -> int:
    """missing phase 后，该玩家手牌里剩余多少张"应该换掉"的缺门牌"""
    missing_suit = game.player_missing_suit(pid)
    if missing_suit < 0:
        return 0
    lo, hi = _SUIT_RANGES[missing_suit]
    hand_tiles = game.player_hand_tiles(pid)
    return sum(1 for t in hand_tiles if lo <= t < hi)


def _run_swap_phase(
    game: Game,
    encoder: ObservationEncoder,
    replay: GameReplay,
    engine: Optional[Engine],
    epsilon: float,
    turn: int,
) -> None:
    """
    Python 手动控制 Swap phase — tile-level 选择 + reward shaping

    升级点：
      - output_dim 27：model 直接选"哪张 tile"作为换牌锚点
      - legal_mask [27]：只标记"花色≥3张"的那些 tile 为合法
      - Python 从选中 tile 的花色抽 3 张（含所选 tile）作为实际换牌
      - reward = (shanten_before - shanten_after) × 0.3
      - next_obs = swap 后 encoder 重编码（正确 Bellman）
    """
    num_players = 4

    # ── Phase 1: 收集所有玩家的 obs + 选动作 + 记录 swap 前 shanten ──
    player_data = []  # [(pid, obs_before, shanten_before, action_idx, picked_3_tiles)]
    selections: List[Optional[List[int]]] = []

    for pid in range(num_players):
        hand_tiles = game.player_hand_tiles(pid)
        obs_flat = np.array(
            encoder.encode(game.board, pid), dtype=np.float32
        )
        legal_mask = _build_swap_legal_mask(hand_tiles)
        sh_before = game.player_shanten(pid)

        # model 选 tile_idx (0-26) — per-pid engine 路由
        _pid_eng = _get_engine(engine, pid)
        if _pid_eng is not None and np.random.random() >= epsilon:
            try:
                action_idx, _ = _pid_eng.select_action(
                    obs_flat, {}, "swap", epsilon=0.0
                )
            except Exception:
                # fallback: 从合法 tile 里随机选一个
                valid_tiles = np.where(legal_mask > 0)[0]
                action_idx = int(np.random.choice(valid_tiles)) if len(valid_tiles) > 0 else 0
        else:
            valid_tiles = np.where(legal_mask > 0)[0]
            action_idx = int(np.random.choice(valid_tiles)) if len(valid_tiles) > 0 else 0

        # 从选中 tile 所在花色抽 3 张（含所选 tile）
        suit = _suit_of_tile(action_idx)
        suit_tiles = _tiles_in_suit(hand_tiles, suit)
        if len(suit_tiles) >= 3:
            # 优先包含 model 选的 tile（如果它在合法花色里）
            if action_idx in suit_tiles:
                remaining = [t for t in suit_tiles if t != action_idx]
                if len(remaining) >= 2:
                    picked_rest = list(np.random.choice(remaining, size=2, replace=False))
                else:
                    # edge case: 花色刚好 3 张且都是同一张 → 复用
                    picked_rest = remaining[:]
                picked = [action_idx] + picked_rest
            else:
                picked = list(np.random.choice(suit_tiles, size=3, replace=False))
        else:
            picked = None  # fallback: Rust AI

        selections.append(picked)
        player_data.append((pid, obs_flat, sh_before, action_idx, legal_mask))

    # ── Phase 2: 执行 swap ──
    game.process_swap_phase_with(selections)
    # phase = Missing

    # ── Phase 3: 重编码 next_obs + 算 reward + 记录 transition ──
    for (pid, obs_before, sh_before, action_idx, legal_mask) in player_data:
        sh_after = game.player_shanten(pid)
        reward = float(sh_before - sh_after) * 0.3
        next_obs = np.array(
            encoder.encode(game.board, pid), dtype=np.float32
        )

        replay.transitions.append(Transition(
            obs=obs_before,
            next_obs=next_obs,
            head_type="swap",
            action_idx=action_idx,
            legal_mask=legal_mask,
            reward=reward,
            done=False,
            player_id=pid,
            turn=turn + pid * 0.1,
            action_str=head_action_str("swap", action_idx),
            phase="swap",
        ))


def _run_missing_phase(
    game: Game,
    encoder: ObservationEncoder,
    replay: GameReplay,
    engine: Optional[Engine],
    epsilon: float,
    turn: int,
) -> None:
    """
    Python 手动控制 Missing phase + reward shaping

    reward shaping:
      - 清理干净 (0 张缺门牌残留): +0.3
      - 残留 1 张: -0.1
      - 残留 2 张: -0.2
      ...

    next_obs: missing 后 encoder 重编码（正确 Bellman）
    """
    num_players = 4

    # Phase 1: 收集 obs + 选动作
    player_data = []  # [(pid, obs_before, action_idx, legal_mask)]
    selections: List[Optional[int]] = []

    for pid in range(num_players):
        obs_flat = np.array(
            encoder.encode(game.board, pid), dtype=np.float32
        )
        legal_mask = _build_missing_legal_mask()

        # model 选 missing suit — per-pid engine 路由
        _pid_eng = _get_engine(engine, pid)
        if _pid_eng is not None and np.random.random() >= epsilon:
            try:
                action_idx, _ = _pid_eng.select_action(
                    obs_flat, {}, "missing", epsilon=0.0
                )
            except Exception:
                action_idx = int(np.random.choice(3))
        else:
            action_idx = int(np.random.choice(3))

        selections.append(action_idx)
        player_data.append((pid, obs_flat, action_idx, legal_mask))

    # Phase 2: 执行
    game.process_missing_phase_with(selections)
    # phase = Playing

    # Phase 3: 算 reward + 重编码 next_obs
    for (pid, obs_before, action_idx, legal_mask) in player_data:
        leftover = _count_missing_leftover(game, pid)
        reward = 0.3 - 0.1 * leftover  # 干净=+0.3, 1残留=+0.2, 3残留=0
        reward = max(reward, 0.0)  # 最低 0（别用负 reward 干扰 Bellman）

        next_obs = np.array(
            encoder.encode(game.board, pid), dtype=np.float32
        )

        replay.transitions.append(Transition(
            obs=obs_before,
            next_obs=next_obs,
            head_type="missing",
            action_idx=action_idx,
            legal_mask=legal_mask,
            reward=reward,
            done=False,
            player_id=pid,
            turn=turn + pid * 0.1 + 0.05,
            action_str=head_action_str("missing", action_idx),
            phase="missing",
        ))


def _select_action_wrapper(
    engine: Optional[Engine],
    legal: Dict,
    epsilon: float,
    obs_flat: Optional[np.ndarray] = None,
) -> Tuple[str, int, int]:
    """
    选动作：优先 discard（phase 决定）

    Args:
        obs_flat: 真实 encoder 输出（用于 model 推理）；None → 随机策略

    Returns:
        (head_type, action_idx, tile_idx_for_log)
    """
    discard_list = legal.get("discard", [])
    kong_list = legal.get("kong", [])

    if engine is not None and np.random.random() >= epsilon:
        # 模型推理 — 用真实 obs
        assert obs_flat is not None, "engine 模式下必须传 obs_flat"
        # kong head 仅在有可杠时触发（0.2 概率让模型尝试杠）
        if kong_list and np.random.random() < 0.2:
            head_type = "kong"
            action_idx, _ = engine.select_action(
                obs_flat, legal, head_type, epsilon=0.0
            )
        else:
            head_type = "discard"
            action_idx, _ = engine.select_action(
                obs_flat, legal, head_type, epsilon=0.0
            )
        # tile_idx_for_log: 只用于日志，实际引擎动作映射用 _map_to_engine_action
        tile_idx = discard_list[0] if discard_list else 0
    else:
        # 随机策略
        if kong_list and np.random.random() < 0.1:
            head_type = "kong"
            action_idx = 1  # kong head: 1=是
            tile_idx = kong_list[0]
        elif discard_list:
            head_type = "discard"
            tile_idx = int(np.random.choice(discard_list))
            action_idx = tile_idx
        else:
            head_type = "discard"
            action_idx = 0
            tile_idx = 0

    return head_type, action_idx, tile_idx


def _map_to_engine_action(
    head_type: str,
    action_idx: int,
    legal: Dict,
) -> Tuple[str, int]:
    """
    把 (head_type, action_idx) → (engine_action_type, tile_index)

    Discard head: action_idx 直接是 tile_index (0..26)
    Kong head:    action_idx=0 → "pass", action_idx=1 → 选 legal["kong"][0]
    其他头:       映射为 discard 兜底（当前 phase 不适用）
    """
    if head_type == "discard":
        tile_idx = int(action_idx)
        # 验证在合法 discard 里（legal_mask 已屏蔽非法）
        legal_discard = legal.get("discard", [])
        if tile_idx not in legal_discard and legal_discard:
            tile_idx = legal_discard[0]
        return "discard", tile_idx

    elif head_type == "kong":
        if action_idx == 1:  # 选杠
            kong_list = legal.get("kong", [])
            if kong_list:
                return "kong", kong_list[0]
        return "pass", 0

    else:
        # 当前 phase 下的被动响应（pong/win/swap/missing）暂不处理
        # 这些会在引擎 check_responses 里由简单 heuristic 处理
        legal_discard = legal.get("discard", [])
        if legal_discard:
            return "discard", legal_discard[0]
        return "pass", 0


def _assign_terminal_rewards(replay: GameReplay) -> None:
    """
    终局 reward 分配（按玩家）：
      - 杠等已在 play_one_game 里做 reward shaping
      - 给每个玩家自己的最后一个 transition 叠加该玩家的终局得分
      - 每个玩家的最后一步 transition.done = True
      - 这样 Bellman target = r_shaped + r_final + γ × max Q(s', a')
        避免单一稀疏 reward
    """
    if not replay.transitions:
        return

    # 按 player_id 找每个玩家的最后一个 transition
    last_by_player: Dict[int, int] = {}  # pid → transition index
    for idx, t in enumerate(replay.transitions):
        last_by_player[t.player_id] = idx

    # 叠加终局分数 + 标记 done
    for pid, idx in last_by_player.items():
        if pid < len(replay.final_scores):
            replay.transitions[idx].reward += float(replay.final_scores[pid])
        replay.transitions[idx].done = True

    # 全局 done（本局结束）
    if replay.transitions:
        replay.transitions[-1].done = True


def _dummy_obs() -> np.ndarray:
    """仅当 engine=None 时不会调用到。占个位置。"""
    return np.zeros(1917, dtype=np.float32)


def generate_synthetic_transitions(
    n_per_head: int = 200,
    head_overrides: Optional[Dict[str, int]] = None,
) -> List[Transition]:
    """
    生成 synthetic transitions 填充低频头 buffer。

    Swap/Missing 已有真实 self-play 数据（每局 4 条），默认大幅降低；
    Discard 全靠真实数据，synthetic 为 0；
    Pong/Kong/Win 仍靠 synthetic 保持冷启动覆盖率。

    Args:
        n_per_head:     默认每个头生成多少条
        head_overrides: 单独覆盖某些头的数量（head_name → count）
                        显式设为 0 可以关闭某个头的 synthetic

    Returns:
        Transition 列表
    """
    import random as _r

    HEAD_OUTPUT_DIMS = {
        "swap": 27, "missing": 3, "discard": 27,
        "pong": 2, "kong": 2, "win": 2,
    }

    # swap/missing 有真实数据 → 降到默认值的 10%；discard 从不 synthetic
    overrides = head_overrides or {}
    if "swap" not in overrides:
        overrides["swap"] = max(20, n_per_head // 10)
    if "missing" not in overrides:
        overrides["missing"] = max(20, n_per_head // 10)
    if "discard" not in overrides:
        overrides["discard"] = 0

    transitions = []

    for head_name, output_dim in HEAD_OUTPUT_DIMS.items():
        n = overrides.get(head_name, n_per_head)
        if n <= 0:
            continue
        for _ in range(n):
            # 随机 obs：用 0-1 均匀（encoder 输出大多是二值 one-hot）
            obs = np.random.rand(1917).astype(np.float32)
            next_obs = np.random.rand(1917).astype(np.float32)

            # legal_mask：全合法（开局阶段全部花色可选；被动响应 pass/respond 都可选）
            legal_mask = np.ones(output_dim, dtype=np.float32)

            transitions.append(Transition(
                obs=obs,
                next_obs=next_obs,
                head_type=head_name,
                action_idx=_r.randrange(output_dim),
                legal_mask=legal_mask,
                reward=0.0,
                done=False,
                player_id=_r.randrange(4),
                turn=0,
            ))

    return transitions


# ────────────────────────────────────────
# 被动响应处理（Pong/Kong/Win）
# ────────────────────────────────────────

_PASSIVE_HEAD_NAME_MAP = {
    "win": "win",       # response type → DQN head
    "kong": "kong",
    "pong": "pong",
}


def _build_passive_legal_mask(
    head_type: str, can_respond: bool
) -> np.ndarray:
    """构造被动响应 head 的 legal_mask"""
    output_dim = HEAD_CONFIG[head_type]["output_dim"]  # 都是 2
    mask = np.zeros(output_dim, dtype=np.float32)
    mask[0] = 1.0  # pass 永远合法
    if can_respond:
        mask[1] = 1.0  # respond（胡/杠/碰）
    return mask


def _handle_passive_responses(
    game: Game,
    encoder: ObservationEncoder,
    replay: GameReplay,
    engine: Optional[Engine],
    epsilon: float,
    turn: int,
) -> None:
    """
    弃牌后处理被动响应机会（胡 > 杠 > 碰）

    为每个有响应机会的玩家：
    1. 用该玩家视角编码 obs
    2. model 选 pass/respond（对应 head 的 action_idx）
    3. 执行 respond 或 pass
    4. 记录 transition 到 replay

    最后 skip_all_passive → advance_turn
    """
    pending = game.get_pending_passive()
    discarder = pending["discarder"]

    if discarder is None:
        # 没有被动机会，直接推进
        game.advance_turn()
        return

    # 按优先级排序响应者
    priority_order = [
        ("win", pending["ron_winners"]),
        ("kong", pending["kong_responders"]),
        ("pong", pending["pong_responders"]),
    ]

    for response_type, responders in priority_order:
        if not responders:
            continue
        head_name = _PASSIVE_HEAD_NAME_MAP[response_type]
        output_dim = HEAD_CONFIG[head_name]["output_dim"]

        for responder in responders:
            if game.is_over():
                break

            # responder 视角编码
            obs_flat = np.array(
                encoder.encode(game.board, responder), dtype=np.float32
            )

            # 构造 legal dict + mask
            can_flag_key = f"can_{response_type}"
            legal = {can_flag_key: True}
            legal_mask = _build_passive_legal_mask(head_name, True)

            # model 选 action — per-responder engine 路由
            _resp_eng = _get_engine(engine, responder)
            if _resp_eng is not None:
                try:
                    action_idx, _ = _resp_eng.select_action(
                        obs_flat, legal, head_name, epsilon=epsilon
                    )
                except Exception:
                    action_idx = 0
            else:
                action_idx = 0 if np.random.random() < 0.7 else 1

            # 编码 next_obs
            is_done = game.is_over()
            if is_done:
                next_obs_flat = np.zeros(1917, dtype=np.float32)
            else:
                # 响应后状态可能变了——重编码（但要在执行后）
                next_obs_flat = obs_flat.copy()  # 先占位

            # Reward shaping: 胡 +5, 杠 +2, 碰 +1
            shape_reward = 0.0
            if action_idx == 1:
                if response_type == "win":
                    shape_reward = 5.0
                elif response_type == "kong":
                    shape_reward = 2.0
                elif response_type == "pong":
                    shape_reward = 1.0

            # 记录 transition（执行前）
            replay.transitions.append(Transition(
                obs=obs_flat,
                next_obs=next_obs_flat,  # placeholder，实际在执行后更新
                head_type=head_name,
                action_idx=action_idx,
                legal_mask=legal_mask,
                reward=shape_reward,
                done=False,
                player_id=responder,
                turn=turn + 0.5,  # 小数表示"主动弃牌后"
                action_str=head_action_str(head_name, action_idx),
                phase=f"passive_{response_type}",
            ))

            # 执行响应
            if action_idx == 1:
                executed = game.execute_passive_response(responder, response_type)
                if executed:
                    # 更新 next_obs（执行后的 responder 视角）
                    last = replay.transitions[-1]
                    if not game.is_over():
                        last.next_obs = np.array(
                            encoder.encode(game.board, responder), dtype=np.float32
                        )
                    else:
                        last.next_obs = np.zeros(1917, dtype=np.float32)
                    last.done = game.is_over()
                    # 胡最高优先级——胡了之后 skip 其他响应者
                    if response_type == "win":
                        break
                # 没执行（可能已被高优先级处理），继续下一个

    # 清剩余 + 推进
    if not game.is_over():
        game.skip_all_passive()
        game.advance_turn()
