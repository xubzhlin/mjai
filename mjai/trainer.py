"""
Trainer — DQN 训练主循环（冻结结构，一次只改一个变量）

主链路：
  for step in range(total_steps):
      1. if step % self_play_freq == 0:
             replays = self_play(model, epsilon)
             buffer.add_replays(replays)
      2. batch = buffer.sample(batch_size)
      3. q_pred = model.forward(batch.obs)[head][action_idx]
      4. target = reward + gamma * max Q_target(next_obs) * (1-done)
      5. loss = MSE(q_pred, target)  +  0.01 * Entropy
      6. loss.backward(); optimizer.step()
      7. if step % target_update_freq == 0: target_net.load_state(model)
      8. 每 N step 打印 loss 分项统计（三级诊断点）

遵循经验 100008214 的冻结主链路原则：
  - 先固定 self_play + buffer + loss 计算 + backward + step 这条线跑通
  - 再加 GRP 辅助头、经验回放优先级等
"""
from __future__ import annotations

import os
import time
import json
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mjai.models import ResNetDQN, ModelConfig, GRP, GRPConfig
from mjai.engine import Engine, HEAD_CONFIG
from mjai.selfplay import play_n_games, GameReplay, generate_synthetic_transitions
from mjai.buffer_v2 import ReplayBuffer, Batch, HEAD_TYPES


class TrainerConfig:
    """训练超参（可从 configs/train.toml 加载）"""

    def __init__(
        self,
        # 自博弈
        num_games_per_iter: int = 50,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,

        # 训练
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        grad_clip: float = 1.0,

        # Target network
        target_update_freq: int = 500,
        tau: float = 0.0,  # 0 = 硬更新；>0 软更新系数

        # ReplayBuffer
        buffer_capacity: int = 100_000,
        warmup_games: int = 200,  # 启动前用 random policy 填 buffer

        # Loss 权重
        entropy_weight: float = 0.01,
        grp_weight: float = 0.1,
        use_grp: bool = False,  # Sub-C 先不开 GRP

        # Checkpoint
        checkpoint_freq: int = 2000,
        eval_freq: int = 500,
        log_freq: int = 50,

        # 设备
        device: str = "cpu",
    ):
        self.num_games_per_iter = num_games_per_iter
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.buffer_capacity = buffer_capacity
        self.warmup_games = warmup_games
        self.entropy_weight = entropy_weight
        self.grp_weight = grp_weight
        self.use_grp = use_grp
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.device = device

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls().__dict__})


class Trainer:
    """DQN 训练器（冻结主链路版）"""

    def __init__(
        self,
        model: ResNetDQN,
        config: TrainerConfig,
        model_config: Optional[ModelConfig] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # 主网络 + Target 网络
        self.model = model.to(self.device)
        self.target_model = self._clone_model(model).to(self.device)
        self.target_model.eval()

        # Buffer
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

        # GRP（可选，Sub-C 先不开）
        self.grp: Optional[GRP] = None
        if config.use_grp:
            self.grp = GRP(GRPConfig(input_dim=1024, hidden_dim=64, num_layers=2))

        # 状态
        self.global_step = 0
        self.epsilon = config.epsilon_start

        # 损失滑动窗口（用于日志）
        self._loss_window: List[float] = []
        self._loss_head_windows: Dict[str, List[float]] = defaultdict(list)

    # ──────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────

    def train(
        self,
        total_iterations: int = 100,
        output_dir: str = "checkpoints",
        max_steps: int = 10_000,
    ):
        """
        Args:
            total_iterations: 自博弈迭代次数（每迭代采 N 局）
            output_dir: checkpoint 输出目录
            max_steps:  最大 DQN 更新步数（安全上限）
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Start training: {total_iterations} iters, max {max_steps} steps")
        print(f"Config: epsilon_start={self.config.epsilon_start}, "
              f"gamma={self.config.gamma}, batch={self.config.batch_size}, "
              f"lr={self.config.learning_rate}")

        # === Warmup ===
        self._warmup()

        # === 训练迭代 ===
        engine = Engine(self.model, device=str(self.device))

        for iteration in range(1, total_iterations + 1):
            # 1. Self-play 采样
            replays = play_n_games(
                self.config.num_games_per_iter,
                engine=engine,
                epsilon=self.epsilon,
                verbose=False,
            )
            for r in replays:
                self.buffer.add_replay(r)

            # 2. DQN 更新若干步
            n_update_steps = min(20, self.config.num_games_per_iter)
            for _ in range(n_update_steps):
                if self.global_step >= max_steps:
                    print(f"Reached max_steps={max_steps}, stopping")
                    return
                self._train_step()

            # 3. 衰减 epsilon
            self._decay_epsilon()

            # 4. 日志
            if iteration % 5 == 0:
                self._log_progress(iteration, replays)

            # 5. Checkpoint
            if iteration % (self.config.checkpoint_freq // self.config.num_games_per_iter + 1) == 0:
                self._save_checkpoint(output_dir, iteration)

        print(f"\nTraining done. {self.global_step} DQN steps.")

    # ──────────────────────────────────────────
    # 核心 step
    # ──────────────────────────────────────────

    def _train_step(self):
        """单次 DQN 更新（冻结结构，批量优化版）"""
        batch = self.buffer.sample(self.config.batch_size)
        if batch is None or batch.size < 4:
            return

        self.model.train()
        self.target_model.eval()

        # ── 打包 tensor ──
        obs = torch.FloatTensor(batch.obs).to(self.device)
        next_obs = torch.FloatTensor(batch.next_obs).to(self.device)
        action_idx = torch.LongTensor(batch.action_idx).to(self.device)
        reward = torch.FloatTensor(batch.reward).to(self.device)
        done = torch.FloatTensor(batch.done).to(self.device)
        head_idx = torch.LongTensor(batch.head_type).to(self.device)

        # ── 一次性 batch forward（优化: 4.8x faster than per-head forward）──
        q_dict = self.model.forward(obs)  # {head: [B, dim]}

        # Target: 一次 forward target_model
        with torch.no_grad():
            next_q_dict = self.target_model.forward(next_obs)

        # ── 按 head 切片计算 Q(s,a) + Bellman target ──
        q_pred_list = []
        target_list = []

        for head_name, head_idx_val in zip(HEAD_TYPES, range(len(HEAD_TYPES))):
            mask = (head_idx == head_idx_val)
            if not mask.any():
                continue

            head_dim_val = HEAD_CONFIG[head_name]["output_dim"]

            # Q(s, a) — 从已有的 q_dict 里切片（不用再 forward）
            q_head = q_dict[head_name][mask, :head_dim_val]
            actions_head = action_idx[mask]
            q_s_a = q_head.gather(1, actions_head.unsqueeze(1)).squeeze(1)

            # Bellman target — 从已有的 next_q_dict 里切片
            with torch.no_grad():
                next_q_head = next_q_dict[head_name][mask, :head_dim_val]
                max_next_q = next_q_head.max(dim=1)[0]
                target = reward[mask] + self.config.gamma * max_next_q * (1 - done[mask])

            q_pred_list.append(q_s_a)
            target_list.append(target)

            # 分项 loss（日志用）
            head_loss = F.mse_loss(q_s_a.detach(), target)
            self._loss_head_windows[head_name].append(head_loss.item())

        if not q_pred_list:
            return

        q_pred = torch.cat(q_pred_list)
        target = torch.cat(target_list)

        # ── DQN Loss ──
        dqn_loss = F.mse_loss(q_pred, target)

        # ── Entropy bonus（可选） ──
        entropy_loss = torch.tensor(0.0, device=self.device)
        if self.config.entropy_weight > 0:
            total_entropy = torch.tensor(0.0, device=self.device)
            n_heads_with_data = 0
            for head_name, head_idx_val in zip(HEAD_TYPES, range(len(HEAD_TYPES))):
                mask = (head_idx == head_idx_val)
                if not mask.any():
                    continue
                head_dim_val = HEAD_CONFIG[head_name]["output_dim"]
                q_head = q_dict[head_name][mask, :head_dim_val]
                tau = 1.0
                log_probs = F.log_softmax(q_head / tau, dim=1)
                probs = log_probs.exp()
                head_entropy = -(probs * log_probs).sum(dim=1).mean()
                total_entropy = total_entropy + head_entropy
                n_heads_with_data += 1
            if n_heads_with_data > 0:
                entropy_loss = -total_entropy / n_heads_with_data

        total_loss = dqn_loss + self.config.entropy_weight * entropy_loss

        # ── 反向传播 ──
        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )

        self.optimizer.step()

        # ── Target update ──
        self.global_step += 1
        if self.global_step % self.config.target_update_freq == 0:
            self._update_target_network()

        # ── 日志 ──
        self._loss_window.append(dqn_loss.item())

        if self.global_step % self.config.log_freq == 0:
            self._diag_output("dqn_loss", dqn_loss)
            self._diag_output("grad_norm", grad_norm)
            self._log_step()

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def _warmup(self):
        """用 random policy + synthetic transitions 填 buffer"""
        print(f"Warming up buffer with {self.config.warmup_games} random games...")
        t0 = time.time()
        replays = play_n_games(
            self.config.warmup_games, engine=None, verbose=False
        )
        for r in replays:
            self.buffer.add_replay(r)

        # 填充 synthetic transitions 到低频头 buffer
        n_synth = 200
        print(f"  Adding {n_synth} synthetic transitions per head (swap/missing/pong/win)...")
        synth = generate_synthetic_transitions(n_per_head=n_synth)
        self.buffer.add_many(synth)

        print(f"  Done in {time.time() - t0:.2f}s. {self.buffer.summary()}")

    def _decay_epsilon(self):
        cfg = self.config
        steps = self.global_step
        if steps < cfg.epsilon_decay_steps:
            frac = steps / cfg.epsilon_decay_steps
            self.epsilon = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac
        else:
            self.epsilon = cfg.epsilon_end

    def _update_target_network(self):
        if self.config.tau > 0:
            # 软更新
            for tp, mp in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.mul_(1 - self.config.tau).add_(self.config.tau * mp.data)
        else:
            # 硬更新
            self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def _clone_model(model: ResNetDQN) -> ResNetDQN:
        import copy
        return copy.deepcopy(model)

    # ──────────────────────────────────────────
    # 三级诊断（经验 100008214）
    # ──────────────────────────────────────────

    @staticmethod
    def _diag_input(name: str, tensor: torch.Tensor):
        with torch.no_grad():
            flat = tensor.detach().float().cpu()
            nan = flat.isnan().sum().item()
            inf = flat.isinf().sum().item()
            if nan > 0 or inf > 0:
                print(f"  ⚠️ [DIAG] {name}: nan={nan}, inf={inf}, "
                      f"min={flat.min():.4f}, max={flat.max():.4f}")

    @staticmethod
    def _diag_output(name: str, value):
        if isinstance(value, torch.Tensor):
            v = value.detach().float().cpu().item()
            nan = value.isnan().any().item() if value.dim() > 0 else value.isnan().item()
            if nan or abs(v) > 100:
                print(f"  ⚠️ [DIAG] {name}={v:.4f}")

    def _log_step(self):
        window = self._loss_window[-self.config.log_freq:]
        avg_loss = float(np.mean(window)) if window else 0.0

        # 每头 loss
        heads_str = ", ".join(
            f"{h}={float(np.mean(ws)):.3f}" for h, ws in self._loss_head_windows.items()
            if ws
        )

        total = self.buffer.size
        epsilon_str = f"{self.epsilon:.3f}"

        print(f"  step={self.global_step:5d}  loss={avg_loss:.4f}  "
              f"ε={epsilon_str}  buffer={total}  [{heads_str}]")

    def _log_progress(self, iteration: int, replays: List[GameReplay]):
        n_over = sum(1 for r in replays if r.is_over)
        avg_trans = np.mean([len(r.transitions) for r in replays]) if replays else 0
        print(f"iter={iteration:3d}  games={len(replays)}  "
              f"over={n_over}/{len(replays)}  avg_trans={avg_trans:.0f}  "
              f"buffer={self.buffer.size}  ε={self.epsilon:.3f}")

    def _save_checkpoint(self, output_dir: str, iteration: int):
        path = os.path.join(output_dir, f"iter_{iteration:04d}.pt")
        torch.save({
            "iteration": iteration,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "target_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "config": self.config.to_dict(),
        }, path)
        print(f"  💾 Checkpoint → {path}")

