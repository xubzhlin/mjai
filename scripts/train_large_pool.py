"""大规模训练脚本 — OpponentPool + 可恢复训练 + 集中落盘

目录结构（牌谱和模型按 iter 集中存放，方便问题排查）:
  checkpoints_opppool5000/
  ├── warmup/                  ← warmup 30 局牌谱
  ├── iter_0001/               ← 每轮一个目录
  │   └── game_00031.json
  ├── iter_0005/
  │   ├── iter_0005.pt         ← checkpoint（含模型+优化器+OpponentPool+game_counter）
  │   └── game_00041.json
  ├── iter_interrupt/
  │   └── iter_interrupt.pt    ← Ctrl+C 中断自动存
  └── model_final.pt           ← 训练完成最终权重

恢复逻辑:
  启动时 glob **/iter_*.pt 找最新 checkpoint + 扫描磁盘最大 game_id，
  两者独立判断——不管 checkpoint 有没有丢，牌谱编号都不重名覆盖。
"""
import os, sys, time, glob, torch, numpy as np
sys.path.insert(0, ".")
os.environ["OMP_NUM_THREADS"] = "1"

from mjai.models import ResNetDQN, ModelConfig
from mjai.trainer import Trainer, TrainerConfig
from mjai.engine import Engine
from mjai.selfplay import play_one_game
from mjai.opponent_pool import OpponentPool, OpponentPoolConfig

# ========== 输出根目录（一切产物都在这里） ==========
OUT = "checkpoints_opppool5000"
os.makedirs(OUT, exist_ok=True)

# ========== 关键配置 ==========
OPPONENT_MIX_PROB = 0.5     # 50% pool 对手 / 50% 自身 (让池真的被用到)
POOL_SIZE = 8
ADMISSION_WR = 0.20
TOTAL_ITER = 500
GAMES_PER_ITER = 10
N_UPDATE = 20
MAX_STEPS = 5000
SNAPSHOT_INTERVAL = 50
CHECKPOINT_INTERVAL = 5     # 每 5 iter 存一次 checkpoint

# ========== 构建组件 ==========
model_cfg = ModelConfig(input_channels=71, feature_dim=1024)

def _make_empty_model(model_id: str = "") -> ResNetDQN:
    return ResNetDQN(model_cfg)

model = ResNetDQN(model_cfg)

cfg = TrainerConfig(
    num_games_per_iter=GAMES_PER_ITER,
    epsilon_start=0.5, epsilon_end=0.05,
    epsilon_decay_steps=800, learning_rate=1e-4, gamma=0.99,
    batch_size=128, warmup_games=30, device="cpu", log_freq=50,
    target_update_freq=200, checkpoint_freq=CHECKPOINT_INTERVAL, buffer_capacity=100000,
)

trainer = Trainer(model, cfg, model_cfg)

pool_cfg = OpponentPoolConfig(
    max_size=POOL_SIZE,
    admission_win_rate=ADMISSION_WR,
    retire_win_rate=0.05,
    snapshot_interval=SNAPSHOT_INTERVAL,
    sample_strategy="uniform",
)
pool = OpponentPool(pool_cfg)

# ========== 路径辅助 ==========
def _iter_dir(iteration: int) -> str:
    """某 iter 的牌谱 + checkpoint 目录"""
    return os.path.join(OUT, f"iter_{iteration:04d}")

def _ckpt_path(iteration: int) -> str:
    """某 iter 的 checkpoint 文件路径"""
    return os.path.join(_iter_dir(iteration), f"iter_{iteration:04d}.pt")

def _warmup_dir() -> str:
    return os.path.join(OUT, "warmup")

# ========== 工具: 找最新 checkpoint / 最大 game_id ==========
def _find_latest_checkpoint(out_dir: str) -> str | None:
    """递归找 **/iter_*.pt，返回最新的那个。

    规则:
      - 正式 checkpoint (iter_0005.pt) 按 iter 号比大小
      - iter_interrupt.pt 永远比正式 checkpoint 新（按 mtime）
      - 其他匹配不上的返回 None
    """
    files = glob.glob(os.path.join(out_dir, "**", "iter_*.pt"), recursive=True)
    if not files:
        return None

    def _key(path: str):
        base = os.path.basename(path)              # iter_0005.pt 或 iter_interrupt.pt
        stem = base.replace("iter_", "").replace(".pt", "")
        if stem == "interrupt":
            return (1, os.path.getmtime(path))     # interrupt 永远排最后
        try:
            return (0, int(stem))                  # 正式 checkpoint 按 iter 号
        except ValueError:
            return (-1, 0)

    return max(files, key=_key)


def _find_max_game_id(out_dir: str) -> int:
    """递归扫描 **/game_XXXXX.json，返回最大 game_id（确保不重名覆盖）"""
    max_id = 0
    for path in glob.glob(os.path.join(out_dir, "**", "game_*.json"), recursive=True):
        base = os.path.basename(path)
        stem = base.replace("game_", "").replace(".json", "")
        try:
            gid = int(stem)
            if gid > max_id:
                max_id = gid
        except ValueError:
            pass
    return max_id


resume_from = _find_latest_checkpoint(OUT)
start_iter = 1
_game_counter = _find_max_game_id(OUT)   # ← 从磁盘实况扫，最靠谱

if resume_from:
    print("=" * 60)
    print(f"🔄 发现 checkpoint，从 {resume_from} 恢复...")
    ckpt = torch.load(resume_from, weights_only=False)

    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.target_model.load_state_dict(ckpt["target_state"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
    trainer.global_step = ckpt.get("step", 0)
    trainer.epsilon = ckpt.get("epsilon", cfg.epsilon_start)
    start_iter = ckpt.get("iteration", 0) + 1

    # game_counter 取两者的 max：磁盘实况 vs checkpoint 内记录
    ckpt_gc = ckpt.get("game_counter", 0)
    if ckpt_gc > _game_counter:
        _game_counter = ckpt_gc

    if "pool_state_dict" in ckpt:
        pool.load_state_dict(ckpt["pool_state_dict"], model_factory=_make_empty_model)

    print(f"  step={trainer.global_step}  ε={trainer.epsilon:.3f}  "
          f"iter={start_iter - 1} → resume iter={start_iter}  "
          f"pool={len(pool)}  已落盘 games={_game_counter}")
    print("=" * 60)
else:
    print("=" * 60)
    print("🆕 Fresh start — 无 checkpoint 可恢复")
    print(f"  目标: {MAX_STEPS} DQN steps, {TOTAL_ITER} iters")
    print(f"  Pool: max={pool_cfg.max_size}, admission={pool_cfg.admission_win_rate:.0%}")
    print(f"  Opponent mix prob: {OPPONENT_MIX_PROB:.0%}")
    print(f"  已落盘 game_counter={_game_counter} (防止重名覆盖)")
    print("=" * 60)

# ========== Warmup（仅 fresh start） ==========
if start_iter == 1:
    warmup_dir = _warmup_dir()
    os.makedirs(warmup_dir, exist_ok=True)
    print(f"\nWarmup {cfg.warmup_games} games → {warmup_dir} ...")
    warmup_engine = Engine(trainer.model, device="cpu")
    warmup_dict = {0: warmup_engine, 1: warmup_engine,
                   2: warmup_engine, 3: warmup_engine}
    for _ in range(cfg.warmup_games):
        r = play_one_game(engine=warmup_dict, epsilon=trainer.epsilon, max_turns=300)
        _game_counter += 1
        r.game_id = _game_counter
        r.iteration = 0
        r.save(os.path.join(warmup_dir, f"game_{_game_counter:05d}.json"))
        trainer.buffer.add_replay(r)
    print(f"  warmup done. buffer={trainer.buffer.size}")

# ========== 自博弈循环 ==========
engine = Engine(trainer.model, device="cpu")

def _make_engines_dict(current_model_engine):
    d = {0: current_model_engine}
    if OPPONENT_MIX_PROB == 0.0:
        for pid in range(1, 4):
            d[pid] = current_model_engine
        return d
    if pool.is_empty() or np.random.random() >= OPPONENT_MIX_PROB:
        for pid in range(1, 4):
            d[pid] = None
        return d
    opponents = pool.sample(num_seats=3)
    for i, opp in enumerate(opponents):
        try:
            m = opp.model
            d[i + 1] = Engine(m, device="cpu") if isinstance(m, torch.nn.Module) else None
        except Exception:
            d[i + 1] = None
    for pid in range(1, 4):
        if pid not in d:
            d[pid] = None
    return d


t0 = time.time()

def _save_checkpoint(path: str, iteration: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": trainer.model.state_dict(),
        "target_state": trainer.target_model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "step": trainer.global_step,
        "epsilon": trainer.epsilon,
        "iteration": iteration,
        "pool_state_dict": pool.state_dict(),
        "game_counter": _game_counter,
    }, path)


try:
    for iteration in range(start_iter, TOTAL_ITER + 1):
        engines = _make_engines_dict(engine)

        # 2. Self-play — 牌谱落盘到 iter_XXXX/
        iter_dir = _iter_dir(iteration)
        os.makedirs(iter_dir, exist_ok=True)

        replays = []
        for _ in range(GAMES_PER_ITER):
            r = play_one_game(engine=engines, epsilon=trainer.epsilon, max_turns=300)
            _game_counter += 1
            r.game_id = _game_counter
            r.iteration = iteration
            r.save(os.path.join(iter_dir, f"game_{_game_counter:05d}.json"))
            replays.append(r)

        # 3. 入 buffer
        for r in replays:
            trainer.buffer.add_replay(r)

        # 4. DQN 更新
        for _ in range(N_UPDATE):
            if trainer.global_step >= MAX_STEPS:
                break
            trainer._train_step()

        # 5. Epsilon decay
        trainer._decay_epsilon()

        # 6. 定期 snapshot 入池
        if iteration % SNAPSHOT_INTERVAL == 0:
            try:
                pool.add_snapshot(
                    trainer.model,
                    iteration=iteration,
                    model_id=f"iter_{iteration:04d}",
                )
            except Exception as e:
                print(f"  [WARN] add_snapshot fail: {e}")

        # 7. 日志
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - t0
            step_rate = trainer.global_step / elapsed if elapsed > 0 else 0
            avg_trans = np.mean([len(r.transitions) for r in replays])
            avg_score_p0 = (np.mean([r.final_scores[0] for r in replays if r.is_over])
                            if any(r.is_over for r in replays) else 0)
            pool_status = f"pool={len(pool)} (mix={OPPONENT_MIX_PROB:.0%})"
            print(f"iter={iteration:4d} step={trainer.global_step:4d} "
                  f"ε={trainer.epsilon:.3f} [speed={step_rate:.1f} step/s] "
                  f"{pool_status} avg_trans={avg_trans:.0f} "
                  f"avg_p0_score={avg_score_p0:+.2f}")

        # 8. 定期 checkpoint — 落在 iter_XXXX/ 里，跟牌谱同目录
        if iteration % CHECKPOINT_INTERVAL == 0:
            ckpt_path = _ckpt_path(iteration)
            _save_checkpoint(ckpt_path, iteration)
            print(f"  💾 {ckpt_path}  "
                  f"(step={trainer.global_step}, pool={len(pool)}, games={_game_counter})")

        if trainer.global_step >= MAX_STEPS:
            print(f"\nReached max_steps={MAX_STEPS}")
            break

except KeyboardInterrupt:
    print(f"\n⚠️  KeyboardInterrupt — 正在保存中断 checkpoint...")
    interrupt_dir = os.path.join(OUT, "iter_interrupt")
    os.makedirs(interrupt_dir, exist_ok=True)
    ckpt_path = os.path.join(interrupt_dir, "iter_interrupt.pt")
    _save_checkpoint(ckpt_path, iteration)
    print(f"  已保存 → {ckpt_path}  step={trainer.global_step}  "
          f"下次自动从 iter={iteration + 1} 继续")
    sys.exit(0)

# ========== Final save ==========
elapsed = time.time() - t0
final_path = os.path.join(OUT, "model_final.pt")
torch.save({
    "model_state_dict": trainer.model.state_dict(),
    "target_state_dict": trainer.target_model.state_dict(),
    "step": trainer.global_step,
}, final_path)

print(f"\n{'='*60}")
print(f"✅ TRAINING DONE")
print(f"  Final step: {trainer.global_step}")
print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"  Final ε: {trainer.epsilon:.3f}")
print(f"  Total games saved: {_game_counter}")
print(f"  OpponentPool: {pool.summary()}")
print(f"  Final model → {final_path}")
print(f"{'='*60}")
