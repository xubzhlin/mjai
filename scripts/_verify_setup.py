"""Setup verification — run after venv + Rust engine are ready"""
import sys, torch, numpy

ok = True

# 1. mjai_engine native
try:
    import mjai_engine
    print(f"  mjai_engine: {mjai_engine.__file__}")
except Exception as e:
    print(f"  FAIL mjai_engine import: {e}")
    ok = False

# 2. Game create
try:
    from mjai_engine.arena import Game
    g = Game(4)
    g.start_game_bare()
    print(f"  Game OK: phase={g.phase()}")
except Exception as e:
    print(f"  FAIL Game: {e}")
    ok = False

# 3. torch + numpy
try:
    print(f"  torch={torch.__version__}  numpy={numpy.__version__}")
except Exception as e:
    print(f"  FAIL torch/numpy: {e}")
    ok = False

# 4. mjai Python 包
try:
    import mjai
    from mjai.selfplay import play_one_game
    from mjai.opponent_pool import OpponentPool
    print("  mjai OK")
except Exception as e:
    print(f"  FAIL mjai: {e}")
    ok = False

sys.exit(0 if ok else 1)
