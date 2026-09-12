"""
mjai.bots — 基线 AI 系统

模块结构：
    base.py          — BotBase 抽象接口
    random_bot.py    — RandomBot（最低基线：合法动作中均匀随机）
    heuristic_bot.py — HeuristicBot（启发式基线：优先打缺门、听牌碰杠等）

设计原则：
    所有 Bot 实现统一接口，便于与 RL 模型对比评估
"""

from .base import BotBase
from .random_bot import RandomBot
from .heuristic_bot import HeuristicBot

__all__ = [
    'BotBase',
    'RandomBot',
    'HeuristicBot',
]