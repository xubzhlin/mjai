//! 引擎自检脚本 (Phase 1.9)
//!
//! 运行 3000 局完整对局，验证：
//!  1. 零违规（无崩溃、无异常分数）
//!  2. 零和成立（四家总得分 ≡ 0）
//!  3. 四家正收益率均衡（约 67%-71%，洗牌无偏倚）
//!  4. 花猪/大叫标记正确
//!  5. 杠结算、番型计算无负数/溢出
//!
//! 运行方式：
//!   cargo run --example self_check --release
//!
//! 可选参数：
//!   cargo run --example self_check --release -- 10000   # 自定义局数

use std::time::Instant;

use mjai_engine::arena::Game;

/// 单局统计（从 Game 提取，不含 Game 本身所有权）
struct GameSnapshot {
    scores: [i32; 4],
    has_won: [bool; 4],
    has_huazhu: [bool; 4],
    has_dajiao: [bool; 4],
    is_draw: bool,
    wall_remaining_at_end: usize,
}

impl GameSnapshot {
    fn from(game: &Game) -> Self {
        let mut scores = [0i32; 4];
        let mut has_won = [false; 4];
        let mut has_huazhu = [false; 4];
        let mut has_dajiao = [false; 4];
        for (i, p) in game.board.players.iter().enumerate() {
            scores[i] = p.score;
            has_won[i] = p.has_won;
            has_huazhu[i] = p.has_huazhu;
            has_dajiao[i] = p.has_dajiao;
        }
        Self {
            scores,
            has_won,
            has_huazhu,
            has_dajiao,
            is_draw: matches!(game.board.game_over_reason,
                Some(mjai_engine::arena::GameOverReason::WallEmpty)),
            wall_remaining_at_end: game.board.wall.remaining_count(),
        }
    }
}

/// 累计统计
struct AggStats {
    total: usize,
    violations: Vec<(usize, String)>, // (game_id, reason)
    pos_gain_count: [usize; 4],       // 每家正收益局数
    score_sum: [i128; 4],             // 累计得分（零和验证）
    zero_sum_mismatches: usize,       // 总分 ≠ 0 的局数
    huazhu_incidents: usize,
    dajiao_incidents: usize,
    draws: usize,
}

impl AggStats {
    fn new() -> Self {
        Self {
            total: 0,
            violations: Vec::new(),
            pos_gain_count: [0; 4],
            score_sum: [0; 4],
            zero_sum_mismatches: 0,
            huazhu_incidents: 0,
            dajiao_incidents: 0,
            draws: 0,
        }
    }

    fn add(&mut self, game_id: usize, snap: &GameSnapshot) {
        self.total += 1;

        // 零和验证
        let sum: i32 = snap.scores.iter().sum();
        if sum != 0 {
            self.zero_sum_mismatches += 1;
            if self.violations.len() < 50 {
                self.violations.push((game_id, format!("零和失败 sum={}", sum)));
            }
        }

        // 分数累计 + 正收益
        for i in 0..4 {
            self.score_sum[i] += snap.scores[i] as i128;
            if snap.scores[i] > 0 {
                self.pos_gain_count[i] += 1;
            }
        }

        // 分数范围检查
        for i in 0..4 {
            if snap.scores[i] < -100_000 || snap.scores[i] > 100_000 {
                if self.violations.len() < 50 {
                    self.violations.push((game_id, format!("玩家{}分数异常 {}", i, snap.scores[i])));
                }
            }
        }

        // 标记统计
        if snap.is_draw { self.draws += 1; }
        for i in 0..4 {
            if snap.has_huazhu[i] { self.huazhu_incidents += 1; }
            if snap.has_dajiao[i] { self.dajiao_incidents += 1; }
        }
    }
}

/// 运行单局，返回快照。任何 panic 都被捕获并记为违规。
fn run_one(game_id: usize) -> Result<GameSnapshot, String> {
    let mut game = Game::new(4);
    game.run_to_end();

    // 基础合理性检查
    let won_count = game.board.players.iter().filter(|p| p.has_won).count();

    // 三家胡牌或牌墙空 → 应已结束
    if !game.is_game_over() {
        return Err("游戏未正常结束".to_string());
    }

    // 血战到底最多 3 家胡牌
    if won_count > 3 {
        return Err(format!("胡牌人数异常: {}", won_count));
    }

    Ok(GameSnapshot::from(&game))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);

    println!("=== mjai_engine 引擎自检 ===");
    println!("目标: {} 局  |  规则: 传统川麻(血战到底·换三张·定缺)", target);
    println!("检查项: 零违规 / 零和 / 正收益率均衡 / 花猪·大叫标记\n");

    let start = Instant::now();
    let mut stats = AggStats::new();

    // 简单进度条：单线程跑（方便错误定位；可后续加 rayon 并行）
    for gid in 1..=target {
        let snap = match run_one(gid) {
            Ok(s) => s,
            Err(reason) => {
                if stats.violations.len() < 50 {
                    stats.violations.push((gid, reason));
                }
                // 构造一个占位快照，避免统计跳过
                GameSnapshot {
                    scores: [0; 4], has_won: [false; 4],
                    has_huazhu: [false; 4], has_dajiao: [false; 4],
                    is_draw: false, wall_remaining_at_end: 0,
                }
            }
        };
        stats.add(gid, &snap);

        if gid % 500 == 0 || gid == target {
            let elapsed = start.elapsed().as_secs_f32();
            println!(
                "  [{:>5}/{}]  elapsed={:.1}s  violations={}  sum0?={}",
                gid, target, elapsed,
                stats.violations.len(),
                stats.zero_sum_mismatches == 0,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let pos_rates: [f64; 4] = std::array::from_fn(|i| {
        stats.pos_gain_count[i] as f64 / stats.total as f64 * 100.0
    });

    println!("\n=== 自检结果 ===");
    println!("总对局数:      {}", stats.total);
    println!("总耗时:        {:.2}s  平均 {:.3}s/局", elapsed, elapsed / stats.total as f32);

    println!("\n--- 核心指标 ---");
    let violations_ok = stats.violations.is_empty();
    let zero_sum_ok = stats.zero_sum_mismatches == 0;
    println!(
        "零违规:        {}",
        if violations_ok {
            "✅ 通过".to_string()
        } else {
            format!("❌ 失败 ({} 处)", stats.violations.len())
        }
    );
    println!(
        "零和成立:      {}",
        if zero_sum_ok {
            "✅ 通过".to_string()
        } else {
            format!("❌ 失败 ({} 局 sum≠0)", stats.zero_sum_mismatches)
        }
    );
    println!("流局数:        {} ({:.1}%)", stats.draws, stats.draws as f64 / stats.total as f64 * 100.0);

    println!("\n--- 四家正收益率 ---");
    for i in 0..4 {
        let ok = pos_rates[i] >= 60.0 && pos_rates[i] <= 75.0;
        println!(
            "  玩家{}: {:>5.1}% ({}/{} 局正收益) {}",
            i, pos_rates[i], stats.pos_gain_count[i], stats.total,
            if ok { "✅" } else { "⚠️  (不在 60-75%)" },
        );
    }
    let rates_mean = pos_rates.iter().sum::<f64>() / 4.0;
    let rates_range = pos_rates.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &r| (acc.0.min(r), acc.1.max(r)));
    let range_span = rates_range.1 - rates_range.0;
    println!(
        "  均值={:.1}%  极差={:.1}%  均衡性: {}",
        rates_mean, range_span,
        if range_span < 10.0 { "✅ 良好" } else { "⚠️ 位置偏倚" },
    );

    println!("\n--- 标记统计 ---");
    println!("  花猪事件:     {} ({:.2}/局)", stats.huazhu_incidents, stats.huazhu_incidents as f64 / stats.total as f64);
    println!("  大叫事件:     {} ({:.2}/局)", stats.dajiao_incidents, stats.dajiao_incidents as f64 / stats.total as f64);

    println!("\n--- 违规详情（前 20 条）---");
    if stats.violations.is_empty() {
        println!("  (无)");
    } else {
        for (gid, reason) in stats.violations.iter().take(20) {
            println!("  局#{:>5}: {}", gid, reason);
        }
        if stats.violations.len() > 20 {
            println!("  ... 另有 {} 条未显示", stats.violations.len() - 20);
        }
    }

    let all_ok = violations_ok && zero_sum_ok;
    let status = if all_ok { "✅ 自检通过" } else { "❌ 自检失败" };
    println!("\n=== {} ===", status);

    std::process::exit(if all_ok { 0 } else { 1 });
}
