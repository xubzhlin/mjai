use std::{fs::File, io::Write, time::Instant};
use mjai_engine::*;

#[derive(Debug)]
struct GameResult {
    game_id: usize,
    winners: Vec<usize>,
    scores: Vec<i32>,
    total_delta: i32,
    violations: Vec<String>,
}

#[derive(Debug)]
struct SelfCheckStats {
    total_games: usize,
    violations_count: usize,
    total_delta_sum: i64,
    max_delta: i32,
    min_delta: i32,
    zero_violation_games: usize,
}

impl SelfCheckStats {
    fn new() -> SelfCheckStats {
        SelfCheckStats {
            total_games: 0,
            violations_count: 0,
            total_delta_sum: 0,
            max_delta: i32::MIN,
            min_delta: i32::MAX,
            zero_violation_games: 0,
        }
    }
    
    fn add_game(&mut self, result: &GameResult) {
        self.total_games += 1;
        self.violations_count += result.violations.len();
        self.total_delta_sum += result.total_delta as i64;
        self.max_delta = self.max_delta.max(result.total_delta);
        self.min_delta = self.min_delta.min(result.total_delta);
        
        if result.violations.is_empty() {
            self.zero_violation_games += 1;
        }
    }
    
    fn is_zero_sum(&self) -> bool {
        self.total_delta_sum == 0
    }
    
    fn zero_violation_rate(&self) -> f64 {
        if self.total_games == 0 {
            0.0
        } else {
            self.zero_violation_games as f64 / self.total_games as f64 * 100.0
        }
    }
}

fn create_random_game() -> Game {
    let config = GameConfig::default();
    Game::new(config)
}

fn run_single_game(game_id: usize) -> GameResult {
    let mut game = create_random_game();
    let mut violations = Vec::new();
    
    // 运行游戏到结束
    game.run_to_end();
    
    // 检查游戏结果
    let winners: Vec<usize> = game.get_winners();
    let scores = game.get_player_scores();
    let total_delta = scores.iter().sum();
    
    // 验证零和
    if total_delta != 0 {
        violations.push(format!("零和验证失败: 总分差 {}", total_delta));
    }
    
    // 验证赢家数量
    if winners.is_empty() {
        violations.push("游戏结束但没有赢家".to_string());
    } else if winners.len() > 1 {
        // 检查是否有多个赢家（流局或特殊规则）
        let winner_scores: Vec<i32> = winners.iter()
            .map(|&idx| scores[idx])
            .collect();
        if !winner_scores.iter().all(|&s| s > 0) {
            violations.push(format!("多赢家但分数异常: {:?}", winner_scores));
        }
    }
    
    // 验证所有玩家分数
    for (i, &score) in scores.iter().enumerate() {
        if score < -100000 || score > 100000 {
            violations.push(format!("玩家{}分数异常: {}", i, score));
        }
    }
    
    GameResult {
        game_id,
        winners,
        scores,
        total_delta,
        violations,
    }
}

fn save_results(results: &[GameResult], stats: &SelfCheckStats) {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("e:\\ai\\mjai\\replays\\self_check_{}.json", timestamp);
    
    let mut file = match File::create(&filename) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("无法创建结果文件: {}", e);
            return;
        }
    };
    
    // 写入统计摘要
    writeln!(file, "=== 引擎自检结果 ({}局) ===", stats.total_games).unwrap();
    writeln!(file, "违规次数: {}", stats.violations_count);
    writeln!(file, "零违规率: {:.2}%", stats.zero_violation_rate());
    writeln!(file, "零和验证: {}", if stats.is_zero_sum() { "通过" } else { "失败" });
    writeln!(file, "总分差总和: {}", stats.total_delta_sum);
    writeln!(file, "最大单局分差: {}", stats.max_delta);
    writeln!(file, "最小单局分差: {}", stats.min_delta);
    writeln!(file).unwrap();
    
    // 写入详细结果
    for result in results {
        writeln!(file, "=== 游戏 {} ===", result.game_id).unwrap();
        writeln!(file, "赢家: {:?}", result.winners).unwrap();
        writeln!(file, "分数: {:?}", result.scores).unwrap();
        writeln!(file, "总分差: {}", result.total_delta).unwrap();
        
        if !result.violations.is_empty() {
            writeln!(file, "违规:").unwrap();
            for violation in &result.violations {
                writeln!(file, "  - {}", violation).unwrap();
            }
        } else {
            writeln!(file, "无违规").unwrap();
        }
        writeln!(file).unwrap();
    }
    
    println!("结果已保存到: {}", filename);
}

fn main() {
    println!("开始引擎自检 - 3000局游戏测试");
    println!("目标: 零违规、零和验证");
    
    let start_time = Instant::now();
    let mut stats = SelfCheckStats::new();
    let mut results = Vec::new();
    
    // 创建replays目录
    std::fs::create_dir_all("e:\\ai\\mjai\\replays").unwrap_or(());
    
    for game_id in 1..=3000 {
        let game_start = Instant::now();
        let result = run_single_game(game_id);
        stats.add_game(&result);
        results.push(result);
        
        // 每100局输出一次进度
        if game_id % 100 == 0 {
            let elapsed = game_start.elapsed();
            println!("完成 {}/3000 局, 用时: {:.2}s, 违规: {}", 
                    game_id, elapsed.as_secs_f32(), stats.violations_count);
        }
        
        // 检查是否需要提前终止
        if stats.violations_count > 0 && game_id > 100 {
            let violation_rate = stats.violations_count as f64 / game_id as f64 * 100.0;
            if violation_rate > 1.0 {
                println!("警告: 违规率过高 {:.2}%, 终止测试", violation_rate);
                break;
            }
        }
    }
    
    let total_time = start_time.elapsed();
    
    // 输出最终统计
    println!("\n=== 自检完成 ===");
    println!("总游戏数: {}", stats.total_games);
    println!("违规次数: {}", stats.violations_count);
    println!("零违规率: {:.2}%", stats.zero_violation_rate());
    println!("零和验证: {}", if stats.is_zero_sum() { "通过" } else { "失败" });
    println!("总分差总和: {}", stats.total_delta_sum);
    println!("最大单局分差: {}", stats.max_delta);
    println!("最小单局分差: {}", stats.min_delta);
    println!("总用时: {:.2}s", total_time.as_secs_f32());
    println!("平均每局: {:.2}s", total_time.as_secs_f32() / stats.total_games as f32);
    
    // 保存结果
    save_results(&results, &stats);
    
    // 判断是否通过自检
    let passed = stats.violations_count == 0 && stats.is_zero_sum();
    if passed {
        println!("✅ 引擎自检通过 - 零违规、零和验证成功");
        std::process::exit(0);
    } else {
        println!("❌ 引擎自检失败 - 发现违规或零和验证失败");
        std::process::exit(1);
    }
}