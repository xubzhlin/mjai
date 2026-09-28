use mjai_engine::arena::Game;

fn main() {
    let mut before_swap: Vec<i32> = Vec::new();
    let mut after_missing: Vec<i32> = Vec::new();
    let mut after_playing10: Vec<i32> = Vec::new();

    for _ in 0..1000 {
        let mut game = Game::new(4);
        
        // 配牌后 shanten（未定缺）
        for p in &game.board.players {
            before_swap.push(p.hand.shanten());
        }

        game.start_game();
        game.process_swap_phase();
        game.process_missing_phase();
        
        // 定缺后 shanten
        for p in &game.board.players {
            after_missing.push(p.hand.shanten());
        }

        // 打 10 个 turn 后 shanten
        for _ in 0..10 {
            game.process_turn();
            if game.is_game_over() { break; }
        }
        if !game.is_game_over() {
            for p in &game.board.players {
                after_playing10.push(p.hand.shanten());
            }
        }
    }

    println!("配牌后(shanten): median={}, p90={}", quantile(&before_swap, 0.5), quantile(&before_swap, 0.9));
    println!("定缺后(shanten): median={}, p90={}", quantile(&after_missing, 0.5), quantile(&after_missing, 0.9));
    println!("打10 turn后(shanten): median={}, p90={}", quantile(&after_playing10, 0.5), quantile(&after_playing10, 0.9));
}

fn quantile(vals: &[i32], q: f64) -> i32 {
    let mut v: Vec<i32> = vals.to_vec();
    v.sort();
    let idx = (v.len() as f64 * q) as usize;
    v[idx.min(v.len()-1)]
}
