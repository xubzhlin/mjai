use std::time::Instant;
use mjai_engine::arena::Game;

fn main() {
    let target = 1000;
    let start = Instant::now();
    let mut total_hu = 0usize;
    let mut total_draw = 0usize;
    let mut turns: Vec<usize> = Vec::new();
    let mut won_players: [usize; 4] = [0; 4];
    let mut tenpai_before_end = 0usize;

    for _ in 0..target {
        let mut game = Game::new(4);
        game.run_to_end();
        let won = game.board.players.iter().filter(|p| p.has_won).count();
        if won > 0 { total_hu += 1; for (i, p) in game.board.players.iter().enumerate() { if p.has_won { won_players[i] += 1; } } }
        else { total_draw += 1; }
        turns.push(game.current_turn);

        // 结算前有多少人在听牌
        let tp: usize = game.board.players.iter().filter(|p| !p.has_won && !p.has_huazhu && p.hand.is_tenpai()).count();
        tenpai_before_end += tp;
    }

    turns.sort();
    println!("1000局: {} 胡, {} 流局 ({:.1}%)", total_hu, total_draw, 100.0 * total_draw as f64 / target as f64);
    println!("胡牌玩家分布: {:?}", won_players);
    println!("Turn 统计: min={}, median={}, p90={}, max={}",
        turns[0], turns[turns.len()/2], turns[(turns.len() as f64 * 0.9) as usize], turns.last().unwrap());
    println!("结算前听牌人数总计: {} (平均每局 {:.1})", tenpai_before_end, tenpai_before_end as f64 / target as f64);
    println!("耗时: {:.2}s", start.elapsed().as_secs_f32());
}
