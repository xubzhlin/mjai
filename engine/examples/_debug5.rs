use mjai_engine::arena::Game;

fn main() {
    let mut pong_count = 0u32;
    let mut kong_count = 0u32;
    let mut shanten_at_draw_end: Vec<i32> = Vec::new();

    for _ in 0..1000 {
        let mut game = Game::new(4);
        game.run_to_end();

        for p in &game.board.players {
            if !p.has_won {
                shanten_at_draw_end.push(p.hand.shanten());
            }
            for m in &p.melds {
                use mjai_engine::state::MeldType;
                match m.meld_type {
                    MeldType::Pong => pong_count += 1,
                    MeldType::ExposedKong | MeldType::ConcealedKong | MeldType::AddKong => kong_count += 1,
                }
            }
        }
    }

    println!("碰总次数: {}, 杠总次数: {}", pong_count, kong_count);
    println!("结算前未胡玩家 shanten 分布 (N={}):", shanten_at_draw_end.len());

    let zero = shanten_at_draw_end.iter().filter(|&&s| s <= 0).count();
    let one = shanten_at_draw_end.iter().filter(|&&s| s == 1).count();
    let two = shanten_at_draw_end.iter().filter(|&&s| s == 2).count();
    let three = shanten_at_draw_end.iter().filter(|&&s| s == 3).count();
    let four_plus = shanten_at_draw_end.iter().filter(|&&s| s >= 4).count();
    println!("  ≤0(已听): {} ({:.1}%)", zero, 100.0*zero as f64/shanten_at_draw_end.len() as f64);
    println!("  =1: {} ({:.1}%)", one, 100.0*one as f64/shanten_at_draw_end.len() as f64);
    println!("  =2: {} ({:.1}%)", two, 100.0*two as f64/shanten_at_draw_end.len() as f64);
    println!("  =3: {} ({:.1}%)", three, 100.0*three as f64/shanten_at_draw_end.len() as f64);
    println!("  ≥4: {} ({:.1}%)", four_plus, 100.0*four_plus as f64/shanten_at_draw_end.len() as f64);
}
