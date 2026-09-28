use mjai_engine::arena::Game;
use mjai_engine::state::MeldType;

fn main() {
    let mut kong_ankan = 0u32;
    let mut kong_minkan = 0u32;
    let mut kong_bukan = 0u32;
    
    for _ in 0..1000 {
        let mut game = Game::new(4);
        game.run_to_end();
        for p in &game.board.players {
            for m in &p.melds {
                match m.meld_type {
                    MeldType::ConcealedKong => kong_ankan += 1,
                    MeldType::ExposedKong => kong_minkan += 1,
                    MeldType::AddKong => kong_bukan += 1,
                    _ => {}
                }
            }
        }
    }
    println!("杠类型分布: 暗杠={}, 明杠={}, 补杠={}", kong_ankan, kong_minkan, kong_bukan);
    println!("总计: {}", kong_ankan + kong_minkan + kong_bukan);
}
