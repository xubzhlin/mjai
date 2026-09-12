use mjai_engine::arena::Game;
fn main() {
    for g in 1..=5000 {
        let mut game = Game::new(4);
        game.run_to_end();
        let scores: Vec<i32> = game.board.players.iter().map(|p| p.score).collect();
        let sum: i32 = scores.iter().sum();
        if sum != 0 {
            println!("game#{} sum={} scores={:?}", g, sum, scores);
            break;
        }
    }
}
