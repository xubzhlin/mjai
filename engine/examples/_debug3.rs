use mjai_engine::arena::Game;
use mjai_engine::algo::WinChecker;
use mjai_engine::algo::winning::WinType;

fn main() {
    let mut game = Game::new(4);
    game.run_to_end();

    let won_count = game.board.players.iter().filter(|p| p.has_won).count();
    println!("胡牌人数: {}", won_count);
    println!("最终得分: {:?}", game.board.players.iter().map(|p| p.score).collect::<Vec<_>>());
    println!("游戏阶段: {:?}", game.phase);
    println!("turn: {}", game.current_turn);

    // 打印最后几个事件
    println!("\n最后10个事件:");
    let events: Vec<_> = game.game_history.iter().rev().take(10).collect();
    for e in events {
        println!("  {:?}", e);
    }

    // 打印每家手牌数量+缺门+shanten
    for (i, p) in game.board.players.iter().enumerate() {
        println!("\n玩家{}: won={}, has_huazhu={}, has_dajiao={}, score={}",
            i, p.has_won, p.has_huazhu, p.has_dajiao, p.score);
        println!("  missing_suit={:?}, shanten={}", p.hand.missing_suit, p.hand.shanten());
        // 模拟: 把所有弃牌河 + 牌墙剩余都收集起来，看该玩家有没有能胡的可能
    }
}
