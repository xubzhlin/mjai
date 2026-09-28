use mjai_engine::arena::Game;

fn main() {
    let mut clear_missing_turns: Vec<Option<usize>> = Vec::new();
    let mut shanten_progress: Vec<Vec<i32>> = vec![vec!(); 20];  // 每 20 turn 记录 shanten

    for _ in 0..500 {
        let mut game = Game::new(4);
        game.start_game();
        game.process_swap_phase();
        game.process_missing_phase();

        let mut turn_count = 0;
        let mut clear_turn: Option<usize> = None;
        let mut last_shanten: Vec<i32> = vec![i32::MAX; 4];
        
        while game.process_turn() {
            turn_count += 1;
            for (i, p) in game.board.players.iter().enumerate() {
                if p.has_won || clear_turn.is_some() { continue; }
                if let Some(s) = p.hand.missing_suit {
                    if p.hand.count_by_suit(s) == 0 {
                        if !clear_missing_turns.iter().any(|x| matches!(x, Some(t) if *t == turn_count && true) && false) {
                            clear_missing_turns.push(Some(turn_count));
                        }
                    }
                }
                last_shanten[i] = p.hand.shanten();
            }
        }
        
        // 没有清干净的玩家
        for p in &game.board.players {
            if !p.has_won {
                if let Some(s) = p.hand.missing_suit {
                    if p.hand.count_by_suit(s) > 0 {
                        clear_missing_turns.push(None);
                    }
                }
            }
        }
    }

    let total = clear_missing_turns.len();
    let cleared: Vec<usize> = clear_missing_turns.iter().filter_map(|&x| x).collect();
    let uncleared = clear_missing_turns.iter().filter(|&x| x.is_none()).count();
    
    println!("总玩家数: {}, 清干净: {}, 没清干净: {}", total, cleared.len(), uncleared);
    if !cleared.is_empty() {
        let mut sorted = cleared.clone();
        sorted.sort();
        println!("清干净所需 turn 数: min={}, median={}, p90={}, max={}",
            sorted[0], sorted[sorted.len()/2], sorted[sorted.len()*9/10], sorted.last().unwrap());
    }
}
