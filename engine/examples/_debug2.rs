use mjai_engine::algo::WinChecker;
use mjai_engine::algo::winning::WinType;
use mjai_engine::tile::{Hand, Tile, Suit};

fn main() {
    // 场景A: 14张完整手牌 + 设了缺门Sou但还留着Sou牌
    let tiles_14 = [
        Tile::M1, Tile::M1, Tile::M1,
        Tile::M2, Tile::M2, Tile::M2,
        Tile::M3, Tile::M3, Tile::M3,
        Tile::P1, Tile::P1, Tile::P1,
        Tile::M5, Tile::M5,
    ];
    let mut hand = Hand::from_tiles(&tiles_14);
    hand.set_missing_suit(Some(Suit::Sou));
    println!("场景A (14张标准胡, 缺门Sou但手牌里不含Sou): can_win={}", WinChecker::can_win(&hand, WinType::Tsumo));

    // 场景B: 同上但手牌还留着一张Sou
    let mut tiles_14b: Vec<Tile> = tiles_14[..13].to_vec();
    tiles_14b.push(Tile::S5); // 替换最后一张为Sou
    let hand_b = Hand::from_tiles(&tiles_14b);
    let mut hand_b = hand_b;
    hand_b.set_missing_suit(Some(Suit::Sou));
    println!("场景B (14张含缺门牌S5): can_win={} (正确应为true因为去掉S5后13张不成胡，但加上一张...)", WinChecker::can_win(&hand_b, WinType::Tsumo));
    
    // 场景C: 13张手牌 + 点炮检测
    let tiles_13 = [
        Tile::M1, Tile::M1, Tile::M1,
        Tile::M2, Tile::M2, Tile::M2,
        Tile::M3, Tile::M3, Tile::M3,
        Tile::P1, Tile::P1, Tile::P1,
        Tile::M5,
    ];
    let mut hand_c = Hand::from_tiles(&tiles_13);
    hand_c.set_missing_suit(Some(Suit::Sou));
    println!("场景C (13张听牌+点炮M5): can_win(Ron)={} (正确应为true, 13张+M5=14张胡)", WinChecker::can_win(&hand_c, WinType::Ron));
    
    // 验证: 13张手牌永远 can_win=false?
    println!("场景D: 13张手牌 can_win 自摸={}, Ron={}", 
        WinChecker::can_win(&hand_c, WinType::Tsumo),
        WinChecker::can_win(&hand_c, WinType::Ron));
}
