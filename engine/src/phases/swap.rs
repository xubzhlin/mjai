use crate::tile::{Hand, Tile, Suit};
use crate::rules::{SwapRule, SwapDirection};
use std::collections::HashMap;

/// 换三张结果
#[derive(Debug, Clone)]
pub struct SwapResult {
    /// 玩家选择的牌
    pub player_selections: Vec<Vec<Tile>>,
    /// 换三张方向
    pub direction: SwapDirection,
    /// 交换后的牌
    pub swapped_tiles: Vec<Vec<Tile>>,
}

/// 血战到底换三张规则
pub struct XueZhanSwapRule {
    pub tian_que_probability: f64,
}

impl Default for XueZhanSwapRule {
    fn default() -> Self {
        Self {
            tian_que_probability: 0.75,
        }
    }
}

impl XueZhanSwapRule {
    /// 验证玩家选择的牌是否合法
    pub fn validate_selection(&self, hand: &Hand, selected_tiles: &[Tile]) -> bool {
        // 检查选择数量
        if selected_tiles.len() != 3 {
            return false;
        }

        // 检查是否都存在于手牌中
        for tile in selected_tiles {
            if !hand.contains(*tile) {
                return false;
            }
        }

        // 检查是否为同花色
        let first_suit = selected_tiles[0].suit();
        for tile in selected_tiles {
            if tile.suit() != first_suit {
                return false;
            }
        }

        true
    }

    /// 执行换三张
    pub fn perform_swap(&self, hands: &mut [Hand], selections: &[Vec<Tile>], direction: SwapDirection) -> SwapResult {
        let num_players = hands.len();
        let mut swapped_tiles = vec![Vec::new(); num_players];

        // 步骤1：从所有手牌中同时移除各自选中的牌
        for (i, selection) in selections.iter().enumerate() {
            for tile in selection {
                hands[i].remove_tile(*tile);
            }
        }

        // 步骤2：按方向同时接收上家/对家/下家的牌
        // Down  → i 从 (i-1) % n 接收（上家传来）
        // Across → i 从 (i-2) % n 接收（对家传来）
        // Up    → i 从 (i+1) % n 接收（下家传来）
        for i in 0..num_players {
            let sender = match direction {
                SwapDirection::Down => (i + num_players - 1) % num_players,
                SwapDirection::Across => (i + num_players - 2) % num_players,
                SwapDirection::Up => (i + 1) % num_players,
            };

            let received = selections[sender].clone();
            for tile in &received {
                hands[i].add_tile(*tile);
            }
            swapped_tiles[i] = received;
        }

        SwapResult {
            player_selections: selections.to_vec(),
            direction,
            swapped_tiles,
        }
    }
}

impl SwapRule for XueZhanSwapRule {
    /// 根据骰子点数确定换三张方向
    /// 骰点 1·2 → 下家, 3·4 → 对家, 5·6 → 上家
    fn determine_direction(&self, dice: u8) -> SwapDirection {
        match dice {
            1 | 2 => SwapDirection::Down,
            3 | 4 => SwapDirection::Across,
            5 | 6 => SwapDirection::Up,
            _ => SwapDirection::Down,
        }
    }

    /// AI 选 3 张同花色换出牌
    fn select_swap_tiles(&self, hand: &Hand) -> Vec<Tile> {
        let mut available_suits = HashMap::new();

        // 统计每种花色的牌数
        for suit in Suit::all() {
            let count = hand.count_by_suit(suit);
            if count > 0 {
                available_suits.insert(suit, count);
            }
        }

        // 选择牌数最少的花色
        if let Some((min_suit, _)) = available_suits.iter().min_by_key(|(_, &count)| count) {
            let mut selected = Vec::new();
            let min_suit_copy = *min_suit;

            // 从最少花色中按实际数量选牌（同花色可能多张相同）
            for tile in Tile::all() {
                if tile.suit() == min_suit_copy {
                    let tile_count = hand.count(tile);
                    for _ in 0..tile_count {
                        if selected.len() < 3 {
                            selected.push(tile);
                        }
                    }
                }
            }

            // 如果该花色不足3张，从次少花色补充
            if selected.len() < 3 {
                let mut sorted_suits: Vec<_> = available_suits.iter().collect();
                sorted_suits.sort_by_key(|(_, &count)| count);

                for (suit, _) in sorted_suits {
                    if *suit != min_suit_copy {
                        for tile in Tile::all() {
                            if tile.suit() == *suit {
                                let tile_count = hand.count(tile);
                                for _ in 0..tile_count {
                                    if selected.len() < 3 {
                                        selected.push(tile);
                                    }
                                }
                            }
                        }
                    }
                    if selected.len() >= 3 {
                        break;
                    }
                }
            }

            selected
        } else {
            Vec::new()
        }
    }
}

/// 换三张处理器
pub struct SwapProcessor;

impl SwapProcessor {
    /// AI 选择换三张的牌
    pub fn ai_select_tiles(hand: &Hand) -> Vec<Tile> {
        let mut available_suits = HashMap::new();

        for suit in Suit::all() {
            let count = hand.count_by_suit(suit);
            if count > 0 {
                available_suits.insert(suit, count);
            }
        }

        if let Some((min_suit, _)) = available_suits.iter().min_by_key(|(_, &count)| count) {
            let mut selected = Vec::new();
            let min_suit_copy = *min_suit;

            for tile in Tile::all() {
                if tile.suit() == min_suit_copy {
                    let tile_count = hand.count(tile);
                    for _ in 0..tile_count {
                        if selected.len() < 3 {
                            selected.push(tile);
                        }
                    }
                }
            }

            if selected.len() < 3 {
                let mut sorted_suits: Vec<_> = available_suits.iter().collect();
                sorted_suits.sort_by_key(|(_, &count)| count);

                for (suit, _) in sorted_suits {
                    if *suit != min_suit_copy {
                        for tile in Tile::all() {
                            if tile.suit() == *suit {
                                let tile_count = hand.count(tile);
                                for _ in 0..tile_count {
                                    if selected.len() < 3 {
                                        selected.push(tile);
                                    }
                                }
                            }
                        }
                    }
                    if selected.len() >= 3 {
                        break;
                    }
                }
            }

            selected
        } else {
            Vec::new()
        }
    }

    /// 处理换三张
    pub fn process_swap(hands: &mut [Hand], rule: &dyn SwapRule, dice: u8) -> SwapResult {
        let num_players = hands.len();
        let mut selections = Vec::with_capacity(num_players);

        // 确定换三张方向
        let direction = rule.determine_direction(dice);

        // 每个玩家选择要交换的牌
        for hand in &*hands {
            let selected = rule.select_swap_tiles(hand);
            selections.push(selected);
        }

        // 验证选择（collect 索引先避免双重借用）
        let need_reselect: Vec<usize> = selections.iter().enumerate()
            .filter(|(i, selection)| {
                selection.len() != 3 || !selection.iter().all(|t| hands[*i].contains(*t))
                    || selection[0].suit() != selection.last().unwrap().suit()
            })
            .map(|(i, _)| i)
            .collect();
        for i in need_reselect {
            selections[i] = Self::ai_select_tiles(&hands[i]);
        }

        // 执行换三张（使用固有方法）
        // 由于 trait 只提供 determine_direction 和 select_swap_tiles，
        // 实际交换逻辑在 SwapProcessor 中实现
        let mut swapped_tiles = vec![Vec::new(); num_players];

        // 步骤1：从所有手牌中同时移除各自选中的牌
        for (i, selection) in selections.iter().enumerate() {
            for tile in selection {
                hands[i].remove_tile(*tile);
            }
        }

        // 步骤2：按方向同时接收
        for i in 0..num_players {
            let sender = match direction {
                SwapDirection::Down => (i + num_players - 1) % num_players,
                SwapDirection::Across => (i + num_players - 2) % num_players,
                SwapDirection::Up => (i + 1) % num_players,
            };

            let received = selections[sender].clone();
            for tile in &received {
                hands[i].add_tile(*tile);
            }
            swapped_tiles[i] = received;
        }

        SwapResult {
            player_selections: selections,
            direction,
            swapped_tiles,
        }
    }

    /// 检查换三张结果是否有效
    pub fn validate_swap_result(hands: &[Hand], swap_result: &SwapResult) -> bool {
        let num_players = hands.len();

        if swap_result.player_selections.len() != num_players {
            return false;
        }

        for selection in &swap_result.player_selections {
            if selection.len() != 3 {
                return false;
            }
        }

        if swap_result.swapped_tiles.len() != num_players {
            return false;
        }

        true
    }

    /// 获取换三张统计信息
    pub fn get_swap_stats(hands: &[Hand], swap_result: &SwapResult) -> SwapStats {
        let mut stats = SwapStats::new();

        for (i, selection) in swap_result.player_selections.iter().enumerate() {
            let suit = selection[0].suit();
            stats.selections_by_suit[suit.index()] += 1;

            let swapped_suit = swap_result.swapped_tiles[i][0].suit();
            stats.swapped_by_suit[swapped_suit.index()] += 1;
        }

        stats.direction = swap_result.direction;
        stats
    }
}

/// 换三张统计信息
#[derive(Debug, Clone)]
pub struct SwapStats {
    pub selections_by_suit: [usize; 3],
    pub swapped_by_suit: [usize; 3],
    pub direction: SwapDirection,
}

impl SwapStats {
    pub fn new() -> Self {
        Self {
            selections_by_suit: [0; 3],
            swapped_by_suit: [0; 3],
            direction: SwapDirection::Down,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.selections_by_suit.iter().sum::<usize>() == self.swapped_by_suit.iter().sum::<usize>() &&
        self.selections_by_suit.iter().sum::<usize>() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_direction() {
        let rule = XueZhanSwapRule::default();

        // 骰点 1·2 → 下家
        assert_eq!(rule.determine_direction(1), SwapDirection::Down);
        assert_eq!(rule.determine_direction(2), SwapDirection::Down);
        // 骰点 3·4 → 对家
        assert_eq!(rule.determine_direction(3), SwapDirection::Across);
        assert_eq!(rule.determine_direction(4), SwapDirection::Across);
        // 骰点 5·6 → 上家
        assert_eq!(rule.determine_direction(5), SwapDirection::Up);
        assert_eq!(rule.determine_direction(6), SwapDirection::Up);
    }

    #[test]
    fn test_swap_validation() {
        let rule = XueZhanSwapRule::default();
        let mut hand = Hand::new();

        for _ in 0..3 {
            hand.add_tile(Tile::M1);
        }
        hand.add_tile(Tile::M2);
        hand.add_tile(Tile::M3);

        let valid_selection = vec![Tile::M1, Tile::M1, Tile::M1];
        assert!(rule.validate_selection(&hand, &valid_selection));

        let invalid_selection = vec![Tile::M1, Tile::M1];
        assert!(!rule.validate_selection(&hand, &invalid_selection));

        let invalid_selection = vec![Tile::M1, Tile::M1, Tile::P1];
        assert!(!rule.validate_selection(&hand, &invalid_selection));

        let invalid_selection = vec![Tile::M1, Tile::M1, Tile::M4];
        assert!(!rule.validate_selection(&hand, &invalid_selection));
    }

    #[test]
    fn test_ai_select_tiles() {
        let mut hand = Hand::new();

        for _ in 0..4 {
            hand.add_tile(Tile::M1);
        }
        for _ in 0..4 {
            hand.add_tile(Tile::P1);
        }
        hand.add_tile(Tile::S1);
        hand.add_tile(Tile::S2);
        hand.add_tile(Tile::S3);

        let selected = SwapProcessor::ai_select_tiles(&hand);

        assert_eq!(selected.len(), 3);
        for tile in &selected {
            assert_eq!(tile.suit(), Suit::Sou);
        }
    }

    #[test]
    fn test_process_swap() {
        let rule = XueZhanSwapRule::default();
        let mut hands = [
            Hand::from_tiles(&[Tile::M1, Tile::M1, Tile::M1, Tile::M2, Tile::M2, Tile::M2]),
            Hand::from_tiles(&[Tile::P1, Tile::P1, Tile::P1, Tile::P2, Tile::P2, Tile::P2]),
        ];

        let result = SwapProcessor::process_swap(&mut hands, &rule, 3);

        assert!(SwapProcessor::validate_swap_result(&hands, &result));
        assert_eq!(hands[0].len(), 6);
        assert_eq!(hands[1].len(), 6);
    }
}
