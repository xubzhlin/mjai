use crate::tile::{Hand, Tile, Suit};
use crate::rules::MissingRule;
use rand::Rng;
use std::collections::HashMap;

/// 定缺结果
#[derive(Debug, Clone)]
pub struct MissingResult {
    /// 玩家选择的缺门
    pub player_selections: Vec<Option<Suit>>,
    /// 天缺玩家
    pub tian_que_players: Vec<usize>,
}

/// 血战到底定缺规则
pub struct XueZhanMissingRule {
    pub tian_que_probability: f64,
}

impl Default for XueZhanMissingRule {
    fn default() -> Self {
        Self {
            tian_que_probability: 0.75,
        }
    }
}

impl XueZhanMissingRule {
    /// 验证玩家选择的缺门是否合法
    pub fn validate_selection(&self, hand: &Hand, selected_suit: Option<Suit>) -> bool {
        match selected_suit {
            Some(suit) => {
                !hand.has_suit(suit)
            },
            None => {
                // 不选择缺门（天缺情况）
                true
            },
        }
    }

    /// 执行定缺
    pub fn perform_missing(&self, hands: &mut [Hand], selections: &[Option<Suit>]) -> MissingResult {
        let mut tian_que_players = Vec::new();

        for (i, &selection) in selections.iter().enumerate() {
            if let Some(suit) = selection {
                hands[i].set_missing_suit(Some(suit));
            } else {
                // 天缺情况
                tian_que_players.push(i);
            }
        }

        MissingResult {
            player_selections: selections.to_vec(),
            tian_que_players,
        }
    }
}

impl MissingRule for XueZhanMissingRule {
    fn is_valid_missing(&self, hand: &Hand, suit: Suit) -> bool {
        // 检查手牌中是否有所选花色的牌
        !hand.has_suit(suit)
    }

    /// AI 选缺门花色
    fn select_missing_suit(&self, hand: &Hand) -> Suit {
        let distribution = hand.suit_distribution();

        // 检查是否天缺（只有一门或两门）
        let present_suits: Vec<_> = distribution.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .collect();

        match present_suits.len() {
            1 => {
                // 只有一门，必须缺其他两门中的一门（随机选择）
                let missing_suits: Vec<_> = Suit::all()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| distribution[*i] == 0)
                    .map(|(_, &suit)| suit)
                    .collect();

                if missing_suits.is_empty() {
                    Suit::Man
                } else {
                    let mut rng = rand::thread_rng();
                    missing_suits[rng.gen_range(0..missing_suits.len())]
                }
            },
            2 => {
                // 有两门，75%概率缺空门，25%概率保留冲清一色
                let mut rng = rand::thread_rng();
                if rng.gen::<f64>() < self.tian_que_probability {
                    // 缺空门
                    let empty_suits: Vec<_> = Suit::all()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| distribution[*i] == 0)
                        .map(|(_, &suit)| suit)
                        .collect();

                    if empty_suits.is_empty() {
                        Suit::Man
                    } else {
                        empty_suits[0]
                    }
                } else {
                    // 保留冲清一色，随机选一个已有花色
                    let mut rng = rand::thread_rng();
                    let idx = rng.gen_range(0..present_suits.len());
                    Suit::from_index(present_suits[idx].0).unwrap()
                }
            },
            3 => {
                // 三门齐全，选择最少门
                if let Some((min_index, _)) = present_suits.iter().min_by_key(|(_, &count)| count) {
                    Suit::from_index(*min_index).unwrap()
                } else {
                    Suit::Man
                }
            },
            _ => {
                Suit::Man
            },
        }
    }
}

/// 定缺处理器
pub struct MissingProcessor;

impl MissingProcessor {
    /// 处理定缺
    pub fn process_missing(hands: &mut [Hand], rule: &dyn MissingRule) -> MissingResult {
        let num_players = hands.len();
        let mut selections = Vec::with_capacity(num_players);

        // 每个玩家选择缺门
        for hand in &*hands {
            let selected = Some(rule.select_missing_suit(hand));
            selections.push(selected);
        }

        // 验证选择（先收集索引再重新选择）
        let need_reselect: Vec<usize> = selections.iter().enumerate()
            .filter(|(i, selection)| {
                if let Some(suit) = selection {
                    !rule.is_valid_missing(&hands[*i], *suit)
                } else {
                    false
                }
            })
            .map(|(i, _)| i)
            .collect();
        for i in need_reselect {
            selections[i] = Some(rule.select_missing_suit(&hands[i]));
        }

        // 执行定缺
        let mut tian_que_players = Vec::new();

        for (i, &selection) in selections.iter().enumerate() {
            if let Some(suit) = selection {
                hands[i].set_missing_suit(Some(suit));
            } else {
                tian_que_players.push(i);
            }
        }

        MissingResult {
            player_selections: selections,
            tian_que_players,
        }
    }

    /// 检查定缺结果是否有效
    pub fn validate_missing_result(hands: &[Hand], missing_result: &MissingResult) -> bool {
        let num_players = hands.len();

        if missing_result.player_selections.len() != num_players {
            return false;
        }

        for (i, &selection) in missing_result.player_selections.iter().enumerate() {
            if let Some(suit) = selection {
                if hands[i].has_suit(suit) {
                    return false;
                }
            }
        }

        true
    }

    /// 获取定缺统计信息
    pub fn get_missing_stats(hands: &[Hand], missing_result: &MissingResult) -> MissingStats {
        let mut stats = MissingStats::new();

        for (i, &selection) in missing_result.player_selections.iter().enumerate() {
            match selection {
                Some(suit) => {
                    stats.missing_by_suit[suit.index()] += 1;
                    stats.normal_missing_count += 1;
                },
                None => {
                    stats.tian_que_count += 1;
                    stats.tian_que_players.push(i);
                },
            }
        }

        stats
    }

    /// 检查是否天缺
    pub fn is_tian_que(hand: &Hand) -> bool {
        let distribution = hand.suit_distribution();
        let present_suits: usize = distribution.iter().filter(|&&count| count > 0).count();
        present_suits <= 2
    }

    /// 获取天缺候选花色
    pub fn get_tian_que_candidates(hand: &Hand) -> Vec<Suit> {
        let distribution = hand.suit_distribution();
        let mut candidates = Vec::new();

        for (i, &count) in distribution.iter().enumerate() {
            if count == 0 {
                if let Some(suit) = Suit::from_index(i) {
                    candidates.push(suit);
                }
            }
        }

        candidates
    }
}

/// 定缺统计信息
#[derive(Debug, Clone)]
pub struct MissingStats {
    pub missing_by_suit: [usize; 3],
    pub normal_missing_count: usize,
    pub tian_que_count: usize,
    pub tian_que_players: Vec<usize>,
}

impl MissingStats {
    pub fn new() -> Self {
        Self {
            missing_by_suit: [0; 3],
            normal_missing_count: 0,
            tian_que_count: 0,
            tian_que_players: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.missing_by_suit.iter().sum::<usize>() == self.normal_missing_count &&
        self.missing_by_suit.iter().sum::<usize>() + self.tian_que_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_validation() {
        let rule = XueZhanMissingRule::default();
        let mut hand = Hand::new();

        for _ in 0..5 {
            hand.add_tile(Tile::M1);
        }
        for _ in 0..4 {
            hand.add_tile(Tile::P1);
            hand.add_tile(Tile::P2);
        }

        assert!(rule.is_valid_missing(&hand, Suit::Sou));
        assert!(!rule.is_valid_missing(&hand, Suit::Man));
    }

    #[test]
    fn test_ai_select_missing() {
        let rule = XueZhanMissingRule::default();

        // 三门齐全的情况
        let mut hand = Hand::new();
        for _ in 0..5 {
            hand.add_tile(Tile::M1);
        }
        for _ in 0..3 {
            hand.add_tile(Tile::P1);
        }
        for _ in 0..2 {
            hand.add_tile(Tile::S1);
        }

        let selected = rule.select_missing_suit(&hand);
        assert_eq!(selected, Suit::Sou);

        // 天缺情况（只有一门）
        let mut hand = Hand::new();
        for _ in 0..10 {
            hand.add_tile(Tile::M1);
        }

        let selected = rule.select_missing_suit(&hand);
        assert!(matches!(selected, Suit::Pin | Suit::Sou));
    }

    #[test]
    fn test_process_missing() {
        let rule = XueZhanMissingRule::default();
        let mut hands = [
            Hand::from_tiles(&[Tile::M1, Tile::M1, Tile::M1, Tile::M2, Tile::M2]),
            Hand::from_tiles(&[Tile::P1, Tile::P1, Tile::P1, Tile::P2, Tile::P2]),
        ];

        let result = MissingProcessor::process_missing(&mut hands, &rule);

        assert!(MissingProcessor::validate_missing_result(&hands, &result));
        assert_eq!(result.player_selections.len(), 2);
    }

    #[test]
    fn test_tian_que_detection() {
        let mut hand = Hand::new();

        for _ in 0..5 {
            hand.add_tile(Tile::M1);
        }
        for _ in 0..4 {
            hand.add_tile(Tile::P1);
            hand.add_tile(Tile::S1);
        }
        assert!(!MissingProcessor::is_tian_que(&hand));

        let mut hand = Hand::new();
        for _ in 0..5 {
            hand.add_tile(Tile::M1);
        }
        for _ in 0..3 {
            hand.add_tile(Tile::P1);
        }
        assert!(MissingProcessor::is_tian_que(&hand));

        let mut hand = Hand::new();
        for _ in 0..10 {
            hand.add_tile(Tile::M1);
        }
        assert!(MissingProcessor::is_tian_que(&hand));
    }
}
