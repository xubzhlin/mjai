use crate::tile::{Hand, Tile, Suit};
use crate::rules::MissingRule;
use anyhow::{Result, anyhow};
use rand::Rng;
use pyo3::prelude::*;

/// 定缺规则实现
#[pyclass(name = "MissingRuleCalculator")]
#[derive(Clone)]
pub struct MissingRuleCalculator {
    #[pyo3(get)]
    pub enabled: bool,
    #[pyo3(get)]
    pub tian_que_threshold: f64,
}

#[pymethods]
impl MissingRuleCalculator {
    #[new]
    pub fn new_py() -> Self {
        Self { enabled: true, tian_que_threshold: 0.75 }
    }
}

impl MissingRuleCalculator {
    pub fn new(enabled: bool, tian_que_threshold: f64) -> Self {
        Self {
            enabled,
            tian_que_threshold,
        }
    }

    /// 检查是否为天缺（只有两个花色）
    pub fn is_tian_que(&self, hand: &Hand) -> Option<(Suit, Suit)> {
        if !self.enabled {
            return None;
        }

        let mut suits_present = std::collections::HashSet::new();
        for i in 0..27 {
            if hand.tiles[i] > 0 {
                let tile = Tile::from_index(i).unwrap_or(Tile::M1);
                suits_present.insert(tile.suit());
            }
        }

        if suits_present.len() == 2 {
            let mut suits: Vec<Suit> = suits_present.into_iter().collect();
            return Some((suits[0], suits[1]));
        }

        None
    }

    /// 检查是否为天缺（只有一个花色）
    pub fn is_tian_que_single(&self, hand: &Hand) -> Option<Suit> {
        if !self.enabled {
            return None;
        }

        let mut suits_present = std::collections::HashSet::new();
        for i in 0..27 {
            if hand.tiles[i] > 0 {
                let tile = Tile::from_index(i).unwrap_or(Tile::M1);
                suits_present.insert(tile.suit());
            }
        }

        if suits_present.len() == 1 {
            return Some(suits_present.into_iter().next().unwrap());
        }

        None
    }

    /// AI策略：选择定缺花色
    pub fn select_missing_suit_internal(&self, hand: &Hand) -> Result<Suit> {
        if !self.enabled {
            return Err(anyhow!("定缺已禁用"));
        }

        // 检查天缺情况
        if let Some(tian_que_suits) = self.is_tian_que(hand) {
            // 两个花色：75%选空门，25%保留冲清一色
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.tian_que_threshold {
                // 选空门（不在天缺花色中的花色）
                let missing_suit = match tian_que_suits {
                    (Suit::Man, Suit::Pin) => Suit::Sou,
                    (Suit::Man, Suit::Sou) => Suit::Pin,
                    (Suit::Pin, Suit::Sou) => Suit::Man,
                    _ => unreachable!(),
                };
                return Ok(missing_suit);
            } else {
                // 随机选择一个天缺花色
                return Ok(rng.gen_bool(0.5).then(|| tian_que_suits.0).unwrap_or(tian_que_suits.1));
            }
        }

        if let Some(single_suit) = self.is_tian_que_single(hand) {
            // 只有一个花色：在空门中选
            let missing_suit = match single_suit {
                Suit::Man => Suit::Pin,
                Suit::Pin => Suit::Man,
                Suit::Sou => Suit::Man,
            };
            return Ok(missing_suit);
        }

        // 正常情况：选择最少花色
        let mut suit_counts = [(Suit::Man, 0), (Suit::Pin, 0), (Suit::Sou, 0)];

        for i in 0..27 {
            let tile = Tile::from_index(i).unwrap_or(Tile::M1);
            if hand.tiles[i] > 0 {
                match tile.suit() {
                    Suit::Man => suit_counts[0].1 += hand.tiles[i] as usize,
                    Suit::Pin => suit_counts[1].1 += hand.tiles[i] as usize,
                    Suit::Sou => suit_counts[2].1 += hand.tiles[i] as usize,
                }
            }
        }

        // 按牌数排序，最少花色优先
        suit_counts.sort_by_key(|(_, count)| *count);

        Ok(suit_counts[0].0)
    }

    /// 验证定缺是否合法
    pub fn is_valid_missing_internal(&self, hand: &Hand, suit: Suit) -> bool {
        if !self.enabled {
            return false;
        }

        // 检查手牌中是否有所选花色的牌
        for i in 0..27 {
            let tile = Tile::from_index(i).unwrap_or(Tile::M1);
            if tile.suit() == suit && hand.tiles[i] > 0 {
                return false;
            }
        }

        true
    }

    /// 定缺后更新手牌状态
    pub fn apply_missing(&self, hand: &mut Hand, suit: Suit) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if !self.is_valid_missing_internal(hand, suit) {
            return Err(anyhow!("定缺不合法"));
        }

        hand.set_missing_suit(Some(suit));

        Ok(())
    }
}

impl MissingRule for MissingRuleCalculator {
    fn is_valid_missing(&self, hand: &Hand, suit: Suit) -> bool {
        self.is_valid_missing_internal(hand, suit)
    }

    fn select_missing_suit(&self, hand: &Hand) -> Suit {
        self.select_missing_suit_internal(hand).unwrap_or(Suit::Man)
    }
}

/// 定缺状态管理
pub struct MissingManager {
    pub missing_rule: MissingRuleCalculator,
    pub missing_completed: bool,
    pub missing_suits: [Option<Suit>; 4],
    pub tian_que_flags: [bool; 4],
}

impl MissingManager {
    pub fn new(enabled: bool, tian_que_threshold: f64) -> Self {
        Self {
            missing_rule: MissingRuleCalculator::new(enabled, tian_que_threshold),
            missing_completed: false,
            missing_suits: [None; 4],
            tian_que_flags: [false; 4],
        }
    }

    pub fn start_missing_phase(&mut self, hands: [&Hand; 4]) -> Result<Vec<(usize, Suit)>> {
        let mut recommendations = Vec::new();

        for (i, hand) in hands.iter().enumerate() {
            if let Ok(suit) = self.missing_rule.select_missing_suit_internal(hand) {
                recommendations.push((i, suit));
            }
        }

        Ok(recommendations)
    }

    pub fn execute_missing(&mut self, player_index: usize, suit: Suit) -> Result<bool> {
        if self.missing_completed {
            return Ok(false);
        }

        // 检查定缺是否合法
        if !self.missing_rule.is_valid_missing_internal(&crate::tile::Hand::default(), suit) {
            return Ok(false);
        }

        // 检查天缺
        let is_tian_que = self.missing_rule.is_tian_que(&crate::tile::Hand::default()).is_some();

        // 执行定缺
        self.missing_suits[player_index] = Some(suit);
        self.tian_que_flags[player_index] = is_tian_que;

        // 检查是否所有玩家都完成了定缺
        let all_completed = self.missing_suits.iter().all(|suit| suit.is_some());
        if all_completed {
            self.missing_completed = true;
        }

        Ok(true)
    }

    pub fn is_missing_completed(&self) -> bool {
        self.missing_completed
    }

    pub fn get_missing_suit(&self, player_index: usize) -> Option<Suit> {
        self.missing_suits[player_index]
    }

    pub fn is_tian_que(&self, player_index: usize) -> bool {
        self.tian_que_flags[player_index]
    }

    pub fn reset(&mut self) {
        self.missing_completed = false;
        self.missing_suits = [None; 4];
        self.tian_que_flags = [false; 4];
    }

    /// 获取所有玩家的定缺信息
    pub fn get_all_missing_info(&self) -> Vec<(usize, Option<Suit>, bool)> {
        self.missing_suits.iter().enumerate()
            .map(|(i, suit)| (i, *suit, self.tian_que_flags[i]))
            .collect()
    }
}

/// 定缺结果
pub struct MissingResult {
    pub suit: Suit,
    pub success: bool,
    pub is_tian_que: bool,
    pub player_index: usize,
}

impl MissingResult {
    pub fn new(suit: Suit, success: bool, is_tian_que: bool, player_index: usize) -> Self {
        Self {
            suit,
            success,
            is_tian_que,
            player_index,
        }
    }
}
