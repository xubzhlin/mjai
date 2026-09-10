use crate::tile::{Hand, Tile, Suit};
use crate::algo::winning::WinType;
use crate::rules::{FanRule, FanResult, MeldView};
use pyo3::prelude::*;

/// 番型计算器实现（无状态，所有方法内部计算）
#[pyclass(name = "FanCalculator")]
#[derive(Clone, Default)]
pub struct FanCalculator;

#[pymethods]
impl FanCalculator {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

impl FanCalculator {
    /// 检查是否为标准胡牌（4面子+1雀头）
    pub fn is_standard_win(&self, hand: &Hand) -> bool {
        crate::algo::winning::WinChecker::can_win(hand, WinType::Tsumo)
            && !self.is_seven_pairs(hand)
    }

    /// 检查是否为七对胡牌
    pub fn is_seven_pairs(&self, hand: &Hand) -> bool {
        let mut pair_count = 0;
        for i in 0..27 {
            let count = hand.tiles[i] as u32;
            if count >= 2 {
                pair_count += count / 2;
            }
        }
        pair_count == 7
    }

    /// 检查是否为清一色
    pub fn is_pure_hand(&self, hand: &Hand) -> bool {
        let mut suits_present = std::collections::HashSet::new();
        for i in 0..27 {
            if hand.tiles[i] > 0 {
                let tile = Tile::from_index(i).unwrap_or(Tile::M1);
                suits_present.insert(tile.suit());
            }
        }
        suits_present.len() == 1
    }

    /// 检查是否为对对胡
    pub fn is_all_triplets(&self, hand: &Hand) -> bool {
        let counts = hand.tiles;
        for i in 0..27 {
            if counts[i] >= 2 {
                let mut remaining = counts;
                remaining[i] -= 2;
                if Self::can_form_all_triplets(&remaining) {
                    return true;
                }
            }
        }
        false
    }

    fn can_form_all_triplets(counts: &[u8; 27]) -> bool {
        let mut remaining = 0u32;
        for &c in counts.iter() {
            remaining += (c / 3) as u32;
        }
        remaining == 4
    }

    /// 检查是否为金钩钓
    pub fn is_golden_hook(&self, hand: &Hand) -> bool {
        hand.tiles.iter().sum::<u8>() == 1
    }

    /// 计算根的数量（手牌 + 副露杠 中四张同牌的组数）
    /// 副露杠（明杠/暗杠/补杠）按 4 张计入，碰不计
    pub fn calculate_gen(&self, hand: &Hand, melds: &[MeldView]) -> u32 {
        // 合并手牌 + 副露杠 的牌数分布
        let mut counts = [0u8; 27];
        for i in 0..27 {
            counts[i] = hand.tiles[i];
        }
        for meld in melds {
            if meld.is_kong() {
                let idx = meld.tile.to_index();
                counts[idx] = counts[idx].saturating_add(4);
            }
        }

        let mut gen_count: u32 = 0;
        for i in 0..27 {
            let count = counts[i];
            if count >= 4 {
                gen_count += (count / 4) as u32;
            }
        }
        gen_count
    }
}

impl FanRule for FanCalculator {
    fn calculate(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult {
        let mut total_fan: u32 = 1;

        let is_seven = self.is_seven_pairs(hand);
        let is_golden = self.is_golden_hook(hand);

        if is_seven {
            total_fan *= 4;
        } else if is_golden {
            total_fan *= 4;
        } else if self.is_standard_win(hand) {
            total_fan *= 1;

            if self.is_all_triplets(hand) {
                total_fan *= 2;
            }
        }

        if self.is_pure_hand(hand) {
            total_fan *= 4;
        }

        // 根：手牌四张同牌 + 副露杠（明杠/暗杠/补杠），按组数叠加 ×2^N
        let gen_count = self.calculate_gen(hand, melds);
        for _ in 0..gen_count {
            total_fan *= 2;
        }

        if matches!(win_type, WinType::KanShang) {
            total_fan *= 2;
        }

        if matches!(win_type, WinType::QiangGang) {
            total_fan *= 2;
        }

        let fan = total_fan.min(16);

        let extra_bases = if matches!(win_type, WinType::Tsumo | WinType::KanShang) {
            1
        } else {
            0
        };

        FanResult { fan, extra_bases }
    }
}
