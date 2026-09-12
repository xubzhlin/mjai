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

    /// 检查是否为金钩钓（胡后 4 副露 + 手牌仅剩 1 个雀头 = 2 张同牌）
    pub fn is_golden_hook(&self, hand: &Hand) -> bool {
        let total: u32 = hand.tiles.iter().map(|&c| c as u32).sum();
        total == 2 && hand.tiles.iter().any(|&c| c == 2)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{Tile, Suit};
    use crate::rules::{FanRule, MeldView, MeldKind};
    use crate::algo::winning::WinType;

    fn fan_calc() -> FanCalculator { FanCalculator::new() }

    fn meld_views(kong_tiles: &[Tile]) -> Vec<MeldView> {
        kong_tiles.iter().map(|&t| MeldView { tile: t, kind: MeldKind::ConcealedKong }).collect()
    }

    // ===== 金钩钓检测 =====
    #[test]
    fn test_golden_hook_sum1_is_false() {
        // 胡前 13 张手牌（听牌时）sum=1 不是胡后金钩钓
        let mut hand = Hand::new();
        hand.add_tile(Tile::M1);  // sum=1，但这是听牌态不是胡后态
        // 真实场景：胡后金钩钓手牌是 sum=2（雀头）
        assert!(!fan_calc().is_golden_hook(&hand), "sum=1 的听牌态不应是金钩钓");
    }

    #[test]
    fn test_golden_hook_correct_detection() {
        // 胡后金钩钓：手牌仅 1 个对子（雀头），sum=2 且有 count==2 的牌
        let mut hand = Hand::new();
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::M1);
        assert!(fan_calc().is_golden_hook(&hand), "胡后金钩钓手牌 sum=2 且有对子");
    }

    #[test]
    fn test_golden_hook_sum3_is_false() {
        // 胡后手牌 sum=3 不可能是金钩钓（金钩钓必须是 4 副露 + 1 雀头 = 手牌 sum=2）
        let mut hand = Hand::new();
        for _ in 0..3 { hand.add_tile(Tile::M1); }
        assert!(!fan_calc().is_golden_hook(&hand));
    }

    // ===== 番型计算 =====
    #[test]
    fn test_pinghu_fan() {
        // 标准胡：1 番（平胡）
        let fc = fan_calc();
        // 构造 4 面子 + 1 雀头，非清一色、非对对胡
        let hand = Hand::from_tiles(&[
            Tile::M1, Tile::M2, Tile::M3,
            Tile::M4, Tile::M5, Tile::M6,
            Tile::P1, Tile::P2, Tile::P3,
            Tile::S1, Tile::S2, Tile::S3,
            Tile::M7, Tile::M7,
        ]);
        let result = fc.calculate(&hand, &[], WinType::Tsumo);
        assert!(!fc.is_seven_pairs(&hand));
        assert!(!fc.is_golden_hook(&hand));
        assert_eq!(result.fan, 1, "平胡应为 1 番");
        assert_eq!(result.extra_bases, 1, "自摸应加底 1");
    }

    #[test]
    fn test_seven_pairs_fan() {
        // 七对：4 番
        let hand = Hand::from_tiles(&[
            Tile::M1, Tile::M1,
            Tile::M2, Tile::M2,
            Tile::M3, Tile::M3,
            Tile::P1, Tile::P1,
            Tile::P2, Tile::P2,
            Tile::P3, Tile::P3,
            Tile::S1, Tile::S1,
        ]);
        let fc = fan_calc();
        let result = fc.calculate(&hand, &[], WinType::Tsumo);
        assert_eq!(result.fan, 4, "七对应为 4 番");
    }

    #[test]
    fn test_gen_multiplier() {
        // 手牌 4 张 M1 + 副露暗杠 M2 = 2 组根 → ×2^2 = ×4
        let mut hand = Hand::from_tiles(&[
            Tile::M1, Tile::M1, Tile::M1, Tile::M1,  // 根 1
            Tile::P1, Tile::P2, Tile::P3,
            Tile::P4, Tile::P5, Tile::P6,
            Tile::S1, Tile::S2, Tile::S3,
            Tile::S4,
        ]);
        // WinChecker 会因手牌 4 张 M1 判定为七对或对对胡，手动验证根
        let fc = fan_calc();
        let views = meld_views(&[Tile::M2]);  // 暗杠 M2 = 根 2
        let gen = fc.calculate_gen(&hand, &views);
        assert_eq!(gen, 2, "手牌 4 张 + 暗杠 = 2 组根");
    }

    #[test]
    fn test_kan_shang_x2() {
        // 杠上花：×2
        let hand = Hand::from_tiles(&[
            Tile::M1, Tile::M2, Tile::M3,
            Tile::M4, Tile::M5, Tile::M6,
            Tile::P1, Tile::P2, Tile::P3,
            Tile::S1, Tile::S2, Tile::S3,
            Tile::M7, Tile::M7,
        ]);
        let fc = fan_calc();
        let tsumo = fc.calculate(&hand, &[], WinType::Tsumo);
        let kanshang = fc.calculate(&hand, &[], WinType::KanShang);
        assert_eq!(kanshang.fan, tsumo.fan * 2, "杠上花应 ×2");
    }

    #[test]
    fn test_qiang_gang_x2() {
        // 抢杠胡：×2
        let hand = Hand::from_tiles(&[
            Tile::M1, Tile::M2, Tile::M3,
            Tile::M4, Tile::M5, Tile::M6,
            Tile::P1, Tile::P2, Tile::P3,
            Tile::S1, Tile::S2, Tile::S3,
            Tile::M7, Tile::M7,
        ]);
        let fc = fan_calc();
        let ron = fc.calculate(&hand, &[], WinType::Ron);
        let qiang = fc.calculate(&hand, &[], WinType::QiangGang);
        assert_eq!(qiang.fan, ron.fan * 2, "抢杠胡应 ×2");
    }

    #[test]
    fn test_fan_cap_16() {
        // 理论最大：清一色(×4) × 七对(×4) × 根(×2×2) = 128 → 封顶 16
        let fc = fan_calc();
        // 构造清龙七对：清一色万 + 两组 4 张
        let mut tiles = vec![];
        for _ in 0..4 { tiles.push(Tile::M1); }
        for _ in 0..4 { tiles.push(Tile::M2); }
        for _ in 0..2 { tiles.push(Tile::M3); }
        for _ in 0..2 { tiles.push(Tile::M4); }
        for _ in 0..2 { tiles.push(Tile::M5); }
        // 4+4+2+2+2 = 14 张
        assert_eq!(tiles.len(), 14);
        let hand = Hand::from_tiles(&tiles);
        let result = fc.calculate(&hand, &[], WinType::Tsumo);
        assert!(result.fan <= 16, "番数必须封顶 16，实际得到 {}", result.fan);
    }
}
