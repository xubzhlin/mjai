//! 胡牌判定核心算法（通用层，不随规则变化）
//!
//! 标准胡牌 = 4×面子(3张顺子或刻子) + 1×雀头(2张对子)
//! 七对 = 7×对子
//! 龙七对 = 七对中至少有 1 组 4 张相同
//! 对对胡 = 全面子为刻子（面子不含顺子）

use crate::tile::{Hand, Tile};

/// 胡牌类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinType {
    Tsumo,      // 自摸
    Ron,        // 点炮
    KanShang,   // 杠上花（杠后补摸自摸）
    QiangGang,  // 抢杠胡（他人补杠时我可胡）
}

/// 胡牌方式（用于番型判定）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinMethod {
    Standard,   // 标准胡（4面子+1雀头）
    SevenPairs, // 七对
    LongPairs,  // 龙七对（七对中有一组4张相同）
    AllTriplets,// 对对胡（全面子为刻子）
    SingleWait, // 金钩钓（全部副露+单张听牌）
}

/// 胡牌判定结果
#[derive(Debug, Clone, Default)]
pub struct WinResult {
    pub can_win: bool,
    pub win_type: Option<WinType>,
    pub win_method: Option<WinMethod>,
    pub fan: u32,
    pub score: i32,
}

impl WinResult {
    pub fn not_win() -> Self { Self::default() }
    pub fn winning(win_type: WinType, win_method: WinMethod, fan: u32, score: i32) -> Self {
        Self { can_win: true, win_type: Some(win_type), win_method: Some(win_method), fan, score }
    }
}

/// 胡牌判定器
pub struct WinChecker;

impl WinChecker {
    /// 完整胡牌检查（标准胡+七对+龙七对），返回最佳 WinResult
    pub fn check_win(hand: &Hand, win_type: WinType) -> WinResult {
        // 七对检查
        if let Some(result) = Self::check_seven_pairs(hand, win_type) {
            return result;
        }
        // 标准胡检查
        if let Some(result) = Self::check_standard_win(hand, win_type) {
            return result;
        }
        WinResult::not_win()
    }

    /// 简化胡牌检查（只返回 can_win: bool）
    pub fn can_win(hand: &Hand, win_type: WinType) -> bool {
        Self::check_win(hand, win_type).can_win
    }

    /// 标准胡牌检查（4面子+1雀头）
    fn check_standard_win(hand: &Hand, win_type: WinType) -> Option<WinResult> {
        let counts = hand.tiles;

        // 尝试每一组可能的雀头位置
        for i in 0..27 {
            if counts[i] >= 2 {
                let mut remaining = counts;
                remaining[i] -= 2;
                if Self::can_form_all_melds(&remaining) {
                    // 检查是否对对胡（没有顺子）
                    let method = if Self::can_form_all_triplets(&remaining) {
                        WinMethod::AllTriplets
                    } else {
                        WinMethod::Standard
                    };
                    return Some(WinResult::winning(win_type, method, 1, 0));
                }
            }
        }
        None
    }

    /// 七对检查
    fn check_seven_pairs(hand: &Hand, win_type: WinType) -> Option<WinResult> {
        let counts = hand.tiles;
        let mut pairs = 0u32;
        let mut has_four = false;

        for &c in counts.iter() {
            if c % 2 != 0 { return None; }
            pairs += (c / 2) as u32;
            if c >= 4 { has_four = true; }
        }

        if pairs == 7 {
            let method = if has_four { WinMethod::LongPairs } else { WinMethod::SevenPairs };
            Some(WinResult::winning(win_type, method, 4, 0))
        } else {
            None
        }
    }

    /// 龙七对（通过七对内部 has_four 标志检测）
    fn check_long_pairs(_hand: &Hand, _win_type: WinType) -> Option<WinResult> {
        None // 由 check_seven_pairs 的 has_four 标志处理
    }

    /// 对对胡（由 check_standard_win 内部 AllTriplets 处理）
    fn check_all_triplets(_hand: &Hand, _win_type: WinType) -> Option<WinResult> {
        None
    }

    /// 金钩钓（简化判断：手牌只有 1 张）
    fn check_single_wait(hand: &Hand, _win_type: WinType) -> Option<WinResult> {
        let total: usize = hand.tiles.iter().map(|&c| c as usize).sum();
        if total == 1 {
            Some(WinResult::winning(_win_type, WinMethod::SingleWait, 4, 0))
        } else {
            None
        }
    }

    /// 检查是否能组成 4 组面子（递归回溯：刻子优先）
    fn can_form_all_melds(counts: &[u8; 27]) -> bool {
        let mut counts = *counts;
        Self::try_melds(&mut counts)
    }

    fn try_melds(counts: &mut [u8; 27]) -> bool {
        // 找到第一个还有牌的位置
        let start = match counts.iter().position(|&c| c > 0) {
            Some(p) => p,
            None => return true, // 全部用完了
        };

        // 尝试作为刻子（同花色同点数3张）
        if counts[start] >= 3 {
            counts[start] -= 3;
            if Self::try_melds(counts) { counts[start] += 3; return true; }
            counts[start] += 3;
        }

        // 尝试作为顺子起点（万筒条各9张，start..start+2 在同花色范围内）
        let suit_start = (start / 9) * 9;
        if start + 2 < suit_start + 9 && counts[start] > 0 && counts[start + 1] > 0 && counts[start + 2] > 0 {
            counts[start] -= 1;
            counts[start + 1] -= 1;
            counts[start + 2] -= 1;
            if Self::try_melds(counts) {
                counts[start] += 1; counts[start + 1] += 1; counts[start + 2] += 1;
                return true;
            }
            counts[start] += 1; counts[start + 1] += 1; counts[start + 2] += 1;
        }

        false
    }

    /// 检查是否能组成 4 组刻子（对对胡）
    fn can_form_all_triplets(counts: &[u8; 27]) -> bool {
        let mut remaining = 4u32;
        for &c in counts.iter() {
            remaining -= (c / 3) as u32;
        }
        remaining == 0
    }

    /// 9 张局部刻子检测（用于预计算）
    fn can_form_group_melds(counts: &[u8; 9]) -> bool {
        let mut counts = *counts;
        Self::try_melds_local(&mut counts)
    }

    fn try_melds_local(counts: &mut [u8; 9]) -> bool {
        let start = match counts.iter().position(|&c| c > 0) {
            Some(p) => p,
            None => return true,
        };
        if counts[start] >= 3 {
            counts[start] -= 3;
            if Self::try_melds_local(counts) { counts[start] += 3; return true; }
            counts[start] += 3;
        }
        if start + 2 < 9 && counts[start] > 0 && counts[start + 1] > 0 && counts[start + 2] > 0 {
            counts[start] -= 1;
            counts[start + 1] -= 1;
            counts[start + 2] -= 1;
            if Self::try_melds_local(counts) {
                counts[start] += 1; counts[start + 1] += 1; counts[start + 2] += 1;
                return true;
            }
            counts[start] += 1; counts[start + 1] += 1; counts[start + 2] += 1;
        }
        false
    }

    pub fn get_possible_win_methods(hand: &Hand, win_type: WinType) -> Vec<WinMethod> {
        let mut methods = Vec::new();
        if Self::check_seven_pairs(hand, win_type).is_some() {
            methods.push(WinMethod::SevenPairs);
        }
        if Self::check_standard_win(hand, win_type).is_some() {
            methods.push(WinMethod::Standard);
        }
        methods
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Tile;

    #[test]
    fn test_basic_win() {
        // 1m1m1m 2m2m2m 3m3m3m 4m4m4m 5m5m
        let tiles = [
            Tile::M1, Tile::M1, Tile::M1,
            Tile::M2, Tile::M2, Tile::M2,
            Tile::M3, Tile::M3, Tile::M3,
            Tile::M4, Tile::M4, Tile::M4,
            Tile::M5, Tile::M5,
        ];
        let hand = Hand::from_tiles(&tiles);
        assert!(WinChecker::can_win(&hand, WinType::Tsumo));
    }

    #[test]
    fn test_seven_pairs() {
        let tiles = [
            Tile::M1, Tile::M1,
            Tile::M2, Tile::M2,
            Tile::M3, Tile::M3,
            Tile::P1, Tile::P1,
            Tile::P2, Tile::P2,
            Tile::S1, Tile::S1,
            Tile::S2, Tile::S2,
        ];
        let hand = Hand::from_tiles(&tiles);
        let result = WinChecker::check_win(&hand, WinType::Tsumo);
        assert!(result.can_win);
        assert!(matches!(result.win_method, Some(WinMethod::SevenPairs)));
    }

    #[test]
    fn test_not_win() {
        let tiles = [Tile::M1, Tile::M2, Tile::M3, Tile::P1];
        let hand = Hand::from_tiles(&tiles);
        assert!(!WinChecker::can_win(&hand, WinType::Tsumo));
    }
}
