use crate::tile::{Hand, Tile, Suit};

/// 听牌数计算结果
///
/// 符号约定（麻将 shanten number 标准）：
/// | 值 | 含义 |
/// |---|------|
/// | -1 | **已经和牌**（14 张完整） |
/// |  0 | **听牌**（13 张，差 1 张即可和牌） |
/// | >0 | 距离听牌还需补 N 张 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShantenResult {
    /// 标准听牌数（4 面子 + 1 雀头）
    pub standard: i32,
    /// 七对听牌数
    pub seven_pairs: i32,
    /// 最小听牌数（standard 和 seven_pairs 中更小者）
    pub min: i32,
}

impl ShantenResult {
    pub fn new(standard: i32, seven_pairs: i32) -> Self {
        Self {
            standard,
            seven_pairs,
            min: standard.min(seven_pairs),
        }
    }

    /// 是否听牌（shanten == 0）
    pub fn is_tenpai(&self) -> bool {
        self.min == 0
    }

    /// 是否已经和牌（shanten < 0，即 == -1）
    pub fn is_complete(&self) -> bool {
        self.min < 0
    }

    // ==================== 计算入口 ====================

    /// 计算听牌数（标准 + 七对，取 min）
    pub fn calculate(hand: &Hand) -> Self {
        let standard = Self::calculate_standard(hand);
        let seven_pairs = Self::calculate_seven_pairs(hand);
        Self::new(standard, seven_pairs)
    }

    pub fn is_tenpai_hand(hand: &Hand) -> bool {
        Self::calculate(hand).is_tenpai()
    }

    pub fn can_potentially_win(hand: &Hand) -> bool {
        Self::calculate(hand).is_complete()
    }

    // ==================== 标准听牌数（4 面子 + 1 雀头） ====================

    /// 标准听牌数
    fn calculate_standard(hand: &Hand) -> i32 {
        let counts = Self::hand_to_counts(hand);
        let total: usize = counts.iter().map(|&c| c as usize).sum();

        if total == 0 {
            return 8;
        }
        if total > 14 {
            return 8;
        }

        // 14 张：检测是否完整和牌
        if total == 14 {
            if Self::is_complete_standard(&counts) {
                return -1;
            }
            // 14 张不完整：返回近似
            return Self::approx_shanten(&counts);
        }

        // 13 张：检测是否听牌（tenpai）
        if total == 13 {
            if Self::is_tenpai_standard(&counts) {
                return 0;
            }
            // 不是 tenpai：近似 shanten >= 1
            return Self::approx_shanten(&counts).max(1);
        }

        // 其他张数（0~12 或 14+）返回近似
        Self::approx_shanten(&counts)
    }

    fn hand_to_counts(hand: &Hand) -> [u8; 27] {
        std::array::from_fn(|i| {
            let t = Tile::from_index(i).unwrap();
            hand.count(t)
        })
    }

    /// 检测 14 张是否完整标准胡（4 面子 + 1 雀头）
    fn is_complete_standard(counts: &[u8; 27]) -> bool {
        // 尝试每种可能的雀头位置
        for i in 0..27usize {
            if counts[i] >= 2 {
                let mut c = *counts;
                c[i] -= 2;
                // 去掉雀头后应该剩下 12 张 = 4 组面子
                if Self::max_melds(&c) >= 4 && counts_remaining(&c) - 4 * 3 == 0 {
                    return true;
                }
            }
        }
        false
    }

    /// 检测 13 张是否 tenpai（加任意 1 张能变完整）
    fn is_tenpai_standard(counts: &[u8; 27]) -> bool {
        // 尝试加每种 tile 后是否 complete
        for i in 0..27usize {
            if counts[i] < 4 {
                let mut c = *counts;
                c[i] += 1;
                if Self::is_complete_standard(&c) {
                    return true;
                }
            }
        }
        false
    }

    /// 贪心计算最多能组成多少组面子（刻子 + 顺子）
    fn max_melds(counts: &[u8; 27]) -> i32 {
        let mut c = *counts;
        let mut melds = 0;

        // 先刻子
        for i in 0..27 {
            while c[i] >= 3 {
                c[i] -= 3;
                melds += 1;
            }
        }

        // 再顺子（每个花色内部）
        for suit in [Suit::Man, Suit::Pin, Suit::Sou] {
            let base = suit_base(suit);
            for num in 0..7usize {
                let i = base + num;
                while c[i] > 0 && c[i + 1] > 0 && c[i + 2] > 0 {
                    c[i] -= 1;
                    c[i + 1] -= 1;
                    c[i + 2] -= 1;
                    melds += 1;
                }
            }
        }

        melds
    }

    /// 近似 shanten（贪心公式，用于非 tenpai/complete 状态）
    fn approx_shanten(counts: &[u8; 27]) -> i32 {
        let total: i32 = counts.iter().sum::<u8>() as i32;

        // 枚举所有可能的雀头
        let mut best = i32::MAX;

        // 不带雀头
        let melds_no_pair = Self::max_melds(counts);
        let rem_no_pair = total - melds_no_pair * 3;
        let s_no_pair = (4 - melds_no_pair).max(0) + rem_no_pair + if total >= 2 { 1 } else { 0 };
        best = best.min(s_no_pair);

        for i in 0..27usize {
            if counts[i] >= 2 {
                let mut c = *counts;
                c[i] -= 2;
                let melds = Self::max_melds(&c);
                let remaining = total - 2 - melds * 3;
                let s = (4 - melds).max(0) + remaining;
                best = best.min(s);
            }
        }

        best.max(0)
    }

    // ==================== 七对听牌数 ====================

    /// 七对 shanten：
    /// - 14 张 + 7 对 → -1
    /// - 13 张 + 6 对 → 0 (tenpai)
    /// - 13 张 + p 对 (p < 6) → (6 - p)
    fn calculate_seven_pairs(hand: &Hand) -> i32 {
        let mut pair_count = 0u8;
        for tile in Tile::all() {
            if hand.count(tile) >= 2 {
                pair_count += 1;
            }
        }

        let total_tiles: usize = Tile::all().into_iter().map(|t| hand.count(t) as usize).sum();

        match total_tiles {
            14 if pair_count == 7 => -1,
            13 => (6 - pair_count) as i32,
            _ => {
                // 其他张数用 6 - pair_count 近似
                (6 - pair_count) as i32
            }
        }
    }

    // ==================== 听牌牌型枚举 ====================

    /// 获取所有能让当前手牌变成和牌的牌
    pub fn get_waiting_tiles(hand: &Hand) -> Vec<Tile> {
        let mut waiting = Vec::new();

        for tile in Tile::all() {
            if !hand.can_add(tile) {
                continue;
            }
            let mut test_hand = hand.copy();
            test_hand.add_tile(tile);

            if Self::calculate(&test_hand).is_complete() {
                waiting.push(tile);
            }
        }

        waiting
    }
}

fn suit_base(suit: Suit) -> usize {
    match suit {
        Suit::Man => 0,
        Suit::Pin => 9,
        Suit::Sou => 18,
    }
}

fn counts_remaining(counts: &[u8; 27]) -> i32 {
    counts.iter().sum::<u8>() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shanten_empty_hand() {
        let hand = Hand::new();
        let result = ShantenResult::calculate(&hand);
        assert!(result.standard >= 0);
        assert_eq!(result.seven_pairs, 6);
    }

    #[test]
    fn test_complete_standard_all_triplets() {
        // 14 张完整标准胡：M1×3 M2×3 M3×3 P1×3 S1×2
        let mut hand = Hand::new();
        for _ in 0..3 { hand.add_tile(Tile::M1); }
        for _ in 0..3 { hand.add_tile(Tile::M2); }
        for _ in 0..3 { hand.add_tile(Tile::M3); }
        for _ in 0..3 { hand.add_tile(Tile::P1); }
        for _ in 0..2 { hand.add_tile(Tile::S1); }

        let result = ShantenResult::calculate(&hand);
        assert_eq!(result.standard, -1, "完整胡应为 -1，实际 {}", result.standard);
        assert!(result.is_complete());
    }

    #[test]
    fn test_complete_standard_with_sequences() {
        // 14 张：M1M2M3 M4M5M6 M7M8M9 P1P1P1 S1S1
        let mut hand = Hand::new();
        for t in [Tile::M1, Tile::M2, Tile::M3,
                  Tile::M4, Tile::M5, Tile::M6,
                  Tile::M7, Tile::M8, Tile::M9] {
            hand.add_tile(t);
        }
        for _ in 0..3 { hand.add_tile(Tile::P1); }
        for _ in 0..2 { hand.add_tile(Tile::S1); }

        let result = ShantenResult::calculate(&hand);
        assert_eq!(result.standard, -1);
        assert!(result.is_complete());
    }

    #[test]
    fn test_tenpai_standard() {
        // 13 张：M1×2 M2×3 M3×3 P1×3 S1×2 → 等 M1 即可完整
        let mut hand = Hand::new();
        for _ in 0..2 { hand.add_tile(Tile::M1); }
        for _ in 0..3 { hand.add_tile(Tile::M2); }
        for _ in 0..3 { hand.add_tile(Tile::M3); }
        for _ in 0..3 { hand.add_tile(Tile::P1); }
        for _ in 0..2 { hand.add_tile(Tile::S1); }

        let result = ShantenResult::calculate(&hand);
        assert_eq!(result.standard, 0, "tenpai shanten 应为 0，实际 {}", result.standard);
        assert!(result.is_tenpai());
    }

    #[test]
    fn test_tenpai_sequence() {
        // 13 张 tenpai: M2M3M4 M5M6 P1×3 S7S8S9 M9×2 (差 M7 组成完整)
        // 加 M7 → pair M9 + 4 组顺/刻子 = 胡
        let mut hand = Hand::new();
        for t in [Tile::M2, Tile::M3, Tile::M4,
                  Tile::M5, Tile::M6,
                  Tile::S7, Tile::S8, Tile::S9] {
            hand.add_tile(t);
        }
        for _ in 0..3 { hand.add_tile(Tile::P1); }
        for _ in 0..2 { hand.add_tile(Tile::M9); }

        let result = ShantenResult::calculate(&hand);
        assert!(result.is_tenpai(), "应该 tenpai，实际 shanten={}", result.min);
    }

    #[test]
    fn test_complete_seven_pairs() {
        // 14 张完整七对：7 对子
        let mut hand = Hand::new();
        let pairs = [Tile::M1, Tile::M2, Tile::M3,
                     Tile::P1, Tile::P2, Tile::P3,
                     Tile::S1];
        for &t in &pairs {
            for _ in 0..2 { hand.add_tile(t); }
        }
        let result = ShantenResult::calculate(&hand);
        assert_eq!(result.seven_pairs, -1, "完整七对应为 -1");
        assert!(result.is_complete());
    }

    #[test]
    fn test_tenpai_seven_pairs() {
        // 13 张：6 对 + 1 单
        let mut hand = Hand::new();
        let pairs = [Tile::M1, Tile::M2, Tile::M3,
                     Tile::P1, Tile::P2, Tile::P3];
        for &t in &pairs {
            for _ in 0..2 { hand.add_tile(t); }
        }
        hand.add_tile(Tile::S1);

        let result = ShantenResult::calculate(&hand);
        assert_eq!(result.seven_pairs, 0, "七对 tenpai 应为 0，实际 {}", result.seven_pairs);
        assert!(result.is_tenpai());
    }

    #[test]
    fn test_waiting_tiles_simple() {
        // tenpai 手牌：等 M1 完整
        let mut hand = Hand::new();
        for _ in 0..2 { hand.add_tile(Tile::M1); }
        for _ in 0..3 { hand.add_tile(Tile::M2); }
        for _ in 0..3 { hand.add_tile(Tile::M3); }
        for _ in 0..3 { hand.add_tile(Tile::P1); }
        for _ in 0..2 { hand.add_tile(Tile::S1); }

        let waiting = ShantenResult::get_waiting_tiles(&hand);
        assert!(
            waiting.contains(&Tile::M1),
            "应等 M1，实际 {:?}",
            waiting
        );
    }
}
