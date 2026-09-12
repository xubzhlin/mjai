//! 观测编码器
//!
//! 将游戏状态编码为 `[C, 27]` 格式的观测张量，展平为 `Vec<f32>` 总长度 = C × 27。
//! 设计文档 §1.8 规定的通道布局（2026-09-10 修订版 v3 — 含弃牌时序）：
//!
//! | 通道组                        | 通道数      | 说明                                                                 |
//! |-------------------------------|-------------|----------------------------------------------------------------------|
//! | 手牌计数标记                   | 5           | 每通道标记哪些牌恰有 0/1/2/3/4 张 (本家)                                |
//! | 四家副露                       | 16          | 4 玩家 × {Pong, ExposedKong, ConcealedKong, AddKong}                   |
//! | 四家弃牌河                     | 4           | 每玩家 1 通道，27 维归一化弃牌计数 (count / 4)                          |
//! | 四家缺门                       | 4           | 缺门花色对应的 9 个位置标记为 1                                         |
//! | 剩余牌                         | 1           | (4 - 已暴露牌数) / 4；已胡玩家手牌不计入                                  |
//! | 轮次标量                       | 1           | turn / 80 广播到 27 维                                                |
//! | 庄家标量                       | 1           | dealer / 3 广播                                                       |
//! | 牌墙剩余                       | 1           | wall.remaining_ratio() 广播（含 NaN 保护）                             |
//! | 四家已胡标记                   | 4           | 玩家已胡 → 全 27 维 = 1                                               |
//! | 换三张方向                     | 3           | one-hot 广播 (Down / Across / Up)                                    |
//! | 杠上花标记                     | 1           | kan_shang_active 广播                                                  |
//! | 本家是否听牌                   | 1           | is_tenpai() ? 1 : 0 广播（缺门清干净 + shanten==0）                     |
//! | 本家听牌数                     | 1           | min(waiting_count, 9) / 9 广播                                         |
//! | 番型进度 (3种)                 | 3           | 清一色/对对胡/七对进度 广播                                              |
//! | 根进度                         | 1           | 手牌中四张同牌组数 / 4 广播                                             |
//! | 四家天缺标记                   | 4           | encoder 自推断：初始缺门牌数 == 0 → 天缺；否则从 PlayerState.is_natural_missing 取 |
//! | 四家 tsumogiri 占比            | 4           | 该玩家 tsumogiri 数 / 弃牌总数 广播                                     |
//! | **四家清缺进度（方案 A）**     | **4**       | **已弃缺门牌数 / 初始缺门牌数；初始==0 则固定为 1.0（天缺瞬间清完）**  |
//! | **四家最近 3 张弃牌（方案 B）**| **4 × 3**   | **每玩家最近 3 张弃牌 one-hot，从最新开始倒序（ch × 3 × 27）**        |
//!
//! **合计 C = 71 通道**，**展平 = 1917 维**（无截断、无填充）。
//!
//! ### 川麻将牌时序为什么需要显式编码？
//!
//! - **日麻没有"定缺"约束**：牌河顺序相对无意义，Suphx/Mortal 只用计数也够
//! - **川麻有强制清缺阶段**：非天缺玩家前 N 回合弃牌几乎都是缺门牌
//!   → "什么时候弃完缺门"是玩家进入正常打牌的**时间分界线**
//! - **方案 A（清缺进度）** 以极低成本捕获这个分界线（每玩家 1 通道广播）
//! - **方案 B（最近 3 张 one-hot）** 捕获"当前状态"下玩家真实弃牌模式，
//!   可直接用于判断"这张牌是否安全"。每张 27 维 one-hot，4 玩家共 12 通道
//!
//! encoder **自包含**：方案 A 的初始缺门牌数 = 已弃缺门牌数 + 手牌中剩余缺门牌数，
//! 实时从 `discarded` + `hand.count_by_suit` 反推，**无需 game.rs 预填充**。
//! 方案 B 直接读 `discarded.iter().rev().take(3)`。
//!
//! ### 已移除通道（2026-09-10 v2）
//!
//! - ~~已知牌累计 (1ch)~~：和剩余牌通道完全冗余，两者互补恒等于 1.0
//! - ~~四家对手手牌推断 (4ch)~~：均匀先验 (4-exposed)/(对手数×4)，信息密度极低
//!
//! ### 为什么"对手听牌/番型进度/根"不显式编码？
//!
//! 对手手牌是黑盒，弃牌河 + 副露 + 缺门花色不足以反推精确手牌结构；
//! 这些特征只能从四家公开通道（缺门/弃牌河/副露/天缺/tsumogiri/清缺进度/最近3张）
//! 自己学习概率推断——这正是残差网络该发挥作用的地方。Suphx/Mortal 也只显式编码本家。

use crate::tile::{Tile, Suit, Hand};
use crate::state::{PlayerState, MeldType};
use crate::arena::Board;
use crate::rules::SwapDirection;
use pyo3::prelude::*;

/// 总通道数 = 55 (v2) + 4 (清缺进度) + 12 (最近 3 张弃牌时序)
const NUM_CHANNELS: usize = 71;
/// 展平后总特征数 = NUM_CHANNELS × 27
const TOTAL_FEATURES: usize = NUM_CHANNELS * 27;

/// 单副牌每种牌的张数上限
const MAX_PER_TILE: u8 = 4;

/// 方案 B 固定参数：每家保留最近几张弃牌 one-hot
const RECENT_DISCARD_WINDOW: usize = 3;

/// 默认观测编码器
#[derive(Debug)]
#[pyclass]
pub struct ObservationEncoder {
    feature_count: usize,
}

impl Default for ObservationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationEncoder {
    pub fn new() -> Self {
        Self {
            feature_count: TOTAL_FEATURES,
        }
    }

    /// 将标量广播为 27 维
    fn broadcast(v: f32) -> [f32; 27] {
        [v; 27]
    }

    /// 编码完整观测，输出 `Vec<f32>` 长度 = C × 27
    pub fn encode(&self, board: &Board, player_id: usize) -> Vec<f32> {
        let player = &board.players[player_id];
        let num_players = board.players.len();

        let mut features = Vec::with_capacity(TOTAL_FEATURES);

        // ========== 1. 手牌计数标记：5 通道 ==========
        for count in 0..=4u8 {
            for tile in Tile::all() {
                let v = if player.hand.count(tile) == count { 1.0 } else { 0.0 };
                features.push(v);
            }
        }

        // ========== 2. 四家副露：4 × 4 = 16 通道 ==========
        let meld_types: [MeldType; 4] = [
            MeldType::Pong,
            MeldType::ExposedKong,
            MeldType::ConcealedKong,
            MeldType::AddKong,
        ];
        for p in &board.players {
            for mt in &meld_types {
                let mut channel = [0.0f32; 27];
                for meld in &p.melds {
                    if meld.meld_type == *mt {
                        let idx = meld.tile.to_index();
                        channel[idx] = 1.0;
                    }
                }
                features.extend_from_slice(&channel);
            }
        }

        // ========== 3. 四家弃牌河：4 通道 ==========
        for p in &board.players {
            let mut channel = [0.0f32; 27];
            for d in &p.discarded {
                let idx = d.tile.to_index();
                channel[idx] += 1.0;
            }
            for v in channel.iter_mut() {
                *v = (*v / MAX_PER_TILE as f32).min(1.0);
            }
            features.extend_from_slice(&channel);
        }

        // ========== 4. 四家缺门：4 通道 ==========
        for p in &board.players {
            let mut channel = [0.0f32; 27];
            if let Some(suit) = p.hand.missing_suit() {
                for tile in Tile::all() {
                    if tile.suit() == suit {
                        channel[tile.to_index()] = 1.0;
                    }
                }
            }
            features.extend_from_slice(&channel);
        }

        // ========== 5. 剩余牌：1 通道 ==========
        // 暴露牌数 = 存活玩家手牌 + 所有副露 + 所有弃牌
        // 已胡玩家不再碰/杠/胡，他们的手牌不计入"剩余牌可被对手持有"的范围
        let mut exposed_count = [0u8; 27];
        for p in &board.players {
            if !p.has_won {
                for tile in Tile::all() {
                    exposed_count[tile.to_index()] += p.hand.count(tile);
                }
            }
            for meld in &p.melds {
                let idx = meld.tile.to_index();
                match meld.meld_type {
                    MeldType::Pong => exposed_count[idx] += 3,
                    MeldType::ExposedKong | MeldType::ConcealedKong | MeldType::AddKong => {
                        exposed_count[idx] += 4
                    }
                }
            }
            for d in &p.discarded {
                exposed_count[d.tile.to_index()] += 1;
            }
        }
        for tile in Tile::all() {
            let idx = tile.to_index();
            let remaining = (MAX_PER_TILE as i16 - exposed_count[idx] as i16).max(0) as u8;
            features.push(remaining as f32 / MAX_PER_TILE as f32);
        }

        // ========== 6. 轮次标量：1 通道 (broadcast) ==========
        let turn_ratio = (board.turn as f32 / 80.0).min(1.0);
        features.extend_from_slice(&Self::broadcast(turn_ratio));

        // ========== 7. 庄家标量：1 通道 (broadcast) ==========
        let dealer_ratio = board.dealer_position as f32 / (num_players.max(1) - 1).max(1) as f32;
        features.extend_from_slice(&Self::broadcast(dealer_ratio));

        // ========== 8. 牌墙剩余：1 通道 (broadcast) ==========
        let wall_ratio_raw = board.wall.remaining_ratio();
        let wall_ratio = if wall_ratio_raw.is_finite() {
            (wall_ratio_raw as f32).min(1.0)
        } else {
            0.0 // 空墙或异常情况
        };
        features.extend_from_slice(&Self::broadcast(wall_ratio));

        // ========== 9. 四家已胡标记：4 通道 (broadcast) ==========
        for p in &board.players {
            let v = if p.has_won { 1.0 } else { 0.0 };
            features.extend_from_slice(&Self::broadcast(v));
        }

        // ========== 10. 换三张方向：3 通道 (one-hot broadcast) ==========
        let direction_onehot: [f32; 3] = match board.swap_direction {
            Some(SwapDirection::Down) => [1.0, 0.0, 0.0],
            Some(SwapDirection::Across) => [0.0, 1.0, 0.0],
            Some(SwapDirection::Up) => [0.0, 0.0, 1.0],
            None => [0.0, 0.0, 0.0],
        };
        for i in 0..3 {
            features.extend_from_slice(&Self::broadcast(direction_onehot[i]));
        }

        // ========== 11. 杠上花标记：1 通道 (broadcast) ==========
        let kan_shang = if board.kan_shang_active { 1.0 } else { 0.0 };
        features.extend_from_slice(&Self::broadcast(kan_shang));

        // ========== 12. 本家是否听牌：1 通道 (broadcast) ==========
        let is_tenpai = if player.hand.is_tenpai() { 1.0 } else { 0.0 };
        features.extend_from_slice(&Self::broadcast(is_tenpai));

        // ========== 13. 本家听牌数：1 通道 (broadcast) ==========
        let wait_count = (player.hand.waiting_count() as f32).min(9.0) / 9.0;
        features.extend_from_slice(&Self::broadcast(wait_count));

        // ========== 14. 番型进度：3 通道 (broadcast) ==========
        let qing_yi_se = self.qing_yi_se_progress(&player.hand);
        let dui_dui = self.dui_dui_hu_progress(&player.hand);
        let qi_dui = self.qi_dui_progress(&player.hand);
        features.extend_from_slice(&Self::broadcast(qing_yi_se));
        features.extend_from_slice(&Self::broadcast(dui_dui));
        features.extend_from_slice(&Self::broadcast(qi_dui));

        // ========== 15. 根进度：1 通道 (broadcast) ==========
        let gen_progress = self.gen_progress(&player.hand);
        features.extend_from_slice(&Self::broadcast(gen_progress));

        // ========== 16. 四家天缺标记：4 通道 (broadcast) ==========
        // encoder 自推断：如果玩家已定缺且初始缺门牌数 == 0 → 天缺
        // 否则回退到 PlayerState.is_natural_missing（供 game.rs 显式设置时覆盖）
        for p in &board.players {
            let is_tian_que = if let Some(suit) = p.hand.missing_suit() {
                let initial = self.initial_missing_count(p, suit);
                initial == 0 || p.is_natural_missing
            } else {
                false
            };
            let v = if is_tian_que { 1.0 } else { 0.0 };
            features.extend_from_slice(&Self::broadcast(v));
        }

        // ========== 17. 四家 tsumogiri 占比：4 通道 (broadcast) ==========
        for p in &board.players {
            let total = p.discarded.len();
            let ratio = if total == 0 {
                0.0
            } else {
                let tsumo_count = p.discarded.iter().filter(|d| d.tsumogiri).count();
                tsumo_count as f32 / total as f32
            };
            features.extend_from_slice(&Self::broadcast(ratio));
        }

        // ========== 18. 四家清缺进度（方案 A）：4 通道 (broadcast) ==========
        // 川麻将牌时序的核心信号：
        // 非天缺玩家前 N 回合弃牌几乎都是缺门牌
        // → 已弃缺门牌数 / 初始缺门牌数 = 该玩家的"清缺进度"
        // → 进度 = 1.0 意味着清完缺门，进入正常打牌阶段
        for p in &board.players {
            let progress = self.compute_clear_missing_progress(p);
            features.extend_from_slice(&Self::broadcast(progress));
        }

        // ========== 19. 四家最近 3 张弃牌（方案 B）：4 × 3 = 12 通道 ==========
        // 每张弃牌 27 维 one-hot，从最新一张开始倒序
        // 如果该玩家弃牌不足 3 张，剩余通道填全 0
        // 4 玩家 × 3 张 = 12 通道
        for p in &board.players {
            // 从最新开始倒序取 RECENT_DISCARD_WINDOW 张
            let recent: Vec<_> = p
                .discarded
                .iter()
                .rev()
                .take(RECENT_DISCARD_WINDOW)
                .collect();
            // 按时间从新到旧填入通道（通道 0 = 最新）
            for slot in 0..RECENT_DISCARD_WINDOW {
                let mut channel = [0.0f32; 27];
                if let Some(d) = recent.get(slot) {
                    let idx = d.tile.to_index();
                    channel[idx] = 1.0;
                }
                features.extend_from_slice(&channel);
            }
        }

        // 最终断言：维度必须精确
        debug_assert_eq!(
            features.len(),
            TOTAL_FEATURES,
            "encoder output size mismatch: expected {} got {}",
            TOTAL_FEATURES,
            features.len()
        );

        features
    }

    pub fn get_feature_count(&self) -> usize {
        self.feature_count
    }

    // ==================== 辅助方法 ====================

    /// 清一色进度：最大花色张数 / 14
    fn qing_yi_se_progress(&self, hand: &Hand) -> f32 {
        let dist = hand.suit_distribution();
        let max = dist.iter().max().copied().unwrap_or(0);
        max as f32 / 14.0
    }

    /// 对对胡进度：刻子潜力数 / 4（每张牌 count>=3 贡献 count/3）
    fn dui_dui_hu_progress(&self, hand: &Hand) -> f32 {
        let mut triplet_room = 0;
        for tile in Tile::all() {
            let c = hand.count(tile);
            if c >= 3 {
                triplet_room += c / 3;
            }
        }
        triplet_room as f32 / 4.0
    }

    /// 七对进度：对子数 / 7
    fn qi_dui_progress(&self, hand: &Hand) -> f32 {
        let mut pairs = 0;
        for tile in Tile::all() {
            if hand.count(tile) >= 2 {
                pairs += 1;
            }
        }
        pairs as f32 / 7.0
    }

    /// 根进度：手牌中四张同牌的组数 / 4
    fn gen_progress(&self, hand: &Hand) -> f32 {
        let mut gen_count = 0;
        for tile in Tile::all() {
            if hand.count(tile) >= 4 {
                gen_count += 1;
            }
        }
        gen_count as f32 / 4.0
    }

    /// 估算玩家**起手**时的缺门牌总数
    ///
    /// 川麻将牌时序自包含推断：
    /// 初始缺门牌数 = 手牌中剩余缺门牌数 + 已弃出的缺门牌数
    /// 如果玩家已定缺但当前手牌中完全没有缺门牌，且弃牌河中也没有缺门牌 → 可能是天缺。
    fn initial_missing_count(&self, p: &PlayerState, suit: Suit) -> u8 {
        let in_hand: u8 = Tile::all()
            .into_iter()
            .filter(|t| t.suit() == suit)
            .map(|t| p.hand.count(t))
            .sum();
        let in_discard: u8 = p
            .discarded
            .iter()
            .filter(|d| d.tile.suit() == suit)
            .count() as u8;
        in_hand + in_discard
    }

    /// 清缺进度（方案 A）：已弃缺门牌数 / 初始缺门牌数，天缺固定 1.0
    fn compute_clear_missing_progress(&self, p: &PlayerState) -> f32 {
        let suit = match p.hand.missing_suit() {
            Some(s) => s,
            None => return 0.0, // 还没定缺
        };

        let initial = self.initial_missing_count(p, suit);
        if initial == 0 {
            // 天缺：起手就没有缺门花色，瞬间清完
            return 1.0;
        }

        let discarded_missing: u8 = p
            .discarded
            .iter()
            .filter(|d| d.tile.suit() == suit)
            .count() as u8;

        (discarded_missing as f32 / initial as f32).min(1.0)
    }
}

#[pymethods]
impl ObservationEncoder {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    #[pyo3(name = "encode")]
    fn py_encode(&self, board: &Board, player_id: usize) -> Vec<f32> {
        Self::encode(self, board, player_id)
    }
}

impl crate::observation::ObservationEncoderTrait for ObservationEncoder {
    fn encode(&self, board: &Board, player_id: usize) -> Vec<f32> {
        ObservationEncoder::encode(self, board, player_id)
    }
    fn feature_count(&self) -> usize {
        self.feature_count
    }
    fn num_channels(&self) -> usize {
        NUM_CHANNELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Wall;
    use crate::state::{Meld, DiscardedTile};

    fn make_empty_board() -> Board {
        let wall = Wall::new(vec![]);
        Board::new(4, 0, wall)
    }

    // ===== 通道编号常量（和 encoder 严格对齐）=====
    const CH_HAND_0: usize = 0; // 手牌 0 张标记
    const CH_MELDS_0: usize = 5; // 副露组起点 = 5 通道手牌之后
    const CH_DISCARD_0: usize = 21; // 弃牌河组起点 = 5+16=21
    const CH_MISSING_0: usize = 25; // 缺门组起点 = 21+4=25
    const CH_REMAINING: usize = 29; // 剩余牌 = 25+4=29
    const CH_TURN: usize = 30; // 轮次 = 29+1=30
    const CH_DEALER: usize = 31; // 庄家 = 31
    const CH_WALL: usize = 32; // 牌墙剩余 = 32
    const CH_HAS_WON_0: usize = 33; // 已胡组起点 = 32+1=33
    const CH_SWAP_DOWN: usize = 37; // 换三张 Down = 33+4=37
    const CH_KAN_SHANG: usize = 40; // 杠上花 = 37+3=40
    const CH_IS_TENPAI: usize = 41; // 本家听牌 = 40+1=41
    const CH_WAIT_COUNT: usize = 42; // 听牌数 = 42
    const CH_QING_YI_SE: usize = 43; // 番型起点 = 41+1=42... 不对，听牌数42 → 番型43
    const CH_GEN: usize = 46; // 根进度 = 43+3=46
    const CH_TIAN_QUE_0: usize = 47; // 天缺组起点 = 46+1=47
    const CH_TSUMOGIRI_0: usize = 51; // tsumogiri = 47+4=51
    const CH_CLEAR_MISS_0: usize = 55; // 清缺进度 = 51+4=55
    const CH_RECENT_0_0: usize = 59; // 最近3张起点 = 55+4=59

    // ===== 基础维度测试 =====

    #[test]
    fn test_dimension_exact() {
        let encoder = ObservationEncoder::new();
        assert_eq!(encoder.get_feature_count(), TOTAL_FEATURES);
        let board = make_empty_board();
        let features = encoder.encode(&board, 0);
        assert_eq!(features.len(), TOTAL_FEATURES, "维度必须精确 = 1917");
    }

    #[test]
    fn test_no_truncate() {
        let encoder = ObservationEncoder::new();
        let board = make_empty_board();
        let f = encoder.encode(&board, 0);
        assert_eq!(f.len(), NUM_CHANNELS * 27);
    }

    // ===== 手牌通道测试 =====

    #[test]
    fn test_hand_count_channels() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        board.players[0].hand.add_tile(Tile::M1);
        board.players[0].hand.add_tile(Tile::M1);
        board.players[0].hand.add_tile(Tile::M2);

        let f = encoder.encode(&board, 0);

        let m1_idx = Tile::M1.to_index();
        let m2_idx = Tile::M2.to_index();

        // ch0 (0张): M1 和 M2 都不是 0 张 → 0
        assert_eq!(f[CH_HAND_0 + m1_idx], 0.0);
        assert_eq!(f[CH_HAND_0 + m2_idx], 0.0);

        // ch1 (1张): M2 恰有 1 → 1
        assert_eq!(f[(CH_HAND_0 + 1) * 27 + m2_idx], 1.0);

        // ch2 (2张): M1 恰有 2 → 1
        assert_eq!(f[(CH_HAND_0 + 2) * 27 + m1_idx], 1.0);
    }

    // ===== 副露通道测试 =====

    #[test]
    fn test_meld_channels() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 1 有一个 M1 碰
        board.players[1].melds.push(Meld {
            meld_type: MeldType::Pong,
            tile: Tile::M1,
            from_player: Some(0),
            turn: 5,
        });

        let f = encoder.encode(&board, 0);

        // player 0: ch5-8, player 1: ch9-12
        // player 1 的 Pong 通道 = CH_MELDS_0 + 4*1 + 0 = 9
        let p1_pong_ch = CH_MELDS_0 + 4 * 1 + 0;
        assert_eq!(f[p1_pong_ch * 27 + Tile::M1.to_index()], 1.0);
    }

    // ===== 弃牌河通道测试 =====

    #[test]
    fn test_discard_channels() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        board.players[2].add_discarded(Tile::P5, 3, false);
        board.players[2].add_discarded(Tile::P5, 4, false);

        let f = encoder.encode(&board, 0);

        let ch = CH_DISCARD_0 + 2; // player 2
        let p5_idx = Tile::P5.to_index();
        assert_eq!(f[ch * 27 + p5_idx], 2.0 / 4.0);
    }

    // ===== 缺门通道测试 =====

    #[test]
    fn test_missing_suit_channel() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        board.players[0].hand.set_missing_suit(Some(Suit::Sou));

        let f = encoder.encode(&board, 0);

        let ch = CH_MISSING_0; // player 0
        // Suit::Sou = 18..=26
        for i in 18..=26 {
            assert_eq!(f[ch * 27 + i], 1.0, "Sou 缺门牌应为 1");
        }
        assert_eq!(f[ch * 27 + 0], 0.0); // M1
        assert_eq!(f[ch * 27 + 9], 0.0); // P1
    }

    // ===== 剩余牌通道测试 =====

    #[test]
    fn test_remaining_excludes_won_player_hand() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 3 胡了, 手牌里有 M1×4
        board.players[3].hand.add_tile(Tile::M1);
        board.players[3].hand.add_tile(Tile::M1);
        board.players[3].hand.add_tile(Tile::M1);
        board.players[3].hand.add_tile(Tile::M1);
        board.players[3].has_won = true;

        let f = encoder.encode(&board, 0);

        let remaining_m1 = f[CH_REMAINING * 27 + Tile::M1.to_index()];
        // 已胡玩家的 M1×4 不计入 exposed → remaining = 4/4 = 1.0
        assert_eq!(remaining_m1, 1.0, "已胡玩家手牌不应计入 exposed");
    }

    // ===== 换三张方向测试 =====

    #[test]
    fn test_swap_direction_channel() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();
        board.swap_direction = Some(SwapDirection::Up);

        let f = encoder.encode(&board, 0);

        assert_eq!(f[CH_SWAP_DOWN * 27], 0.0); // Down
        assert_eq!(f[(CH_SWAP_DOWN + 1) * 27], 0.0); // Across
        assert_eq!(f[(CH_SWAP_DOWN + 2) * 27], 1.0); // Up
    }

    // ===== 本家听牌通道（空手牌肯定没听）=====

    #[test]
    fn test_is_tenpai_channel() {
        let encoder = ObservationEncoder::new();
        let board = make_empty_board();
        let f = encoder.encode(&board, 0);
        assert_eq!(f[CH_IS_TENPAI * 27], 0.0);
    }

    // ===== 牌墙剩余通道 =====

    #[test]
    fn test_wall_remaining_channel() {
        let encoder = ObservationEncoder::new();
        let wall = Wall::new(vec![Tile::M1, Tile::M2]);
        let board = Board::new(4, 0, wall);
        let f = encoder.encode(&board, 0);
        assert!((f[CH_WALL * 27] - 1.0).abs() < 0.01);
    }

    // ===== 方案 A：清缺进度测试 =====

    #[test]
    fn test_clear_missing_progress_normal() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 0 定缺 Sou，手牌里有 Sou×2，已弃 Sou×2 → 初始 4，进度 0.5
        board.players[0].hand.add_tile(Tile::S1);
        board.players[0].hand.add_tile(Tile::S2);
        board.players[0].hand.set_missing_suit(Some(Suit::Sou));
        board.players[0].add_discarded(Tile::S3, 1, false);
        board.players[0].add_discarded(Tile::S4, 2, false);

        let f = encoder.encode(&board, 0);

        // player 0 的清缺进度通道 = CH_CLEAR_MISS_0 + 0
        let ch = CH_CLEAR_MISS_0;
        let progress = f[ch * 27]; // 广播的，任意位置值一样
        // 初始 = 手牌Sou(2) + 弃Sou(2) = 4
        // 已弃Sou = 2
        // 进度 = 2/4 = 0.5
        assert!(
            (progress - 0.5).abs() < 0.01,
            "正常清缺进度应为 0.5，实际 {}",
            progress
        );
    }

    #[test]
    fn test_clear_missing_progress_complete() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 1 定缺 Man，弃完最后一张 Man，手牌已无 Man
        board.players[1].add_discarded(Tile::M1, 1, false);
        board.players[1].add_discarded(Tile::M2, 2, false);
        board.players[1].add_discarded(Tile::M3, 3, false);
        board.players[1].add_discarded(Tile::M4, 4, false);
        board.players[1].hand.set_missing_suit(Some(Suit::Man));

        let f = encoder.encode(&board, 0);

        let ch = CH_CLEAR_MISS_0 + 1; // player 1
        let progress = f[ch * 27];
        // 初始 = 手牌Man(0) + 弃Man(4) = 4
        // 已弃Man = 4
        assert!(
            (progress - 1.0).abs() < 0.01,
            "清完缺门应为 1.0，实际 {}",
            progress
        );
    }

    #[test]
    fn test_clear_missing_progress_tian_que() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 2 定缺 Pin，但手牌和弃牌里完全没有 Pin → 天缺，瞬间清完
        board.players[2].hand.set_missing_suit(Some(Suit::Pin));

        let f = encoder.encode(&board, 0);

        let ch = CH_CLEAR_MISS_0 + 2; // player 2
        let progress = f[ch * 27];
        assert_eq!(progress, 1.0, "天缺进度应为 1.0");

        // 同时检查天缺标记通道
        let tian_que_ch = CH_TIAN_QUE_0 + 2;
        assert_eq!(f[tian_que_ch * 27], 1.0, "天缺标记应为 1");
    }

    // ===== 方案 B：最近 3 张弃牌 one-hot 测试 =====

    #[test]
    fn test_recent_discard_one_hot() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 0 弃了 3 张，最新 → 最旧：P5, P3, M1
        // DiscardedTile 顺序是按时间追加的，所以 .rev() 拿到最新
        board.players[0].add_discarded(Tile::M1, 1, false); // 最早
        board.players[0].add_discarded(Tile::P3, 2, false);
        board.players[0].add_discarded(Tile::P5, 3, false); // 最新

        let f = encoder.encode(&board, 0);

        // player 0 的 3 个 one-hot 通道 = CH_RECENT_0_0 + 0, +1, +2
        // 顺序：通道 0 = 最新 (P5), 通道 1 = 中间 (P3), 通道 2 = 最旧 (M1)
        let ch_latest = CH_RECENT_0_0;
        let ch_mid = CH_RECENT_0_0 + 1;
        let ch_oldest = CH_RECENT_0_0 + 2;

        assert_eq!(
            f[ch_latest * 27 + Tile::P5.to_index()],
            1.0,
            "通道 0 = 最新弃牌 P5"
        );
        assert_eq!(f[ch_latest * 27 + Tile::M1.to_index()], 0.0);

        assert_eq!(
            f[ch_mid * 27 + Tile::P3.to_index()],
            1.0,
            "通道 1 = 中间弃牌 P3"
        );

        assert_eq!(
            f[ch_oldest * 27 + Tile::M1.to_index()],
            1.0,
            "通道 2 = 最旧弃牌 M1"
        );
    }

    #[test]
    fn test_recent_discard_fewer_than_three() {
        let encoder = ObservationEncoder::new();
        let mut board = make_empty_board();

        // player 1 只弃了 1 张牌
        board.players[1].add_discarded(Tile::S7, 1, false);

        let f = encoder.encode(&board, 0);

        // player 1 的通道 = CH_RECENT_0_0 + 3, +4, +5
        let ch_latest = CH_RECENT_0_0 + 3;
        let ch_others = [CH_RECENT_0_0 + 4, CH_RECENT_0_0 + 5];

        assert_eq!(
            f[ch_latest * 27 + Tile::S7.to_index()],
            1.0,
            "通道 0 = 唯一弃牌 S7"
        );
        // 通道 1 和 2 应该全 0（没足够弃牌）
        for &ch in &ch_others {
            let sum: f32 = f[ch * 27..(ch + 1) * 27].iter().sum();
            assert_eq!(sum, 0.0, "不足 3 张时后续通道应为全 0");
        }
    }

    // ===== 全局安全检查 =====

    #[test]
    fn test_no_nan_or_inf() {
        let encoder = ObservationEncoder::new();
        let board = make_empty_board();
        let f = encoder.encode(&board, 0);
        for v in &f {
            assert!(!v.is_nan(), "发现 NaN");
            assert!(!v.is_infinite(), "发现 Inf");
            assert!(*v >= 0.0 && *v <= 1.0, "越界：{}", v);
        }
    }
}
