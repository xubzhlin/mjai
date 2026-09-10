pub mod fan;
pub mod scoring;
pub mod swap;
pub mod missing;
pub mod presets;

use crate::tile::{Hand, Tile, Suit};
use crate::algo::winning::WinType;
use anyhow::Result;

// ===== 公共重导出 =====
pub use fan::FanCalculator;
pub use scoring::ScoringCalculator;
pub use swap::SwapRuleCalculator;
pub use missing::MissingRuleCalculator;
pub use presets::xue_zhan::XueZhanRuleSet;

/// fan.rs 输出：番数 + 不受封顶限制的额外底数
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FanResult {
    /// 参与封顶 16 的番数（含所有乘法修饰）
    pub fan: u32,
    /// 不受封顶限制的额外底数（目前只有自摸加底=1）
    pub extra_bases: u32,
}

/// 杠类型（用于杠结算）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KongType {
    /// 暗杠
    AnKan,
    /// 明杠（点杠）
    MinKan,
    /// 补杠
    BuKan,
}

/// 副露类型（rules 层轻量视图，不依赖 state::MeldType）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeldKind {
    /// 碰（3 张相同牌）
    Pong,
    /// 明杠（点杠，4 张相同牌）
    ExposedKong,
    /// 暗杠（4 张相同牌）
    ConcealedKong,
    /// 补杠（碰后补，4 张相同牌）
    AddKong,
}

/// 副露轻量视图（rules 层使用，仅含番型计算所需信息）
#[derive(Debug, Clone, Copy)]
pub struct MeldView {
    pub tile: Tile,
    pub kind: MeldKind,
}

impl MeldView {
    pub fn is_kong(&self) -> bool {
        matches!(self.kind, MeldKind::ExposedKong | MeldKind::ConcealedKong | MeldKind::AddKong)
    }

    /// 该副露占用的牌张数（碰=3，杠=4）
    pub fn tile_count(&self) -> u8 {
        match self.kind {
            MeldKind::Pong => 3,
            _ => 4,
        }
    }
}

/// 规则集 trait - 定义完整的麻将规则接口
pub trait RuleSet: Send + Sync {
    /// 计算番型（所有规则修饰）
    /// melds: 副露视图列表，用于根的统计（杠=4张同牌计入根）
    fn fan(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult;

    /// 胡牌得分
    fn score_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;

    /// 杠结算
    fn score_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;

    /// 获取换三张规则
    fn swap_rule(&self) -> &dyn SwapRule;

    /// 获取定缺规则
    fn missing_rule(&self) -> &dyn MissingRule;

    /// 获取规则名称
    fn name(&self) -> &'static str;
}

/// 番型计算 trait
pub trait FanRule {
    /// 计算手牌的番型（所有规则修饰）
    /// melds: 副露视图列表，用于根的统计（杠=4张同牌计入根）
    fn calculate(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult;
}

/// 计分规则 trait
pub trait ScoringRule {
    /// 胡牌得分计算
    fn calculate_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32;

    /// 杠结算计算
    fn calculate_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32;
}

/// 换三张规则 trait
pub trait SwapRule {
    /// 根据骰子确定换三张方向
    fn determine_direction(&self, dice: u8) -> SwapDirection;

    /// AI 选 3 张同花色换出牌
    fn select_swap_tiles(&self, hand: &Hand) -> Vec<Tile>;
}

/// 定缺规则 trait
pub trait MissingRule {
    /// 验证定缺是否合法
    fn is_valid_missing(&self, hand: &Hand, suit: Suit) -> bool;

    /// AI 选缺门花色
    fn select_missing_suit(&self, hand: &Hand) -> Suit;
}

/// 换三张方向枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapDirection {
    /// 下家交换
    Down = 0,
    /// 对家交换
    Across = 1,
    /// 上家交换
    Up = 2,
}

/// 换三张结果
#[derive(Debug, Clone)]
pub struct SwapResult {
    pub direction: SwapDirection,
    pub tiles_swapped: [[Tile; 3]; 4],
    pub success: bool,
}

/// 番型类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanType {
    PingHu,      // 平胡
    DuiDuiHu,    // 对对胡
    QingYiSe,    // 清一色
    QiDui,       // 七对
    JinGouDiao,  // 金钩钓
    // 组合番型
    QingDui,     // 清对
    QingQiDui,   // 清七对
    // 乘法修饰
    Gen,         // 根
    KanAbove,    // 杠上花
    RobKong,     // 抢杠胡
}

impl FanType {
    pub fn base_fan_value(&self) -> u32 {
        match self {
            FanType::PingHu => 1,
            FanType::DuiDuiHu => 2,
            FanType::QingYiSe => 4,
            FanType::QiDui => 4,
            FanType::JinGouDiao => 4,
            FanType::QingDui => 8,      // 4×2
            FanType::QingQiDui => 16,   // 4×4
            FanType::Gen => 2,
            FanType::KanAbove => 2,
            FanType::RobKong => 2,
        }
    }

    pub fn is_base(&self) -> bool {
        matches!(self,
            FanType::PingHu | FanType::DuiDuiHu | FanType::QingYiSe |
            FanType::QiDui | FanType::JinGouDiao
        )
    }

    pub fn is_composite(&self) -> bool {
        matches!(self,
            FanType::QingDui | FanType::QingQiDui
        )
    }

    pub fn is_modifier(&self) -> bool {
        matches!(self, FanType::Gen | FanType::KanAbove | FanType::RobKong)
    }
}
