use crate::tile::{Hand, Tile, Suit};
use crate::algo::winning::WinType;
use crate::rules::{
    RuleSet, FanRule, ScoringRule, SwapRule, MissingRule,
    FanCalculator, ScoringCalculator, SwapRuleCalculator, MissingRuleCalculator,
    FanResult, KongType, MeldView,
};
use pyo3::prelude::*;

/// 血战到底规则集
#[pyclass(name = "XueZhanRuleSet")]
#[derive(Clone)]
pub struct XueZhanRuleSet {
    #[pyo3(get)]
    pub fan_calculator: FanCalculator,
    #[pyo3(get)]
    pub scoring_calculator: ScoringCalculator,
    #[pyo3(get)]
    pub swap_rule: SwapRuleCalculator,
    #[pyo3(get)]
    pub missing_rule: MissingRuleCalculator,
}

#[pymethods]
impl XueZhanRuleSet {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        "血战到底"
    }
}

impl XueZhanRuleSet {
    /// 创建默认的血战到底规则集
    pub fn new() -> Self {
        Self {
            fan_calculator: FanCalculator::new(),
            scoring_calculator: ScoringCalculator::new_with_params(16, 1), // 封顶16番，自摸加1底
            swap_rule: SwapRuleCalculator::new(true, 3),     // 启用换三张，每人3张
            missing_rule: MissingRuleCalculator::new(true, 0.75), // 启用定缺，天缺75%选空门
        }
    }

    /// 创建自定义参数的血战到底规则集
    pub fn new_with_params(
        max_fan: u32,
        self_draw_bonus: u32,
        swap_enabled: bool,
        missing_enabled: bool,
        tian_que_threshold: f64,
    ) -> Self {
        Self {
            fan_calculator: FanCalculator::new(),
            scoring_calculator: ScoringCalculator::new_with_params(max_fan, self_draw_bonus),
            swap_rule: SwapRuleCalculator::new(swap_enabled, 3),
            missing_rule: MissingRuleCalculator::new(missing_enabled, tian_que_threshold),
        }
    }

    /// 检查是否为胡牌状态
    pub fn is_winning_hand(&self, hand: &Hand, win_type: WinType) -> bool {
        match win_type {
            WinType::Tsumo | WinType::Ron => {
                self.fan_calculator.is_standard_win(hand) ||
                self.fan_calculator.is_seven_pairs(hand)
            },
            WinType::KanShang => {
                self.fan_calculator.is_standard_win(hand) ||
                self.fan_calculator.is_seven_pairs(hand)
            },
            WinType::QiangGang => {
                self.fan_calculator.is_standard_win(hand) ||
                self.fan_calculator.is_seven_pairs(hand)
            },
        }
    }

    /// 检查是否为流局
    pub fn is_riichi_game(&self, hands: &[Hand]) -> bool {
        let mut winning_count = 0;
        for hand in hands {
            if hand.is_winning() {
                winning_count += 1;
            }
        }
        winning_count >= 3
    }

    /// 检查抢杠胡
    /// 补杠的 kan_tile 正好是 hand 听的牌 → 可抢杠胡（暗杠不可抢，由调用方区分）
    pub fn check_rob_kong(&self, hand: &Hand, kan_tile: &Tile) -> bool {
        let mut h = hand.copy();
        h.add_tile(*kan_tile);
        crate::algo::WinChecker::can_win(&h, crate::algo::winning::WinType::QiangGang)
    }

    /// 检查花猪
    pub fn check_hua_zhu(&self, hands: &[Hand], missing_suits: &[Option<Suit>]) -> Option<usize> {
        for (i, hand) in hands.iter().enumerate() {
            if let Some(missing_suit) = missing_suits[i] {
                let other_suits: Vec<Suit> = Suit::all().into_iter()
                    .filter(|s| *s != missing_suit).collect();
                let missing_others = other_suits.iter()
                    .filter(|s| hand.is_missing_suit(**s)).count();
                if missing_others >= 1 {
                    return Some(i);
                }
            }
        }
        None
    }

    /// 检查大叫
    pub fn check_da_jiao(&self, hands: &[Hand]) -> Vec<usize> {
        let mut da_jiao_players = Vec::new();

        for (i, hand) in hands.iter().enumerate() {
            if hand.is_waiting() && !hand.is_winning() {
                da_jiao_players.push(i);
            }
        }

        da_jiao_players
    }
}

impl RuleSet for XueZhanRuleSet {
    fn fan(&self, hand: &Hand, melds: &[MeldView], win_type: WinType) -> FanResult {
        self.fan_calculator.calculate(hand, melds, win_type)
    }

    fn score_hu(&self, result: FanResult, win_type: WinType, base: u32, num_payers: usize) -> i32 {
        self.scoring_calculator.calculate_hu(result, win_type, base, num_payers)
    }

    fn score_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32 {
        self.scoring_calculator.calculate_kong(kong_type, base, num_payers)
    }

    fn swap_rule(&self) -> &dyn SwapRule {
        &self.swap_rule
    }

    fn missing_rule(&self) -> &dyn MissingRule {
        &self.missing_rule
    }

    fn name(&self) -> &'static str {
        "血战到底"
    }
}

/// 血战到底规则配置
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct XueZhanConfig {
    pub max_fan: u32,
    pub self_draw_bonus: u32,
    pub swap_enabled: bool,
    pub missing_enabled: bool,
    pub tian_que_threshold: f64,
    pub base_score: u32,
    pub kan_scoring: KanScoringConfig,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct KanScoringConfig {
    pub ankan_multiplier: u32,
    pub minkan_multiplier: u32,
    pub bukang_multiplier: u32,
}

impl Default for XueZhanConfig {
    fn default() -> Self {
        Self {
            max_fan: 16,
            self_draw_bonus: 1,
            swap_enabled: true,
            missing_enabled: true,
            tian_que_threshold: 0.75,
            base_score: 1,
            kan_scoring: KanScoringConfig {
                ankan_multiplier: 2,
                minkan_multiplier: 2,
                bukang_multiplier: 1,
            },
        }
    }
}

impl XueZhanRuleSet {
    /// 从配置创建规则集
    pub fn from_config(config: XueZhanConfig) -> Self {
        Self {
            fan_calculator: FanCalculator::new(),
            scoring_calculator: ScoringCalculator::new_with_params(config.max_fan, config.self_draw_bonus),
            swap_rule: SwapRuleCalculator::new(config.swap_enabled, 3),
            missing_rule: MissingRuleCalculator::new(config.missing_enabled, config.tian_que_threshold),
        }
    }

    /// 获取配置
    pub fn get_config(&self) -> XueZhanConfig {
        XueZhanConfig {
            max_fan: self.scoring_calculator.max_fan(),
            self_draw_bonus: self.scoring_calculator.self_draw_bonus(),
            swap_enabled: self.swap_rule.enabled,
            missing_enabled: self.missing_rule.enabled,
            tian_que_threshold: self.missing_rule.tian_que_threshold,
            base_score: 1,
            kan_scoring: KanScoringConfig {
                ankan_multiplier: 2,
                minkan_multiplier: 2,
                bukang_multiplier: 1,
            },
        }
    }
}

/// 血战到底游戏状态
#[derive(Debug, Clone)]
pub struct XueZhanGameState {
    pub players: [Hand; 4],
    pub current_player: usize,
    pub round_number: u32,
    pub is_game_over: bool,
    pub winning_players: Vec<usize>,
    pub scores: [i32; 4],
}

impl XueZhanGameState {
    pub fn new() -> Self {
        Self {
            players: [Hand::new(), Hand::new(), Hand::new(), Hand::new()],
            current_player: 0,
            round_number: 0,
            is_game_over: false,
            winning_players: Vec::new(),
            scores: [0; 4],
        }
    }

    pub fn add_winning_player(&mut self, player_index: usize) {
        if !self.winning_players.contains(&player_index) {
            self.winning_players.push(player_index);
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.is_game_over || self.winning_players.len() >= 3
    }
}

