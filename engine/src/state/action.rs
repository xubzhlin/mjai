use crate::tile::{Tile, Suit};
use serde::{Serialize, Deserialize};

/// 动作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    Discard,  // 打牌
    Pong,     // 碰
    Kong,     // 杠
    Win,      // 胡牌
    Pass,     // 过
    Swap,     // 换三张
    Missing,  // 定缺
    KanShang, // 杠上花
    QiangGang,// 抢杠胡
}

/// 动作优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActionPriority {
    Win = 4,      // 胡牌最高优先级
    Kong = 3,     // 杠
    Pong = 2,     // 碰
    Discard = 1,  // 打牌
    Pass = 0,     // 过
}

/// 动作定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: ActionType,
    pub tile: Option<Tile>,
    pub target_player: Option<usize>,
    pub priority: ActionPriority,
    pub turn: usize,
}

impl Action {
    pub fn discard(tile: Tile, turn: usize) -> Self {
        Self { action_type: ActionType::Discard, tile: Some(tile), target_player: None, priority: ActionPriority::Discard, turn }
    }
    pub fn pong(tile: Tile, target_player: usize, turn: usize) -> Self {
        Self { action_type: ActionType::Pong, tile: Some(tile), target_player: Some(target_player), priority: ActionPriority::Pong, turn }
    }
    pub fn kong(tile: Tile, target_player: Option<usize>, turn: usize) -> Self {
        Self { action_type: ActionType::Kong, tile: Some(tile), target_player, priority: ActionPriority::Kong, turn }
    }
    pub fn win(tile: Tile, turn: usize) -> Self {
        Self { action_type: ActionType::Win, tile: Some(tile), target_player: None, priority: ActionPriority::Win, turn }
    }
    pub fn pass(turn: usize) -> Self {
        Self { action_type: ActionType::Pass, tile: None, target_player: None, priority: ActionPriority::Pass, turn }
    }
    pub fn swap(_suit: Suit, turn: usize) -> Self {
        Self { action_type: ActionType::Swap, tile: None, target_player: None, priority: ActionPriority::Pass, turn }
    }
    pub fn missing(_suit: Suit, turn: usize) -> Self {
        Self { action_type: ActionType::Missing, tile: None, target_player: None, priority: ActionPriority::Pass, turn }
    }
    pub fn kan_shang(tile: Tile, turn: usize) -> Self {
        Self { action_type: ActionType::KanShang, tile: Some(tile), target_player: None, priority: ActionPriority::Win, turn }
    }
    pub fn qiang_gang(tile: Tile, target_player: usize, turn: usize) -> Self {
        Self { action_type: ActionType::QiangGang, tile: Some(tile), target_player: Some(target_player), priority: ActionPriority::Win, turn }
    }

    pub fn description(&self) -> String {
        match self.action_type {
            ActionType::Discard => self.tile.map(|t| format!("打牌 {}", t)).unwrap_or_else(|| "打牌".into()),
            ActionType::Pong => self.tile.map(|t| format!("碰 {}", t)).unwrap_or_else(|| "碰".into()),
            ActionType::Kong => self.tile.map(|t| format!("杠 {}", t)).unwrap_or_else(|| "杠".into()),
            ActionType::Win => self.tile.map(|t| format!("胡牌 {}", t)).unwrap_or_else(|| "胡牌".into()),
            ActionType::Pass => "过".into(),
            ActionType::Swap => "换三张".into(),
            ActionType::Missing => "定缺".into(),
            ActionType::KanShang => self.tile.map(|t| format!("杠上花 {}", t)).unwrap_or_else(|| "杠上花".into()),
            ActionType::QiangGang => self.tile.map(|t| format!("抢杠胡 {}", t)).unwrap_or_else(|| "抢杠胡".into()),
        }
    }
    pub fn is_valid(&self) -> bool {
        match self.action_type {
            ActionType::Discard | ActionType::Pong | ActionType::Kong |
            ActionType::Win | ActionType::KanShang | ActionType::QiangGang => self.tile.is_some(),
            ActionType::Pass | ActionType::Swap | ActionType::Missing => self.tile.is_none(),
        }
    }
    pub fn has_higher_priority(&self, other: &Action) -> bool { self.priority > other.priority }
    pub fn is_win_action(&self) -> bool { matches!(self.action_type, ActionType::Win | ActionType::KanShang | ActionType::QiangGang) }
    pub fn is_meld_action(&self) -> bool { matches!(self.action_type, ActionType::Pong | ActionType::Kong) }
    pub fn is_opening_action(&self) -> bool { matches!(self.action_type, ActionType::Swap | ActionType::Missing) }
}

/// 动作候选
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    pub actions: Vec<Action>,
    pub selected_action: Option<Action>,
}

impl ActionCandidate {
    pub fn new() -> Self { Self { actions: Vec::new(), selected_action: None } }
    pub fn add_action(&mut self, action: Action) { self.actions.push(action); }
    pub fn get_highest_priority_action(&self) -> Option<&Action> { self.actions.iter().max_by_key(|a| a.priority) }
    pub fn get_win_actions(&self) -> Vec<&Action> { self.actions.iter().filter(|a| a.is_win_action()).collect() }
    pub fn get_meld_actions(&self) -> Vec<&Action> { self.actions.iter().filter(|a| a.is_meld_action()).collect() }
    pub fn auto_select(&mut self) { if let Some(h) = self.get_highest_priority_action() { self.selected_action = Some(h.clone()); } }
    pub fn has_win_actions(&self) -> bool { !self.get_win_actions().is_empty() }
    pub fn has_meld_actions(&self) -> bool { !self.get_meld_actions().is_empty() }
    pub fn clear(&mut self) { self.actions.clear(); self.selected_action = None; }
    pub fn len(&self) -> usize { self.actions.len() }
    pub fn is_empty(&self) -> bool { self.actions.is_empty() }
}

impl Default for ActionCandidate {
    fn default() -> Self { Self::new() }
}

/// 动作验证器
pub struct ActionValidator;

impl ActionValidator {
    pub fn validate_discard(player: &crate::state::PlayerState, tile: Tile) -> bool {
        player.hand.contains(tile) && !player.has_won
    }
    pub fn validate_pong(player: &crate::state::PlayerState, tile: Tile) -> bool {
        player.can_pong(tile)
    }
    pub fn validate_kong(player: &crate::state::PlayerState, tile: Tile) -> bool {
        player.can_kong(tile)
    }
    pub fn validate_add_kong(player: &crate::state::PlayerState, tile: Tile) -> bool {
        player.can_add_kong(tile)
    }
    pub fn validate_win(player: &crate::state::PlayerState, _tile: Tile) -> bool {
        player.can_win()
    }
    pub fn validate_swap(player: &crate::state::PlayerState, _suit: Suit) -> bool {
        !player.has_swapped && !player.has_missing_declared()
    }
    pub fn validate_missing(player: &crate::state::PlayerState, _suit: Suit) -> bool {
        !player.has_missing_declared() && player.has_swapped
    }
    pub fn generate_legal_actions(player: &crate::state::PlayerState, turn: usize) -> ActionCandidate {
        let mut c = ActionCandidate::new();
        for tile in Tile::all() {
            if player.hand.contains(tile) { c.add_action(Action::discard(tile, turn)); }
            if player.can_pong(tile) { c.add_action(Action::pong(tile, 0, turn)); }
            if player.can_kong(tile) { c.add_action(Action::kong(tile, None, turn)); }
        }
        c
    }
}
