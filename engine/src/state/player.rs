use crate::tile::{Hand, Tile, Suit};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 玩家状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    /// 玩家ID
    pub id: usize,
    /// 手牌
    pub hand: Hand,
    /// 副露列表
    pub melds: Vec<Meld>,
    /// 弃牌序列
    pub discarded: Vec<DiscardedTile>,
    /// 是否已胡牌
    pub has_won: bool,
    /// 是否花猪
    pub has_huazhu: bool,
    /// 是否大叫
    pub has_dajiao: bool,
    /// 是否已定缺
    pub has_missing_declared: bool,
    /// 是否为天缺（定缺前手牌只有 1-2 门花色）
    pub is_natural_missing: bool,
    /// 是否已换三张
    pub has_swapped: bool,
    /// 得分
    pub score: i32,
    /// 番型
    pub fan: u32,
}

/// 副露类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeldType {
    Pong,      // 碰
    ExposedKong, // 明杠
    ConcealedKong, // 暗杠
    AddKong,   // 补杠
}

/// 副露信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meld {
    pub meld_type: MeldType,
    pub tile: Tile,
    pub from_player: Option<usize>, // 碰杠来源玩家
    pub turn: usize, // 回合数
}

/// 弃牌信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscardedTile {
    pub tile: Tile,
    pub turn: usize, // 回合数
    pub tsumogiri: bool, // 是否为摸牌后打出
}

impl PlayerState {
    /// 创建新玩家
    pub fn new(id: usize) -> Self {
        Self {
            id,
            hand: Hand::new(),
            melds: Vec::new(),
            discarded: Vec::new(),
            has_won: false,
            has_huazhu: false,
            has_dajiao: false,
            has_missing_declared: false,
            is_natural_missing: false,
            has_swapped: false,
            score: 0,
            fan: 0,
        }
    }
    
    /// 设置手牌
    pub fn set_hand(&mut self, hand: Hand) {
        self.hand = hand;
    }
    
    /// 添加手牌
    pub fn add_tile(&mut self, tile: Tile) -> bool {
        self.hand.add_tile(tile)
    }
    
    /// 移除手牌
    pub fn remove_tile(&mut self, tile: Tile) -> bool {
        self.hand.remove_tile(tile)
    }
    
    /// 获取手牌数量
    pub fn hand_size(&self) -> usize {
        self.hand.len()
    }
    
    /// 检查是否可以添加指定牌
    pub fn can_add_tile(&self, tile: Tile) -> bool {
        self.hand.can_add(tile)
    }
    
    /// 添加副露
    pub fn add_meld(&mut self, meld: Meld) {
        self.melds.push(meld);
    }
    
    /// 移除副露
    pub fn remove_meld(&mut self, index: usize) -> Option<Meld> {
        if index < self.melds.len() {
            Some(self.melds.remove(index))
        } else {
            None
        }
    }
    
    /// 添加弃牌
    pub fn add_discarded(&mut self, tile: Tile, turn: usize, tsumogiri: bool) {
        self.discarded.push(DiscardedTile {
            tile,
            turn,
            tsumogiri,
        });
    }
    
    /// 获取弃牌数量
    pub fn discarded_count(&self) -> usize {
        self.discarded.len()
    }
    
    /// 设置已胡牌
    pub fn set_won(&mut self, fan: u32, score: i32) {
        self.has_won = true;
        self.fan = fan;
        self.score = score;
    }
    
    /// 设置花猪
    pub fn set_huazhu(&mut self) {
        self.has_huazhu = true;
    }
    
    /// 设置大叫
    pub fn set_dajiao(&mut self) {
        self.has_dajiao = true;
    }
    
    /// 设置定缺（保守估计天缺状态为 false，建议显式调用 set_missing_declared_with_tian_que）
    pub fn set_missing_declared(&mut self, suit: Option<Suit>) {
        self.has_missing_declared = true;
        self.hand.set_missing_suit(suit);
    }

    /// 设置定缺并显式指定是否为天缺
    pub fn set_missing_declared_with_tian_que(&mut self, suit: Option<Suit>, is_tian_que: bool) {
        self.has_missing_declared = true;
        self.is_natural_missing = is_tian_que;
        self.hand.set_missing_suit(suit);
    }
    
    /// 设置换三张
    pub fn set_swapped(&mut self) {
        self.has_swapped = true;
    }
    
    /// 获取副露数量
    pub fn meld_count(&self) -> usize {
        self.melds.len()
    }
    
    /// 获取指定类型的副露数量
    pub fn meld_count_by_type(&self, meld_type: MeldType) -> usize {
        self.melds.iter().filter(|m| m.meld_type == meld_type).count()
    }
    
    /// 检查是否可以碰
    pub fn can_pong(&self, tile: Tile) -> bool {
        self.hand.count(tile) >= 2 && !self.has_won
    }
    
    /// 检查是否可以杠
    pub fn can_kong(&self, tile: Tile) -> bool {
        self.hand.count(tile) >= 3 && !self.has_won
    }
    
    /// 检查是否可以补杠（有碰副露 + 手牌有那张牌 + 未胡）
    pub fn can_add_kong(&self, tile: Tile) -> bool {
        self.melds.iter().any(|m| 
            m.meld_type == MeldType::Pong && m.tile == tile
        ) && self.hand.contains(tile) && !self.has_won
    }
    
    /// 检查是否可以胡牌
    pub fn can_win(&self) -> bool {
        !self.has_won
    }
    
    /// 检查是否已定缺
    pub fn has_missing_declared(&self) -> bool {
        self.has_missing_declared
    }
    
    /// 检查是否已换三张
    pub fn has_swapped(&self) -> bool {
        self.has_swapped
    }
    
    /// 获取副露中的牌
    pub fn get_melded_tiles(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for meld in &self.melds {
            tiles.push(meld.tile);
        }
        tiles
    }
    
    /// 获取所有牌（手牌 + 副露）
    pub fn get_all_tiles(&self) -> Vec<Tile> {
        let mut all_tiles = self.hand.get_all_tiles();
        all_tiles.extend(self.get_melded_tiles());
        all_tiles
    }
    
    /// 检查是否包含指定牌
    pub fn contains_tile(&self, tile: Tile) -> bool {
        self.hand.contains(tile) || self.get_melded_tiles().contains(&tile)
    }
    
    /// 获取指定牌的总数
    pub fn get_tile_count(&self, tile: Tile) -> u8 {
        self.hand.count(tile) + self.get_melded_tiles().iter().filter(|&t| *t == tile).count() as u8
    }
    
    /// 复制玩家状态
    pub fn copy(&self) -> PlayerState {
        Self {
            id: self.id,
            hand: self.hand.copy(),
            melds: self.melds.clone(),
            discarded: self.discarded.clone(),
            has_won: self.has_won,
            has_huazhu: self.has_huazhu,
            has_dajiao: self.has_dajiao,
            has_missing_declared: self.has_missing_declared,
            is_natural_missing: self.is_natural_missing,
            has_swapped: self.has_swapped,
            score: self.score,
            fan: self.fan,
        }
    }
    
    /// 检查玩家是否有效
    pub fn is_valid(&self) -> bool {
        self.id < 4 && 
        self.hand.is_valid() && 
        self.melds.len() <= 5 && // 最多5个副露
        self.discarded.len() <= 80 // 最多80张弃牌
    }
    
    /// 获取玩家状态摘要
    pub fn get_summary(&self) -> PlayerSummary {
        PlayerSummary {
            id: self.id,
            hand_size: self.hand_size(),
            meld_count: self.meld_count(),
            discarded_count: self.discarded_count(),
            has_won: self.has_won,
            has_huazhu: self.has_huazhu,
            has_dajiao: self.has_dajiao,
            score: self.score,
            fan: self.fan,
        }
    }
}

/// 玩家状态摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerSummary {
    pub id: usize,
    pub hand_size: usize,
    pub meld_count: usize,
    pub discarded_count: usize,
    pub has_won: bool,
    pub has_huazhu: bool,
    pub has_dajiao: bool,
    pub score: i32,
    pub fan: u32,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Tile;
    
    #[test]
    fn test_player_state_basic() {
        let mut player = PlayerState::new(0);
        
        assert_eq!(player.id, 0);
        assert!(player.hand.is_empty());
        assert_eq!(player.melds.len(), 0);
        assert_eq!(player.discarded.len(), 0);
        assert!(!player.has_won);
        assert_eq!(player.score, 0);
        
        // 添加手牌
        player.add_tile(Tile::M1);
        assert_eq!(player.hand_size(), 1);
        assert!(player.contains_tile(Tile::M1));
    }
    
    #[test]
    fn test_player_melds() {
        let mut player = PlayerState::new(0);
        
        let meld = Meld {
            meld_type: MeldType::Pong,
            tile: Tile::M1,
            from_player: Some(1),
            turn: 1,
        };
        
        player.add_meld(meld);
        assert_eq!(player.meld_count(), 1);
        assert_eq!(player.get_melded_tiles(), vec![Tile::M1]);
    }
    
    #[test]
    fn test_player_discarded() {
        let mut player = PlayerState::new(0);
        
        player.add_discarded(Tile::M1, 1, false);
        assert_eq!(player.discarded_count(), 1);
        assert_eq!(player.discarded[0].tile, Tile::M1);
        assert!(!player.discarded[0].tsumogiri);
    }
    
    #[test]
    fn test_player_win() {
        let mut player = PlayerState::new(0);
        
        player.set_won(4, 16);
        assert!(player.has_won);
        assert_eq!(player.fan, 4);
        assert_eq!(player.score, 16);
    }
    
    #[test]
    fn test_player_missing() {
        let mut player = PlayerState::new(0);
        
        assert!(!player.has_missing_declared());
        
        player.set_missing_declared(Some(Suit::Man));
        assert!(player.has_missing_declared());
        assert!(player.hand.is_missing_suit(Suit::Man));
    }
    
    #[test]
    fn test_player_swap() {
        let mut player = PlayerState::new(0);
        
        assert!(!player.has_swapped());
        
        player.set_swapped();
        assert!(player.has_swapped());
    }
    
    #[test]
    fn test_player_can_pong() {
        let mut player = PlayerState::new(0);
        
        // 添加两张相同的牌
        player.add_tile(Tile::M1);
        player.add_tile(Tile::M1);
        
        assert!(player.can_pong(Tile::M1));
        assert!(!player.can_pong(Tile::M2));
        
        // 设置已胡牌，不能碰
        player.set_won(1, 1);
        assert!(!player.can_pong(Tile::M1));
    }
    
    #[test]
    fn test_player_can_kong() {
        let mut player = PlayerState::new(0);
        
        // 添加三张相同的牌
        for _ in 0..3 {
            player.add_tile(Tile::M1);
        }
        
        assert!(player.can_kong(Tile::M1));
        assert!(!player.can_kong(Tile::M2));
    }
    
    #[test]
    fn test_player_summary() {
        let mut player = PlayerState::new(0);
        player.add_tile(Tile::M1);
        player.add_discarded(Tile::P1, 1, true);
        
        let summary = player.get_summary();
        assert_eq!(summary.id, 0);
        assert_eq!(summary.hand_size, 1);
        assert_eq!(summary.discarded_count, 1);
        assert!(!summary.has_won);
    }
    
    #[test]
    fn test_player_copy() {
        let mut player = PlayerState::new(0);
        player.add_tile(Tile::M1);
        player.set_won(1, 1);
        
        let copy = player.copy();
        assert_eq!(copy.id, player.id);
        assert_eq!(copy.hand_size(), player.hand_size());
        assert_eq!(copy.has_won, player.has_won);
}
}