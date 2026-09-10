use std::fmt;
use serde::{Serialize, Deserialize};
use super::{Tile, Suit};

/// 手牌结构 - 使用计数数组表示
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hand {
    /// 牌的计数数组 [u8; 27]，索引对应 Tile::to_index()
    pub tiles: [u8; 27],
    /// 缺门花色（可选）
    pub missing_suit: Option<Suit>,
}

impl Hand {
    /// 创建空手牌
    pub fn new() -> Self {
        Self {
            tiles: [0; 27],
            missing_suit: None,
        }
    }
    
    /// 创建手牌并设置缺门
    pub fn with_missing_suit(missing_suit: Suit) -> Self {
        Self {
            tiles: [0; 27],
            missing_suit: Some(missing_suit),
        }
    }
    
    /// 从牌列表创建手牌
    pub fn from_tiles(tiles: &[Tile]) -> Self {
        let mut hand = Self::new();
        for tile in tiles {
            hand.add_tile(*tile);
        }
        hand
    }
    
    /// 获取牌的总数
    pub fn len(&self) -> usize {
        self.tiles.iter().sum::<u8>() as usize
    }
    
    /// 检查手牌是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 添加一张牌
    pub fn add_tile(&mut self, tile: Tile) -> bool {
        let index = tile.to_index();
        if self.tiles[index] < 4 {
            self.tiles[index] += 1;
            true
        } else {
            false
        }
    }
    
    /// 移除一张牌
    pub fn remove_tile(&mut self, tile: Tile) -> bool {
        let index = tile.to_index();
        if self.tiles[index] > 0 {
            self.tiles[index] -= 1;
            true
        } else {
            false
        }
    }
    
    /// 获取指定牌的数量
    pub fn count(&self, tile: Tile) -> u8 {
        let index = tile.to_index();
        self.tiles[index]
    }
    
    /// 检查是否包含指定牌
    pub fn contains(&self, tile: Tile) -> bool {
        self.count(tile) > 0
    }
    
    /// 检查是否可以添加指定牌（不超过4张）
    pub fn can_add(&self, tile: Tile) -> bool {
        self.count(tile) < 4
    }
    
    /// 检查是否可以移除指定牌
    pub fn can_remove(&self, tile: Tile) -> bool {
        self.count(tile) > 0
    }
    
    /// 设置缺门花色
    pub fn set_missing_suit(&mut self, suit: Option<Suit>) {
        self.missing_suit = suit;
    }
    
    /// 获取缺门花色
    pub fn missing_suit(&self) -> Option<Suit> {
        self.missing_suit
    }
    
    /// 检查是否缺指定花色
    pub fn is_missing_suit(&self, suit: Suit) -> bool {
        self.missing_suit == Some(suit)
    }
    
    /// 检查是否缺门（三门齐全）
    pub fn is_missing_clear(&self) -> bool {
        self.missing_suit.is_some()
    }
    
    /// 获取所有牌（按数量展开）
    pub fn get_all_tiles(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for (index, count) in self.tiles.iter().enumerate() {
            if *count > 0 {
                let tile = Tile::from_index(index).unwrap();
                for _ in 0..*count {
                    tiles.push(tile);
                }
            }
        }
        tiles
    }
    
    /// 获取指定花色的所有牌
    pub fn get_tiles_by_suit(&self, suit: Suit) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for (index, count) in self.tiles.iter().enumerate() {
            if *count > 0 {
                let tile = Tile::from_index(index).unwrap();
                if tile.suit() == suit {
                    for _ in 0..*count {
                        tiles.push(tile);
                    }
                }
            }
        }
        tiles
    }
    
    /// 获取指定花色的牌数量
    pub fn count_by_suit(&self, suit: Suit) -> u8 {
        let mut count = 0;
        for (index, tile_count) in self.tiles.iter().enumerate() {
            if *tile_count > 0 {
                let tile = Tile::from_index(index).unwrap();
                if tile.suit() == suit {
                    count += tile_count;
                }
            }
        }
        count
    }
    
    /// 获取花色分布
    pub fn suit_distribution(&self) -> [u8; 3] {
        let mut distribution = [0; 3];
        for (index, count) in self.tiles.iter().enumerate() {
            if *count > 0 {
                let tile = Tile::from_index(index).unwrap();
                let suit_index = tile.suit().index();
                distribution[suit_index] += count;
            }
        }
        distribution
    }
    
    /// 检查是否包含指定花色
    pub fn has_suit(&self, suit: Suit) -> bool {
        self.count_by_suit(suit) > 0
    }
    
    /// 获取数量最多的花色
    pub fn get_max_suit(&self) -> Option<Suit> {
        let distribution = self.suit_distribution();
        let max_count = distribution.iter().max()?;
        
        if *max_count == 0 {
            return None;
        }
        
        let max_index = distribution.iter().position(|&x| x == *max_count)?;
        Suit::from_index(max_index)
    }
    
    /// 获取数量最少的花色（排除缺门）
    pub fn get_min_suit(&self) -> Option<Suit> {
        let distribution = self.suit_distribution();
        let mut min_count = u8::MAX;
        let mut min_index = None;
        
        for (i, &count) in distribution.iter().enumerate() {
            if count > 0 && !self.is_missing_suit(Suit::from_index(i).unwrap()) {
                if count < min_count {
                    min_count = count;
                    min_index = Some(i);
                }
            }
        }
        
        min_index.and_then(Suit::from_index)
    }
    
    /// 复制手牌
    pub fn copy(&self) -> Hand {
        Hand {
            tiles: self.tiles,
            missing_suit: self.missing_suit,
        }
    }
    
    /// 获取牌的计数数组
    pub fn get_tile_counts(&self) -> &[u8; 27] {
        &self.tiles
    }
    
    /// 设置牌的计数数组
    pub fn set_tile_counts(&mut self, counts: [u8; 27]) {
        self.tiles = counts;
    }
    
    /// 获取牌的总数（手牌大小）
    pub fn hand_size(&self) -> usize {
        self.len()
    }
    
    /// 返回排除缺门花色后的临时手牌（用于川麻 shanten 计算）
    pub fn hand_without_missing(&self) -> Hand {
        match self.missing_suit {
            None => self.copy(),
            Some(suit) => {
                let mut h = Hand::with_missing_suit(suit);
                for tile in Tile::all() {
                    if tile.suit() != suit {
                        for _ in 0..self.tiles[tile.to_index()] {
                            h.add_tile(tile);
                        }
                    }
                }
                h
            }
        }
    }

    /// 基于排除缺门牌的临时手牌，计算 shanten 结果
    pub fn shanten(&self) -> i32 {
        let h = self.hand_without_missing();
        crate::algo::shanten::ShantenResult::calculate(&h).min
    }

    /// 是否听牌（shanten == 0 且已清缺门）
    pub fn is_tenpai(&self) -> bool {
        // 川麻必须先清缺门才能听牌
        if self.missing_suit.is_some() {
            if let Some(s) = self.missing_suit {
                if self.has_suit(s) {
                    return false;
                }
            }
        }
        self.shanten() == 0
    }

    /// 检查是否听牌（is_tenpai 的别名）
    pub fn is_waiting(&self) -> bool {
        self.is_tenpai()
    }

    /// 听牌数（多少种不同的牌能成和），没听牌返回 0
    pub fn waiting_count(&self) -> u8 {
        if !self.is_tenpai() {
            return 0;
        }
        let h = self.hand_without_missing();
        crate::algo::shanten::ShantenResult::get_waiting_tiles(&h).len() as u8
    }

    /// 检查是否已胡牌
    pub fn is_winning(&self) -> bool {
        self.shanten() < 0
    }
    
    /// 设置已胡牌
    pub fn set_won(&mut self, _fan: u32, _score: i32) {
        // 简化实现，实际应该更新状态
    }
    
    /// 检查是否已定缺
    pub fn has_missing_declared(&self) -> bool {
        self.missing_suit.is_some()
    }
    
    /// 设置定缺声明
    pub fn set_missing_declared(&mut self, suit: Option<Suit>) {
        self.missing_suit = suit;
    }
    
    /// 检查是否已换三张
    pub fn has_swapped(&self) -> bool {
        // 简化实现，实际应该检查换三张状态
        false
    }
    
    /// 设置已换三张
    pub fn set_swapped(&mut self) {
        // 简化实现，实际应该设置换三张状态
    }
    
    /// 检查手牌是否有效（总数不超过14张）
    pub fn is_valid(&self) -> bool {
        self.len() <= 14
    }
    
    /// 检查手牌是否完整（14张）
    pub fn is_complete(&self) -> bool {
        self.len() == 14
    }
    
    /// 检查手牌是否可以胡牌（初步检查）
    pub fn can_potentially_win(&self) -> bool {
        self.len() >= 13 && self.len() <= 14
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Hand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tiles = Vec::new();
        for (index, count) in self.tiles.iter().enumerate() {
            if *count > 0 {
                let tile = Tile::from_index(index).unwrap();
                for _ in 0..*count {
                    tiles.push(tile.to_string());
                }
            }
        }
        tiles.sort();
        write!(f, "[{}]", tiles.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hand_basic() {
        let mut hand = Hand::new();
        assert_eq!(hand.len(), 0);
        assert!(hand.is_empty());
        
        hand.add_tile(Tile::M1);
        assert_eq!(hand.len(), 1);
        assert!(!hand.is_empty());
        assert_eq!(hand.count(Tile::M1), 1);
        assert!(hand.contains(Tile::M1));
    }
    
    #[test]
    fn test_hand_remove() {
        let mut hand = Hand::from_tiles(&[Tile::M1, Tile::M1, Tile::M2]);
        
        assert_eq!(hand.count(Tile::M1), 2);
        assert!(hand.remove_tile(Tile::M1));
        assert_eq!(hand.count(Tile::M1), 1);
        assert!(hand.remove_tile(Tile::M1));
        assert_eq!(hand.count(Tile::M1), 0);
        assert!(!hand.remove_tile(Tile::M1));
        assert_eq!(hand.count(Tile::M1), 0);
    }
    
    #[test]
    fn test_hand_suit() {
        let mut hand = Hand::new();
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::M2);
        hand.add_tile(Tile::P1);
        hand.add_tile(Tile::P2);
        hand.add_tile(Tile::S1);
        
        assert_eq!(hand.count_by_suit(Suit::Man), 2);
        assert_eq!(hand.count_by_suit(Suit::Pin), 2);
        assert_eq!(hand.count_by_suit(Suit::Sou), 1);
        
        assert!(hand.has_suit(Suit::Man));
        assert!(hand.has_suit(Suit::Pin));
        assert!(hand.has_suit(Suit::Sou));
    }
    
    #[test]
    fn test_hand_missing_suit() {
        let mut hand = Hand::with_missing_suit(Suit::Man);
        assert_eq!(hand.missing_suit(), Some(Suit::Man));
        assert!(hand.is_missing_suit(Suit::Man));
        assert!(!hand.is_missing_suit(Suit::Pin));
        assert!(hand.is_missing_clear());
    }
    
    #[test]
    fn test_hand_max_min_suit() {
        let mut hand = Hand::new();
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::M2);
        hand.add_tile(Tile::P1);
        hand.add_tile(Tile::P1);
        hand.add_tile(Tile::P1);
        hand.add_tile(Tile::S1);
        
        assert_eq!(hand.get_max_suit(), Some(Suit::Pin));
        assert_eq!(hand.get_min_suit(), Some(Suit::Sou));
    }
    
    #[test]
    fn test_hand_all_tiles() {
        let mut hand = Hand::new();
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::P5);
        
        let all_tiles = hand.get_all_tiles();
        assert_eq!(all_tiles.len(), 3);
        assert_eq!(all_tiles.iter().filter(|&t| *t == Tile::M1).count(), 2);
        assert_eq!(all_tiles.iter().filter(|&t| *t == Tile::P5).count(), 1);
    }
    
    #[test]
    fn test_hand_validity() {
        let mut hand = Hand::new();
        assert!(hand.is_valid());
        assert!(!hand.is_complete());
        
        // Build a valid 13-tile hand (1 of each M1-M9, P1-P4)
        let tiles13 = [
            Tile::M1, Tile::M2, Tile::M3, Tile::M4, Tile::M5,
            Tile::M6, Tile::M7, Tile::M8, Tile::M9,
            Tile::P1, Tile::P2, Tile::P3, Tile::P4,
        ];
        for t in &tiles13 {
            hand.add_tile(*t);
        }
        assert_eq!(hand.len(), 13);
        assert!(hand.is_valid());
        assert!(!hand.is_complete());
        
        // 14th tile — complete
        hand.add_tile(Tile::P5);
        assert_eq!(hand.len(), 14);
        assert!(hand.is_valid());
        assert!(hand.is_complete());
        
        // 15th tile — invalid
        hand.add_tile(Tile::P6);
        assert!(!hand.is_valid());
    }
    
    #[test]
    fn test_hand_potential_win() {
        let mut hand = Hand::new();
        assert!(!hand.can_potentially_win());
        
        // Valid 13-tile hand
        let tiles13 = [
            Tile::M1, Tile::M2, Tile::M3, Tile::M4, Tile::M5,
            Tile::M6, Tile::M7, Tile::M8, Tile::M9,
            Tile::P1, Tile::P2, Tile::P3, Tile::P4,
        ];
        for t in &tiles13 {
            hand.add_tile(*t);
        }
        assert_eq!(hand.len(), 13);
        assert!(hand.can_potentially_win());
        
        // 14th tile — still potentially winning
        hand.add_tile(Tile::P5);
        assert_eq!(hand.len(), 14);
        assert!(hand.can_potentially_win());
        
        // 15th tile — too many
        hand.add_tile(Tile::P6);
        assert_eq!(hand.len(), 15);
        assert!(!hand.can_potentially_win());
    }
}