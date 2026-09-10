use crate::tile::{Hand, Tile};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// 配牌结果
#[derive(Debug, Clone)]
pub struct DealResult {
    /// 玩家手牌
    pub player_hands: Vec<Hand>,
    /// 剩余牌墙
    pub wall: Vec<Tile>,
    /// 庄家位置
    pub dealer_position: usize,
    /// 骰子点数
    pub dice: (u8, u8),
}

/// 配牌器
pub struct Dealer;

impl Dealer {
    /// 创建新牌墙（27种牌，每种4张 = 108张）
    pub fn create_wall() -> Vec<Tile> {
        let mut wall = Vec::new();
        
        for tile in Tile::all() {
            for _ in 0..4 {
                wall.push(tile);
            }
        }
        
        // 洗牌
        let mut rng = rand::thread_rng();
        wall.shuffle(&mut rng);
        
        wall
    }
    
    /// 配牌
    pub fn deal(num_players: usize) -> DealResult {
        let wall = Self::create_wall();
        let mut rng = rand::thread_rng();
        
        let dice1 = rng.gen_range(1..=6);
        let dice2 = rng.gen_range(1..=6);
        let dice = (dice1, dice2);
        
        // 庄家固定为位置0
        let dealer_position = 0;
        
        // 按顺序从牌墙配牌：庄家14张，其余玩家各13张
        let mut player_hands = vec![Hand::new(); num_players];
        let mut wall_index = 0;
        
        for player in 0..num_players {
            let hand_size = if player == dealer_position { 14 } else { 13 };
            for _ in 0..hand_size {
                if wall_index < wall.len() {
                    let tile = wall[wall_index];
                    player_hands[player].add_tile(tile);
                    wall_index += 1;
                }
            }
        }
        
        let wall = wall[wall_index..].to_vec();
        
        DealResult {
            player_hands,
            wall,
            dealer_position,
            dice,
        }
    }
    
    /// 检查配牌是否有效
    pub fn validate_deal(deal_result: &DealResult) -> bool {
        let num_players = deal_result.player_hands.len();
        
        // 检查玩家数量
        if num_players == 0 || num_players > 4 {
            return false;
        }
        
        // 检查手牌数量
        for (i, hand) in deal_result.player_hands.iter().enumerate() {
            let expected_size = if i == deal_result.dealer_position { 14 } else { 13 };
            if hand.len() != expected_size {
                return false;
            }
        }
        
        // 检查牌墙
        let total_tiles: usize = deal_result.player_hands.iter().map(|h| h.len()).sum();
        let expected_wall_size = 108 - total_tiles; // 4副牌共108张
        
        if deal_result.wall.len() != expected_wall_size {
            return false;
        }
        
        // 检查是否有重复牌（同一张牌在多个玩家手中）
        let mut tile_counts = HashMap::new();
        for hand in &deal_result.player_hands {
            for tile in hand.get_all_tiles() {
                *tile_counts.entry(tile).or_insert(0) += 1;
            }
        }
        
        // 检查牌墙中的牌
        for tile in &deal_result.wall {
            *tile_counts.entry(*tile).or_insert(0) += 1;
        }
        
        // 每种牌应该有4张
        for (&tile, &count) in &tile_counts {
            if count != 4 {
                println!("牌 {} 数量异常: {}", tile, count);
                return false;
            }
        }
        
        true
    }
    
    /// 获取配牌统计信息
    pub fn get_deal_stats(deal_result: &DealResult) -> DealStats {
        let mut stats = DealStats::new();
        
        for (i, hand) in deal_result.player_hands.iter().enumerate() {
            let hand_size = hand.len();
            let is_dealer = i == deal_result.dealer_position;
            
            stats.total_hands += hand_size;
            stats.dealer_hands += if is_dealer { hand_size } else { 0 };
            stats.player_hands += if is_dealer { 0 } else { hand_size };
            
            // 统计花色分布
            let distribution = hand.suit_distribution();
            for (j, &count) in distribution.iter().enumerate() {
                stats.suit_counts[j] += count;
            }
        }
        
        stats.wall_size = deal_result.wall.len();
        stats.dice_sum = deal_result.dice.0 + deal_result.dice.1;
        
        stats
    }
}

/// 配牌统计信息
#[derive(Debug, Clone)]
pub struct DealStats {
    pub total_hands: usize,
    pub dealer_hands: usize,
    pub player_hands: usize,
    pub suit_counts: [u8; 3],
    pub wall_size: usize,
    pub dice_sum: u8,
}

impl DealStats {
    pub fn new() -> Self {
        Self {
            total_hands: 0,
            dealer_hands: 0,
            player_hands: 0,
            suit_counts: [0; 3],
            wall_size: 0,
            dice_sum: 0,
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.total_hands == 108 - self.wall_size &&
        self.dealer_hands == 14 &&
        self.suit_counts.iter().sum::<u8>() as usize == self.total_hands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_wall() {
        let wall = Dealer::create_wall();
        
        assert_eq!(wall.len(), 108); // 4副牌，每副27张
        
        // 检查每种牌都有4张
        let mut tile_counts = std::collections::HashMap::new();
        for tile in &wall {
            *tile_counts.entry(*tile).or_insert(0) += 1;
        }
        
        for &count in tile_counts.values() {
            assert_eq!(count, 4);
        }
    }
    
    #[test]
    fn test_deal() {
        let deal_result = Dealer::deal(4);
        
        assert_eq!(deal_result.player_hands.len(), 4);
        assert_eq!(deal_result.dealer_position, 0);
        
        // 检查配牌是否有效
        assert!(Dealer::validate_deal(&deal_result));
        
        // 检查手牌数量
        assert_eq!(deal_result.player_hands[0].len(), 14); // 庄家
        for i in 1..4 {
            assert_eq!(deal_result.player_hands[i].len(), 13); // 闲家
        }
        
        // 检查牌墙
        assert_eq!(deal_result.wall.len(), 55); // 108 - 14 - 13*3 = 55
        
        // 检查骰子
        assert!(deal_result.dice.0 >= 1 && deal_result.dice.0 <= 6);
        assert!(deal_result.dice.1 >= 1 && deal_result.dice.1 <= 6);
    }
    
    #[test]
    fn test_deal_different_player_counts() {
        // 测试不同玩家数量的配牌
        for num_players in [2, 3, 4] {
            let deal_result = Dealer::deal(num_players);
            
            assert_eq!(deal_result.player_hands.len(), num_players);
            assert!(Dealer::validate_deal(&deal_result));
            
            // 检查庄家手牌数量
            assert_eq!(deal_result.player_hands[deal_result.dealer_position].len(), 14);
            
            // 检查闲家手牌数量
            for i in 0..num_players {
                if i != deal_result.dealer_position {
                    assert_eq!(deal_result.player_hands[i].len(), 13);
                }
            }
        }
    }
    
    #[test]
    fn test_deal_validation() {
        let deal_result = Dealer::deal(4);
        assert!(Dealer::validate_deal(&deal_result));
        
        // 测试无效配牌
        let mut invalid_deal = deal_result.clone();
        invalid_deal.player_hands[0].add_tile(Tile::M1); // 庄家多一张牌
        assert!(!Dealer::validate_deal(&invalid_deal));
    }
    
    #[test]
    fn test_deal_stats() {
        let deal_result = Dealer::deal(4);
        let stats = Dealer::get_deal_stats(&deal_result);
        
        assert!(stats.is_valid());
        assert_eq!(stats.total_hands, 108 - stats.wall_size);
        assert_eq!(stats.dealer_hands, 14);
        assert_eq!(stats.player_hands, 39);
        assert_eq!(stats.suit_counts.iter().sum::<u8>(), (108 - stats.wall_size) as u8);
    }
    
    #[test]
    fn test_wall_distribution() {
        let deal_result = Dealer::deal(4);
        
        // 检查牌墙中的牌是否都是完整的4张
        let mut tile_counts = std::collections::HashMap::new();
        for tile in &deal_result.wall {
            *tile_counts.entry(*tile).or_insert(0) += 1;
        }
        
        // 牌墙中的牌数量可能少于4张，因为已经被分配给玩家
        for &count in tile_counts.values() {
            assert!(count <= 4);
        }
    }
}
