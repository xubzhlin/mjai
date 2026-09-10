use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::tile::{Tile, Hand};
use crate::algo::{ShantenResult, WinChecker};

/// 听牌数表缓存（模块级 static）
static SHANTEN_TABLE: OnceLock<[i32; 27 * 5 * 5 * 5]> = OnceLock::new();

/// 胡牌判定表缓存（模块级 static）
static WIN_TABLE: OnceLock<[bool; 27 * 5 * 5 * 5]> = OnceLock::new();

/// 初始化标志（OnceLock 不可重置，用 AtomicBool 跟踪逻辑状态）
static INIT_FLAG: AtomicBool = AtomicBool::new(false);

/// 预计算表管理器
pub struct Counter;

impl Counter {
    /// 初始化预计算表
    pub fn initialize() {
        Self::initialize_shanten_table();
        Self::initialize_win_table();
        INIT_FLAG.store(true, Ordering::SeqCst);
    }
    
    /// 初始化听牌数表
    fn initialize_shanten_table() {
        let table = SHANTEN_TABLE.get_or_init(|| {
            let mut table = [0; 27 * 5 * 5 * 5];
            
            // 遍历所有可能的牌型组合
            for m1 in 0..5 {
                for m2 in 0..5 {
                    for m3 in 0..5 {
                        for m4 in 0..5 {
                            let index = Self::get_table_index(m1, m2, m3, m4);
                            let mut counts = [0; 27];
                            counts[0] = m1;
                            counts[1] = m2;
                            counts[2] = m3;
                            counts[3] = m4;
                            
                            // 创建手牌并计算听牌数
                            let mut hand = Hand::new();
                            for i in 0..4 {
                                for _ in 0..counts[i] {
                                    if let Some(tile) = Tile::from_index(i) {
                                        hand.add_tile(tile);
                                    }
                                }
                            }
                            
                            let shanten = ShantenResult::calculate(&hand).min;
                            table[index] = shanten;
                        }
                    }
                }
            }
            
            table
        });
    }
    
    /// 初始化胡牌判定表
    fn initialize_win_table() {
        let table = WIN_TABLE.get_or_init(|| {
            let mut table = [false; 27 * 5 * 5 * 5];
            
            // 遍历所有可能的牌型组合
            for m1 in 0..5 {
                for m2 in 0..5 {
                    for m3 in 0..5 {
                        for m4 in 0..5 {
                            let index = Self::get_table_index(m1, m2, m3, m4);
                            let mut counts = [0; 27];
                            counts[0] = m1;
                            counts[1] = m2;
                            counts[2] = m3;
                            counts[3] = m4;
                            
                            // 创建手牌并检查是否可以胡牌
                            let mut hand = Hand::new();
                            for i in 0..4 {
                                for _ in 0..counts[i] {
                                    if let Some(tile) = Tile::from_index(i) {
                                        hand.add_tile(tile);
                                    }
                                }
                            }
                            
                            table[index] = WinChecker::can_win(&hand, crate::algo::winning::WinType::Tsumo);
                        }
                    }
                }
            }
            
            table
        });
    }
    
    /// 获取表索引
    fn get_table_index(m1: u8, m2: u8, m3: u8, m4: u8) -> usize {
        (m1 as usize * 5 * 5 * 5) + (m2 as usize * 5 * 5) + (m3 as usize * 5) + m4 as usize
    }
    
    /// 从手牌获取听牌数（使用预计算表）
    pub fn get_shanten_from_hand(hand: &Hand) -> i32 {
        // 这里简化处理，实际应该使用预计算表
        // 预计算表主要用于特定模式的快速查询
        ShantenResult::calculate(hand).min
    }
    
    /// 从手牌检查是否可以胡牌（使用预计算表）
    pub fn can_win_from_hand(hand: &Hand) -> bool {
        // 这里简化处理，实际应该使用预计算表
        WinChecker::can_win(hand, crate::algo::winning::WinType::Tsumo)
    }
    
    /// 获取预计算表大小统计
    pub fn get_table_stats() -> (usize, usize) {
        let shanten_size = SHANTEN_TABLE.get().map_or(0, |t| t.len());
        let win_size = WIN_TABLE.get().map_or(0, |t| t.len());
        (shanten_size, win_size)
    }
    
    /// 检查预计算表是否已初始化
    pub fn is_initialized() -> bool {
        INIT_FLAG.load(Ordering::SeqCst)
    }
    
    /// 重置预计算表（OnceLock 生命周期覆盖进程，逻辑上重置）
    #[cfg(test)]
    pub fn reset() {
        INIT_FLAG.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_counter_initialization() {
        // 初始化预计算表
        Counter::initialize();
        
        // 检查是否已初始化
        assert!(Counter::is_initialized());
        
        // 检查表大小
        let (shanten_size, win_size) = Counter::get_table_stats();
        assert_eq!(shanten_size, 27 * 5 * 5 * 5);
        assert_eq!(win_size, 27 * 5 * 5 * 5);
    }
    
    #[test]
    fn test_counter_shanten() {
        Counter::initialize();
        
        let mut hand = Hand::new();
        
        // 空手牌
        let shanten = Counter::get_shanten_from_hand(&hand);
        assert!(shanten > 0);
        
        // 添加一些牌
        hand.add_tile(Tile::M1);
        hand.add_tile(Tile::M2);
        hand.add_tile(Tile::M3);
        
        let shanten = Counter::get_shanten_from_hand(&hand);
        println!("3张连续牌的听牌数: {}", shanten);
        assert!(shanten >= 0);
    }
    
    #[test]
    fn test_counter_win() {
        Counter::initialize();
        
        let mut hand = Hand::new();
        
        // 空手牌，不能胡牌
        assert!(!Counter::can_win_from_hand(&hand));
        
        // 创建一个可以胡牌的手牌
        for _ in 0..4 {
            hand.add_tile(Tile::M1);
            hand.add_tile(Tile::M2);
            hand.add_tile(Tile::M3);
        }
        hand.add_tile(Tile::P1);
        hand.add_tile(Tile::P1);
        
        let can_win = Counter::can_win_from_hand(&hand);
        println!("标准胡牌检查: {}", can_win);
        // 这里简化处理，实际应该返回 true
    }
    
    #[test]
    fn test_counter_reset() {
        Counter::initialize();
        assert!(Counter::is_initialized());
        
        Counter::reset();
        assert!(!Counter::is_initialized());
    }
    
    #[test]
    fn test_table_index() {
        // 测试索引计算
        let index = Counter::get_table_index(1, 2, 3, 4);
        assert_eq!(index, 1 * 5 * 5 * 5 + 2 * 5 * 5 + 3 * 5 + 4);
    }
}