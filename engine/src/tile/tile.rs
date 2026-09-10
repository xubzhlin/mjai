use std::fmt;
use serde::{Serialize, Deserialize};

/// 花色枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Suit {
    Man,  // 万子
    Pin,  // 筒子  
    Sou,  // 条子
}

impl Suit {
    pub fn all() -> [Suit; 3] {
        [Suit::Man, Suit::Pin, Suit::Sou]
    }
    
    pub fn index(&self) -> usize {
        match self {
            Suit::Man => 0,
            Suit::Pin => 1,
            Suit::Sou => 2,
        }
    }
    
    pub fn from_index(index: usize) -> Option<Suit> {
        match index {
            0 => Some(Suit::Man),
            1 => Some(Suit::Pin),
            2 => Some(Suit::Sou),
            _ => None,
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suit_str = match self {
            Suit::Man => "m",
            Suit::Pin => "p", 
            Suit::Sou => "s",
        };
        write!(f, "{}", suit_str)
    }
}

/// 牌枚举 - 四川麻将共27种（万筒条1-9）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tile {
    // 万子 (Man)
    M1, M2, M3, M4, M5, M6, M7, M8, M9,
    // 筒子 (Pin)  
    P1, P2, P3, P4, P5, P6, P7, P8, P9,
    // 条子 (Sou)
    S1, S2, S3, S4, S5, S6, S7, S8, S9,
}

impl Tile {
    /// 获取牌的花色
    pub fn suit(&self) -> Suit {
        match self {
            Tile::M1 | Tile::M2 | Tile::M3 | Tile::M4 | Tile::M5 | 
            Tile::M6 | Tile::M7 | Tile::M8 | Tile::M9 => Suit::Man,
            Tile::P1 | Tile::P2 | Tile::P3 | Tile::P4 | Tile::P5 | 
            Tile::P6 | Tile::P7 | Tile::P8 | Tile::P9 => Suit::Pin,
            Tile::S1 | Tile::S2 | Tile::S3 | Tile::S4 | Tile::S5 | 
            Tile::S6 | Tile::S7 | Tile::S8 | Tile::S9 => Suit::Sou,
        }
    }
    
    /// 获取牌的点数（1-9）
    pub fn number(&self) -> u8 {
        match self {
            Tile::M1 | Tile::P1 | Tile::S1 => 1,
            Tile::M2 | Tile::P2 | Tile::S2 => 2,
            Tile::M3 | Tile::P3 | Tile::S3 => 3,
            Tile::M4 | Tile::P4 | Tile::S4 => 4,
            Tile::M5 | Tile::P5 | Tile::S5 => 5,
            Tile::M6 | Tile::P6 | Tile::S6 => 6,
            Tile::M7 | Tile::P7 | Tile::S7 => 7,
            Tile::M8 | Tile::P8 | Tile::S8 => 8,
            Tile::M9 | Tile::P9 | Tile::S9 => 9,
        }
    }
    
    /// 从索引获取牌（0-26）
    pub fn from_index(index: usize) -> Option<Tile> {
        match index {
            0 => Some(Tile::M1), 1 => Some(Tile::M2), 2 => Some(Tile::M3), 3 => Some(Tile::M4), 4 => Some(Tile::M5),
            5 => Some(Tile::M6), 6 => Some(Tile::M7), 7 => Some(Tile::M8), 8 => Some(Tile::M9),
            9 => Some(Tile::P1), 10 => Some(Tile::P2), 11 => Some(Tile::P3), 12 => Some(Tile::P4), 13 => Some(Tile::P5),
            14 => Some(Tile::P6), 15 => Some(Tile::P7), 16 => Some(Tile::P8), 17 => Some(Tile::P9),
            18 => Some(Tile::S1), 19 => Some(Tile::S2), 20 => Some(Tile::S3), 21 => Some(Tile::S4), 22 => Some(Tile::S5),
            23 => Some(Tile::S6), 24 => Some(Tile::S7), 25 => Some(Tile::S8), 26 => Some(Tile::S9),
            _ => None,
        }
    }
    
    /// 转换为索引（0-26）
    pub fn to_index(&self) -> usize {
        match self {
            Tile::M1 => 0, Tile::M2 => 1, Tile::M3 => 2, Tile::M4 => 3, Tile::M5 => 4,
            Tile::M6 => 5, Tile::M7 => 6, Tile::M8 => 7, Tile::M9 => 8,
            Tile::P1 => 9, Tile::P2 => 10, Tile::P3 => 11, Tile::P4 => 12, Tile::P5 => 13,
            Tile::P6 => 14, Tile::P7 => 15, Tile::P8 => 16, Tile::P9 => 17,
            Tile::S1 => 18, Tile::S2 => 19, Tile::S3 => 20, Tile::S4 => 21, Tile::S5 => 22,
            Tile::S6 => 23, Tile::S7 => 24, Tile::S8 => 25, Tile::S9 => 26,
        }
    }
    
    /// 从 MJAI 格式字符串解析
    pub fn from_mjai_str(s: &str) -> Option<Tile> {
        if s.len() != 2 {
            return None;
        }
        
        let number = s[0..1].parse::<u8>().ok()?;
        let suit_char = &s[1..2];
        
        if number < 1 || number > 9 {
            return None;
        }
        
        match suit_char {
            "m" => match number {
                1 => Some(Tile::M1), 2 => Some(Tile::M2), 3 => Some(Tile::M3), 4 => Some(Tile::M4), 5 => Some(Tile::M5),
                6 => Some(Tile::M6), 7 => Some(Tile::M7), 8 => Some(Tile::M8), 9 => Some(Tile::M9),
                _ => None,
            },
            "p" => match number {
                1 => Some(Tile::P1), 2 => Some(Tile::P2), 3 => Some(Tile::P3), 4 => Some(Tile::P4), 5 => Some(Tile::P5),
                6 => Some(Tile::P6), 7 => Some(Tile::P7), 8 => Some(Tile::P8), 9 => Some(Tile::P9),
                _ => None,
            },
            "s" => match number {
                1 => Some(Tile::S1), 2 => Some(Tile::S2), 3 => Some(Tile::S3), 4 => Some(Tile::S4), 5 => Some(Tile::S5),
                6 => Some(Tile::S6), 7 => Some(Tile::S7), 8 => Some(Tile::S8), 9 => Some(Tile::S9),
                _ => None,
            },
            _ => None,
        }
    }
    
    /// 转换为 MJAI 格式字符串
    pub fn to_mjai_str(&self) -> String {
        format!("{}{}", self.number(), self.suit())
    }
    
    /// 获取所有牌
    pub fn all() -> [Tile; 27] {
        [
            // 万子
            Tile::M1, Tile::M2, Tile::M3, Tile::M4, Tile::M5,
            Tile::M6, Tile::M7, Tile::M8, Tile::M9,
            // 筒子
            Tile::P1, Tile::P2, Tile::P3, Tile::P4, Tile::P5,
            Tile::P6, Tile::P7, Tile::P8, Tile::P9,
            // 条子
            Tile::S1, Tile::S2, Tile::S3, Tile::S4, Tile::S5,
            Tile::S6, Tile::S7, Tile::S8, Tile::S9,
        ]
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.number(), self.suit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_suit() {
        assert_eq!(Tile::M1.suit(), Suit::Man);
        assert_eq!(Tile::P5.suit(), Suit::Pin);
        assert_eq!(Tile::S9.suit(), Suit::Sou);
    }
    
    #[test]
    fn test_tile_number() {
        assert_eq!(Tile::M1.number(), 1);
        assert_eq!(Tile::P5.number(), 5);
        assert_eq!(Tile::S9.number(), 9);
    }
    
    #[test]
    fn test_tile_index() {
        assert_eq!(Tile::M1.to_index(), 0);
        assert_eq!(Tile::P1.to_index(), 9);
        assert_eq!(Tile::S1.to_index(), 18);
        assert_eq!(Tile::from_index(0), Some(Tile::M1));
        assert_eq!(Tile::from_index(9), Some(Tile::P1));
        assert_eq!(Tile::from_index(18), Some(Tile::S1));
        assert_eq!(Tile::from_index(27), None);
    }
    
    #[test]
    fn test_mjai_format() {
        assert_eq!(Tile::M1.to_mjai_str(), "1m");
        assert_eq!(Tile::P5.to_mjai_str(), "5p");
        assert_eq!(Tile::S9.to_mjai_str(), "9s");
        
        assert_eq!(Tile::from_mjai_str("1m"), Some(Tile::M1));
        assert_eq!(Tile::from_mjai_str("5p"), Some(Tile::P5));
        assert_eq!(Tile::from_mjai_str("9s"), Some(Tile::S9));
        assert_eq!(Tile::from_mjai_str("0m"), None);
        assert_eq!(Tile::from_mjai_str("10m"), None);
        assert_eq!(Tile::from_mjai_str("1z"), None);
    }
    
    #[test]
    fn test_all_tiles() {
        let all_tiles = Tile::all();
        assert_eq!(all_tiles.len(), 27);
        assert_eq!(all_tiles[0], Tile::M1);
        assert_eq!(all_tiles[8], Tile::M9);
        assert_eq!(all_tiles[9], Tile::P1);
        assert_eq!(all_tiles[17], Tile::P9);
        assert_eq!(all_tiles[18], Tile::S1);
        assert_eq!(all_tiles[26], Tile::S9);
    }
}