use crate::tile::{Hand, Tile, Suit};
use crate::rules::{SwapRule, SwapDirection};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;

/// 换三张规则实现
#[pyclass(name = "SwapRuleCalculator")]
#[derive(Clone)]
pub struct SwapRuleCalculator {
    #[pyo3(get)]
    pub enabled: bool,
    #[pyo3(get)]
    pub tiles_per_player: usize,
}

#[pymethods]
impl SwapRuleCalculator {
    #[new]
    pub fn new_py() -> Self {
        Self { enabled: true, tiles_per_player: 3 }
    }
}

impl SwapRuleCalculator {
    pub fn new(enabled: bool, tiles_per_player: usize) -> Self {
        Self {
            enabled,
            tiles_per_player,
        }
    }

    /// AI策略：选择最少花色的3张牌进行交换
    /// 如果同花色不足3张，则选择次少花色
    pub fn select_tiles_for_swap(&self, hand: &Hand) -> Result<(Suit, Vec<Tile>)> {
        if !self.enabled {
            return Err(anyhow!("换三张已禁用"));
        }

        // 统计各花色牌数
        let mut suit_counts = [(Suit::Man, 0), (Suit::Pin, 0), (Suit::Sou, 0)];

        for i in 0..27 {
            let tile = Tile::from_index(i).unwrap_or(Tile::M1);
            if hand.tiles[i] > 0 {
                match tile.suit() {
                    Suit::Man => suit_counts[0].1 += hand.tiles[i] as usize,
                    Suit::Pin => suit_counts[1].1 += hand.tiles[i] as usize,
                    Suit::Sou => suit_counts[2].1 += hand.tiles[i] as usize,
                }
            }
        }

        // 按牌数排序，最少花色优先
        suit_counts.sort_by_key(|(_, count)| *count);

        // 选择最少花色（如果该花色牌数足够）或次少花色
        let mut selected_suit = suit_counts[0].0;
        let mut selected_tiles = Vec::new();

        // 尝试从最少花色选择
        for i in 0..27 {
            let tile = Tile::from_index(i).unwrap_or(Tile::M1);
            if tile.suit() == selected_suit && hand.tiles[i] > 0 {
                selected_tiles.push(tile);
                if selected_tiles.len() == self.tiles_per_player {
                    break;
                }
            }
        }

        // 如果最少花色牌数不足，尝试次少花色
        if selected_tiles.len() < self.tiles_per_player {
            selected_suit = suit_counts[1].0;
            selected_tiles.clear();

            for i in 0..27 {
                let tile = Tile::from_index(i).unwrap_or(Tile::M1);
                if tile.suit() == selected_suit && hand.tiles[i] > 0 {
                    selected_tiles.push(tile);
                    if selected_tiles.len() == self.tiles_per_player {
                        break;
                    }
                }
            }
        }

        if selected_tiles.len() < self.tiles_per_player {
            return Err(anyhow!("无法选择足够的牌进行交换"));
        }

        Ok((selected_suit, selected_tiles))
    }

    /// 验证换三张是否合法
    pub fn is_valid_swap(&self, hand: &Hand, tiles: &[Tile], suit: Suit) -> bool {
        if !self.enabled {
            return false;
        }

        // 检查选择的牌是否都是同花色
        for tile in tiles {
            if tile.suit() != suit {
                return false;
            }
        }

        // 检查手牌中是否有这些牌
        for tile in tiles {
            let index = tile.to_index();
            if hand.tiles[index] == 0 {
                return false;
            }
        }

        // 检查牌数是否正确
        if tiles.len() != self.tiles_per_player {
            return false;
        }

        true
    }

    /// 执行换三张
    pub fn perform_swap(&self, hands: [&mut Hand; 4], tiles: [[Tile; 3]; 4], direction: SwapDirection) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // 根据方向确定交换关系
        let swap_pairs: Vec<(usize, usize)> = match direction {
            SwapDirection::Down => vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            SwapDirection::Across => vec![(0, 2), (1, 3)],
            SwapDirection::Up => vec![(0, 3), (1, 0), (2, 1), (3, 2)],
        };

        // 执行交换
        for (player1, player2) in swap_pairs {
            // 临时存储玩家1的牌
            let mut temp_tiles = Vec::new();
            for tile in tiles[player1] {
                let index = tile.to_index();
                let count = hands[player1].tiles[index];
                if count > 0 {
                    temp_tiles.push((tile, count));
                    hands[player1].tiles[index] = 0;
                }
            }

            // 将玩家2的牌给玩家1
            for tile in tiles[player2] {
                let index = tile.to_index();
                hands[player1].tiles[index] += 1;
            }

            // 将临时存储的牌给玩家2
            for (tile, count) in temp_tiles {
                let index = tile.to_index();
                hands[player2].tiles[index] += count;
            }
        }

        Ok(())
    }
}

impl SwapRule for SwapRuleCalculator {
    /// 根据骰子确定换三张方向
    /// 骰点 1·2 → 下家, 3·4 → 对家, 5·6 → 上家
    fn determine_direction(&self, dice: u8) -> SwapDirection {
        match dice {
            1 | 2 => SwapDirection::Down,
            3 | 4 => SwapDirection::Across,
            5 | 6 => SwapDirection::Up,
            _ => SwapDirection::Down, // 默认下家
        }
    }

    /// AI 选 3 张同花色换出牌
    fn select_swap_tiles(&self, hand: &Hand) -> Vec<Tile> {
        self.select_tiles_for_swap(hand)
            .map(|(_, tiles)| tiles)
            .unwrap_or_default()
    }
}

/// 换三张状态管理
pub struct SwapManager {
    pub swap_rule: SwapRuleCalculator,
    pub swap_completed: bool,
    pub swap_direction: Option<SwapDirection>,
    pub tiles_swapped: Option<[[Tile; 3]; 4]>,
}

impl SwapManager {
    pub fn new(enabled: bool, tiles_per_player: usize) -> Self {
        Self {
            swap_rule: SwapRuleCalculator::new(enabled, tiles_per_player),
            swap_completed: false,
            swap_direction: None,
            tiles_swapped: None,
        }
    }

    pub fn start_swap_phase(&mut self, dice: u8) -> SwapDirection {
        let direction = self.swap_rule.determine_direction(dice);
        self.swap_direction = Some(direction);
        direction
    }

    pub fn execute_swap(&mut self, hands: [&mut Hand; 4], tiles: [[Tile; 3]; 4]) -> Result<crate::rules::SwapResult> {
        if self.swap_completed {
            return Ok(crate::rules::SwapResult {
                direction: self.swap_direction.unwrap_or(SwapDirection::Down),
                tiles_swapped: tiles,
                success: false,
            });
        }

        let direction = self.swap_direction.unwrap_or(SwapDirection::Down);

        // 验证交换是否合法
        for i in 0..4 {
            let suit = tiles[i][0].suit();
            if !self.swap_rule.is_valid_swap(hands[i], &tiles[i], suit) {
                return Ok(crate::rules::SwapResult {
                    direction,
                    tiles_swapped: tiles,
                    success: false,
                });
            }
        }

        // 执行交换
        self.swap_rule.perform_swap(hands, tiles, direction)?;

        self.swap_completed = true;
        self.tiles_swapped = Some(tiles);

        Ok(crate::rules::SwapResult {
            direction,
            tiles_swapped: tiles,
            success: true,
        })
    }

    pub fn is_swap_completed(&self) -> bool {
        self.swap_completed
    }

    pub fn reset(&mut self) {
        self.swap_completed = false;
        self.swap_direction = None;
        self.tiles_swapped = None;
    }
}
