use crate::tile::{Tile, Hand};
use crate::state::PlayerState;
use crate::rules::SwapDirection;
use pyo3::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

/// 牌墙管理器
#[derive(Debug, Clone)]
pub struct Wall {
    tiles: VecDeque<Tile>,
    initial_size: usize,
}

impl Wall {
    pub fn new(tiles: Vec<Tile>) -> Self {
        let initial_size = tiles.len();
        Self {
            tiles: VecDeque::from(tiles),
            initial_size,
        }
    }
    
    pub fn len(&self) -> usize {
        self.tiles.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
    
    pub fn draw(&mut self) -> Option<Tile> {
        self.tiles.pop_front()
    }
    
    pub fn draw_from_end(&mut self, count: usize) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for _ in 0..count {
            if let Some(tile) = self.tiles.pop_back() {
                tiles.push(tile);
            }
        }
        tiles
    }
    
    pub fn remaining_count(&self) -> usize {
        self.tiles.len()
    }
    
    pub fn remaining_ratio(&self) -> f64 {
        self.remaining_count() as f64 / self.initial_size as f64
    }
}

/// 牌桌状态
#[derive(Debug, Clone)]
#[pyclass]
pub struct Board {
    /// 玩家状态
    pub players: Vec<PlayerState>,
    /// 牌墙
    pub wall: Wall,
    /// 当前回合数
    pub turn: usize,
    /// 当前玩家索引
    pub current_player: usize,
    /// 庄家位置
    pub dealer_position: usize,
    /// 游戏是否结束
    pub game_over: bool,
    /// 游戏结束原因
    pub game_over_reason: Option<GameOverReason>,
    /// 换三张方向（开局后设定）
    pub swap_direction: Option<SwapDirection>,
    /// 当前是否为杠上花状态（刚杠完摸牌）
    pub kan_shang_active: bool,
}

/// 游戏结束原因
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameOverReason {
    ThreeWins,   // 三家胡牌
    WallEmpty,   // 牌墙摸完
    AllPlayersWon, // 所有玩家都胡牌
}

/// 牌桌管理器
impl Board {
    pub fn new(num_players: usize, dealer_position: usize, wall: Wall) -> Self {
        let mut players = Vec::new();
        for i in 0..num_players {
            players.push(PlayerState::new(i));
        }
        
        Self {
            players,
            wall,
            turn: 0,
            current_player: dealer_position,
            dealer_position,
            game_over: false,
            game_over_reason: None,
            swap_direction: None,
            kan_shang_active: false,
        }
    }
    
    pub fn get_current_player(&self) -> &PlayerState {
        &self.players[self.current_player]
    }
    
    pub fn get_current_player_mut(&mut self) -> &mut PlayerState {
        &mut self.players[self.current_player]
    }
    
    pub fn get_player(&self, player_id: usize) -> &PlayerState {
        &self.players[player_id]
    }
    
    pub fn get_player_mut(&mut self, player_id: usize) -> &mut PlayerState {
        &mut self.players[player_id]
    }
    
    pub fn get_other_players(&self, exclude_player: usize) -> Vec<&PlayerState> {
        self.players.iter()
            .enumerate()
            .filter(|(i, _)| *i != exclude_player)
            .map(|(_, player)| player)
            .collect()
    }
    
    pub fn get_other_players_mut(&mut self, exclude_player: usize) -> Vec<&mut PlayerState> {
        self.players.iter_mut()
            .enumerate()
            .filter(|(i, _)| *i != exclude_player)
            .map(|(_, player)| player)
            .collect()
    }
    
    pub fn next_turn(&mut self) {
        self.turn += 1;
        self.current_player = (self.current_player + 1) % self.players.len();
    }
    
    pub fn skip_turn(&mut self) {
        self.next_turn();
    }
    
    pub fn draw_tile(&mut self) -> Option<Tile> {
        let tile = self.wall.draw();
        if let Some(t) = tile {
            self.players[self.current_player].add_tile(t);
        }
        tile
    }
    
    pub fn draw_from_wall_end(&mut self, count: usize) -> Vec<Tile> {
        let tiles = self.wall.draw_from_end(count);
        for tile in &tiles {
            self.players[self.current_player].add_tile(*tile);
        }
        tiles
    }
    
    pub fn discard_tile(&mut self, tile: Tile, tsumogiri: bool) -> bool {
        if self.players[self.current_player].remove_tile(tile) {
            self.players[self.current_player].add_discarded(tile, self.turn, tsumogiri);
            true
        } else {
            false
        }
    }
    
    pub fn add_meld(&mut self, meld_type: crate::state::MeldType, tile: Tile, from_player: Option<usize>) {
        let meld = crate::state::Meld {
            meld_type,
            tile,
            from_player,
            turn: self.turn,
        };
        self.players[self.current_player].add_meld(meld);
    }
    
    pub fn set_player_won(&mut self, player_id: usize, fan: u32, score: i32) {
        self.players[player_id].set_won(fan, score);
    }
    
    pub fn set_player_huazhu(&mut self, player_id: usize) {
        self.players[player_id].set_huazhu();
    }
    
    pub fn set_player_dajiao(&mut self, player_id: usize) {
        self.players[player_id].set_dajiao();
    }
    
    pub fn check_game_over(&mut self) -> bool {
        // 检查是否三家胡牌
        let won_count = self.players.iter().filter(|p| p.has_won).count();
        if won_count >= 3 {
            self.game_over = true;
            self.game_over_reason = Some(GameOverReason::ThreeWins);
            return true;
        }
        
        // 检查是否牌墙摸完
        if self.wall.is_empty() {
            self.game_over = true;
            self.game_over_reason = Some(GameOverReason::WallEmpty);
            return true;
        }
        
        // 检查是否所有玩家都胡牌
        if self.players.iter().all(|p| p.has_won) {
            self.game_over = true;
            self.game_over_reason = Some(GameOverReason::AllPlayersWon);
            return true;
        }
        
        false
    }
    
    pub fn get_game_summary(&self) -> GameSummary {
        GameSummary {
            turn: self.turn,
            current_player: self.current_player,
            dealer_position: self.dealer_position,
            wall_remaining: self.wall.remaining_count(),
            players_won: self.players.iter().filter(|p| p.has_won).count(),
            game_over: self.game_over,
            game_over_reason: self.game_over_reason.clone(),
        }
    }
    
    pub fn is_game_over(&self) -> bool {
        self.game_over
    }
    
    pub fn get_winner_players(&self) -> Vec<usize> {
        self.players.iter()
            .enumerate()
            .filter(|(_, p)| p.has_won)
            .map(|(i, _)| i)
            .collect()
    }
    
    pub fn get_loser_players(&self) -> Vec<usize> {
        self.players.iter()
            .enumerate()
            .filter(|(_, p)| !p.has_won)
            .map(|(i, _)| i)
            .collect()
    }
}

#[pymethods]
impl Board {
    #[pyo3(name = "current_player")]
    fn py_current_player(&self) -> usize {
        self.current_player
    }

    #[pyo3(name = "kan_shang_active")]
    fn py_kan_shang_active(&self) -> bool {
        self.kan_shang_active
    }

    #[pyo3(name = "wall_remaining")]
    fn py_wall_remaining(&self) -> usize {
        self.wall.remaining_count()
    }

    #[pyo3(name = "num_players")]
    fn py_num_players(&self) -> usize {
        self.players.len()
    }

    #[pyo3(name = "player_score")]
    fn py_player_score(&self, pid: usize) -> i32 {
        self.players.get(pid).map(|p| p.score).unwrap_or(0)
    }

    #[pyo3(name = "player_has_won")]
    fn py_player_has_won(&self, pid: usize) -> bool {
        self.players.get(pid).map(|p| p.has_won).unwrap_or(false)
    }

    #[pyo3(name = "game_summary")]
    fn py_game_summary(&self) -> String {
        // 简单 JSON-like 摘要字符串，便于 Python 打印
        let scores: Vec<String> = self.players.iter()
            .map(|p| format!("{}:+{}", p.id, p.score))
            .collect();
        format!("GameOver wall_remaining={} players=[{}]",
            self.wall.remaining_count(), scores.join(", "))
    }
}

/// 游戏摘要
#[derive(Debug, Clone)]
pub struct GameSummary {
    pub turn: usize,
    pub current_player: usize,
    pub dealer_position: usize,
    pub wall_remaining: usize,
    pub players_won: usize,
    pub game_over: bool,
    pub game_over_reason: Option<GameOverReason>,
}

impl GameSummary {
    pub fn is_valid(&self) -> bool {
        self.turn >= 0 && 
        self.current_player < 4 && 
        self.dealer_position < 4 &&
        self.wall_remaining >= 0 &&
        self.players_won <= 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::Tile;
    
    #[test]
    fn test_wall() {
        let mut wall = Wall::new(vec![Tile::M1, Tile::M2, Tile::M3]);
        
        assert_eq!(wall.len(), 3);
        assert!(!wall.is_empty());
        
        let tile1 = wall.draw();
        assert_eq!(tile1, Some(Tile::M1));
        assert_eq!(wall.len(), 2);
        
        let tile2 = wall.draw();
        assert_eq!(tile2, Some(Tile::M2));
        assert_eq!(wall.len(), 1);
        
        let tile3 = wall.draw();
        assert_eq!(tile3, Some(Tile::M3));
        assert_eq!(wall.len(), 0);
        
        let tile4 = wall.draw();
        assert_eq!(tile4, None);
    }
    
    #[test]
    fn test_board_creation() {
        let wall = Wall::new(vec![Tile::M1, Tile::M2, Tile::M3, Tile::M4]);
        let board = Board::new(4, 0, wall);
        
        assert_eq!(board.players.len(), 4);
        assert_eq!(board.current_player, 0);
        assert_eq!(board.dealer_position, 0);
        assert!(!board.game_over);
        assert_eq!(board.turn, 0);
    }
    
    #[test]
    fn test_board_turn_management() {
        let wall = Wall::new(vec![Tile::M1, Tile::M2, Tile::M3, Tile::M4]);
        let mut board = Board::new(4, 0, wall);
        
        assert_eq!(board.current_player, 0);
        
        board.next_turn();
        assert_eq!(board.current_player, 1);
        
        board.next_turn();
        assert_eq!(board.current_player, 2);
        
        board.next_turn();
        assert_eq!(board.current_player, 3);
        
        board.next_turn();
        assert_eq!(board.current_player, 0);
    }
    
    #[test]
    fn test_board_tile_operations() {
        let wall = Wall::new(vec![Tile::M1, Tile::M2, Tile::M3, Tile::M4]);
        let mut board = Board::new(4, 0, wall);
        
        // 摸牌
        let tile = board.draw_tile();
        assert_eq!(tile, Some(Tile::M1));
        assert_eq!(board.players[0].hand_size(), 1);
        
        // 打牌
        let discarded = board.discard_tile(Tile::M1, false);
        assert!(discarded);
        assert_eq!(board.players[0].hand_size(), 0);
        assert_eq!(board.players[0].discarded_count(), 1);
    }
    
    #[test]
    fn test_board_meld_operations() {
        let wall = Wall::new(vec![Tile::M1, Tile::M2, Tile::M3, Tile::M4]);
        let mut board = Board::new(4, 0, wall);
        
        // 添加副露
        board.add_meld(crate::state::MeldType::Pong, Tile::M1, Some(1));
        assert_eq!(board.players[0].meld_count(), 1);
    }
    
    #[test]
    fn test_board_game_over() {
        let wall = Wall::new(vec![]);
        let mut board = Board::new(4, 0, wall);
        
        // 牌墙为空，游戏应该结束
        assert!(board.check_game_over());
        assert!(board.game_over);
        assert_eq!(board.game_over_reason, Some(GameOverReason::WallEmpty));
    }
    
    #[test]
    fn test_board_game_summary() {
        let wall = Wall::new(vec![Tile::M1, Tile::M2]);
        let mut board = Board::new(4, 0, wall);
        
        let summary = board.get_game_summary();
        assert!(summary.is_valid());
        assert_eq!(summary.turn, 0);
        assert_eq!(summary.current_player, 0);
        assert_eq!(summary.dealer_position, 0);
        assert_eq!(summary.wall_remaining, 2);
        assert!(!summary.game_over);
    }
    
    #[test]
    fn test_board_winners_and_losers() {
        let wall = Wall::new(vec![]);
        let mut board = Board::new(4, 0, wall);
        
        // 设置一些玩家获胜
        board.set_player_won(0, 1, 1);
        board.set_player_won(1, 1, 1);
        
        let winners = board.get_winner_players();
        let losers = board.get_loser_players();
        
        assert_eq!(winners.len(), 2);
        assert_eq!(losers.len(), 2);
        assert!(winners.contains(&0));
        assert!(winners.contains(&1));
        assert!(losers.contains(&2));
        assert!(losers.contains(&3));
    }
}