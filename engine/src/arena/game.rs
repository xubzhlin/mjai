use crate::tile::{Tile, Suit, Hand};
use crate::state::{PlayerState, Action, MeldType, Meld, ActionValidator};
use crate::arena::Board;
use crate::phases::{Dealer, SwapProcessor, MissingProcessor};
use crate::algo::WinChecker;
use crate::rules::{
    RuleSet, FanResult, KongType, MeldView, MeldKind, SwapDirection,
    presets::xue_zhan::XueZhanRuleSet,
};
use rand::Rng;

/// 游戏阶段
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    Deal,
    Swap,
    Missing,
    Playing,
    GameOver,
}

/// 游戏结果
#[derive(Debug, Clone)]
pub struct GameResult {
    pub final_scores: Vec<i32>,
    pub winners: Vec<usize>,
    pub losers: Vec<usize>,
    pub total_turns: usize,
    pub game_over_reason: String,
}

/// 完整游戏流程管理器（集成 RuleSet trait）
pub struct Game {
    pub board: Board,
    pub phase: GamePhase,
    pub ruleset: XueZhanRuleSet,
    pub game_history: Vec<GameEvent>,
    pub current_turn: usize,
    pub base_score: u32,
}

/// 游戏事件
#[derive(Debug, Clone)]
pub enum GameEvent {
    StartGame,
    DealTiles,
    SwapTiles { player: usize, selected_tiles: Vec<Tile>, direction: String },
    Missing { player: usize, selected_suit: Option<Suit> },
    DrawTile { player: usize, tile: Tile },
    DiscardTile { player: usize, tile: Tile, tsumogiri: bool },
    Meld { player: usize, meld_type: MeldType, tile: Tile, from_player: Option<usize> },
    KongSettle { player: usize, kong_type: KongType, amount: i32 },
    Win { player: usize, tile: Tile, fan: u32, score: i32 },
    Pass { player: usize },
    GameOver { reason: String },
}

// ===== 辅助：state::Meld → rules::MeldView 转换 =====
fn convert_meld_type(mt: MeldType) -> MeldKind {
    match mt {
        MeldType::Pong => MeldKind::Pong,
        MeldType::ExposedKong => MeldKind::ExposedKong,
        MeldType::ConcealedKong => MeldKind::ConcealedKong,
        MeldType::AddKong => MeldKind::AddKong,
    }
}

fn convert_melds_to_view(melds: &[Meld]) -> Vec<MeldView> {
    melds.iter().map(|m| MeldView {
        tile: m.tile,
        kind: convert_meld_type(m.meld_type),
    }).collect()
}

impl Game {
    pub fn new(num_players: usize) -> Self {
        let ruleset = XueZhanRuleSet::new();

        let deal_result = Dealer::deal(num_players);
        let wall = crate::arena::Wall::new(deal_result.wall);

        let mut board = Board::new(num_players, deal_result.dealer_position, wall);

        for (i, hand) in deal_result.player_hands.into_iter().enumerate() {
            board.players[i].set_hand(hand);
        }

        Self {
            board,
            phase: GamePhase::Deal,
            ruleset,
            game_history: vec![GameEvent::StartGame],
            current_turn: 0,
            base_score: 1,
        }
    }

    pub fn start_game(&mut self) {
        self.phase = GamePhase::Swap;
        self.game_history.push(GameEvent::DealTiles);
    }

    /// 换三张阶段：掷骰子定向 + 四家同步交换
    pub fn process_swap_phase(&mut self) {
        if self.phase != GamePhase::Swap { return; }

        let mut hands: Vec<_> = self.board.players.iter_mut().map(|p| p.hand.copy()).collect();
        // 掷 1 枚骰子（点数 1-6，奇数方向由 SwapRule 映射）
        let dice: u8 = rand::thread_rng().gen_range(1..=6);
        let direction = self.ruleset.swap_rule().determine_direction(dice);

        let swap_result = SwapProcessor::process_swap(&mut hands, self.ruleset.swap_rule(), dice);

        for (i, hand) in hands.into_iter().enumerate() {
            self.board.players[i].hand = hand;
        }

        for (i, selection) in swap_result.player_selections.iter().enumerate() {
            self.game_history.push(GameEvent::SwapTiles {
                player: i,
                selected_tiles: selection.clone(),
                direction: format!("{:?}", direction),
            });
        }

        self.phase = GamePhase::Missing;
    }

    /// 定缺阶段
    pub fn process_missing_phase(&mut self) {
        if self.phase != GamePhase::Missing { return; }

        let mut hands: Vec<_> = self.board.players.iter_mut().map(|p| p.hand.copy()).collect();

        let missing_result = MissingProcessor::process_missing(&mut hands, self.ruleset.missing_rule());

        for (i, hand) in hands.into_iter().enumerate() {
            self.board.players[i].hand = hand;
        }

        for (i, selection) in missing_result.player_selections.iter().enumerate() {
            self.game_history.push(GameEvent::Missing {
                player: i,
                selected_suit: selection.clone(),
            });
        }

        self.phase = GamePhase::Playing;
        self.current_turn = 0;
    }

    /// 获取未胡玩家数（用于计分中的各家支付）
    fn non_winner_count(&self, exclude: Option<usize>) -> usize {
        self.board.players.iter()
            .enumerate()
            .filter(|(i, p)| !p.has_won && exclude.map(|ex| *i != ex).unwrap_or(true))
            .count()
    }

    /// 胡牌结算：调 RuleSet.fan + score_hu
    fn settle_win(&mut self, player_id: usize, win_type: crate::algo::winning::WinType, num_payers: usize) {
        let melds_view = convert_melds_to_view(&self.board.players[player_id].melds);
        let fan_result = self.ruleset.fan(&self.board.players[player_id].hand, &melds_view, win_type);
        let total = self.ruleset.score_hu(fan_result, win_type, self.base_score, num_payers);

        self.board.set_player_won(player_id, fan_result.fan, total);

        self.game_history.push(GameEvent::Win {
            player: player_id,
            tile: Tile::M1, // placeholder, 实际 tile 由调用方传入
            fan: fan_result.fan,
            score: total,
        });
    }

    /// 杠即时结算（刮风下雨）
    fn settle_kong(&mut self, player_id: usize, kong_type: KongType, num_payers: usize) {
        let amount = self.ruleset.score_kong(kong_type, self.base_score, num_payers);
        let current_turn = self.board.turn;
        let event_type = match kong_type {
            KongType::AnKan => MeldType::ConcealedKong,
            KongType::MinKan => MeldType::ExposedKong,
            KongType::BuKan => MeldType::AddKong,
        };
        self.board.get_player_mut(player_id).add_meld(Meld {
            meld_type: event_type,
            tile: Tile::M1,
            from_player: None,
            turn: current_turn,
        });

        self.game_history.push(GameEvent::KongSettle {
            player: player_id,
            kong_type,
            amount,
        });
    }

    pub fn process_turn(&mut self) -> bool {
        if self.phase != GamePhase::Playing || self.board.is_game_over() {
            return false;
        }

        let current_player = self.board.current_player;

        // 已胡玩家跳过
        if self.board.players[current_player].has_won {
            self.board.next_turn();
            self.current_turn += 1;
            return true;
        }

        // 1. 摸牌
        let drawn_tile = match self.board.draw_tile() {
            Some(t) => t,
            None => {
                // 牌墙空 → 流局
                self.resolve_draw();
                return false;
            }
        };

        self.game_history.push(GameEvent::DrawTile {
            player: current_player,
            tile: drawn_tile,
        });

        // 2. 杠上花检测（刚杠完摸的牌）
        if self.board.kan_shang_active {
            let can_win_kanshang = WinChecker::can_win(
                &self.board.players[current_player].hand,
                crate::algo::winning::WinType::KanShang,
            );
            if can_win_kanshang {
                let n = self.non_winner_count(Some(current_player));
                self.settle_win(current_player, crate::algo::winning::WinType::KanShang, n.max(1));
                self.board.kan_shang_active = false;
                self.check_and_finalize();
                return !self.board.is_game_over();
            }
            self.board.kan_shang_active = false;
        }

        // 3. 自摸检测
        let can_tsumo = WinChecker::can_win(
            &self.board.players[current_player].hand,
            crate::algo::winning::WinType::Tsumo,
        );
        if can_tsumo {
            let n = self.non_winner_count(Some(current_player));
            self.settle_win(current_player, crate::algo::winning::WinType::Tsumo, n.max(1));
            self.check_and_finalize();
            return !self.board.is_game_over();
        }

        // 4. AI 选择动作（简化：优先暗杠，否则打牌）
        let action = self.ai_select_action(current_player);

        match action.action_type {
            crate::state::ActionType::Discard => {
                if let Some(tile) = action.tile {
                    self.board.discard_tile(tile, true);
                    self.game_history.push(GameEvent::DiscardTile {
                        player: current_player,
                        tile,
                        tsumogiri: true,
                    });
                    // 检查其他玩家响应（一炮多响：先胡，再杠，再碰）
                    self.check_responses(current_player, tile);
                }
            },
            crate::state::ActionType::Kong => {
                if let Some(tile) = action.tile {
                    let ps = &self.board.players[current_player];

                    // 判断：补杠（有碰副露） vs 暗杠（手牌 4 张）
                    let is_add_kong = ps.melds.iter().any(|m|
                        m.meld_type == MeldType::Pong && m.tile == tile
                    );

                    if is_add_kong {
                        // === 补杠：先检查抢杠胡 ===
                        let robbers: Vec<usize> = (0..self.board.players.len())
                            .filter(|p| *p != current_player
                                && !self.board.players[*p].has_won
                                && self.ruleset.check_rob_kong(&self.board.players[*p].hand, &tile))
                            .collect();

                        if !robbers.is_empty() {
                            // 抢杠胡：补杠者独付（WinType::QiangGang → fan×2 + extra_bases）
                            self.board.players[current_player].remove_tile(tile);
                            for robber in &robbers {
                                self.settle_win(*robber, crate::algo::winning::WinType::QiangGang, 1);
                            }
                            // 补杠者不补摸，跳过本轮后续
                        } else {
                            // 正常补杠 + 补杠结算 + 杠后补摸
                            self.board.players[current_player].remove_tile(tile);
                            self.board.add_meld(MeldType::AddKong, tile, None);
                            let n = self.non_winner_count(Some(current_player));
                            let amount = self.ruleset.score_kong(KongType::BuKan, self.base_score, n.max(1));
                            self.board.players[current_player].score += amount;
                            self.game_history.push(GameEvent::KongSettle {
                                player: current_player, kong_type: KongType::BuKan, amount,
                            });
                            if let Some(nt) = self.board.draw_from_wall_end(1).first().copied() {
                                self.game_history.push(GameEvent::DrawTile {
                                    player: current_player, tile: nt,
                                });
                                self.board.kan_shang_active = true;
                            }
                        }
                    } else {
                        // 暗杠：手牌 4 张相同，直接杠 + 杠后补摸（暗杠不可抢）
                        self.settle_kong(current_player, KongType::AnKan, self.non_winner_count(Some(current_player)));
                        if let Some(nt) = self.board.draw_from_wall_end(1).first().copied() {
                            self.game_history.push(GameEvent::DrawTile {
                                player: current_player, tile: nt,
                            });
                            self.board.kan_shang_active = true;
                        }
                    }
                }
            },
            _ => {}
        }

        self.check_and_finalize();
        if self.board.is_game_over() { return false; }

        self.board.next_turn();
        self.current_turn += 1;
        true
    }

    /// 检查其他人对弃牌的响应（胡 > 杠 > 碰），支持一炮多响
    fn check_responses(&mut self, discarder: usize, tile: Tile) {
        let mut ron_winners = Vec::new();
        let mut kong_responders = Vec::new();
        let mut pong_responders = Vec::new();

        for player in 0..self.board.players.len() {
            if player == discarder || self.board.players[player].has_won { continue; }
            let p = &self.board.players[player];
            if WinChecker::can_win(&p.hand, crate::algo::winning::WinType::Ron) {
                ron_winners.push(player);
            } else if p.can_kong(tile) {
                kong_responders.push(player);
            } else if p.can_pong(tile) {
                pong_responders.push(player);
            }
        }

        // 优先级：胡 > 杠 > 碰
        if !ron_winners.is_empty() {
            // 一炮多响：多家同时胡，各自与 discarder 结算
            for winner in &ron_winners {
                self.board.players[*winner].remove_tile(tile);
                self.settle_win(*winner, crate::algo::winning::WinType::Ron, 1);
            }
        } else if !kong_responders.is_empty() {
            // 明杠：先到先得
            let responder = kong_responders[0];
            self.board.players[responder].remove_tile(tile);
            self.board.add_meld(MeldType::ExposedKong, tile, Some(discarder));
            // 明杠：点杠者付 2 底（点杠者已打，也算未胡；num_payers=1 固定）
            let amount = self.ruleset.score_kong(KongType::MinKan, self.base_score, 1);
            self.board.players[responder].score += amount;
            self.game_history.push(GameEvent::KongSettle { player: responder, kong_type: KongType::MinKan, amount });
            // 杠后补摸
            if let Some(new_tile) = self.board.draw_from_wall_end(1).first().copied() {
                self.game_history.push(GameEvent::DrawTile { player: responder, tile: new_tile });
                self.board.kan_shang_active = true;
            }
        } else if !pong_responders.is_empty() {
            let responder = pong_responders[0];
            self.board.players[responder].remove_tile(tile);
            self.board.add_meld(MeldType::Pong, tile, Some(discarder));
        }
    }

    /// 检查结束条件
    fn check_and_finalize(&mut self) {
        let _ = self.board.check_game_over();
        if self.board.is_game_over() {
            self.phase = GamePhase::GameOver;
            // 流局结算
            if matches!(self.board.game_over_reason, Some(crate::arena::GameOverReason::WallEmpty)) {
                self.resolve_draw();
            }
            let reason = format!("{:?}", self.board.game_over_reason);
            self.game_history.push(GameEvent::GameOver { reason });
        }
    }

    /// 流局结算：查花猪 + 查大叫
    /// 规则：花猪赔所有未胡者其牌型全额；大叫未听者赔已听者
    fn resolve_draw(&mut self) {
        let non_winners: Vec<usize> = self.board.players.iter()
            .enumerate()
            .filter(|(_, p)| !p.has_won)
            .map(|(i, _)| i)
            .collect();

        if non_winners.is_empty() { return; }

        // 1. 识别花猪/大叫
        let mut huazhu_players = Vec::new();
        let mut dajiao_players = Vec::new();
        let mut tenpai_players = Vec::new();

        for &p in &non_winners {
            let ps = &self.board.players[p];
            // 花猪：手牌仍有缺门牌
            let has_missing_remaining = ps.hand.hand_without_missing().tiles.iter().any(|t| *t > 0);
            if has_missing_remaining {
                huazhu_players.push(p);
                self.board.set_player_huazhu(p);
                continue; // 花猪同时也是大叫，不再单独标记
            }
            // 听牌判定：shanten == 0（通用听牌）+ 缺门已清干净
            let is_tenpai = ps.hand.is_tenpai() || ps.hand.is_waiting();
            if is_tenpai {
                tenpai_players.push(p);
            } else {
                dajiao_players.push(p);
                self.board.set_player_dajiao(p);
            }
        }

        // 2. 花猪结算：每只花猪赔所有未胡者（包括花猪自己）"全额"
        // 全额 = 该花猪假设胡牌时能拿到的番数 × base（简化用 RuleSet::fan 平胡计算）
        for &huazhu in &huazhu_players {
            let melds_view = convert_melds_to_view(&self.board.players[huazhu].melds);
            // 花猪的"全额"：用 Tsumo 假设一个番数作为赔付基数
            let fan_result = self.ruleset.fan(
                &self.board.players[huazhu].hand,
                &melds_view,
                crate::algo::winning::WinType::Tsumo,
            );
            // 简化：花猪按 fan 底赔付给所有未胡者（每人赔 fan 底）
            let per_loser = (fan_result.fan as i32) * (self.base_score as i32);
            for &receiver in &non_winners {
                self.board.players[huazhu].score -= per_loser;
                self.board.players[receiver].score += per_loser;
            }
        }

        // 3. 查大叫：未听者向已听者赔付（简化：每人赔 1 底 × 已听者数）
        let tenpai_count = tenpai_players.len();
        if !dajiao_players.is_empty() && tenpai_count > 0 {
            let per_tenpai = self.base_score as i32;
            for &dajiao in &dajiao_players {
                // 每个大叫赔所有听牌者
                let total_pay = per_tenpai * tenpai_count as i32;
                self.board.players[dajiao].score -= total_pay;
                for &tp in &tenpai_players {
                    self.board.players[tp].score += per_tenpai;
                }
            }
        }
    }

    fn ai_select_action(&self, player_id: usize) -> Action {
        let ps = &self.board.players[player_id];

        // 优先：用 ActionValidator 生成全部合法动作，取优先级最高
        let mut candidate = ActionValidator::generate_legal_actions(ps, self.current_turn);
        candidate.auto_select();
        if let Some(a) = candidate.selected_action.clone() {
            // 跳过胡牌/杠/碰等需要外部触发的动作（这里只处理本家主动动作）
            return a;
        }

        // 兜底：随便打一张
        for tile in Tile::all() {
            if ps.hand.contains(tile) {
                return Action::discard(tile, self.current_turn);
            }
        }
        Action::pass(self.current_turn)
    }

    pub fn get_game_result(&self) -> Option<GameResult> {
        if self.phase != GamePhase::GameOver { return None; }

        Some(GameResult {
            final_scores: self.board.players.iter().map(|p| p.score).collect(),
            winners: self.board.get_winner_players(),
            losers: self.board.get_loser_players(),
            total_turns: self.current_turn,
            game_over_reason: format!("{:?}", self.board.game_over_reason),
        })
    }

    pub fn get_game_summary(&self) -> GameSummary {
        GameSummary {
            phase: self.phase.clone(),
            turn: self.current_turn,
            current_player: self.board.current_player,
            dealer_position: self.board.dealer_position,
            wall_remaining: self.board.wall.remaining_count(),
            players_won: self.board.players.iter().filter(|p| p.has_won).count(),
            game_over: self.board.is_game_over(),
            game_history_count: self.game_history.len(),
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.phase == GamePhase::GameOver
    }

    pub fn run_to_end(&mut self) {
        self.start_game();
        self.process_swap_phase();
        self.process_missing_phase();
        while self.process_turn() {
            // 继续直到结束
        }
    }
}

#[derive(Debug, Clone)]
pub struct GameSummary {
    pub phase: GamePhase,
    pub turn: usize,
    pub current_player: usize,
    pub dealer_position: usize,
    pub wall_remaining: usize,
    pub players_won: usize,
    pub game_over: bool,
    pub game_history_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_creation() {
        let game = Game::new(4);
        assert_eq!(game.board.players.len(), 4);
        assert_eq!(game.phase, GamePhase::Deal);
    }

    #[test]
    fn test_game_run_to_end() {
        let mut game = Game::new(4);
        game.run_to_end();
        assert!(game.is_game_over());
        let result = game.get_game_result();
        assert!(result.is_some());
    }
}

