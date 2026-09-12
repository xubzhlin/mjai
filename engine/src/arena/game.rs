use crate::tile::{Tile, Suit, Hand};
use crate::state::{PlayerState, Action, MeldType, Meld, ActionValidator};
use crate::arena::Board;
use crate::phases::{Dealer, SwapProcessor, MissingProcessor};
use crate::algo::WinChecker;
use crate::rules::{
    RuleSet, FanResult, KongType, MeldView, MeldKind, SwapDirection,
    presets::xue_zhan::XueZhanRuleSet,
};
use pyo3::prelude::*;
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
#[pyclass]
pub struct Game {
    pub board: Board,
    pub phase: GamePhase,
    pub ruleset: XueZhanRuleSet,
    pub game_history: Vec<GameEvent>,
    pub current_turn: usize,
    pub base_score: u32,

    // 手动被动响应模式：弃牌后暂停，让 Python 决策碰/杠/胡
    pub process_passive_manually: bool,
    pub pending_passive: Option<PendingPassive>,
}

/// 弃牌后待处理的被动响应（胡 > 杠 > 碰）
#[derive(Debug, Clone)]
pub struct PendingPassive {
    pub discarder: usize,
    pub tile: Tile,
    pub tile_index: usize,
    pub ron_winners: Vec<usize>,
    pub kong_responders: Vec<usize>,
    pub pong_responders: Vec<usize>,
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
            process_passive_manually: false,
            pending_passive: None,
        }
    }

    pub fn start_game(&mut self) {
        self.phase = GamePhase::Swap;
        self.game_history.push(GameEvent::DealTiles);
    }

    /// 换三张阶段：掷骰子定向 + 四家同步交换
    pub fn process_swap_phase(&mut self) {
        if self.phase != GamePhase::Swap { return; }
        self._process_swap_phase_internal(None);
    }

    /// 手动版：Python 可选传入每个玩家的换牌 (tile indices)
    pub fn process_swap_phase_with_selections(&mut self, python_selections: Option<&[Option<Vec<usize>>]>) {
        if self.phase != GamePhase::Swap { return; }
        self._process_swap_phase_internal(python_selections);
    }

    fn _process_swap_phase_internal(&mut self, python_selections: Option<&[Option<Vec<usize>>]>) {
        let mut hands: Vec<_> = self.board.players.iter_mut().map(|p| p.hand.copy()).collect();
        let dice: u8 = rand::thread_rng().gen_range(1..=6);
        let direction = self.ruleset.swap_rule().determine_direction(dice);

        let swap_result = SwapProcessor::process_swap(
            &mut hands, self.ruleset.swap_rule(), dice, python_selections,
        );

        for (i, hand) in hands.into_iter().enumerate() {
            self.board.players[i].hand = hand;
        }
        for (i, selection) in swap_result.player_selections.iter().enumerate() {
            self.game_history.push(GameEvent::SwapTiles {
                player: i, selected_tiles: selection.clone(),
                direction: format!("{:?}", direction),
            });
        }
        self.phase = GamePhase::Missing;
    }

    /// 定缺阶段
    pub fn process_missing_phase(&mut self) {
        if self.phase != GamePhase::Missing { return; }
        self._process_missing_phase_internal(None);
    }

    /// 手动版：Python 可选传入每个玩家的缺门 (0=Man,1=Pin,2=Sou)
    pub fn process_missing_phase_with_selections(&mut self, python_selections: Option<&[Option<usize>]>) {
        if self.phase != GamePhase::Missing { return; }
        self._process_missing_phase_internal(python_selections);
    }

    fn _process_missing_phase_internal(&mut self, python_selections: Option<&[Option<usize>]>) {
        let mut hands: Vec<_> = self.board.players.iter_mut().map(|p| p.hand.copy()).collect();

        let missing_result = MissingProcessor::process_missing(
            &mut hands, self.ruleset.missing_rule(), python_selections,
        );

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

    /// 获取未胡玩家列表（不含 exclude 指定者）
    fn unwon_players_excluding(&self, exclude: Option<usize>) -> Vec<usize> {
        self.board.players.iter()
            .enumerate()
            .filter(|(i, p)| !p.has_won && exclude.map(|ex| *i != ex).unwrap_or(true))
            .map(|(i, _)| i)
            .collect()
    }

    /// 胡牌结算：调 RuleSet.fan 计算番，然后同时处理赢家加和输家减（零和）
    /// - Tsumo/KanShang: 其余未胡者各家支付 (base×fan + extra_bases×base) × 1（winner 总共收 n 倍）
    /// - Ron/PaoHu/QiangGang: 仅点炮者支付 1 倍
    fn settle_win(&mut self, player_id: usize, win_type: crate::algo::winning::WinType, base: u32,
                  payers: &[usize]) {
        let melds_view = convert_melds_to_view(&self.board.players[player_id].melds);
        let fan_result = self.ruleset.fan(&self.board.players[player_id].hand, &melds_view, win_type);

        let per_payer = (base as i32) * (fan_result.fan as i32) + (fan_result.extra_bases as i32) * (base as i32);
        let total = per_payer * (payers.len() as i32);

        self.board.players[player_id].score += total;
        self.board.set_player_won(player_id, fan_result.fan as u32, total);

        for &payer in payers {
            self.board.players[payer].score -= per_payer;
        }

        self.game_history.push(GameEvent::Win {
            player: player_id,
            tile: Tile::M1,
            fan: fan_result.fan,
            score: total,
        });
    }

    /// 杠即时结算（刮风下雨）：同时处理赢家加和输家减（零和）
    /// - AnKan: 其余未胡者各付 2 base（不含自己）
    /// - MinKan: 点杠者付 2 base
    /// - BuKan: 其余未胡者各付 1 base（不含自己）
    fn settle_kong(&mut self, player_id: usize, kong_type: KongType, base: u32,
                   payers: &[usize], discarder: Option<usize>) {
        let base_i = base as i32;
        let per_payer = match kong_type {
            KongType::AnKan => 2 * base_i,
            KongType::MinKan => 2 * base_i,
            KongType::BuKan => base_i,
        };

        let total = match kong_type {
            KongType::MinKan => per_payer,
            _ => per_payer * (payers.len() as i32),
        };

        self.board.players[player_id].score += total;

        match kong_type {
            KongType::MinKan => {
                if let Some(d) = discarder {
                    self.board.players[d].score -= per_payer;
                }
            }
            _ => {
                for &p in payers {
                    self.board.players[p].score -= per_payer;
                }
            }
        }

        let event_type = match kong_type {
            KongType::AnKan => MeldType::ConcealedKong,
            KongType::MinKan => MeldType::ExposedKong,
            KongType::BuKan => MeldType::AddKong,
        };
        let turn = self.board.turn;
        self.board.get_player_mut(player_id).add_meld(Meld {
            meld_type: event_type,
            tile: Tile::M1,
            from_player: None,
            turn,
        });

        self.game_history.push(GameEvent::KongSettle {
            player: player_id,
            kong_type,
            amount: total,
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
                let payers = self.unwon_players_excluding(Some(current_player));
                self.settle_win(current_player, crate::algo::winning::WinType::KanShang, self.base_score, &payers);
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
            let payers = self.unwon_players_excluding(Some(current_player));
            self.settle_win(current_player, crate::algo::winning::WinType::Tsumo, self.base_score, &payers);
            self.check_and_finalize();
            return !self.board.is_game_over();
        }

        // 4. AI 选择动作
        let action = self.ai_select_action(current_player);
        self.execute_action(current_player, &action);

        self.check_and_finalize();
        if self.board.is_game_over() { return false; }

        self.board.next_turn();
        self.current_turn += 1;
        true
    }

    /// 执行动作（Discard / Kong / Pass），从原 process_turn 中抽离，便于 Python 侧注入动作
    fn execute_action(&mut self, current_player: usize, action: &Action) {
        use crate::state::ActionType;
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile) = action.tile {
                    self.board.discard_tile(tile, true);
                    self.game_history.push(GameEvent::DiscardTile {
                        player: current_player,
                        tile,
                        tsumogiri: true,
                    });
                    if self.process_passive_manually {
                        // 手动模式：只检测存起来，让 Python 决策
                        self.pending_passive = Some(self._detect_responses(current_player, tile));
                    } else {
                        // 默认：自动执行（兼容旧行为）
                        self._execute_detect_and_resolve(current_player, tile);
                    }
                }
            },
            ActionType::Kong => {
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
                                self.settle_win(*robber, crate::algo::winning::WinType::QiangGang,
                                    self.base_score, &[current_player]);
                            }
                        } else {
                            // 正常补杠 + 杠结算 + 杠后补摸
                            self.board.players[current_player].remove_tile(tile);
                            let payers = self.unwon_players_excluding(Some(current_player));
                            self.settle_kong(current_player, KongType::BuKan, self.base_score, &payers, None);
                            if let Some(nt) = self.board.draw_from_wall_end(1).first().copied() {
                                self.game_history.push(GameEvent::DrawTile {
                                    player: current_player, tile: nt,
                                });
                                self.board.kan_shang_active = true;
                            }
                        }
                    } else {
                        // 暗杠：手牌 4 张相同，直接杠 + 杠后补摸（暗杠不可抢）
                        self.board.players[current_player].hand.remove_n_tiles(tile, 4);
                        let payers = self.unwon_players_excluding(Some(current_player));
                        self.settle_kong(current_player, KongType::AnKan, self.base_score, &payers, None);
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
    }

    /// 检查其他人对弃牌的响应（胡 > 杠 > 碰），支持一炮多响
    /// 只检测被动响应机会（不执行），用于手动模式
    fn _detect_responses(&self, discarder: usize, tile: Tile) -> PendingPassive {
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

        PendingPassive {
            discarder,
            tile,
            tile_index: tile.to_index(),
            ron_winners,
            kong_responders,
            pong_responders,
        }
    }

    /// 自动模式：检测 + 直接执行（兼容旧行为）
    fn _execute_detect_and_resolve(&mut self, discarder: usize, tile: Tile) {
        let pending = self._detect_responses(discarder, tile);
        self._execute_all_passives(&pending);
    }

    /// 按优先级（胡 > 杠 > 碰）自动执行所有被动响应
    fn _execute_all_passives(&mut self, pending: &PendingPassive) {
        let tile = pending.tile;
        let discarder = pending.discarder;

        if !pending.ron_winners.is_empty() {
            // 一炮多响
            for winner in &pending.ron_winners {
                self.board.players[*winner].remove_tile(tile);
                self.settle_win(*winner, crate::algo::winning::WinType::Ron, self.base_score, &[discarder]);
            }
        } else if !pending.kong_responders.is_empty() {
            let responder = pending.kong_responders[0];
            self.board.players[responder].hand.remove_n_tiles(tile, 3);
            self.settle_kong(responder, KongType::MinKan, self.base_score, &[], Some(discarder));
            if let Some(new_tile) = self.board.draw_from_wall_end(1).first().copied() {
                self.game_history.push(GameEvent::DrawTile { player: responder, tile: new_tile });
                self.board.kan_shang_active = true;
            }
        } else if !pending.pong_responders.is_empty() {
            let responder = pending.pong_responders[0];
            self.board.players[responder].remove_tile(tile);
            self.board.add_meld(MeldType::Pong, tile, Some(discarder));
        }
    }

    /// Python 侧调用：执行某玩家的被动响应决策
    /// action_type: "win"/"kong"/"pong" 或 "pass"
    /// 返回 true 表示执行了动作，false 表示 pass 或不合法
    pub fn execute_passive_response(&mut self, responder: usize, action_type: &str) -> bool {
        let Some(pending) = &self.pending_passive else { return false; };
        let tile = pending.tile;
        let discarder = pending.discarder;

        match action_type {
            "win" => {
                if pending.ron_winners.contains(&responder) {
                    self.board.players[responder].remove_tile(tile);
                    self.settle_win(responder, crate::algo::winning::WinType::Ron, self.base_score, &[discarder]);
                    self.pending_passive = None;  // 胡后清掉（胡最高优先级）
                    return true;
                }
            },
            "kong" => {
                if pending.kong_responders.contains(&responder) {
                    self.board.players[responder].hand.remove_n_tiles(tile, 3);
                    self.settle_kong(responder, KongType::MinKan, self.base_score, &[], Some(discarder));
                    if let Some(new_tile) = self.board.draw_from_wall_end(1).first().copied() {
                        self.game_history.push(GameEvent::DrawTile { player: responder, tile: new_tile });
                        self.board.kan_shang_active = true;
                    }
                    // 清掉 kong/pong 候选（胡不会发生了）
                    let p = self.pending_passive.as_mut().unwrap();
                    p.kong_responders.clear();
                    p.pong_responders.clear();
                    return true;
                }
            },
            "pong" => {
                if pending.pong_responders.contains(&responder) {
                    self.board.players[responder].remove_tile(tile);
                    self.board.add_meld(MeldType::Pong, tile, Some(discarder));
                    // 清掉 pong 候选（胡/杠不会发生了）
                    let p = self.pending_passive.as_mut().unwrap();
                    p.ron_winners.clear();
                    p.kong_responders.clear();
                    p.pong_responders.clear();
                    return true;
                }
            },
            "pass" => {
                return false;
            },
            _ => {}
        }
        false
    }

    /// Python 侧调用：跳过所有剩余被动响应，让回合继续
    pub fn skip_all_passive(&mut self) {
        self.pending_passive = None;
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
    /// 流局结算：查花猪 + 查大叫（严格分离，保证零和）
    /// - 查花猪：花猪之间互相赔自己的 fan（全额 = fan × base），大叫和听牌者不参与
    /// - 查大叫：未听者向已听者赔（每人赔 1 底，听牌者间瓜分）
    fn resolve_draw(&mut self) {
        let non_winners: Vec<usize> = self.board.players.iter()
            .enumerate()
            .filter(|(_, p)| !p.has_won)
            .map(|(i, _)| i)
            .collect();

        if non_winners.is_empty() { return; }

        let mut huazhu_players = Vec::new();
        let mut dajiao_players = Vec::new();
        let mut tenpai_players = Vec::new();

        for &p in &non_winners {
            let ps = &self.board.players[p];
            // 花猪 = 手牌中还留着缺门牌没清理干净
            let is_huazhu = ps.hand.missing_suit
                .map(|s| ps.hand.count_by_suit(s) > 0)
                .unwrap_or(false);
            if is_huazhu {
                huazhu_players.push(p);
                self.board.set_player_huazhu(p);
                continue;
            }
            let is_tenpai = ps.hand.is_tenpai() || ps.hand.is_waiting();
            if is_tenpai {
                tenpai_players.push(p);
            } else {
                dajiao_players.push(p);
                self.board.set_player_dajiao(p);
            }
        }

        for &huazhu in &huazhu_players {
            let melds_view = convert_melds_to_view(&self.board.players[huazhu].melds);
            let fan_result = self.ruleset.fan(
                &self.board.players[huazhu].hand,
                &melds_view,
                crate::algo::winning::WinType::Tsumo,
            );
            let per_payer = (fan_result.fan as i32) * (self.base_score as i32);
            for &receiver in &huazhu_players {
                if receiver == huazhu { continue; }
                self.board.players[huazhu].score -= per_payer;
                self.board.players[receiver].score += per_payer;
            }
        }

        let tenpai_count = tenpai_players.len();
        if !dajiao_players.is_empty() && tenpai_count > 0 {
            let per_dajiao = self.base_score as i32;
            let per_tenpai_total = per_dajiao * dajiao_players.len() as i32;
            let per_tenpai = per_tenpai_total / tenpai_count as i32;
            let remainder = per_tenpai_total % tenpai_count as i32;

            for &dajiao in &dajiao_players {
                self.board.players[dajiao].score -= per_dajiao;
            }
            for (idx, &tp) in tenpai_players.iter().enumerate() {
                let extra = if idx == 0 { remainder } else { 0 };
                self.board.players[tp].score += per_tenpai + extra;
            }
        }
    }

    fn ai_select_action(&self, player_id: usize) -> Action {
        let ps = &self.board.players[player_id];

        // 优先：用 ActionValidator 生成全部合法动作，过滤掉被动响应（Pong/Pass/Win/QiangGang），只取本家主动动作
        let candidate = ActionValidator::generate_legal_actions(ps, self.current_turn);
        let active_actions: Vec<&Action> = candidate.actions.iter()
            .filter(|a| matches!(a.action_type,
                crate::state::ActionType::Discard
              | crate::state::ActionType::Kong))
            .collect();

        if let Some(a) = active_actions.into_iter().max_by_key(|a| a.priority) {
            return a.clone();
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

    /// Step-level control: draw → kan_shang/tsumo check → execute Python-supplied action → finalize → next_turn.
    /// Returns false if preconditions fail or game ended.
    pub fn step_with_action(&mut self, player_id: usize, action: Action) -> bool {
        if self.phase != GamePhase::Playing || self.board.is_game_over() {
            return false;
        }
        if self.board.current_player != player_id {
            return false;
        }
        if self.board.players[player_id].has_won {
            self.board.next_turn();
            self.current_turn += 1;
            return true;
        }

        // 1. 摸牌
        let current_player = player_id;
        let drawn_tile = match self.board.draw_tile() {
            Some(t) => t,
            None => {
                self.resolve_draw();
                return false;
            }
        };

        self.game_history.push(GameEvent::DrawTile {
            player: current_player,
            tile: drawn_tile,
        });

        // 2. 杠上花检测
        if self.board.kan_shang_active {
            let can_win_kanshang = WinChecker::can_win(
                &self.board.players[current_player].hand,
                crate::algo::winning::WinType::KanShang,
            );
            if can_win_kanshang {
                let payers = self.unwon_players_excluding(Some(current_player));
                self.settle_win(current_player, crate::algo::winning::WinType::KanShang, self.base_score, &payers);
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
            let payers = self.unwon_players_excluding(Some(current_player));
            self.settle_win(current_player, crate::algo::winning::WinType::Tsumo, self.base_score, &payers);
            self.check_and_finalize();
            return !self.board.is_game_over();
        }

        // 4. 执行 Python 提供的动作
        self.execute_action(current_player, &action);

        self.check_and_finalize();
        if self.board.is_game_over() { return false; }

        // 手动被动响应模式：弃牌后暂停，等 Python 决策完被动响应再推进
        if self.process_passive_manually && self.pending_passive.is_some() {
            return true;
        }

        self.board.next_turn();
        self.current_turn += 1;
        true
    }

    /// 获取指定玩家当前的合法主动动作（Discard/Kong + 自摸/杠上花布尔）
    pub fn get_legal_actions(&self, player_id: usize) -> LegalActions {
        let ps = &self.board.players[player_id];
        let candidate = ActionValidator::generate_legal_actions(ps, self.current_turn);

        let mut discard_tiles: Vec<usize> = Vec::new();
        let mut kong_tiles: Vec<usize> = Vec::new();

        for a in &candidate.actions {
            match a.action_type {
                crate::state::ActionType::Discard => {
                    if let Some(t) = a.tile { discard_tiles.push(t.to_index()); }
                },
                crate::state::ActionType::Kong => {
                    if let Some(t) = a.tile { kong_tiles.push(t.to_index()); }
                },
                _ => {}
            }
        }

        let can_tsumo = WinChecker::can_win(
            &self.board.players[player_id].hand,
            crate::algo::winning::WinType::Tsumo,
        );
        let can_kan_shang = self.board.kan_shang_active && WinChecker::can_win(
            &self.board.players[player_id].hand,
            crate::algo::winning::WinType::KanShang,
        );

        LegalActions {
            discard: discard_tiles,
            kong: kong_tiles,
            can_tsumo,
            can_kan_shang,
        }
    }

    pub fn run_to_end(&mut self) {
        self.start_game();
        self.process_swap_phase();
        self.process_missing_phase();
        while self.process_turn() {}
    }
}

#[pymethods]
impl Game {
    #[new]
    fn py_new(num_players: usize) -> Self {
        Self::new(num_players)
    }

    #[pyo3(name = "run_to_end")]
    fn py_run_to_end(&mut self) {
        self.start_game();
        self.process_swap_phase();
        self.process_missing_phase();
        while self.process_turn() {}
    }

    #[getter]
    fn board(&self) -> Board {
        self.board.clone()
    }

    #[pyo3(name = "start_game")]
    fn py_start_game(&mut self) -> PyResult<()> {
        self.start_game();
        self.process_swap_phase();
        self.process_missing_phase();
        Ok(())
    }

    #[pyo3(name = "phase")]
    fn py_phase(&self) -> String {
        match self.phase {
            GamePhase::Deal => "Deal".into(),
            GamePhase::Swap => "Swap".into(),
            GamePhase::Missing => "Missing".into(),
            GamePhase::Playing => "Playing".into(),
            GamePhase::GameOver => "GameOver".into(),
        }
    }

    #[pyo3(name = "is_over")]
    fn py_is_over(&self) -> bool {
        self.board.is_game_over()
    }

    #[pyo3(name = "current_player")]
    fn py_current_player(&self) -> usize {
        self.board.current_player
    }

    #[pyo3(name = "step_with_action")]
    fn py_step_with_action(
        &mut self,
        player_id: usize,
        action_type: &str,
        tile_index: u8,
    ) -> PyResult<bool> {
        let tile = Tile::from_index(tile_index as usize);
        let action = match action_type {
            "discard" => Action::discard(tile.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tile_index: {}", tile_index))
            })?, self.current_turn),
            "kong" => Action::kong(tile.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tile_index: {}", tile_index))
            })?, None, self.current_turn),
            "pass" => Action::pass(self.current_turn),
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown action_type: {} (expected 'discard'/'kong'/'pass')", other
            ))),
        };
        Ok(self.step_with_action(player_id, action))
    }

    #[pyo3(name = "get_legal_actions")]
    fn py_get_legal_actions(&self, player_id: usize) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let la = self.get_legal_actions(player_id);
        let phase_str = match self.phase {
            GamePhase::Swap => "Swap",
            GamePhase::Missing => "Missing",
            GamePhase::Playing => "Playing",
            _ => "GameOver",
        };
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("phase", phase_str)?;
            dict.set_item("discard", la.discard)?;
            dict.set_item("kong", la.kong)?;
            dict.set_item("can_tsumo", la.can_tsumo)?;
            dict.set_item("can_kan_shang", la.can_kan_shang)?;

            // 被动响应（仅 Playing phase 且有最近弃牌时才有意义）
            let (passive_discarder, passive_tile_idx) = self._last_discard_info();
            let can_passive = if self.phase == GamePhase::Playing && passive_discarder.is_some()
                && passive_discarder.unwrap() != player_id
                && !self.board.players[player_id].has_won
            {
                if let Some(tile) = Tile::from_index(passive_tile_idx) {
                    let ps = &self.board.players[player_id];
                    let can_win = WinChecker::can_win(&ps.hand, crate::algo::winning::WinType::Ron);
                    let can_kong = ps.can_kong(tile);
                    let can_pong = ps.can_pong(tile);
                    (can_win, can_kong, can_pong)
                } else {
                    (false, false, false)
                }
            } else {
                (false, false, false)
            };
            dict.set_item("can_win", can_passive.0)?;
            dict.set_item("can_kong", can_passive.1)?;
            dict.set_item("can_pong", can_passive.2)?;

            Ok(dict.into())
        })
    }

    // ── Phase 分阶段控制（替代自动 start_game → Playing） ──

    #[pyo3(name = "start_game_bare")]
    fn py_start_game_bare(&mut self) -> PyResult<()> {
        self.start_game();  // phase = Swap，不自动推进
        Ok(())
    }

    #[pyo3(name = "process_swap_phase")]
    fn py_process_swap_phase(&mut self) -> PyResult<()> {
        self.process_swap_phase();
        Ok(())
    }

    #[pyo3(name = "process_missing_phase")]
    fn py_process_missing_phase(&mut self) -> PyResult<()> {
        self.process_missing_phase();
        Ok(())
    }

    /// Python 查询某玩家手牌 (tile indices, 0-26)
    #[pyo3(name = "player_hand_tiles")]
    fn py_player_hand_tiles(&self, player_id: usize) -> PyResult<Vec<usize>> {
        if player_id >= self.board.players.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("player_id out of range"));
        }
        // Hand.tiles 是 [u8; 27] 计数数组 → 展开成 tile index 列表
        let counts = self.board.players[player_id].hand.get_tile_counts();
        let mut tiles = Vec::with_capacity(counts.iter().sum::<u8>() as usize);
        for (idx, &cnt) in counts.iter().enumerate() {
            for _ in 0..cnt {
                tiles.push(idx);
            }
        }
        Ok(tiles)
    }

    /// Python 查询某玩家 shanten number（川麻: 已排除缺门牌后计算；无缺门=完整手牌）
    #[pyo3(name = "player_shanten")]
    fn py_player_shanten(&self, player_id: usize) -> PyResult<i32> {
        if player_id >= self.board.players.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("player_id out of range"));
        }
        Ok(self.board.players[player_id].hand.shanten())
    }

    /// Python 查询某玩家缺门花色 (0=Man, 1=Pin, 2=Sou; -1 表示未定缺)
    #[pyo3(name = "player_missing_suit")]
    fn py_player_missing_suit(&self, player_id: usize) -> PyResult<i32> {
        if player_id >= self.board.players.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("player_id out of range"));
        }
        Ok(match self.board.players[player_id].hand.missing_suit {
            Some(s) => s.index() as i32,
            None => -1,
        })
    }

    /// Python 版换三张：传入每玩家的换牌 indices (可选 None 表示用默认 AI)
    /// selections: List[Optional[List[int]]]，每个元素是该玩家换出的 3 张 tile index
    #[pyo3(name = "process_swap_phase_with")]
    fn py_process_swap_phase_with(
        &mut self,
        selections: Vec<Option<Vec<usize>>>,
    ) -> PyResult<()> {
        self.process_swap_phase_with_selections(Some(&selections));
        Ok(())
    }

    /// Python 版定缺：传入每玩家的缺门 (0=Man,1=Pin,2=Sou; None 用默认 AI)
    #[pyo3(name = "process_missing_phase_with")]
    fn py_process_missing_phase_with(
        &mut self,
        selections: Vec<Option<usize>>,
    ) -> PyResult<()> {
        self.process_missing_phase_with_selections(Some(&selections));
        Ok(())
    }

    // ── 被动响应检测 ──

    #[pyo3(name = "last_discard_info")]
    fn py_last_discard_info(&self) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let (discarder, tile_idx) = self._last_discard_info();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("discarder", discarder)?;
            dict.set_item("tile_index", tile_idx)?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "check_passive")]
    fn py_check_passive(&self, player_id: usize) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let (discarder_opt, tile_idx) = self._last_discard_info();
        let discarder = match discarder_opt {
            Some(d) => d,
            None => {
                return Python::with_gil(|py| Ok(PyDict::new(py).into()));
            }
        };
        if player_id == discarder || self.board.players[player_id].has_won {
            return Python::with_gil(|py| Ok(PyDict::new(py).into()));
        }
        let tile = match Tile::from_index(tile_idx) {
            Some(t) => t,
            None => {
                return Python::with_gil(|py| Ok(PyDict::new(py).into()));
            }
        };
        let ps = &self.board.players[player_id];
        let can_win = WinChecker::can_win(&ps.hand, crate::algo::winning::WinType::Ron);
        let can_kong = ps.can_kong(tile);
        let can_pong = ps.can_pong(tile);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("can_win", can_win)?;
            dict.set_item("can_kong", can_kong)?;
            dict.set_item("can_pong", can_pong)?;
            dict.set_item("discarder", discarder)?;
            dict.set_item("tile_index", tile_idx)?;
            Ok(dict.into())
        })
    }

    // ── 手动被动响应控制 ──

    #[pyo3(name = "set_passive_manual")]
    fn py_set_passive_manual(&mut self, enabled: bool) {
        self.process_passive_manually = enabled;
    }

    #[pyo3(name = "get_pending_passive")]
    fn py_get_pending_passive(&self) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            match &self.pending_passive {
                Some(p) => {
                    dict.set_item("discarder", p.discarder)?;
                    dict.set_item("tile_index", p.tile_index)?;
                    dict.set_item("ron_winners", &p.ron_winners)?;
                    dict.set_item("kong_responders", &p.kong_responders)?;
                    dict.set_item("pong_responders", &p.pong_responders)?;
                },
                None => {
                    dict.set_item("discarder", pyo3::ToPyObject::to_object(&py.None(), py))?;
                    dict.set_item("tile_index", 0usize)?;
                    dict.set_item("ron_winners", Vec::<usize>::new())?;
                    dict.set_item("kong_responders", Vec::<usize>::new())?;
                    dict.set_item("pong_responders", Vec::<usize>::new())?;
                }
            }
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute_passive_response")]
    fn py_execute_passive_response(&mut self, responder: usize, action_type: &str) -> bool {
        self.execute_passive_response(responder, action_type)
    }

    #[pyo3(name = "skip_all_passive")]
    fn py_skip_all_passive(&mut self) {
        self.skip_all_passive();
    }

    #[pyo3(name = "advance_turn")]
    fn py_advance_turn(&mut self) -> bool {
        if self.board.is_game_over() { return false; }
        self.board.next_turn();
        self.current_turn += 1;
        true
    }
}

impl Game {
    /// 返回最后一次弃牌的 (discarder_id, tile_index)，无则返回 (None, 0)
    /// 从后往前扫 game_history，跳过 Pong/Kong/Ron/GameOver 等事件
    fn _last_discard_info(&self) -> (Option<usize>, usize) {
        for evt in self.game_history.iter().rev() {
            if let GameEvent::DiscardTile { player, tile, .. } = evt {
                return (Some(*player), tile.to_index());
            }
        }
        (None, 0)
    }
}

#[derive(Debug, Clone)]
pub struct LegalActions {
    pub discard: Vec<usize>,
    pub kong: Vec<usize>,
    pub can_tsumo: bool,
    pub can_kan_shang: bool,
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











