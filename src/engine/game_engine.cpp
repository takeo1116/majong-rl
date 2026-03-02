#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>

namespace mahjong {

// ============================
// フェーズ遷移の検証（CQ-0006）
// ============================

bool GameEngine::is_valid_transition(Phase from, Phase to) {
    switch (from) {
        case Phase::StartMatch:
            return to == Phase::StartRound;
        case Phase::StartRound:
            return to == Phase::DrawPhase || to == Phase::SelfActionPhase;
        case Phase::DrawPhase:
            return to == Phase::SelfActionPhase || to == Phase::ResolveDrawPhase;
        case Phase::SelfActionPhase:
            return to == Phase::ResponsePhase || to == Phase::DrawPhase
                || to == Phase::ResolveWinPhase || to == Phase::ResolveDrawPhase
                || to == Phase::EndRound;
        case Phase::ResponsePhase:
            return to == Phase::ResolveResponsePhase;
        case Phase::ResolveResponsePhase:
            return to == Phase::DrawPhase || to == Phase::SelfActionPhase
                || to == Phase::ResolveWinPhase || to == Phase::EndRound;
        case Phase::ResolveWinPhase:
            return to == Phase::EndRound;
        case Phase::ResolveDrawPhase:
            return to == Phase::EndRound;
        case Phase::EndRound:
            return to == Phase::StartRound || to == Phase::EndMatch;
        case Phase::EndMatch:
            return false;
    }
    return false;
}

std::vector<ActionType> GameEngine::allowed_action_types(Phase phase) {
    switch (phase) {
        case Phase::SelfActionPhase:
            return {ActionType::Discard, ActionType::TsumoWin, ActionType::Ankan,
                    ActionType::Kakan, ActionType::Kyuushu};
        case Phase::ResponsePhase:
            return {ActionType::Ron, ActionType::Pon, ActionType::Daiminkan,
                    ActionType::Chi, ActionType::Skip};
        default:
            return {};
    }
}

// ============================
// リセット（CQ-0007, CQ-0008）
// ============================

void GameEngine::reset_match(EnvironmentState& env, uint64_t seed, RunMode mode) {
    env.reset(seed, mode);
    init_round(env);
}

void GameEngine::reset_match(EnvironmentState& env, uint64_t seed, PlayerId first_dealer, RunMode mode) {
    env.reset(seed, first_dealer, mode);
    init_round(env);
}

void GameEngine::init_round(EnvironmentState& env) {
    auto& rs = env.round_state;
    auto& ms = env.match_state;

    rs.reset(ms.round_number, ms.current_dealer, ms.honba, ms.kyotaku);

    // 各プレイヤーの得点を MatchState から同期
    for (PlayerId i = 0; i < kNumPlayers; ++i) {
        rs.players[i].score = ms.scores[i];
    }

    generate_wall(env);
    deal_tiles(env);
    setup_dora(env);

    // 親は14枚を持って SelfActionPhase に入る
    rs.phase = Phase::SelfActionPhase;
    rs.current_player = rs.dealer;
}

void GameEngine::generate_wall(EnvironmentState& env) {
    auto& rs = env.round_state;
    // 0-135 の牌ID配列を作り、シャッフルする
    std::iota(rs.wall.begin(), rs.wall.end(), 0);
    env.rng.shuffle(rs.wall.begin(), rs.wall.end());
}

void GameEngine::deal_tiles(EnvironmentState& env) {
    auto& rs = env.round_state;

    // 各プレイヤーに13枚ずつ配る
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        // 親から反時計回りに配る
        PlayerId player = (rs.dealer + p) % kNumPlayers;
        rs.players[player].hand.reserve(14);
        for (int i = 0; i < 13; ++i) {
            rs.players[player].hand.push_back(rs.wall[rs.wall_position++]);
        }
    }

    // 親に14枚目（最初のツモ）を配る
    rs.players[rs.dealer].hand.push_back(rs.wall[rs.wall_position++]);
}

void GameEngine::setup_dora(EnvironmentState& env) {
    auto& rs = env.round_state;
    // 最初のドラ表示牌: wall[kNumTiles - 6] = wall[130]
    TileId indicator = rs.wall[kNumTiles - 6];
    rs.dora_indicators.push_back(indicator);
    // 裏ドラ表示牌: wall[kNumTiles - 5] = wall[131]
    TileId uradora = rs.wall[kNumTiles - 5];
    rs.uradora_indicators.push_back(uradora);
}

// ============================
// 1局モードリセット（CQ-0028）
// ============================

ErrorCode GameEngine::reset_round(EnvironmentState& env, const RoundConfig& config, RunMode mode) {
    // --- 入力検証 ---

    // 親が有効範囲か
    if (config.dealer >= kNumPlayers) {
        return ErrorCode::InvalidActor;
    }

    // 局番号が有効範囲か（0=東1局 .. 8=延長局）
    if (config.round_number > 8) {
        return ErrorCode::InconsistentState;
    }

    // 手牌枚数の検証: 親14枚、子13枚
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        size_t expected = (p == config.dealer) ? 14 : 13;
        if (config.hands[p].size() != expected) {
            return ErrorCode::InconsistentState;
        }
    }

    // 山の牌IDが有効範囲かつ重複なし
    {
        std::array<bool, kNumTiles> seen{};
        for (int i = 0; i < kNumTiles; ++i) {
            TileId t = config.wall[i];
            if (t >= kNumTiles) {
                return ErrorCode::InvalidTile;
            }
            if (seen[t]) {
                return ErrorCode::InconsistentState;  // 山内で重複
            }
            seen[t] = true;
        }
    }

    // 手牌の牌IDが有効範囲かつ重複なし、かつ山に含まれていること
    {
        std::array<bool, kNumTiles> hand_seen{};
        std::array<bool, kNumTiles> wall_set{};
        for (int i = 0; i < kNumTiles; ++i) {
            wall_set[config.wall[i]] = true;
        }

        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            for (TileId t : config.hands[p]) {
                if (t >= kNumTiles) {
                    return ErrorCode::InvalidTile;
                }
                if (hand_seen[t]) {
                    return ErrorCode::InconsistentState;  // 手牌間で重複
                }
                if (!wall_set[t]) {
                    return ErrorCode::InconsistentState;  // 山にない牌
                }
                hand_seen[t] = true;
            }
        }
    }

    // --- 状態セットアップ ---

    env.run_mode = mode;
    env.logging_enabled = (mode == RunMode::Debug);

    // MatchState を設定
    auto& ms = env.match_state;
    ms.round_number = config.round_number;
    ms.current_dealer = config.dealer;
    ms.first_dealer = config.dealer;  // 1局モードでは起家＝親
    ms.honba = config.honba;
    ms.kyotaku = config.kyotaku;
    ms.scores = config.scores;
    ms.is_extra_round = (config.round_number >= 8);
    ms.is_match_over = false;

    // RoundState を初期化
    auto& rs = env.round_state;
    rs.reset(config.round_number, config.dealer, config.honba, config.kyotaku);

    // 山をコピー
    rs.wall = config.wall;

    // 配牌をコピー
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        rs.players[p].hand = config.hands[p];
        rs.players[p].score = config.scores[p];
    }

    // wall_position を配牌済みの位置に設定（4人×13 + 親の1枚 = 53）
    rs.wall_position = 53;

    // ドラ設定（山のレイアウトに基づく）
    setup_dora(env);

    // 親は14枚を持って SelfActionPhase に入る
    rs.phase = Phase::SelfActionPhase;
    rs.current_player = config.dealer;

    return ErrorCode::Ok;
}

// ============================
// step()（CQ-0007, CQ-0029）
// ============================

StepResult GameEngine::step(EnvironmentState& env, const Action& action) {
    auto& rs = env.round_state;

    // 局が既に終了している場合
    if (rs.is_round_over()) {
        StepResult result;
        result.error = ErrorCode::WrongPhase;
        return result;
    }

    // フェーズごとの許可アクション種別を検証
    auto allowed = allowed_action_types(rs.phase);
    if (std::find(allowed.begin(), allowed.end(), action.type) == allowed.end()) {
        StepResult result;
        result.error = ErrorCode::WrongPhase;
        return result;
    }

    // フェーズに応じたアクション処理
    switch (rs.phase) {
        case Phase::SelfActionPhase:
            return process_self_action(env, action);
        case Phase::ResponsePhase:
            return process_response(env, action);
        default: {
            StepResult result;
            result.error = ErrorCode::WrongPhase;
            return result;
        }
    }
}

// ============================
// SelfActionPhase の処理
// ============================

StepResult GameEngine::process_self_action(EnvironmentState& env, const Action& action) {
    auto& rs = env.round_state;
    StepResult result;

    // アクター検証
    if (action.actor != rs.current_player) {
        result.error = ErrorCode::InvalidActor;
        return result;
    }

    auto& player = rs.players[action.actor];

    switch (action.type) {
        case ActionType::Discard: {
            // 手牌から対象牌を除去
            auto it = std::find(player.hand.begin(), player.hand.end(), action.tile);
            if (it == player.hand.end()) {
                result.error = ErrorCode::InvalidTile;
                return result;
            }

            // 喰い替えチェック
            if (rs.just_called && (action.tile / 4) == rs.last_call_tile_type) {
                result.error = ErrorCode::IllegalAction;
                return result;
            }

            player.hand.erase(it);

            // 一発の失効: 立直宣言以外の打牌で自身の一発が消える
            // （立直宣言打牌の場合は直後に ippatsu = true で上書きされる）
            if (!action.riichi) {
                player.ippatsu = false;
            }

            // 立直処理
            if (action.riichi) {
                player.is_riichi = true;
                player.ippatsu = true;
                player.score -= 1000;
                rs.kyotaku++;
                result.events.push_back(Event::make_discard(action.actor, action.tile, true));
            } else {
                result.events.push_back(Event::make_discard(action.actor, action.tile, false));
            }

            // 河に追加
            DiscardInfo di;
            di.tile = action.tile;
            di.riichi_discard = action.riichi;
            di.called = false;
            player.discards.push_back(di);

            // 直前打牌を記録
            rs.last_discard = action.tile;
            rs.last_discarder = action.actor;

            // 喰い替えフラグをリセット
            rs.just_called = false;
            rs.last_call_tile_type = 255;

            // 第一ツモ巡を終了
            rs.first_draw[action.actor] = false;

            // 槓ドラ公開予約がある場合、この打牌で公開
            if (rs.pending_kan_dora) {
                int dora_count = static_cast<int>(rs.dora_indicators.size());
                // 次のドラ表示位置: wall[kNumTiles - 6 - 2*dora_count]
                int idx = kNumTiles - 6 - 2 * dora_count;
                if (idx >= kNumTiles - 14) {  // 王牌範囲内であることを確認
                    TileId new_indicator = rs.wall[idx];
                    rs.dora_indicators.push_back(new_indicator);
                    TileId new_uradora = rs.wall[idx + 1];
                    rs.uradora_indicators.push_back(new_uradora);
                    result.events.push_back(Event::make_dora_reveal(new_indicator));
                }
                rs.pending_kan_dora = false;
            }

            // 応答チェック
            if (has_any_response(env, action.actor, action.tile)) {
                setup_response_phase(env, action.actor, action.tile);
            } else {
                // 応答なし → 次のプレイヤーのツモ
                advance_to_next_draw(env, result);
            }
            break;
        }
        case ActionType::TsumoWin: {
            // ツモ和了
            rs.end_reason = RoundEndReason::Tsumo;
            rs.phase = Phase::EndRound;
            result.round_over = true;
            result.events.push_back(Event::make_tsumo(action.actor));
            result.events.push_back(Event::make_round_end(RoundEndReason::Tsumo));
            break;
        }
        case ActionType::Ankan: {
            // 暗槓: action.tile には TileType が格納されている
            TileType kan_type = action.tile;
            if (rs.total_kan_count >= 4) {
                result.error = ErrorCode::IllegalAction;
                return result;
            }

            // 手牌から4枚を探す
            std::vector<TileId> kan_tiles;
            for (TileId id : player.hand) {
                if (id / 4 == kan_type) {
                    kan_tiles.push_back(id);
                }
            }
            if (kan_tiles.size() < 4) {
                result.error = ErrorCode::IllegalAction;
                return result;
            }

            // 手牌から4枚除去
            for (TileId id : kan_tiles) {
                auto it = std::find(player.hand.begin(), player.hand.end(), id);
                player.hand.erase(it);
            }

            // 暗槓メルドを追加
            Meld m = Meld::make_ankan(kan_tiles[0], kan_tiles[1], kan_tiles[2], kan_tiles[3], action.actor);
            player.melds.push_back(m);
            rs.total_kan_count++;

            // 暗槓は即ドラ公開
            int dora_count = static_cast<int>(rs.dora_indicators.size());
            int idx = kNumTiles - 6 - 2 * dora_count;
            if (idx >= kNumTiles - 14) {
                TileId new_indicator = rs.wall[idx];
                rs.dora_indicators.push_back(new_indicator);
                TileId new_uradora = rs.wall[idx + 1];
                rs.uradora_indicators.push_back(new_uradora);
                result.events.push_back(Event::make_dora_reveal(new_indicator));
            }

            result.events.push_back(Event::make_kan(action.actor, MeldType::Ankan));

            // 全員の第一ツモ巡を終了
            rs.first_draw.fill(false);

            // 嶺上ツモ
            draw_rinshan(env, action.actor, result);
            break;
        }
        case ActionType::Kakan: {
            // 加槓: action.tile には加える TileId が格納
            if (rs.total_kan_count >= 4) {
                result.error = ErrorCode::IllegalAction;
                return result;
            }

            TileId added = action.tile;
            TileType added_type = added / 4;

            // 対応するポンを見つける
            int pon_idx = -1;
            for (int i = 0; i < static_cast<int>(player.melds.size()); ++i) {
                if (player.melds[i].type == MeldType::Pon && player.melds[i].base_type() == added_type) {
                    pon_idx = i;
                    break;
                }
            }
            if (pon_idx < 0) {
                result.error = ErrorCode::IllegalAction;
                return result;
            }

            // 手牌から1枚除去
            auto it = std::find(player.hand.begin(), player.hand.end(), added);
            if (it == player.hand.end()) {
                result.error = ErrorCode::InvalidTile;
                return result;
            }
            player.hand.erase(it);

            // ポン → 加槓に変換
            player.melds[pon_idx] = Meld::make_kakan(player.melds[pon_idx], added);
            rs.total_kan_count++;

            // 大明槓/加槓は次巡捨牌時にドラ公開
            rs.pending_kan_dora = true;

            result.events.push_back(Event::make_kan(action.actor, MeldType::Kakan));

            // 全員の第一ツモ巡を終了
            rs.first_draw.fill(false);

            // TODO: 槍槓チェック（CQ-0016 で実装）

            // 嶺上ツモ
            draw_rinshan(env, action.actor, result);
            break;
        }
        case ActionType::Kyuushu: {
            // 九種九牌
            rs.end_reason = RoundEndReason::AbortiveKyuushu;
            rs.phase = Phase::EndRound;
            result.round_over = true;
            result.events.push_back(Event::make_abortive_draw());
            result.events.push_back(Event::make_round_end(RoundEndReason::AbortiveKyuushu));
            break;
        }
        default:
            result.error = ErrorCode::WrongPhase;
            return result;
    }

    return result;
}

// ============================
// ResponsePhase の処理（CQ-0010）
// ============================

StepResult GameEngine::process_response(EnvironmentState& env, const Action& action) {
    auto& rs = env.round_state;
    auto& ctx = rs.response_context;
    StepResult result;

    if (!ctx.active) {
        result.error = ErrorCode::WrongPhase;
        return result;
    }

    // アクター検証
    if (action.actor != rs.current_player) {
        result.error = ErrorCode::InvalidActor;
        return result;
    }

    // 応答を記録
    ctx.responses[action.actor] = action;
    ctx.has_responded[action.actor] = true;

    // 次の応答者を探す
    PlayerId next = find_next_responder(env);
    if (next != 255) {
        // まだ応答待ちのプレイヤーがいる
        rs.current_player = next;
    } else {
        // 全員応答済み → 解決
        rs.phase = Phase::ResolveResponsePhase;
        resolve_responses(env, result);
    }

    return result;
}

void GameEngine::resolve_responses(EnvironmentState& env, StepResult& result) {
    auto& rs = env.round_state;
    auto& ctx = rs.response_context;

    // ロン判定（最優先、複数可）
    std::vector<PlayerId> ron_players;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (ctx.has_responded[p] && ctx.responses[p].type == ActionType::Ron) {
            ron_players.push_back(p);
        }
    }

    if (!ron_players.empty()) {
        // ロン成立
        for (PlayerId winner : ron_players) {
            result.events.push_back(Event::make_ron(winner, ctx.discarder));
        }
        // 河の牌を「鳴かれた」に
        auto& discarder = rs.players[ctx.discarder];
        if (!discarder.discards.empty()) {
            discarder.discards.back().called = true;
        }
        rs.end_reason = RoundEndReason::Ron;
        rs.phase = Phase::EndRound;
        result.round_over = true;
        result.events.push_back(Event::make_round_end(RoundEndReason::Ron));
        ctx.active = false;
        return;
    }

    // ポン / 大明槓判定（放銃者から反時計回りの近い順）
    PlayerId pon_player = 255;
    PlayerId daiminkan_player = 255;
    for (int offset = 1; offset <= 3; ++offset) {
        PlayerId p = (ctx.discarder + offset) % kNumPlayers;
        if (!ctx.has_responded[p]) continue;
        if (ctx.responses[p].type == ActionType::Pon && pon_player == 255) {
            pon_player = p;
        }
        if (ctx.responses[p].type == ActionType::Daiminkan && daiminkan_player == 255) {
            daiminkan_player = p;
        }
    }

    // ポン処理
    if (pon_player != 255) {
        auto& resp = ctx.responses[pon_player];
        auto& caller = rs.players[pon_player];

        // 手牌から2枚除去
        for (int i = 0; i < 2; ++i) {
            TileId consumed = resp.consumed_tiles[i];
            auto it = std::find(caller.hand.begin(), caller.hand.end(), consumed);
            if (it != caller.hand.end()) {
                caller.hand.erase(it);
            }
        }

        // メルド追加
        Meld m = Meld::make_pon(ctx.discard_tile, resp.consumed_tiles[0], resp.consumed_tiles[1], ctx.discarder);
        caller.melds.push_back(m);
        caller.is_menzen = false;

        // 河の牌を「鳴かれた」に
        auto& discarder = rs.players[ctx.discarder];
        if (!discarder.discards.empty()) {
            discarder.discards.back().called = true;
        }

        // 全員の一発を消す
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            rs.players[p].ippatsu = false;
        }

        // 全員の第一ツモ巡を終了
        rs.first_draw.fill(false);

        // 同巡内フリテンをリセット（ポンした人以外の全員）
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            if (p != pon_player) {
                rs.players[p].is_temporary_furiten = false;
            }
        }

        // 喰い替え追跡
        rs.just_called = true;
        rs.last_call_tile_type = ctx.discard_tile / 4;

        result.events.push_back(Event::make_pon(pon_player, ctx.discard_tile, ctx.discarder));

        rs.phase = Phase::SelfActionPhase;
        rs.current_player = pon_player;
        ctx.active = false;
        return;
    }

    // 大明槓処理
    if (daiminkan_player != 255) {
        auto& caller = rs.players[daiminkan_player];
        TileType kan_type = ctx.discard_tile / 4;

        if (rs.total_kan_count >= 4) {
            // 槓不可 → スキップ扱い
        } else {
            // 手牌から同種3枚を除去
            std::vector<TileId> consumed;
            for (auto it = caller.hand.begin(); it != caller.hand.end(); ) {
                if (*it / 4 == kan_type && consumed.size() < 3) {
                    consumed.push_back(*it);
                    it = caller.hand.erase(it);
                } else {
                    ++it;
                }
            }

            if (consumed.size() == 3) {
                Meld m = Meld::make_daiminkan(ctx.discard_tile, consumed[0], consumed[1], consumed[2], ctx.discarder);
                caller.melds.push_back(m);
                caller.is_menzen = false;
                rs.total_kan_count++;

                // 河の牌を「鳴かれた」に
                auto& discarder = rs.players[ctx.discarder];
                if (!discarder.discards.empty()) {
                    discarder.discards.back().called = true;
                }

                // 全員の一発を消す
                for (PlayerId p = 0; p < kNumPlayers; ++p) {
                    rs.players[p].ippatsu = false;
                }

                // 全員の第一ツモ巡を終了
                rs.first_draw.fill(false);

                // 大明槓は次巡捨牌時にドラ公開
                rs.pending_kan_dora = true;

                result.events.push_back(Event::make_kan(daiminkan_player, MeldType::Daiminkan));

                // 嶺上ツモ
                draw_rinshan(env, daiminkan_player, result);
                ctx.active = false;
                return;
            }
        }
    }

    // チー判定（下家のみ）
    PlayerId shimocha = (ctx.discarder + 1) % kNumPlayers;
    if (ctx.has_responded[shimocha] && ctx.responses[shimocha].type == ActionType::Chi) {
        auto& resp = ctx.responses[shimocha];
        auto& caller = rs.players[shimocha];

        // 手牌から2枚除去
        for (int i = 0; i < 2; ++i) {
            TileId consumed = resp.consumed_tiles[i];
            auto it = std::find(caller.hand.begin(), caller.hand.end(), consumed);
            if (it != caller.hand.end()) {
                caller.hand.erase(it);
            }
        }

        // メルド追加
        Meld m = Meld::make_chi(ctx.discard_tile, resp.consumed_tiles[0], resp.consumed_tiles[1], ctx.discarder);
        caller.melds.push_back(m);
        caller.is_menzen = false;

        // 河の牌を「鳴かれた」に
        auto& discarder = rs.players[ctx.discarder];
        if (!discarder.discards.empty()) {
            discarder.discards.back().called = true;
        }

        // 全員の一発を消す
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            rs.players[p].ippatsu = false;
        }

        // 全員の第一ツモ巡を終了
        rs.first_draw.fill(false);

        // 喰い替え追跡
        rs.just_called = true;
        rs.last_call_tile_type = ctx.discard_tile / 4;

        result.events.push_back(Event::make_chi(shimocha, ctx.discard_tile));

        rs.phase = Phase::SelfActionPhase;
        rs.current_player = shimocha;
        ctx.active = false;
        return;
    }

    // 全員スキップ → 次のプレイヤーのツモ
    ctx.active = false;

    // 同巡内フリテンをリセット
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        rs.players[p].is_temporary_furiten = false;
    }

    advance_to_next_draw(env, result);
}

// ============================
// ツモ処理
// ============================

void GameEngine::draw_tile(EnvironmentState& env, PlayerId player, StepResult& result) {
    auto& rs = env.round_state;

    if (rs.remaining_draws() <= 0) {
        // 荒牌平局
        rs.end_reason = RoundEndReason::ExhaustiveDraw;
        rs.phase = Phase::EndRound;
        result.round_over = true;
        result.events.push_back(Event::make_exhaustive_draw());
        result.events.push_back(Event::make_round_end(RoundEndReason::ExhaustiveDraw));
        return;
    }

    TileId drawn = rs.wall[rs.wall_position++];
    rs.players[player].hand.push_back(drawn);
    rs.players[player].rinshan_draw = false;
    rs.current_player = player;
    rs.phase = Phase::SelfActionPhase;
    rs.turn_number++;

    result.events.push_back(Event::make_draw(player, drawn));
}

void GameEngine::draw_rinshan(EnvironmentState& env, PlayerId player, StepResult& result) {
    auto& rs = env.round_state;

    // 嶺上牌: wall[kNumTiles - 1 - rinshan_draw_count]
    int rinshan_idx = kNumTiles - 1 - rs.rinshan_draw_count;
    TileId drawn = rs.wall[rinshan_idx];
    rs.rinshan_draw_count++;

    rs.players[player].hand.push_back(drawn);
    rs.players[player].rinshan_draw = true;
    rs.current_player = player;
    rs.phase = Phase::SelfActionPhase;

    result.events.push_back(Event::make_draw(player, drawn));
}

void GameEngine::advance_to_next_draw(EnvironmentState& env, StepResult& result) {
    auto& rs = env.round_state;
    PlayerId next = (rs.last_discarder + 1) % kNumPlayers;
    draw_tile(env, next, result);
}

// ============================
// 応答チェック・セットアップ
// ============================

bool GameEngine::has_any_response(const EnvironmentState& env, PlayerId discarder, TileId tile) const {
    const auto& rs = env.round_state;
    TileType discard_type = tile / 4;

    for (int offset = 1; offset <= 3; ++offset) {
        PlayerId p = (discarder + offset) % kNumPlayers;
        const auto& player = rs.players[p];

        // ロンチェック（簡易: 手形のみ、フリテンはCQ-0017で実装）
        auto counts = hand_utils::make_type_counts(player.hand);
        counts[discard_type]++;
        if (hand_utils::is_agari(counts)) return true;
        counts[discard_type]--;

        // ポンチェック
        int same_count = 0;
        for (TileId id : player.hand) {
            if (id / 4 == discard_type) same_count++;
        }
        if (same_count >= 2) return true;

        // 大明槓チェック
        if (same_count >= 3 && rs.total_kan_count < 4) return true;

        // チーチェック（下家のみ）
        if (p == (discarder + 1) % kNumPlayers && !Tile::is_jihai(discard_type)) {
            int num = Tile::number_of(discard_type);
            Suit suit = Tile::suit_of(discard_type);
            int base = static_cast<int>(suit) * 9;

            // 3パターンの順子チェック
            // パターン1: (N-2, N-1, N) → 手牌に N-2 と N-1 があるか
            if (num >= 3) {
                TileType t1 = base + (num - 3);
                TileType t2 = base + (num - 2);
                bool has_t1 = false, has_t2 = false;
                for (TileId id : player.hand) {
                    if (id / 4 == t1) has_t1 = true;
                    if (id / 4 == t2) has_t2 = true;
                }
                if (has_t1 && has_t2) return true;
            }
            // パターン2: (N-1, N, N+1)
            if (num >= 2 && num <= 8) {
                TileType t1 = base + (num - 2);
                TileType t2 = base + num;
                bool has_t1 = false, has_t2 = false;
                for (TileId id : player.hand) {
                    if (id / 4 == t1) has_t1 = true;
                    if (id / 4 == t2) has_t2 = true;
                }
                if (has_t1 && has_t2) return true;
            }
            // パターン3: (N, N+1, N+2)
            if (num <= 7) {
                TileType t1 = base + num;
                TileType t2 = base + (num + 1);
                bool has_t1 = false, has_t2 = false;
                for (TileId id : player.hand) {
                    if (id / 4 == t1) has_t1 = true;
                    if (id / 4 == t2) has_t2 = true;
                }
                if (has_t1 && has_t2) return true;
            }
        }
    }
    return false;
}

void GameEngine::setup_response_phase(EnvironmentState& env, PlayerId discarder, TileId tile) {
    auto& rs = env.round_state;
    auto& ctx = rs.response_context;

    ctx.reset();
    ctx.discarder = discarder;
    ctx.discard_tile = tile;
    ctx.active = true;

    // 応答可能プレイヤーを判定
    // 応答オプションが Skip のみのプレイヤーは自動スキップ
    TileType discard_type = tile / 4;

    for (int offset = 1; offset <= 3; ++offset) {
        PlayerId p = (discarder + offset) % kNumPlayers;
        const auto& player = rs.players[p];
        bool has_option = false;

        // ロンチェック
        auto counts = hand_utils::make_type_counts(player.hand);
        counts[discard_type]++;
        if (hand_utils::is_agari(counts)) has_option = true;
        counts[discard_type]--;

        // ポンチェック
        int same_count = 0;
        for (TileId id : player.hand) {
            if (id / 4 == discard_type) same_count++;
        }
        if (same_count >= 2) has_option = true;

        // 大明槓チェック
        if (same_count >= 3 && rs.total_kan_count < 4) has_option = true;

        // チーチェック（下家のみ）
        if (p == (discarder + 1) % kNumPlayers && !Tile::is_jihai(discard_type)) {
            int num = Tile::number_of(discard_type);
            Suit suit = Tile::suit_of(discard_type);
            int base = static_cast<int>(suit) * 9;

            if (num >= 3) {
                TileType t1 = base + (num - 3), t2 = base + (num - 2);
                bool h1 = false, h2 = false;
                for (TileId id : player.hand) { if (id/4==t1) h1=true; if (id/4==t2) h2=true; }
                if (h1 && h2) has_option = true;
            }
            if (num >= 2 && num <= 8) {
                TileType t1 = base + (num - 2), t2 = base + num;
                bool h1 = false, h2 = false;
                for (TileId id : player.hand) { if (id/4==t1) h1=true; if (id/4==t2) h2=true; }
                if (h1 && h2) has_option = true;
            }
            if (num <= 7) {
                TileType t1 = base + num, t2 = base + (num + 1);
                bool h1 = false, h2 = false;
                for (TileId id : player.hand) { if (id/4==t1) h1=true; if (id/4==t2) h2=true; }
                if (h1 && h2) has_option = true;
            }
        }

        if (has_option) {
            ctx.needs_response[p] = true;
        } else {
            // 自動スキップ
            ctx.has_responded[p] = true;
            ctx.responses[p] = Action::make_skip(p);
        }
    }
    // 打牌者自身は応答不要
    ctx.has_responded[discarder] = true;
    ctx.responses[discarder] = Action::make_skip(discarder);

    rs.phase = Phase::ResponsePhase;
    PlayerId first_responder = find_next_responder(env);
    if (first_responder != 255) {
        rs.current_player = first_responder;
    } else {
        // 全員自動スキップ（通常ここには来ない）
        StepResult dummy;
        rs.phase = Phase::ResolveResponsePhase;
        resolve_responses(env, dummy);
    }
}

PlayerId GameEngine::find_next_responder(const EnvironmentState& env) const {
    const auto& ctx = env.round_state.response_context;
    // 打牌者から反時計回りに探す
    for (int offset = 1; offset <= 3; ++offset) {
        PlayerId p = (ctx.discarder + offset) % kNumPlayers;
        if (ctx.needs_response[p] && !ctx.has_responded[p]) {
            return p;
        }
    }
    return 255;
}

// ============================
// 合法手列挙（CQ-0009, CQ-0010）
// ============================

std::vector<Action> GameEngine::get_legal_actions(const EnvironmentState& env) const {
    const auto& rs = env.round_state;
    switch (rs.phase) {
        case Phase::SelfActionPhase:
            return get_self_actions(env);
        case Phase::ResponsePhase:
            return get_response_actions(env);
        default:
            return {};
    }
}

std::vector<Action> GameEngine::get_self_actions(const EnvironmentState& env) const {
    const auto& rs = env.round_state;
    PlayerId player_id = rs.current_player;
    const auto& player = rs.players[player_id];
    std::vector<Action> actions;

    auto counts = hand_utils::make_type_counts(player.hand);

    // --- 打牌 ---
    std::set<TileId> added_discards;
    for (TileId tile : player.hand) {
        // 喰い替えチェック
        if (rs.just_called && (tile / 4) == rs.last_call_tile_type) continue;
        if (added_discards.insert(tile).second) {
            actions.push_back(Action::make_discard(player_id, tile));
        }
    }

    // --- 立直打牌 ---
    if (player.is_menzen && !player.is_riichi && player.score >= 1000) {
        // 各牌を仮に捨てたときにテンパイかチェック
        std::set<TileId> added_riichi;
        for (TileId tile : player.hand) {
            if (rs.just_called && (tile / 4) == rs.last_call_tile_type) continue;
            if (!added_riichi.insert(tile).second) continue;

            auto temp = counts;
            temp[tile / 4]--;
            if (hand_utils::is_tenpai(temp)) {
                actions.push_back(Action::make_discard(player_id, tile, true));
            }
        }
    }

    // --- ツモ和了 ---
    if (hand_utils::is_agari(counts)) {
        actions.push_back(Action::make_tsumo_win(player_id));
    }

    // --- 暗槓 ---
    if (!rs.just_called && rs.total_kan_count < 4) {
        std::set<TileType> checked;
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (counts[t] == 4 && checked.insert(static_cast<TileType>(t)).second) {
                // 立直中は待ち不変チェックが必要（CQ-0016で実装、ここでは単純に許可）
                actions.push_back(Action::make_ankan(player_id, static_cast<TileType>(t)));
            }
        }
    }

    // --- 加槓 ---
    if (!rs.just_called && rs.total_kan_count < 4) {
        for (const auto& meld : player.melds) {
            if (meld.type == MeldType::Pon) {
                TileType pon_type = meld.base_type();
                for (TileId tile : player.hand) {
                    if (tile / 4 == pon_type) {
                        actions.push_back(Action::make_kakan(player_id, tile));
                        break;  // 同種の加槓は1つ
                    }
                }
            }
        }
    }

    // --- 九種九牌 ---
    if (rs.first_draw[player_id]) {
        if (hand_utils::count_yaochu_types(counts) >= 9) {
            actions.push_back(Action::make_kyuushu(player_id));
        }
    }

    return actions;
}

std::vector<Action> GameEngine::get_response_actions(const EnvironmentState& env) const {
    const auto& rs = env.round_state;
    const auto& ctx = rs.response_context;
    PlayerId player_id = rs.current_player;
    const auto& player = rs.players[player_id];
    std::vector<Action> actions;

    TileType discard_type = ctx.discard_tile / 4;

    // --- ロン ---
    auto counts = hand_utils::make_type_counts(player.hand);
    counts[discard_type]++;
    if (hand_utils::is_agari(counts)) {
        // フリテンチェックは CQ-0017 で実装。ここでは形のみ。
        actions.push_back(Action::make_ron(player_id, ctx.discarder));
    }

    // --- ポン ---
    {
        std::vector<TileId> same_tiles;
        for (TileId id : player.hand) {
            if (id / 4 == discard_type) {
                same_tiles.push_back(id);
            }
        }
        if (same_tiles.size() >= 2) {
            // 最初の2枚を使用（赤牌の選択は今後拡張可能）
            actions.push_back(Action::make_pon(player_id, ctx.discard_tile,
                                                same_tiles[0], same_tiles[1], ctx.discarder));
        }
    }

    // --- 大明槓 ---
    {
        int same_count = 0;
        for (TileId id : player.hand) {
            if (id / 4 == discard_type) same_count++;
        }
        if (same_count >= 3 && rs.total_kan_count < 4) {
            actions.push_back(Action::make_daiminkan(player_id, ctx.discard_tile, ctx.discarder));
        }
    }

    // --- チー（下家のみ）---
    PlayerId shimocha = (ctx.discarder + 1) % kNumPlayers;
    if (player_id == shimocha && !Tile::is_jihai(discard_type)) {
        int num = Tile::number_of(discard_type);
        Suit suit = Tile::suit_of(discard_type);
        int base = static_cast<int>(suit) * 9;

        // パターン1: 鳴き牌が最大 (N-2, N-1, N)
        if (num >= 3) {
            TileType t1 = base + (num - 3);
            TileType t2 = base + (num - 2);
            TileId id1 = 255, id2 = 255;
            for (TileId id : player.hand) {
                if (id / 4 == t1 && id1 == 255) id1 = id;
                if (id / 4 == t2 && id2 == 255) id2 = id;
            }
            if (id1 != 255 && id2 != 255) {
                actions.push_back(Action::make_chi(player_id, ctx.discard_tile, id1, id2));
            }
        }
        // パターン2: 鳴き牌が中央 (N-1, N, N+1)
        if (num >= 2 && num <= 8) {
            TileType t1 = base + (num - 2);
            TileType t2 = base + num;
            TileId id1 = 255, id2 = 255;
            for (TileId id : player.hand) {
                if (id / 4 == t1 && id1 == 255) id1 = id;
                if (id / 4 == t2 && id2 == 255) id2 = id;
            }
            if (id1 != 255 && id2 != 255) {
                actions.push_back(Action::make_chi(player_id, ctx.discard_tile, id1, id2));
            }
        }
        // パターン3: 鳴き牌が最小 (N, N+1, N+2)
        if (num <= 7) {
            TileType t1 = base + num;
            TileType t2 = base + (num + 1);
            TileId id1 = 255, id2 = 255;
            for (TileId id : player.hand) {
                if (id / 4 == t1 && id1 == 255) id1 = id;
                if (id / 4 == t2 && id2 == 255) id2 = id;
            }
            if (id1 != 255 && id2 != 255) {
                actions.push_back(Action::make_chi(player_id, ctx.discard_tile, id1, id2));
            }
        }
    }

    // --- スキップ（常に可能）---
    actions.push_back(Action::make_skip(player_id));

    return actions;
}

}  // namespace mahjong
