#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/player_state.h"
#include "core/action.h"
#include "core/event.h"
#include <array>
#include <vector>

namespace mahjong {

// 応答コンテキスト（他家打牌に対する応答待ち情報）
struct ResponseContext {
    PlayerId discarder = 255;           // 打牌者
    TileId discard_tile = 255;          // 打牌された牌
    bool active = false;                // 応答待ち中かどうか

    // 各プレイヤーの応答（ResponsePhase 中に収集）
    std::array<Action, kNumPlayers> responses;
    std::array<bool, kNumPlayers> has_responded = {};
    // 応答が必要なプレイヤーの一覧
    std::array<bool, kNumPlayers> needs_response = {};

    void reset() {
        discarder = 255;
        discard_tile = 255;
        active = false;
        has_responded.fill(false);
        needs_response.fill(false);
    }

    bool operator==(const ResponseContext&) const = default;
};

// 局状態
// 値コピー可能な構造体
struct RoundState {
    // 局番号（0=東1局, 1=東2局, ..., 7=南4局, 8=延長局）
    uint8_t round_number = 0;

    // 親プレイヤーID
    PlayerId dealer = 0;

    // 現在手番プレイヤーID
    PlayerId current_player = 0;

    // 山（136枚、シャッフル済み）
    std::array<TileId, kNumTiles> wall;

    // 山からの次ツモ位置
    uint8_t wall_position = 0;

    // 王牌の開始位置（山の末尾14枚が王牌）
    // wall[kNumTiles - 14] から wall[kNumTiles - 1] まで
    // ドラ表示牌: wall[kNumTiles - 6], wall[kNumTiles - 8], ...
    // 裏ドラ:    wall[kNumTiles - 5], wall[kNumTiles - 7], ...
    // 嶺上牌:    wall[kNumTiles - 1], wall[kNumTiles - 2], wall[kNumTiles - 3], wall[kNumTiles - 4]

    // ドラ表示牌一覧（開示済み）
    std::vector<TileId> dora_indicators;

    // 裏ドラ表示牌一覧
    std::vector<TileId> uradora_indicators;

    // 槓ドラ公開予約状態（大明槓/加槓で次巡捨牌時に公開）
    bool pending_kan_dora = false;

    // 各プレイヤー状態
    std::array<PlayerState, kNumPlayers> players;

    // 本場数
    uint8_t honba = 0;

    // 供託本数
    uint8_t kyotaku = 0;

    // 巡目
    uint8_t turn_number = 0;

    // 直前打牌情報
    TileId last_discard = 255;
    PlayerId last_discarder = 255;

    // 直前イベント列
    std::vector<Event> recent_events;

    // 局終了フラグ
    RoundEndReason end_reason = RoundEndReason::None;

    // 現在のフェーズ
    Phase phase = Phase::StartRound;

    // 応答コンテキスト
    ResponseContext response_context;

    // 槓の実行回数（全体、5回目の槓は不可）
    uint8_t total_kan_count = 0;

    // 嶺上ツモ位置（王牌の末尾から取る）
    uint8_t rinshan_draw_count = 0;

    // 第一ツモ巡かどうか（九種九牌の判定用）
    std::array<bool, kNumPlayers> first_draw;

    // 喰い替えチェック用
    bool just_called = false;
    TileType last_call_tile_type = 255;

    // 局が終了しているかどうか
    bool is_round_over() const { return end_reason != RoundEndReason::None; }

    // ツモ可能な残り枚数を返す
    int remaining_draws() const;

    // 比較
    bool operator==(const RoundState&) const = default;

    // 初期化
    void reset(uint8_t round_num, PlayerId dealer_id, uint8_t honba_count, uint8_t kyotaku_count);
};

}  // namespace mahjong
