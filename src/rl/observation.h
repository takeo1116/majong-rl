#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/meld.h"
#include "core/player_state.h"
#include "core/match_state.h"
#include "core/environment_state.h"
#include <array>
#include <vector>

namespace mahjong {

// 部分観測（エージェントに渡す通常の観測）
struct PartialObservation {
    PlayerId observer;

    // 自家情報
    std::vector<TileId> hand;
    std::vector<Meld> melds;
    bool is_riichi = false;
    bool is_menzen = true;
    bool is_furiten = false;
    bool is_temporary_furiten = false;
    bool is_riichi_furiten = false;

    // 全員の公開情報
    std::array<std::vector<DiscardInfo>, kNumPlayers> discards;
    std::array<std::vector<Meld>, kNumPlayers> public_melds;
    std::array<int32_t, kNumPlayers> scores = {};
    std::array<bool, kNumPlayers> riichi_declared = {};

    // 局情報
    uint8_t round_number = 0;
    PlayerId dealer = 0;
    Wind bakaze = Wind::East;
    Wind jikaze = Wind::East;
    uint8_t honba = 0;
    uint8_t kyotaku = 0;
    uint8_t turn_number = 0;
    PlayerId current_player = 0;
    Phase phase = Phase::StartRound;
    std::vector<TileId> dora_indicators;
};

// 完全観測（デバッグ・学習補助・探索補助用）
struct FullObservation {
    // 全プレイヤー情報
    std::array<std::vector<TileId>, kNumPlayers> hands;
    std::array<std::vector<Meld>, kNumPlayers> melds;
    std::array<std::vector<DiscardInfo>, kNumPlayers> discards;
    std::array<int32_t, kNumPlayers> scores = {};

    // 山・王牌
    std::array<TileId, kNumTiles> wall = {};
    uint8_t wall_position = 0;
    std::vector<TileId> dora_indicators;
    std::vector<TileId> uradora_indicators;

    // 局情報
    uint8_t round_number = 0;
    PlayerId dealer = 0;
    PlayerId current_player = 0;
    Phase phase = Phase::StartRound;
    uint8_t honba = 0;
    uint8_t kyotaku = 0;
    uint8_t turn_number = 0;
    RoundEndReason end_reason = RoundEndReason::None;

    // 半荘情報
    MatchState match_state;
};

// 部分観測を生成する
PartialObservation make_partial_observation(const EnvironmentState& env, PlayerId observer);

// 完全観測を生成する
FullObservation make_full_observation(const EnvironmentState& env);

// 統一 API: 部分観測を生成する（make_observation のオーバーロード）
PartialObservation make_observation(const EnvironmentState& env, PlayerId observer);

// 統一 API: 完全観測を生成する（make_observation のオーバーロード）
// タグディスパッチ用構造体
struct FullObservationTag {};
constexpr FullObservationTag full_observation{};

FullObservation make_observation(const EnvironmentState& env, FullObservationTag);

}  // namespace mahjong
