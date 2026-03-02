#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/action.h"
#include "core/event.h"
#include <array>
#include <cstdint>
#include <vector>

namespace mahjong {

// 1局分の記録
struct RoundRecord {
    uint8_t round_number = 0;
    PlayerId dealer = 0;
    uint8_t honba = 0;
    uint8_t kyotaku = 0;
    std::array<int32_t, kNumPlayers> start_scores = {};
    std::array<int32_t, kNumPlayers> end_scores = {};
    RoundEndReason end_reason = RoundEndReason::None;
    std::vector<Action> actions;    // 再現用アクション列
    std::vector<Event> events;      // イベントログ
};

// 半荘全体の記録
struct GameRecord {
    uint64_t seed = 0;
    PlayerId first_dealer = 0;
    std::vector<RoundRecord> rounds;
    std::array<int32_t, kNumPlayers> final_scores = {};
    std::array<uint8_t, kNumPlayers> final_ranking = {};
    bool is_complete = false;
};

// ゲーム記録器
struct GameRecorder {
    GameRecord record;
    bool enabled = true;  // Fast モードで無効化可能

    void on_match_start(uint64_t seed, PlayerId first_dealer,
                        const std::array<int32_t, kNumPlayers>& scores);
    void on_round_start(uint8_t round_number, PlayerId dealer,
                        uint8_t honba, uint8_t kyotaku,
                        const std::array<int32_t, kNumPlayers>& scores);
    void on_action(const Action& action);
    void on_events(const std::vector<Event>& events);
    void on_round_end(RoundEndReason reason,
                      const std::array<int32_t, kNumPlayers>& end_scores);
    void on_match_end(const std::array<int32_t, kNumPlayers>& final_scores,
                      const std::array<uint8_t, kNumPlayers>& final_ranking);
};

}  // namespace mahjong
