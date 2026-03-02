#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/action.h"
#include "core/event.h"
#include "core/environment_state.h"
#include <string>
#include <vector>
#include <array>

namespace mahjong {
namespace display {

// 局番号を日本語名に変換（"東1局" 〜 "南4局", "延長局"）
std::string round_name(uint8_t round_number);

// 局ヘッダ（例: "=== 東1局 0本場 供託0本 親:P0 ==="）
std::string round_header(uint8_t round_number, uint8_t honba, uint8_t kyotaku, PlayerId dealer);

// スコア表示（例: "P0:25000 P1:25000 P2:25000 P3:25000"）
std::string scores_to_string(const std::array<int32_t, kNumPlayers>& scores);

// 手牌表示（ソート済み、例: "1m 2m 3m 5p 6p 7p 1s 2s 3s 東 東 東 白"）
std::string hand_to_string(const std::vector<TileId>& hand);

// アクション表示（例: "[P0] 打牌: 1m"）
std::string action_display(const Action& action);

// イベント表示（例: "[P0] ツモ: 5m"）
std::string event_display(const Event& event);

// 局終了サマリ
std::string round_end_summary(const EnvironmentState& env);

// 半荘終了サマリ
std::string match_end_summary(const EnvironmentState& env);

}  // namespace display
}  // namespace mahjong
