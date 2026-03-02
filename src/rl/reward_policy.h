#pragma once

#include "core/tile.h"
#include "core/match_state.h"
#include "core/environment_state.h"
#include <array>
#include <cstdint>

namespace mahjong {

// RewardPolicyType, kDefaultRankRewards, RewardPolicyConfig は
// core/environment_state.h で定義されている

namespace reward_policy {

// 点差報酬: (scores_after - scores_before) * scale
std::array<float, kNumPlayers> compute_point_delta(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    float scale = 1.0f);

// 順位報酬: 半荘終了時のみ非ゼロ
std::array<float, kNumPlayers> compute_final_rank(
    const MatchState& ms,
    const std::array<float, kNumPlayers>& rank_rewards = kDefaultRankRewards,
    float scale = 1.0f);

// 混合報酬
std::array<float, kNumPlayers> compute_combined(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    const MatchState& ms,
    bool match_over,
    const RewardPolicyConfig& config);

// RewardPolicyConfig に基づいて報酬を計算する
std::array<float, kNumPlayers> compute(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    const MatchState& ms,
    bool match_over,
    const RewardPolicyConfig& config);

}  // namespace reward_policy
}  // namespace mahjong
