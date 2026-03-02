#include "rl/reward_policy.h"

namespace mahjong {
namespace reward_policy {

std::array<float, kNumPlayers> compute_point_delta(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    float scale) {
    std::array<float, kNumPlayers> rewards = {};
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        rewards[p] = static_cast<float>(after[p] - before[p]) * scale;
    }
    return rewards;
}

std::array<float, kNumPlayers> compute_final_rank(
    const MatchState& ms,
    const std::array<float, kNumPlayers>& rank_rewards,
    float scale) {
    std::array<float, kNumPlayers> rewards = {};
    if (!ms.is_match_over) {
        return rewards;  // 半荘終了していなければゼロ
    }
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        uint8_t rank = ms.final_ranking[p];  // 0-based 順位
        rewards[p] = rank_rewards[rank] * scale;
    }
    return rewards;
}

std::array<float, kNumPlayers> compute_combined(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    const MatchState& ms,
    bool match_over,
    const RewardPolicyConfig& config) {
    std::array<float, kNumPlayers> rewards = {};

    // 点差報酬
    auto pd = compute_point_delta(before, after, config.point_delta_scale);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        rewards[p] += pd[p];
    }

    // 順位報酬（半荘終了時のみ）
    if (match_over) {
        auto fr = compute_final_rank(ms, config.rank_rewards, config.rank_scale);
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            rewards[p] += fr[p];
        }
    }

    return rewards;
}

std::array<float, kNumPlayers> compute(
    const std::array<int32_t, kNumPlayers>& before,
    const std::array<int32_t, kNumPlayers>& after,
    const MatchState& ms,
    bool match_over,
    const RewardPolicyConfig& config) {
    switch (config.type) {
        case RewardPolicyType::PointDelta:
            return compute_point_delta(before, after, config.point_delta_scale);
        case RewardPolicyType::FinalRank:
            if (match_over) {
                return compute_final_rank(ms, config.rank_rewards, config.rank_scale);
            }
            return {};
        case RewardPolicyType::Combined:
            return compute_combined(before, after, ms, match_over, config);
    }
    return {};
}

}  // namespace reward_policy
}  // namespace mahjong
