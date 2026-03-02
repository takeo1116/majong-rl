#pragma once

#include "core/types.h"
#include "core/tile.h"
#include "core/match_state.h"
#include "core/round_state.h"
#include <array>
#include <cstdint>
#include <random>

namespace mahjong {

// 報酬ポリシー種別
enum class RewardPolicyType : uint8_t {
    PointDelta = 0,
    FinalRank  = 1,
    Combined   = 2,
};

// デフォルト順位報酬（1位〜4位）
constexpr std::array<float, kNumPlayers> kDefaultRankRewards = {90.0f, 45.0f, -45.0f, -90.0f};

// 報酬ポリシー設定
struct RewardPolicyConfig {
    RewardPolicyType type = RewardPolicyType::PointDelta;
    std::array<float, kNumPlayers> rank_rewards = kDefaultRankRewards;
    float point_delta_scale = 1.0f;
    float rank_scale = 1.0f;

    bool operator==(const RewardPolicyConfig&) const = default;
};

// RNG ラッパー（値コピー可能）
// mt19937 は値コピーで内部状態が複製される
struct Rng {
    std::mt19937 engine;

    bool operator==(const Rng& other) const { return engine == other.engine; }

    // seed で初期化
    void seed(uint64_t s) { engine.seed(static_cast<std::mt19937::result_type>(s)); }

    // [0, n) の一様整数乱数を返す
    int next_int(int n) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(engine);
    }

    // 配列をシャッフルする
    template<typename RandomIt>
    void shuffle(RandomIt first, RandomIt last) {
        std::shuffle(first, last, engine);
    }
};

// 環境状態（値コピー可能）
// MatchState + 現在の RoundState + RNG + 実行モード
struct EnvironmentState {
    MatchState match_state;
    RoundState round_state;
    Rng rng;
    RunMode run_mode = RunMode::Fast;
    bool logging_enabled = true;
    RewardPolicyConfig reward_policy_config;  // 報酬ポリシー設定

    // seed で環境を初期化する
    void reset(uint64_t seed, RunMode mode = RunMode::Fast);

    // 比較
    bool operator==(const EnvironmentState&) const = default;

    // seed + 起家指定で初期化する
    void reset(uint64_t seed, PlayerId first_dealer, RunMode mode = RunMode::Fast);
};

}  // namespace mahjong
