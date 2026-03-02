#pragma once

#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <vector>

namespace mahjong {
namespace test_helpers {

// 半荘完走の結果
struct MatchResult {
    EnvironmentState final_env;
    std::vector<Action> all_actions;
    // 各局終了時のスコアスナップショット
    std::vector<std::array<int32_t, kNumPlayers>> round_end_scores;
    int total_steps = 0;
};

// 方策: 常に最初の合法手を選ぶ（deterministic、RNG 不使用）
inline Action first_action_policy(const std::vector<Action>& actions, const EnvironmentState&) {
    return actions[0];
}

// 方策: 常に最後の合法手を選ぶ（first_action_policy と対になる）
inline Action last_action_policy(const std::vector<Action>& actions, const EnvironmentState&) {
    return actions.back();
}

// 方策: Discard(非立直) か Skip を優先、なければ最初の合法手
inline Action discard_skip_policy(const std::vector<Action>& actions, const EnvironmentState&) {
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && !a.riichi) return a;
        if (a.type == ActionType::Skip) return a;
    }
    return actions[0];
}

// 半荘を完走するテンプレート関数
// Policy: Action(const std::vector<Action>&, const EnvironmentState&)
template<typename Policy>
MatchResult run_full_match(GameEngine& engine, uint64_t seed, PlayerId first_dealer,
                           RunMode mode, Policy policy, int max_steps = 10000) {
    MatchResult result;
    EnvironmentState env;
    engine.reset_match(env, seed, first_dealer, mode);

    int steps = 0;
    while (!env.match_state.is_match_over && steps < max_steps) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;

        Action chosen = policy(actions, env);
        result.all_actions.push_back(chosen);

        auto step_result = engine.step(env, chosen);
        if (step_result.error != ErrorCode::Ok) break;

        if (step_result.round_over) {
            // 局終了時のスコアをスナップショット
            result.round_end_scores.push_back(env.match_state.scores);
            if (!step_result.match_over) {
                engine.advance_round(env);
            }
        }
        ++steps;
    }

    result.final_env = env;
    result.total_steps = steps;
    return result;
}

}  // namespace test_helpers
}  // namespace mahjong
