#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "io/game_record.h"
#include "io/display.h"
#include "core/environment_state.h"
#include "../test_helpers.h"

using namespace mahjong;
using namespace mahjong::test_helpers;

class ReplayTest : public ::testing::Test {
protected:
    GameEngine engine;
};

// ============================
// CQ-0026: Debug/Fast モード一致テスト
// ============================

// 同一 seed で Debug/Fast 両方半荘完走 → 行動列・最終スコア・順位一致
TEST_F(ReplayTest, DebugFastFullMatchConsistency) {
    auto result_debug = run_full_match(engine, 42, 0, RunMode::Debug, first_action_policy);
    auto result_fast  = run_full_match(engine, 42, 0, RunMode::Fast,  first_action_policy);

    EXPECT_TRUE(result_debug.final_env.match_state.is_match_over) << "Debug 半荘が完走していない";
    EXPECT_TRUE(result_fast.final_env.match_state.is_match_over)  << "Fast 半荘が完走していない";

    EXPECT_EQ(result_debug.total_steps, result_fast.total_steps)
        << "Debug/Fast でステップ数が不一致";
    EXPECT_EQ(result_debug.final_env.match_state.scores, result_fast.final_env.match_state.scores)
        << "Debug/Fast で最終スコアが不一致";
    EXPECT_EQ(result_debug.final_env.match_state.final_ranking,
              result_fast.final_env.match_state.final_ranking)
        << "Debug/Fast で最終順位が不一致";

    // 行動列一致
    ASSERT_EQ(result_debug.all_actions.size(), result_fast.all_actions.size())
        << "Debug/Fast で行動列の長さが不一致";
    for (size_t i = 0; i < result_debug.all_actions.size(); ++i) {
        EXPECT_EQ(result_debug.all_actions[i], result_fast.all_actions[i])
            << "Debug/Fast で行動列が step " << i << " で不一致";
    }
}

// 複数 seed で Debug/Fast 一致を確認
TEST_F(ReplayTest, DebugFastConsistencyMultipleSeeds) {
    for (uint64_t seed : {0ULL, 7ULL, 42ULL, 256ULL, 1000ULL}) {
        auto result_debug = run_full_match(engine, seed, 0, RunMode::Debug, first_action_policy);
        auto result_fast  = run_full_match(engine, seed, 0, RunMode::Fast,  first_action_policy);

        EXPECT_TRUE(result_debug.final_env.match_state.is_match_over)
            << "seed=" << seed << " Debug 半荘が完走していない";
        EXPECT_TRUE(result_fast.final_env.match_state.is_match_over)
            << "seed=" << seed << " Fast 半荘が完走していない";

        EXPECT_EQ(result_debug.total_steps, result_fast.total_steps)
            << "seed=" << seed << " でステップ数が不一致";
        EXPECT_EQ(result_debug.final_env.match_state.scores,
                  result_fast.final_env.match_state.scores)
            << "seed=" << seed << " で最終スコアが不一致";
        EXPECT_EQ(result_debug.all_actions.size(), result_fast.all_actions.size())
            << "seed=" << seed << " で行動列の長さが不一致";
    }
}

// ============================
// CQ-0026: リプレイテスト
// ============================

// 半荘完走で行動列を保存 → 同一 seed で再生 → 最終状態一致
TEST_F(ReplayTest, FullMatchReplayFromActionSequence) {
    // 半荘完走して行動列を記録
    auto original = run_full_match(engine, 42, 0, RunMode::Fast, first_action_policy);
    ASSERT_TRUE(original.final_env.match_state.is_match_over) << "半荘が完走していない";

    // 同一 seed でリセットし、記録した行動列をリプレイ
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    size_t action_idx = 0;
    int steps = 0;
    while (!env.match_state.is_match_over && action_idx < original.all_actions.size()) {
        ASSERT_LT(action_idx, original.all_actions.size()) << "行動列が途中で尽きた";

        auto result = engine.step(env, original.all_actions[action_idx]);
        ASSERT_EQ(result.error, ErrorCode::Ok) << "リプレイ step " << action_idx << " でエラー";

        if (result.round_over && !result.match_over) {
            engine.advance_round(env);
        }
        ++action_idx;
        ++steps;
    }

    // 最終状態一致
    EXPECT_TRUE(env.match_state.is_match_over) << "リプレイが半荘完了しない";
    EXPECT_EQ(steps, original.total_steps) << "ステップ数が不一致";
    EXPECT_EQ(env.match_state.scores, original.final_env.match_state.scores)
        << "最終スコアが不一致";
    EXPECT_EQ(env.match_state.final_ranking, original.final_env.match_state.final_ranking)
        << "最終順位が不一致";
}

// 各局終了時のスコアスナップショットを保存 → 再生 → 一致
TEST_F(ReplayTest, MultiRoundReplayPreservesIntermediateState) {
    // 半荘完走して各局のスコアスナップショットを記録
    auto original = run_full_match(engine, 42, 0, RunMode::Fast, first_action_policy);
    ASSERT_TRUE(original.final_env.match_state.is_match_over) << "半荘が完走していない";
    ASSERT_GT(original.round_end_scores.size(), 0u) << "局が1つも終了していない";

    // リプレイ
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    std::vector<std::array<int32_t, kNumPlayers>> replay_scores;
    size_t action_idx = 0;
    while (!env.match_state.is_match_over && action_idx < original.all_actions.size()) {
        auto result = engine.step(env, original.all_actions[action_idx]);
        ASSERT_EQ(result.error, ErrorCode::Ok) << "リプレイ step " << action_idx << " でエラー";

        if (result.round_over) {
            replay_scores.push_back(env.match_state.scores);
            if (!result.match_over) {
                engine.advance_round(env);
            }
        }
        ++action_idx;
    }

    // 各局終了時のスコアが一致
    ASSERT_EQ(replay_scores.size(), original.round_end_scores.size())
        << "局終了回数が不一致";
    for (size_t r = 0; r < replay_scores.size(); ++r) {
        EXPECT_EQ(replay_scores[r], original.round_end_scores[r])
            << "局 " << r << " のスコアスナップショットが不一致";
    }
}

// GameRecorder で記録 → seed + action 列から再生 → 一致
TEST_F(ReplayTest, ReplayWithRecorder) {
    // GameRecorder を使って半荘完走を記録
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    GameRecorder recorder;
    recorder.on_match_start(42, 0, env.match_state.scores);
    recorder.on_round_start(env.round_state.round_number, env.round_state.dealer,
                            env.round_state.honba, env.round_state.kyotaku,
                            env.match_state.scores);

    int steps = 0;
    while (!env.match_state.is_match_over && steps < 10000) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;

        Action chosen = first_action_policy(actions, env);
        recorder.on_action(chosen);

        auto result = engine.step(env, chosen);
        if (result.error != ErrorCode::Ok) break;
        recorder.on_events(result.events);

        if (result.round_over) {
            recorder.on_round_end(env.round_state.end_reason, env.match_state.scores);
            if (!result.match_over) {
                engine.advance_round(env);
                recorder.on_round_start(env.round_state.round_number, env.round_state.dealer,
                                        env.round_state.honba, env.round_state.kyotaku,
                                        env.match_state.scores);
            }
        }
        ++steps;
    }
    recorder.on_match_end(env.match_state.scores, env.match_state.final_ranking);

    // 記録の基本検証
    ASSERT_TRUE(recorder.record.is_complete) << "記録が完了していない";
    ASSERT_GT(recorder.record.rounds.size(), 0u) << "記録に局が含まれていない";

    // GameRecorder の記録から再現
    EnvironmentState env2;
    engine.reset_match(env2, recorder.record.seed,
                       recorder.record.first_dealer);

    for (size_t r = 0; r < recorder.record.rounds.size(); ++r) {
        const auto& round = recorder.record.rounds[r];
        for (const auto& action : round.actions) {
            auto result = engine.step(env2, action);
            ASSERT_EQ(result.error, ErrorCode::Ok)
                << "局 " << r << " のリプレイでエラー";

            if (result.round_over && !result.match_over) {
                engine.advance_round(env2);
            }
        }
    }

    // 再現後の状態一致
    EXPECT_TRUE(env2.match_state.is_match_over) << "リプレイが半荘完了しない";
    EXPECT_EQ(env2.match_state.scores, env.match_state.scores)
        << "最終スコアが不一致";
    EXPECT_EQ(env2.match_state.final_ranking, env.match_state.final_ranking)
        << "最終順位が不一致";
    EXPECT_EQ(env2.match_state.scores, recorder.record.final_scores)
        << "記録の最終スコアと不一致";
}

// ============================
// CQ-0033: イベント表示安定性テスト
// ============================

// random 方策で半荘完走し、全イベントの display が例外なく通ることを確認
TEST_F(ReplayTest, EventDisplayStabilityWithRandomPolicy) {
    // random 方策（env.rng 使用で再現可能）
    auto random_policy = [](const std::vector<Action>& actions, const EnvironmentState& env) -> Action {
        // const 参照なので rng は直接使えないが、actions.size() ベースで選択
        // 再現性のために seed ベースのインデックスを使う
        int idx = static_cast<int>(env.round_state.wall_position % actions.size());
        return actions[idx];
    };

    for (uint64_t seed : {1ULL, 42ULL, 256ULL}) {
        EnvironmentState env;
        engine.reset_match(env, seed, static_cast<PlayerId>(0));

        int steps = 0;
        while (!env.match_state.is_match_over && steps < 10000) {
            auto actions = engine.get_legal_actions(env);
            if (actions.empty()) break;

            Action chosen = random_policy(actions, env);
            auto result = engine.step(env, chosen);
            if (result.error != ErrorCode::Ok) break;

            // 全イベントの display が正常に動作することを確認
            for (const auto& evt : result.events) {
                std::string s = display::event_display(evt);
                EXPECT_FALSE(s.empty())
                    << "seed=" << seed << " step=" << steps
                    << " event type=" << to_string(evt.type) << " の表示が空";
            }

            // アクション表示も確認
            std::string a = display::action_display(chosen);
            EXPECT_FALSE(a.empty());

            if (result.round_over && !result.match_over) {
                engine.advance_round(env);
            }
            ++steps;
        }
        EXPECT_TRUE(env.match_state.is_match_over)
            << "seed=" << seed << " で半荘が完走していない";
    }
}
