#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/state_validator.h"
#include "core/environment_state.h"
#include "../test_helpers.h"

using namespace mahjong;
using namespace mahjong::test_helpers;

class ReproducibilityTest : public ::testing::Test {
protected:
    GameEngine engine;
};

// ============================
// CQ-0026: seed 再現性テスト
// ============================

// 同一 seed で半荘完走 ×2 → 最終スコア・順位・行動列・ステップ数が一致
TEST_F(ReproducibilityTest, FullMatchSeedReproducibility) {
    auto result1 = run_full_match(engine, 42, 0, RunMode::Fast, first_action_policy);
    auto result2 = run_full_match(engine, 42, 0, RunMode::Fast, first_action_policy);

    EXPECT_TRUE(result1.final_env.match_state.is_match_over) << "半荘が完走していない";
    EXPECT_TRUE(result2.final_env.match_state.is_match_over) << "半荘が完走していない";

    EXPECT_EQ(result1.total_steps, result2.total_steps) << "ステップ数が不一致";
    EXPECT_EQ(result1.final_env.match_state.scores, result2.final_env.match_state.scores)
        << "最終スコアが不一致";
    EXPECT_EQ(result1.final_env.match_state.final_ranking, result2.final_env.match_state.final_ranking)
        << "最終順位が不一致";
    EXPECT_EQ(result1.all_actions.size(), result2.all_actions.size())
        << "行動列の長さが不一致";
    for (size_t i = 0; i < result1.all_actions.size(); ++i) {
        EXPECT_EQ(result1.all_actions[i], result2.all_actions[i])
            << "行動列が step " << i << " で不一致";
    }
}

// 複数 seed で再現性を確認
TEST_F(ReproducibilityTest, FullMatchSeedReproducibilityMultipleSeeds) {
    for (uint64_t seed : {0ULL, 1ULL, 42ULL, 100ULL, 999ULL}) {
        auto result1 = run_full_match(engine, seed, 0, RunMode::Fast, first_action_policy);
        auto result2 = run_full_match(engine, seed, 0, RunMode::Fast, first_action_policy);

        EXPECT_TRUE(result1.final_env.match_state.is_match_over)
            << "seed=" << seed << " で半荘が完走していない";
        EXPECT_EQ(result1.total_steps, result2.total_steps)
            << "seed=" << seed << " でステップ数が不一致";
        EXPECT_EQ(result1.final_env.match_state.scores, result2.final_env.match_state.scores)
            << "seed=" << seed << " で最終スコアが不一致";
        EXPECT_EQ(result1.all_actions.size(), result2.all_actions.size())
            << "seed=" << seed << " で行動列の長さが不一致";
    }
}

// ============================
// CQ-0026: 状態コピー独立性テスト
// ============================

// コピー後にコピーを進行しても元が不変
TEST_F(ReproducibilityTest, StateCopyIndependentProgression) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    // 30ステップ進める
    for (int i = 0; i < 30; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto result = engine.step(env, first_action_policy(actions, env));
        if (result.round_over && !result.match_over) {
            engine.advance_round(env);
        }
        if (result.match_over) break;
    }

    // 元の状態をスナップショット
    auto original_snapshot = env;

    // コピーを作成
    auto env_copy = env;

    // コピーを20ステップ進める
    for (int i = 0; i < 20; ++i) {
        if (env_copy.match_state.is_match_over) break;
        auto actions = engine.get_legal_actions(env_copy);
        if (actions.empty()) break;
        auto result = engine.step(env_copy, first_action_policy(actions, env_copy));
        if (result.round_over && !result.match_over) {
            engine.advance_round(env_copy);
        }
    }

    // 元の状態がスナップショットと一致（不変であること）
    EXPECT_EQ(env.match_state.scores, original_snapshot.match_state.scores)
        << "コピーの進行で元のスコアが変化した";
    EXPECT_EQ(env.round_state.phase, original_snapshot.round_state.phase)
        << "コピーの進行で元のフェーズが変化した";
    EXPECT_EQ(env.round_state.current_player, original_snapshot.round_state.current_player)
        << "コピーの進行で元の手番が変化した";
    EXPECT_EQ(env.rng, original_snapshot.rng)
        << "コピーの進行で元の RNG が変化した";
}

// コピー後に両方を異なる方策で進行 → 両方 valid かつ状態が異なる
TEST_F(ReproducibilityTest, StateCopyBothProgressIndependently) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    // 20ステップ進める
    for (int i = 0; i < 20; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto result = engine.step(env, first_action_policy(actions, env));
        if (result.round_over && !result.match_over) engine.advance_round(env);
        if (result.match_over) break;
    }
    if (env.match_state.is_match_over) GTEST_SKIP() << "半荘が20ステップ以内に終了";

    // コピーを作成
    auto env_copy = env;

    // 元を first_action_policy で10ステップ進める
    for (int i = 0; i < 10; ++i) {
        if (env.match_state.is_match_over) break;
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto result = engine.step(env, first_action_policy(actions, env));
        if (result.round_over && !result.match_over) engine.advance_round(env);
    }

    // コピーを last_action_policy で10ステップ進める
    for (int i = 0; i < 10; ++i) {
        if (env_copy.match_state.is_match_over) break;
        auto actions = engine.get_legal_actions(env_copy);
        if (actions.empty()) break;
        auto result = engine.step(env_copy, last_action_policy(actions, env_copy));
        if (result.round_over && !result.match_over) engine.advance_round(env_copy);
    }

    // 両方が異なる状態であること（行動選択が異なるため）
    bool differ = (env.match_state.scores != env_copy.match_state.scores) ||
                  (env.round_state.current_player != env_copy.round_state.current_player) ||
                  (env.round_state.phase != env_copy.round_state.phase);
    // 手牌の内容も比較
    for (PlayerId p = 0; p < kNumPlayers && !differ; ++p) {
        differ = (env.round_state.players[p].hand != env_copy.round_state.players[p].hand) ||
                 (env.round_state.players[p].discards.size() !=
                  env_copy.round_state.players[p].discards.size());
    }
    EXPECT_TRUE(differ) << "異なる方策で進行したのに状態が同一";
}

// コピー直後の rng 出力が一致、進行後は独立
TEST_F(ReproducibilityTest, CopyPreservesRngState) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    // 10ステップ進める
    for (int i = 0; i < 10; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto result = engine.step(env, first_action_policy(actions, env));
        if (result.round_over && !result.match_over) engine.advance_round(env);
        if (result.match_over) break;
    }

    // コピー
    auto env_copy = env;

    // コピー直後は rng が一致
    EXPECT_EQ(env.rng, env_copy.rng) << "コピー直後の RNG 状態が不一致";

    // 元の rng を進める
    int val1 = env.rng.next_int(1000);
    int val2 = env_copy.rng.next_int(1000);
    EXPECT_EQ(val1, val2) << "コピー直後の最初の乱数出力が不一致";

    // さらに元のみ進める
    env.rng.next_int(1000);
    env.rng.next_int(1000);

    // この時点で rng 状態は異なるはず
    EXPECT_NE(env.rng, env_copy.rng) << "進行後も RNG 状態が同一（独立でない）";
}
