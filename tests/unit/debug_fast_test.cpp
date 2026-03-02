#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/state_validator.h"
#include "core/environment_state.h"

using namespace mahjong;

// ============================
// Debug モードでの自動検証テスト（CQ-0031）
// ============================

class DebugFastTest : public ::testing::Test {
protected:
    GameEngine engine;
};

TEST_F(DebugFastTest, DebugModeDetectsInconsistencyInStep) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0), RunMode::Debug);

    // 状態を壊してから step を呼ぶ
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];

    // 手牌を壊す（重複牌を追加）
    env.round_state.players[cp].hand.push_back(env.round_state.wall[env.round_state.wall_position]);

    auto result = engine.step(env, Action::make_discard(cp, tile));
    // Debug モードでは InconsistentState エラーが返る
    EXPECT_EQ(result.error, ErrorCode::InconsistentState);
}

TEST_F(DebugFastTest, FastModeSkipsValidation) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0), RunMode::Fast);

    // 正常な step が OK で返ることを確認
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    auto result = engine.step(env, Action::make_discard(cp, tile));
    EXPECT_EQ(result.error, ErrorCode::Ok);
}

TEST_F(DebugFastTest, DebugModeValidAfterReset) {
    // Debug モードで reset して検証が通ることを確認
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0), RunMode::Debug);
    auto vr = state_validator::validate(env);
    EXPECT_TRUE(vr.valid);
}

TEST_F(DebugFastTest, FastModeValidAfterReset) {
    // Fast モードでも reset 結果は正しい
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0), RunMode::Fast);
    auto vr = state_validator::validate(env);
    EXPECT_TRUE(vr.valid);
}

TEST_F(DebugFastTest, DebugAndFastProduceSameResult) {
    // 同じ seed で Debug/Fast モードの結果が一致
    EnvironmentState env_debug, env_fast;
    engine.reset_match(env_debug, 42, static_cast<PlayerId>(0), RunMode::Debug);
    engine.reset_match(env_fast, 42, static_cast<PlayerId>(0), RunMode::Fast);

    // 山が一致
    EXPECT_EQ(env_debug.round_state.wall, env_fast.round_state.wall);

    // 手牌が一致
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_debug.round_state.players[p].hand,
                  env_fast.round_state.players[p].hand);
    }

    // 数ステップ実行して結果が一致
    for (int i = 0; i < 10; ++i) {
        auto actions_debug = engine.get_legal_actions(env_debug);
        auto actions_fast = engine.get_legal_actions(env_fast);
        if (actions_debug.empty()) break;

        // 同じアクション列数
        EXPECT_EQ(actions_debug.size(), actions_fast.size());

        // 最初の合法手を実行
        auto result_debug = engine.step(env_debug, actions_debug[0]);
        auto result_fast = engine.step(env_fast, actions_fast[0]);

        EXPECT_EQ(result_debug.error, ErrorCode::Ok);
        EXPECT_EQ(result_fast.error, ErrorCode::Ok);

        // 報酬が一致（デフォルト PointDelta）
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            EXPECT_FLOAT_EQ(result_debug.rewards[p], result_fast.rewards[p]);
        }

        if (result_debug.round_over) break;
    }

    // ステップ後のスコアが一致
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_debug.round_state.players[p].score,
                  env_fast.round_state.players[p].score);
    }
}

TEST_F(DebugFastTest, DebugModeValidAfterMultipleSteps) {
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0), RunMode::Debug);

    // 20ステップ実行して全て OK であることを確認
    for (int i = 0; i < 20; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto result = engine.step(env, actions[0]);
        EXPECT_EQ(result.error, ErrorCode::Ok)
            << "step " << i << " failed";
        if (result.round_over || result.match_over) break;
    }
}
