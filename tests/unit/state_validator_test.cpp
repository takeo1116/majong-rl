#include <gtest/gtest.h>
#include "engine/state_validator.h"
#include "engine/game_engine.h"
#include "core/environment_state.h"

using namespace mahjong;

class StateValidatorTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }
};

// ============================
// reset 直後の valid
// ============================

TEST_F(StateValidatorTest, ValidAfterReset) {
    auto result = state_validator::validate(env);
    EXPECT_TRUE(result.valid) << "errors:";
    for (const auto& e : result.errors) {
        std::cerr << "  " << e << std::endl;
    }
}

TEST_F(StateValidatorTest, ValidAfterResetWithDifferentSeed) {
    EnvironmentState env2;
    engine.reset_match(env2, 123, static_cast<PlayerId>(2));
    auto result = state_validator::validate(env2);
    EXPECT_TRUE(result.valid);
}

// ============================
// 数ステップ後の valid
// ============================

TEST_F(StateValidatorTest, ValidAfterSeveralSteps) {
    // 数ターン打牌して検証
    for (int i = 0; i < 10; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        // 最初の合法手を実行
        auto result = engine.step(env, actions[0]);
        if (result.error != ErrorCode::Ok) break;
        if (result.round_over || result.match_over) break;
    }
    auto vr = state_validator::validate(env);
    EXPECT_TRUE(vr.valid) << "errors:";
    for (const auto& e : vr.errors) {
        std::cerr << "  " << e << std::endl;
    }
}

// ============================
// 手動で壊した状態の invalid 検出
// ============================

TEST_F(StateValidatorTest, DetectDuplicateTile) {
    // 手牌に重複牌を挿入
    PlayerId dealer = env.round_state.dealer;
    PlayerId other = (dealer + 1) % kNumPlayers;
    // other の手牌の最初の牌を dealer の手牌にも入れる（重複）
    TileId dup = env.round_state.players[other].hand[0];
    env.round_state.players[dealer].hand.push_back(dup);
    auto result = state_validator::validate(env);
    EXPECT_FALSE(result.valid);
    // 重複エラーか総数エラーが含まれている
    bool found = false;
    for (const auto& e : result.errors) {
        if (e.find("重複") != std::string::npos || e.find("総数") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(StateValidatorTest, DetectHandCountError) {
    // 手牌を余計に追加（存在しない牌を使わず、wall から取る）
    PlayerId cp = env.round_state.current_player;
    // wall_position の1つ前（まだツモされていない牌）を手牌に追加
    // これにより手牌枚数と牌総数の両方が崩れる
    // ただし wall から削除はしないので重複が出る → 重複 or 手牌枚数でエラー
    env.round_state.players[cp].hand.push_back(env.round_state.wall[env.round_state.wall_position]);
    auto result = state_validator::validate(env);
    EXPECT_FALSE(result.valid);
}

TEST_F(StateValidatorTest, DetectWallPositionOverflow) {
    env.round_state.wall_position = kNumTiles + 1;
    auto result = state_validator::validate(env);
    EXPECT_FALSE(result.valid);
    bool found = false;
    for (const auto& e : result.errors) {
        if (e.find("wall_position") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(StateValidatorTest, DetectDoraIndicatorMismatch) {
    // ドラ表示牌を追加して裏ドラとの数を不一致にする
    env.round_state.dora_indicators.push_back(0);
    auto result = state_validator::validate(env);
    EXPECT_FALSE(result.valid);
    bool found = false;
    for (const auto& e : result.errors) {
        if (e.find("ドラ") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(StateValidatorTest, DetectPhaseEndReasonContradiction) {
    // 局終了理由があるのに進行フェーズにいる
    env.round_state.end_reason = RoundEndReason::Tsumo;
    // phase は SelfActionPhase（SetUp後のデフォルト）
    auto result = state_validator::validate(env);
    EXPECT_FALSE(result.valid);
}
