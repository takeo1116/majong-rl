#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <algorithm>
#include <set>

using namespace mahjong;

class GameEngineTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }
};

// ============================
// reset_match テスト（CQ-0007, CQ-0008）
// ============================

TEST_F(GameEngineTest, ResetMatchSetsPhase) {
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
}

TEST_F(GameEngineTest, ResetMatchDealerHas14Tiles) {
    PlayerId dealer = env.round_state.dealer;
    EXPECT_EQ(env.round_state.players[dealer].hand.size(), 14u);
}

TEST_F(GameEngineTest, ResetMatchNonDealersHave13Tiles) {
    PlayerId dealer = env.round_state.dealer;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p != dealer) {
            EXPECT_EQ(env.round_state.players[p].hand.size(), 13u);
        }
    }
}

TEST_F(GameEngineTest, ResetMatchWallPosition) {
    // 4人×13枚 + 親の1枚 = 53枚
    EXPECT_EQ(env.round_state.wall_position, 53);
}

TEST_F(GameEngineTest, ResetMatchAllTilesUnique) {
    // 配られた牌が全てユニークであること
    std::set<TileId> all_tiles;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        for (TileId t : env.round_state.players[p].hand) {
            EXPECT_TRUE(all_tiles.insert(t).second) << "Duplicate tile: " << static_cast<int>(t);
        }
    }
    EXPECT_EQ(all_tiles.size(), 53u);
}

TEST_F(GameEngineTest, ResetMatchDoraSetup) {
    EXPECT_EQ(env.round_state.dora_indicators.size(), 1u);
    EXPECT_EQ(env.round_state.uradora_indicators.size(), 1u);
}

TEST_F(GameEngineTest, ResetMatchFirstDealer) {
    EnvironmentState env2;
    engine.reset_match(env2, 42, static_cast<PlayerId>(2));
    EXPECT_EQ(env2.round_state.dealer, 2);
    EXPECT_EQ(env2.round_state.current_player, 2);
    EXPECT_EQ(env2.round_state.players[2].hand.size(), 14u);
}

TEST_F(GameEngineTest, ResetMatchDeterministic) {
    // 同じseedで同じ結果
    EnvironmentState env2;
    engine.reset_match(env2, 42, static_cast<PlayerId>(0));
    for (int i = 0; i < kNumTiles; ++i) {
        EXPECT_EQ(env.round_state.wall[i], env2.round_state.wall[i]);
    }
}

TEST_F(GameEngineTest, ResetMatchDifferentSeed) {
    EnvironmentState env2;
    engine.reset_match(env2, 99, static_cast<PlayerId>(0));
    bool any_diff = false;
    for (int i = 0; i < kNumTiles; ++i) {
        if (env.round_state.wall[i] != env2.round_state.wall[i]) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

TEST_F(GameEngineTest, ResetMatchPlayerScores) {
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env.round_state.players[p].score, 25000);
    }
}

TEST_F(GameEngineTest, ResetMatchFirstDrawFlags) {
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_TRUE(env.round_state.first_draw[p]);
    }
}

// ============================
// step() 基本テスト（CQ-0007）
// ============================

TEST_F(GameEngineTest, StepWrongPhaseError) {
    // フェーズをEndRoundに書き換えてstepを呼ぶ
    env.round_state.end_reason = RoundEndReason::ExhaustiveDraw;
    auto result = engine.step(env, Action::make_discard(0, 0));
    EXPECT_EQ(result.error, ErrorCode::WrongPhase);
}

TEST_F(GameEngineTest, StepInvalidActorError) {
    // 現在のプレイヤーでない人がアクション
    PlayerId wrong = (env.round_state.current_player + 1) % kNumPlayers;
    auto result = engine.step(env, Action::make_discard(wrong, 0));
    EXPECT_EQ(result.error, ErrorCode::InvalidActor);
}

TEST_F(GameEngineTest, StepDiscardInvalidTileError) {
    PlayerId cp = env.round_state.current_player;
    // 手牌にない牌を捨てようとする
    auto result = engine.step(env, Action::make_discard(cp, 200));
    EXPECT_EQ(result.error, ErrorCode::InvalidTile);
}

TEST_F(GameEngineTest, StepDiscardSuccess) {
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    auto result = engine.step(env, Action::make_discard(cp, tile));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    // 打牌イベントが含まれている
    bool has_discard = false;
    for (const auto& e : result.events) {
        if (e.type == EventType::Discard) has_discard = true;
    }
    EXPECT_TRUE(has_discard);
}

TEST_F(GameEngineTest, StepDiscardRemovesTileFromHand) {
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    size_t before = env.round_state.players[cp].hand.size();
    engine.step(env, Action::make_discard(cp, tile));
    EXPECT_EQ(env.round_state.players[cp].hand.size(), before - 1);
}

// ============================
// 合法手列挙テスト（CQ-0009）
// ============================

TEST_F(GameEngineTest, GetLegalActionsInSelfPhase) {
    auto actions = engine.get_legal_actions(env);
    EXPECT_FALSE(actions.empty());

    // 少なくとも打牌が含まれているはず
    bool has_discard = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard) has_discard = true;
    }
    EXPECT_TRUE(has_discard);
}

TEST_F(GameEngineTest, LegalActionsAllHaveCorrectActor) {
    auto actions = engine.get_legal_actions(env);
    PlayerId cp = env.round_state.current_player;
    for (const auto& a : actions) {
        EXPECT_EQ(a.actor, cp);
    }
}

TEST_F(GameEngineTest, LegalDiscardTilesAreInHand) {
    PlayerId cp = env.round_state.current_player;
    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && !a.riichi) {
            auto& hand = env.round_state.players[cp].hand;
            auto it = std::find(hand.begin(), hand.end(), a.tile);
            EXPECT_NE(it, hand.end()) << "Discard tile " << static_cast<int>(a.tile) << " not in hand";
        }
    }
}

TEST_F(GameEngineTest, LegalActionsNoResponseInSelfPhase) {
    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        EXPECT_NE(a.type, ActionType::Ron);
        EXPECT_NE(a.type, ActionType::Chi);
        EXPECT_NE(a.type, ActionType::Pon);
        EXPECT_NE(a.type, ActionType::Skip);
    }
}

// ============================
// ツモ和了テスト
// ============================

TEST_F(GameEngineTest, StepTsumoWin) {
    // ツモ和了可能な手を作る: 面子+雀頭の14枚
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    // 1m*3 2m*3 3m*3 4m*3 5m*2 = 14枚（和了形）
    for (int i = 0; i < 3; ++i) hand.push_back(0 + i);    // 1m: 0,1,2
    for (int i = 0; i < 3; ++i) hand.push_back(4 + i);    // 2m: 4,5,6
    for (int i = 0; i < 3; ++i) hand.push_back(8 + i);    // 3m: 8,9,10
    for (int i = 0; i < 3; ++i) hand.push_back(12 + i);   // 4m: 12,13,14
    hand.push_back(16);                                     // 5m: 16
    hand.push_back(17);                                     // 5m: 17

    auto actions = engine.get_legal_actions(env);
    bool has_tsumo = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::TsumoWin) has_tsumo = true;
    }
    EXPECT_TRUE(has_tsumo);

    auto result = engine.step(env, Action::make_tsumo_win(cp));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_EQ(env.round_state.end_reason, RoundEndReason::Tsumo);
}

// ============================
// 九種九牌テスト
// ============================

TEST_F(GameEngineTest, KyuushuAvailableOnFirstDraw) {
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    // 14枚: 1m 9m 1p 9p 1s 9s 東 南 西 北 白 發 中 + 1m
    hand = {0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132, 1};
    env.round_state.first_draw[cp] = true;

    auto actions = engine.get_legal_actions(env);
    bool has_kyuushu = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Kyuushu) has_kyuushu = true;
    }
    EXPECT_TRUE(has_kyuushu);
}

TEST_F(GameEngineTest, KyuushuNotAvailableAfterFirstDraw) {
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    hand = {0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132, 1};
    env.round_state.first_draw[cp] = false;

    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        EXPECT_NE(a.type, ActionType::Kyuushu);
    }
}

TEST_F(GameEngineTest, StepKyuushu) {
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    hand = {0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132, 1};
    env.round_state.first_draw[cp] = true;

    auto result = engine.step(env, Action::make_kyuushu(cp));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_EQ(env.round_state.end_reason, RoundEndReason::AbortiveKyuushu);
}
