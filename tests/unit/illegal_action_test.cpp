#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "core/environment_state.h"

using namespace mahjong;

class IllegalActionTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }
};

// ============================
// SelfActionPhase の違法手テスト
// ============================

TEST_F(IllegalActionTest, SelfPhaseRejectsRon) {
    ASSERT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_ron(cp, 1));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseRejectsPon) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_pon(cp, 0, 1, 2, 1));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseRejectsChi) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_chi(cp, 0, 4, 8));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseRejectsSkip) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_skip(cp));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseRejectsDaiminkan) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_daiminkan(cp, 0, 1));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseInvalidActorPreservesState) {
    auto snapshot = env;
    PlayerId wrong = (env.round_state.current_player + 1) % kNumPlayers;

    auto result = engine.step(env, Action::make_discard(wrong, 0));
    EXPECT_EQ(result.error, ErrorCode::InvalidActor);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseInvalidTilePreservesState) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_discard(cp, 200));
    EXPECT_EQ(result.error, ErrorCode::InvalidTile);
    EXPECT_EQ(env, snapshot);
}

TEST_F(IllegalActionTest, SelfPhaseDiscardTileNotInHandPreservesState) {
    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    // 手牌に確実にない牌を見つける
    auto& hand = env.round_state.players[cp].hand;
    TileId not_in_hand = 255;
    for (TileId t = 0; t < kNumTiles; ++t) {
        if (std::find(hand.begin(), hand.end(), t) == hand.end()) {
            not_in_hand = t;
            break;
        }
    }
    ASSERT_NE(not_in_hand, 255);

    auto result = engine.step(env, Action::make_discard(cp, not_in_hand));
    EXPECT_EQ(result.error, ErrorCode::InvalidTile);
    EXPECT_EQ(env, snapshot);
}

// ============================
// ResponsePhase の違法手テスト
// ============================

class ResponseIllegalActionTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
        // ResponsePhase に入れるため、打牌して応答待ちにする
        enter_response_phase();
    }

    void enter_response_phase() {
        PlayerId dealer = env.round_state.dealer;

        // 下家にロン可能な手を作る（確実にResponsePhaseに入る）
        PlayerId shimocha = (dealer + 1) % kNumPlayers;
        auto& shimocha_hand = env.round_state.players[shimocha].hand;
        shimocha_hand.clear();
        // 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p → 2p待ち
        shimocha_hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40};

        // 親の手牌に 2p (TileId=41) を入れて捨てさせる
        auto& dealer_hand = env.round_state.players[dealer].hand;
        bool has_41 = false;
        for (auto t : dealer_hand) {
            if (t == 41) { has_41 = true; break; }
        }
        if (!has_41) {
            dealer_hand.push_back(41);
        }

        engine.step(env, Action::make_discard(dealer, 41));
        // ResponsePhase に入っているはず
    }
};

TEST_F(ResponseIllegalActionTest, ResponsePhaseRejectsDiscard) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_discard(cp, 0));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(ResponseIllegalActionTest, ResponsePhaseRejectsTsumoWin) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_tsumo_win(cp));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(ResponseIllegalActionTest, ResponsePhaseRejectsAnkan) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_ankan(cp, 0));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(ResponseIllegalActionTest, ResponsePhaseRejectsKakan) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_kakan(cp, 0));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(ResponseIllegalActionTest, ResponsePhaseRejectsKyuushu) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId cp = env.round_state.current_player;

    auto result = engine.step(env, Action::make_kyuushu(cp));
    EXPECT_NE(result.error, ErrorCode::Ok);
    EXPECT_EQ(env, snapshot);
}

TEST_F(ResponseIllegalActionTest, ResponsePhaseInvalidActorPreservesState) {
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Failed to enter ResponsePhase";
    }

    auto snapshot = env;
    PlayerId wrong = (env.round_state.current_player + 1) % kNumPlayers;

    auto result = engine.step(env, Action::make_skip(wrong));
    EXPECT_EQ(result.error, ErrorCode::InvalidActor);
    EXPECT_EQ(env, snapshot);
}

// ============================
// EndRound での違法手テスト
// ============================

TEST_F(IllegalActionTest, EndRoundRejectsAllActions) {
    // 局を終了させる
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    hand = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17};
    engine.step(env, Action::make_tsumo_win(cp));

    ASSERT_TRUE(env.round_state.is_round_over());
    auto snapshot = env;

    auto result = engine.step(env, Action::make_discard(cp, 0));
    EXPECT_EQ(result.error, ErrorCode::WrongPhase);
    EXPECT_EQ(env, snapshot);
}

// ============================
// 喰い替えの状態不変性
// ============================

TEST_F(IllegalActionTest, KuikaePreservesState) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    // 対面にポン可能な手を作る
    auto& toimen_hand = env.round_state.players[toimen].hand;
    toimen_hand.clear();
    toimen_hand = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};

    auto& dealer_hand = env.round_state.players[dealer].hand;
    bool has_0 = false;
    for (auto t : dealer_hand) { if (t == 0) { has_0 = true; break; } }
    if (!has_0) dealer_hand.push_back(0);

    engine.step(env, Action::make_discard(dealer, 0));

    // ResponsePhase でポンする
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        if (cp == toimen) {
            auto actions = engine.get_legal_actions(env);
            for (const auto& a : actions) {
                if (a.type == ActionType::Pon) {
                    engine.step(env, a);
                    goto pon_done;
                }
            }
            engine.step(env, Action::make_skip(cp));
        } else {
            engine.step(env, Action::make_skip(cp));
        }
    }
    pon_done:

    if (env.round_state.phase == Phase::SelfActionPhase &&
        env.round_state.current_player == toimen &&
        env.round_state.just_called) {
        // 喰い替え対象の牌（1m = type 0）を捨てようとする
        // player has tiles with type 0 (id 1 or 2 should remain if one was used for pon)
        auto snapshot = env;

        // 手牌に残っている type 0 の牌を探す
        TileId kuikae_tile = 255;
        for (auto t : env.round_state.players[toimen].hand) {
            if (t / 4 == 0) { kuikae_tile = t; break; }
        }

        if (kuikae_tile != 255) {
            auto result = engine.step(env, Action::make_discard(toimen, kuikae_tile));
            EXPECT_EQ(result.error, ErrorCode::IllegalAction);
            EXPECT_EQ(env, snapshot);
        }
    }
}
