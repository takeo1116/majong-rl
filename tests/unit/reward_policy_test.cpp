#include <gtest/gtest.h>
#include "rl/reward_policy.h"
#include "engine/game_engine.h"
#include "core/environment_state.h"

using namespace mahjong;

// ============================
// PointDelta テスト
// ============================

TEST(RewardPolicyTest, PointDeltaZeroWhenNoChange) {
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    auto rewards = reward_policy::compute_point_delta(scores, scores);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(rewards[p], 0.0f);
    }
}

TEST(RewardPolicyTest, PointDeltaReflectsScoreDiff) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    auto rewards = reward_policy::compute_point_delta(before, after);
    EXPECT_FLOAT_EQ(rewards[0], 8000.0f);
    EXPECT_FLOAT_EQ(rewards[1], 0.0f);
    EXPECT_FLOAT_EQ(rewards[2], 0.0f);
    EXPECT_FLOAT_EQ(rewards[3], -8000.0f);
}

TEST(RewardPolicyTest, PointDeltaScale) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    auto rewards = reward_policy::compute_point_delta(before, after, 0.001f);
    EXPECT_FLOAT_EQ(rewards[0], 8.0f);
    EXPECT_FLOAT_EQ(rewards[3], -8.0f);
}

// ============================
// FinalRank テスト
// ============================

TEST(RewardPolicyTest, FinalRankZeroWhenMatchNotOver) {
    MatchState ms;
    ms.is_match_over = false;
    auto rewards = reward_policy::compute_final_rank(ms);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(rewards[p], 0.0f);
    }
}

TEST(RewardPolicyTest, FinalRankReturnsRankRewards) {
    MatchState ms;
    ms.is_match_over = true;
    // player0=1位, player1=2位, player2=3位, player3=4位
    ms.final_ranking = {0, 1, 2, 3};
    auto rewards = reward_policy::compute_final_rank(ms);
    EXPECT_FLOAT_EQ(rewards[0], 90.0f);
    EXPECT_FLOAT_EQ(rewards[1], 45.0f);
    EXPECT_FLOAT_EQ(rewards[2], -45.0f);
    EXPECT_FLOAT_EQ(rewards[3], -90.0f);
}

TEST(RewardPolicyTest, FinalRankCustomRewards) {
    MatchState ms;
    ms.is_match_over = true;
    ms.final_ranking = {3, 0, 1, 2};  // p0=4位, p1=1位, p2=2位, p3=3位
    std::array<float, kNumPlayers> custom = {100.0f, 50.0f, -50.0f, -100.0f};
    auto rewards = reward_policy::compute_final_rank(ms, custom);
    EXPECT_FLOAT_EQ(rewards[0], -100.0f);  // 4位
    EXPECT_FLOAT_EQ(rewards[1], 100.0f);   // 1位
    EXPECT_FLOAT_EQ(rewards[2], 50.0f);    // 2位
    EXPECT_FLOAT_EQ(rewards[3], -50.0f);   // 3位
}

TEST(RewardPolicyTest, FinalRankScale) {
    MatchState ms;
    ms.is_match_over = true;
    ms.final_ranking = {0, 1, 2, 3};
    auto rewards = reward_policy::compute_final_rank(ms, kDefaultRankRewards, 0.5f);
    EXPECT_FLOAT_EQ(rewards[0], 45.0f);
    EXPECT_FLOAT_EQ(rewards[1], 22.5f);
}

// ============================
// Combined テスト
// ============================

TEST(RewardPolicyTest, CombinedSumsBothRewards) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    MatchState ms;
    ms.is_match_over = true;
    ms.final_ranking = {0, 1, 2, 3};

    RewardPolicyConfig config;
    config.type = RewardPolicyType::Combined;
    config.point_delta_scale = 1.0f;
    config.rank_scale = 1.0f;

    auto rewards = reward_policy::compute_combined(before, after, ms, true, config);
    // player0: 8000 + 90 = 8090
    EXPECT_FLOAT_EQ(rewards[0], 8090.0f);
    // player3: -8000 + (-90) = -8090
    EXPECT_FLOAT_EQ(rewards[3], -8090.0f);
}

TEST(RewardPolicyTest, CombinedNoRankWhenMatchNotOver) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    MatchState ms;
    ms.is_match_over = false;

    RewardPolicyConfig config;
    config.type = RewardPolicyType::Combined;

    auto rewards = reward_policy::compute_combined(before, after, ms, false, config);
    // 点差のみ
    EXPECT_FLOAT_EQ(rewards[0], 8000.0f);
    EXPECT_FLOAT_EQ(rewards[3], -8000.0f);
}

// ============================
// compute() ディスパッチテスト
// ============================

TEST(RewardPolicyTest, ComputeDispatchesPointDelta) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    MatchState ms;
    RewardPolicyConfig config;
    config.type = RewardPolicyType::PointDelta;
    auto rewards = reward_policy::compute(before, after, ms, false, config);
    EXPECT_FLOAT_EQ(rewards[0], 8000.0f);
    EXPECT_FLOAT_EQ(rewards[3], -8000.0f);
}

TEST(RewardPolicyTest, ComputeDispatchesFinalRankZeroWhenNotOver) {
    std::array<int32_t, kNumPlayers> before = {25000, 25000, 25000, 25000};
    std::array<int32_t, kNumPlayers> after  = {33000, 25000, 25000, 17000};
    MatchState ms;
    RewardPolicyConfig config;
    config.type = RewardPolicyType::FinalRank;
    auto rewards = reward_policy::compute(before, after, ms, false, config);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(rewards[p], 0.0f);
    }
}

// ============================
// step() 統合テスト（CQ-0029）
// ============================

TEST(RewardPolicyTest, StepUsesDefaultPointDelta) {
    GameEngine engine;
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));
    // デフォルトは PointDelta
    EXPECT_EQ(env.reward_policy_config.type, RewardPolicyType::PointDelta);

    // 通常打牌では点数差0なので報酬は0
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    auto result = engine.step(env, Action::make_discard(cp, tile));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(result.rewards[p], 0.0f);
    }
}

TEST(RewardPolicyTest, StepRespectsRewardPolicyConfig) {
    GameEngine engine;
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));

    // FinalRank に切り替え
    env.reward_policy_config.type = RewardPolicyType::FinalRank;

    // 通常打牌では半荘終了していないので FinalRank は0
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    auto result = engine.step(env, Action::make_discard(cp, tile));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(result.rewards[p], 0.0f);
    }
}

TEST(RewardPolicyTest, StepPointDeltaOnRiichiDeduction) {
    GameEngine engine;
    EnvironmentState env;
    engine.reset_match(env, 42, static_cast<PlayerId>(0));
    env.reward_policy_config.type = RewardPolicyType::PointDelta;

    // 立直可能な手を作る
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;
    hand.clear();
    // テンパイ形: 1m*3 2m*3 3m*3 4m*3 + 5m 6m = 14枚
    for (int i = 0; i < 3; ++i) hand.push_back(0 + i);    // 1m
    for (int i = 0; i < 3; ++i) hand.push_back(4 + i);    // 2m
    for (int i = 0; i < 3; ++i) hand.push_back(8 + i);    // 3m
    for (int i = 0; i < 3; ++i) hand.push_back(12 + i);   // 4m
    hand.push_back(16);  // 5m
    hand.push_back(20);  // 6m

    // 立直宣言打牌で1000点供託 → 本人に -1000 の報酬
    auto actions = engine.get_legal_actions(env);
    bool found_riichi = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && a.riichi) {
            auto result = engine.step(env, a);
            EXPECT_EQ(result.error, ErrorCode::Ok);
            EXPECT_FLOAT_EQ(result.rewards[cp], -1000.0f);
            found_riichi = true;
            break;
        }
    }
    EXPECT_TRUE(found_riichi);
}

// ============================
// CQ-0032: match_over が step() で正しく設定されるテスト
// ============================

class MatchOverRewardTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    // オーラス親ツモで半荘終了する状態をセットアップ
    // dealer がトップかつツモ和了形を持ち、step(TsumoWin) で match_over になる
    void setup_oorasu_tsumo_win() {
        auto& rs = env.round_state;
        auto& ms = env.match_state;

        // 南4局にする
        ms.round_number = 7;
        ms.current_dealer = 0;
        rs.round_number = 7;
        rs.dealer = 0;
        rs.current_player = 0;
        rs.phase = Phase::SelfActionPhase;

        // 親をトップにする
        ms.scores = {40000, 20000, 20000, 20000};
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            rs.players[p].score = ms.scores[p];
        }

        // 和了形の手牌: 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p2p = 14枚
        rs.players[0].hand.clear();
        rs.players[0].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
        rs.players[0].is_menzen = true;
    }

    // 飛び終了する状態をセットアップ
    // player3 が低得点で、player0 がツモ和了すると player3 が 0 点未満になる
    void setup_tobi_tsumo_win() {
        auto& rs = env.round_state;
        auto& ms = env.match_state;

        rs.current_player = 0;
        rs.dealer = 0;
        rs.phase = Phase::SelfActionPhase;

        // player3 を飛びギリギリに
        ms.scores = {40000, 30000, 30000, 100};
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            rs.players[p].score = ms.scores[p];
        }

        // 和了形の手牌
        rs.players[0].hand.clear();
        rs.players[0].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
        rs.players[0].is_menzen = true;
    }
};

// step() で半荘終了時に match_over=true が返る（和了止め）
TEST_F(MatchOverRewardTest, StepReturnsMatchOverOnAgariDome) {
    setup_oorasu_tsumo_win();
    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_TRUE(result.match_over) << "match_over should be true on agari-dome";
    EXPECT_TRUE(env.match_state.is_match_over);
}

// step() で半荘終了時に match_over=true が返る（飛び）
TEST_F(MatchOverRewardTest, StepReturnsMatchOverOnTobi) {
    setup_tobi_tsumo_win();
    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_TRUE(result.match_over) << "match_over should be true on tobi";
    EXPECT_TRUE(env.match_state.is_match_over);
}

// 通常の局終了では match_over=false
TEST_F(MatchOverRewardTest, StepMatchOverFalseOnNormalRoundEnd) {
    // 東1局の親ツモ（renchan → match は終わらない）
    auto& rs = env.round_state;
    PlayerId dealer = rs.dealer;
    rs.current_player = dealer;
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
    rs.players[dealer].is_menzen = true;

    auto result = engine.step(env, Action::make_tsumo_win(dealer));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_FALSE(result.match_over) << "match_over should be false on normal round end";
}

// FinalRank: 半荘終了時に順位報酬が返る
TEST_F(MatchOverRewardTest, FinalRankRewardOnMatchEnd) {
    setup_oorasu_tsumo_win();
    env.reward_policy_config.type = RewardPolicyType::FinalRank;

    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.match_over);

    // player0 がトップなので 1位報酬
    EXPECT_GT(result.rewards[0], 0.0f) << "Winner (1st place) should get positive rank reward";

    // 非ゼロの報酬が返っていること
    bool has_nonzero = false;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (result.rewards[p] != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "FinalRank should return non-zero rewards at match end";
}

// FinalRank: 半荘未終了時は報酬ゼロ
TEST_F(MatchOverRewardTest, FinalRankZeroOnNonMatchEnd) {
    // 東1局
    auto& rs = env.round_state;
    PlayerId dealer = rs.dealer;
    rs.current_player = dealer;
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
    rs.players[dealer].is_menzen = true;

    env.reward_policy_config.type = RewardPolicyType::FinalRank;

    auto result = engine.step(env, Action::make_tsumo_win(dealer));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_FALSE(result.match_over);

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(result.rewards[p], 0.0f)
            << "FinalRank should return 0 when match is not over";
    }
}

// Combined: 半荘終了時に点差報酬 + 順位報酬が返る
TEST_F(MatchOverRewardTest, CombinedRewardOnMatchEnd) {
    setup_oorasu_tsumo_win();
    env.reward_policy_config.type = RewardPolicyType::Combined;
    env.reward_policy_config.point_delta_scale = 1.0f;
    env.reward_policy_config.rank_scale = 1.0f;

    // 精算前のスコアを記録
    std::array<int32_t, kNumPlayers> scores_before = {};
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        scores_before[p] = env.round_state.players[p].score;
    }

    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.match_over);

    // 点差報酬のみの値
    std::array<int32_t, kNumPlayers> scores_after = {};
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        scores_after[p] = env.round_state.players[p].score;
    }
    auto point_delta_only = reward_policy::compute_point_delta(scores_before, scores_after);

    // Combined 報酬は点差報酬 + 順位報酬なので、点差報酬だけより大きい（1位の場合）
    EXPECT_GT(result.rewards[0], point_delta_only[0])
        << "Combined should add rank reward on top of point delta for the winner";
}

// PointDelta: 半荘終了時も点差報酬のみ（既存動作のリグレッション確認）
TEST_F(MatchOverRewardTest, PointDeltaUnchangedOnMatchEnd) {
    setup_oorasu_tsumo_win();
    env.reward_policy_config.type = RewardPolicyType::PointDelta;

    std::array<int32_t, kNumPlayers> scores_before = {};
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        scores_before[p] = env.round_state.players[p].score;
    }

    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.match_over);

    // 点差報酬のみ
    std::array<int32_t, kNumPlayers> scores_after = {};
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        scores_after[p] = env.round_state.players[p].score;
    }

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        float expected = static_cast<float>(scores_after[p] - scores_before[p]);
        EXPECT_FLOAT_EQ(result.rewards[p], expected)
            << "PointDelta should only return point difference for player " << static_cast<int>(p);
    }
}

// advance_round() は check_match_over() 後も正しく動作する
TEST_F(MatchOverRewardTest, AdvanceRoundIdempotentAfterMatchOver) {
    setup_oorasu_tsumo_win();
    auto result = engine.step(env, Action::make_tsumo_win(0));
    ASSERT_TRUE(result.match_over);

    // advance_round を呼んでも問題ない（is_match_over で即 return）
    engine.advance_round(env);
    EXPECT_TRUE(env.match_state.is_match_over) << "Match should still be over after advance_round";
}

// advance_round() は半荘未終了時に正常に動作する
TEST_F(MatchOverRewardTest, AdvanceRoundWorksNormallyWhenNotMatchOver) {
    // 東1局親ツモ
    auto& rs = env.round_state;
    PlayerId dealer = rs.dealer;
    rs.current_player = dealer;
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
    rs.players[dealer].is_menzen = true;

    auto result = engine.step(env, Action::make_tsumo_win(dealer));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(result.round_over);
    ASSERT_FALSE(result.match_over);

    // advance_round で次局が始まる
    engine.advance_round(env);
    EXPECT_FALSE(env.match_state.is_match_over);
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
}
