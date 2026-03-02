#include <gtest/gtest.h>
#include "rl/observation.h"
#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <algorithm>

using namespace mahjong;

class ObservationTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }
};

// ============================
// PartialObservation テスト
// ============================

TEST_F(ObservationTest, PartialContainsOwnHand) {
    PlayerId observer = 0;
    auto obs = make_partial_observation(env, observer);
    EXPECT_EQ(obs.observer, observer);
    EXPECT_EQ(obs.hand, env.round_state.players[observer].hand);
}

TEST_F(ObservationTest, PartialDoesNotContainOtherHands) {
    PlayerId observer = 0;
    auto obs = make_partial_observation(env, observer);
    // PartialObservation には他家手牌のフィールドが存在しない
    // public_melds は副露のみ（手牌ではない）
    // 他家の手牌牌IDが obs.hand に含まれていないことを確認
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        for (TileId t : env.round_state.players[p].hand) {
            auto it = std::find(obs.hand.begin(), obs.hand.end(), t);
            EXPECT_EQ(it, obs.hand.end()) << "他家(player" << static_cast<int>(p)
                << ")の手牌 " << static_cast<int>(t) << " が含まれている";
        }
    }
}

TEST_F(ObservationTest, PartialContainsDoraIndicators) {
    auto obs = make_partial_observation(env, 0);
    EXPECT_EQ(obs.dora_indicators, env.round_state.dora_indicators);
}

TEST_F(ObservationTest, PartialContainsScores) {
    auto obs = make_partial_observation(env, 0);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(obs.scores[p], env.round_state.players[p].score);
    }
}

TEST_F(ObservationTest, PartialContainsRoundInfo) {
    auto obs = make_partial_observation(env, 0);
    EXPECT_EQ(obs.round_number, env.round_state.round_number);
    EXPECT_EQ(obs.dealer, env.round_state.dealer);
    EXPECT_EQ(obs.honba, env.round_state.honba);
    EXPECT_EQ(obs.kyotaku, env.round_state.kyotaku);
    EXPECT_EQ(obs.current_player, env.round_state.current_player);
    EXPECT_EQ(obs.phase, env.round_state.phase);
}

TEST_F(ObservationTest, PartialJikazeIsCorrect) {
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        auto obs = make_partial_observation(env, p);
        EXPECT_EQ(obs.jikaze, env.round_state.players[p].jikaze);
    }
}

TEST_F(ObservationTest, PartialContainsDiscards) {
    // 1手打牌してから部分観測を確認
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    engine.step(env, Action::make_discard(cp, tile));

    // 応答フェーズをスキップ
    while (env.round_state.phase == Phase::ResponsePhase) {
        auto actions = engine.get_legal_actions(env);
        for (const auto& a : actions) {
            if (a.type == ActionType::Skip) {
                engine.step(env, a);
                break;
            }
        }
    }

    auto obs = make_partial_observation(env, 0);
    EXPECT_EQ(obs.discards[cp].size(), env.round_state.players[cp].discards.size());
}

// ============================
// FullObservation テスト
// ============================

TEST_F(ObservationTest, FullContainsAllHands) {
    auto obs = make_full_observation(env);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(obs.hands[p], env.round_state.players[p].hand);
    }
}

TEST_F(ObservationTest, FullContainsWall) {
    auto obs = make_full_observation(env);
    EXPECT_EQ(obs.wall, env.round_state.wall);
    EXPECT_EQ(obs.wall_position, env.round_state.wall_position);
}

TEST_F(ObservationTest, FullContainsUradora) {
    auto obs = make_full_observation(env);
    EXPECT_EQ(obs.uradora_indicators, env.round_state.uradora_indicators);
}

TEST_F(ObservationTest, FullContainsMatchState) {
    auto obs = make_full_observation(env);
    EXPECT_EQ(obs.match_state, env.match_state);
}

TEST_F(ObservationTest, FullContainsRoundInfo) {
    auto obs = make_full_observation(env);
    EXPECT_EQ(obs.round_number, env.round_state.round_number);
    EXPECT_EQ(obs.dealer, env.round_state.dealer);
    EXPECT_EQ(obs.phase, env.round_state.phase);
    EXPECT_EQ(obs.honba, env.round_state.honba);
    EXPECT_EQ(obs.kyotaku, env.round_state.kyotaku);
    EXPECT_EQ(obs.end_reason, env.round_state.end_reason);
}

// ============================
// make_observation 統一 API テスト（CQ-0028）
// ============================

TEST_F(ObservationTest, MakeObservationPartialMatchesDirect) {
    PlayerId observer = 0;
    auto obs1 = make_partial_observation(env, observer);
    auto obs2 = make_observation(env, observer);

    EXPECT_EQ(obs1.observer, obs2.observer);
    EXPECT_EQ(obs1.hand, obs2.hand);
    EXPECT_EQ(obs1.melds, obs2.melds);
    EXPECT_EQ(obs1.is_riichi, obs2.is_riichi);
    EXPECT_EQ(obs1.scores, obs2.scores);
    EXPECT_EQ(obs1.dora_indicators, obs2.dora_indicators);
    EXPECT_EQ(obs1.round_number, obs2.round_number);
    EXPECT_EQ(obs1.phase, obs2.phase);
}

TEST_F(ObservationTest, MakeObservationFullMatchesDirect) {
    auto obs1 = make_full_observation(env);
    auto obs2 = make_observation(env, full_observation);

    EXPECT_EQ(obs1.wall, obs2.wall);
    EXPECT_EQ(obs1.wall_position, obs2.wall_position);
    EXPECT_EQ(obs1.uradora_indicators, obs2.uradora_indicators);
    EXPECT_EQ(obs1.match_state, obs2.match_state);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(obs1.hands[p], obs2.hands[p]);
    }
}

TEST_F(ObservationTest, MakeObservationPartialForAllPlayers) {
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        auto obs = make_observation(env, p);
        EXPECT_EQ(obs.observer, p);
        EXPECT_EQ(obs.hand, env.round_state.players[p].hand);
    }
}
