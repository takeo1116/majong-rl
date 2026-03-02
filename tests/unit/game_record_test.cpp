#include <gtest/gtest.h>
#include "io/game_record.h"
#include "engine/game_engine.h"
#include "core/environment_state.h"

using namespace mahjong;

// ============================
// 記録の蓄積テスト
// ============================

TEST(GameRecordTest, RecordMatchStart) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    EXPECT_EQ(recorder.record.seed, 42u);
    EXPECT_EQ(recorder.record.first_dealer, 0);
}

TEST(GameRecordTest, RecordRoundStart) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    recorder.on_round_start(0, 0, 0, 0, scores);
    EXPECT_EQ(recorder.record.rounds.size(), 1u);
    EXPECT_EQ(recorder.record.rounds[0].round_number, 0);
    EXPECT_EQ(recorder.record.rounds[0].dealer, 0);
}

TEST(GameRecordTest, RecordActions) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    recorder.on_round_start(0, 0, 0, 0, scores);

    auto action = Action::make_discard(0, 10);
    recorder.on_action(action);
    EXPECT_EQ(recorder.record.rounds[0].actions.size(), 1u);
    EXPECT_EQ(recorder.record.rounds[0].actions[0].type, ActionType::Discard);
}

TEST(GameRecordTest, RecordEvents) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    recorder.on_round_start(0, 0, 0, 0, scores);

    std::vector<Event> events = {Event::make_discard(0, 10, false)};
    recorder.on_events(events);
    EXPECT_EQ(recorder.record.rounds[0].events.size(), 1u);
}

TEST(GameRecordTest, RecordRoundEnd) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    recorder.on_round_start(0, 0, 0, 0, scores);

    std::array<int32_t, kNumPlayers> end_scores = {33000, 25000, 25000, 17000};
    recorder.on_round_end(RoundEndReason::Tsumo, end_scores);
    EXPECT_EQ(recorder.record.rounds[0].end_reason, RoundEndReason::Tsumo);
    EXPECT_EQ(recorder.record.rounds[0].end_scores, end_scores);
}

TEST(GameRecordTest, RecordMatchEnd) {
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);

    std::array<int32_t, kNumPlayers> final_scores = {40000, 30000, 20000, 10000};
    std::array<uint8_t, kNumPlayers> ranking = {0, 1, 2, 3};
    recorder.on_match_end(final_scores, ranking);
    EXPECT_TRUE(recorder.record.is_complete);
    EXPECT_EQ(recorder.record.final_scores, final_scores);
    EXPECT_EQ(recorder.record.final_ranking, ranking);
}

// ============================
// enabled=false で記録なし
// ============================

TEST(GameRecordTest, DisabledRecorderDoesNotRecord) {
    GameRecorder recorder;
    recorder.enabled = false;

    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};
    recorder.on_match_start(42, 0, scores);
    recorder.on_round_start(0, 0, 0, 0, scores);
    recorder.on_action(Action::make_discard(0, 10));
    recorder.on_events({Event::make_discard(0, 10, false)});
    recorder.on_round_end(RoundEndReason::Tsumo, scores);
    recorder.on_match_end(scores, {0, 1, 2, 3});

    // 無効時は何も記録されない
    EXPECT_EQ(recorder.record.seed, 0u);
    EXPECT_TRUE(recorder.record.rounds.empty());
    EXPECT_FALSE(recorder.record.is_complete);
}

// ============================
// seed + action 列からの再現テスト
// ============================

TEST(GameRecordTest, ReplayFromRecord) {
    GameEngine engine;
    EnvironmentState env;
    const uint64_t seed = 42;
    const PlayerId first_dealer = 0;

    engine.reset_match(env, seed, first_dealer);

    // 記録しながら数ステップ実行
    GameRecorder recorder;
    std::array<int32_t, kNumPlayers> scores;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        scores[p] = env.round_state.players[p].score;
    }
    recorder.on_match_start(seed, first_dealer, scores);
    recorder.on_round_start(env.round_state.round_number, env.round_state.dealer,
                            env.round_state.honba, env.round_state.kyotaku, scores);

    // 20ステップ実行して記録
    for (int i = 0; i < 20; ++i) {
        auto actions = engine.get_legal_actions(env);
        if (actions.empty()) break;
        auto& action = actions[0];
        recorder.on_action(action);
        auto result = engine.step(env, action);
        recorder.on_events(result.events);
        if (result.round_over || result.match_over) break;
    }

    // 記録から再現
    EnvironmentState env2;
    engine.reset_match(env2, recorder.record.seed, recorder.record.first_dealer);

    for (const auto& action : recorder.record.rounds[0].actions) {
        auto result = engine.step(env2, action);
        EXPECT_EQ(result.error, ErrorCode::Ok);
    }

    // 再現後の状態が一致することを確認
    EXPECT_EQ(env.round_state.phase, env2.round_state.phase);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env.round_state.players[p].hand, env2.round_state.players[p].hand);
        EXPECT_EQ(env.round_state.players[p].score, env2.round_state.players[p].score);
        EXPECT_EQ(env.round_state.players[p].discards.size(),
                  env2.round_state.players[p].discards.size());
    }
}
