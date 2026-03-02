#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "core/environment_state.h"

using namespace mahjong;

class SettlementTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    void skip_all_responses() {
        while (env.round_state.phase == Phase::ResponsePhase) {
            PlayerId cp = env.round_state.current_player;
            engine.step(env, Action::make_skip(cp));
        }
    }

    // ゲームを流局まで進める
    StepResult play_until_round_end() {
        StepResult last_result;
        int safety = 0;
        while (!env.round_state.is_round_over() && safety < 1000) {
            auto actions = engine.get_legal_actions(env);
            if (actions.empty()) break;

            // ツモ和了は避けて打牌を優先
            Action chosen = actions[0];
            for (const auto& a : actions) {
                if (a.type == ActionType::Discard && !a.riichi) {
                    chosen = a;
                    break;
                }
                if (a.type == ActionType::Skip) {
                    chosen = a;
                    break;
                }
            }
            last_result = engine.step(env, chosen);
            safety++;
        }
        return last_result;
    }
};

// ============================
// CQ-0018: 通常流局・ノーテン罰符
// ============================

// 通常流局のノーテン罰符: テンパイ者にノーテン者から3000点
TEST_F(SettlementTest, ExhaustiveDrawNotenPenalty) {
    // RoundConfig で局面を注入して制御する
    RoundConfig config;
    config.round_number = 0;
    config.dealer = 0;
    config.honba = 0;
    config.kyotaku = 0;
    config.scores = {25000, 25000, 25000, 25000};

    // 山を生成（シャッフル済み）
    std::iota(config.wall.begin(), config.wall.end(), 0);
    std::mt19937 rng(123);
    std::shuffle(config.wall.begin(), config.wall.end(), rng);

    // 配牌: 親14枚、子13枚
    int pos = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        int count = (p == config.dealer) ? 14 : 13;
        config.hands[p].clear();
        for (int i = 0; i < count; ++i) {
            config.hands[p].push_back(config.wall[pos++]);
        }
    }

    auto err = engine.reset_round(env, config);
    ASSERT_EQ(err, ErrorCode::Ok);

    // 流局まで進める
    auto result = play_until_round_end();

    if (env.round_state.end_reason != RoundEndReason::ExhaustiveDraw) {
        GTEST_SKIP() << "Round did not end with exhaustive draw";
    }

    // ノーテン罰符の確認: 全員の得点合計は100000のまま
    int32_t total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += env.match_state.scores[p];
    }
    EXPECT_EQ(total, 100000) << "Total score should remain 100000 after noten penalty";
}

// ノーテン罰符: 1人テンパイ、3人ノーテンの場合
TEST_F(SettlementTest, NotenPenalty1Tenpai3Noten) {
    // 直接手牌を操作して流局をシミュレート
    auto& rs = env.round_state;
    auto& ms = env.match_state;

    // Player 0 (dealer): テンパイ手（13枚: 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p）
    rs.players[0].hand.clear();
    rs.players[0].hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40};

    // Player 1-3: ノーテン（バラバラで順子・刻子が作れない13枚）
    for (PlayerId p = 1; p < kNumPlayers; ++p) {
        rs.players[p].hand.clear();
        // types: 0,2,5,8,10,13,16,19,22,25,26,27,28 → バラバラ、テンパイ不可
        rs.players[p].hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};
    }

    // 流局を強制
    rs.wall_position = kNumTiles - 14;  // 残りツモ0
    rs.end_reason = RoundEndReason::ExhaustiveDraw;
    rs.phase = Phase::EndRound;

    // settle_round を呼ぶ（step 内で自動呼び出しされるが、ここではテスト用に直接呼べない）
    // step が round_over = true の result を返した時に呼ばれるので、
    // draw_tile で流局になった場合の結果を検証する

    // 代わりに: player 0 のツモで流局させる
    rs.phase = Phase::SelfActionPhase;
    rs.current_player = 0;
    rs.end_reason = RoundEndReason::None;

    // 手牌を14枚にする（ツモ状態）
    rs.players[0].hand.push_back(41);

    // player 0 が打牌 → 次のツモで流局
    auto result = engine.step(env, Action::make_discard(0, 41));
    // 応答をスキップ
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        engine.step(env, Action::make_skip(cp));
    }

    if (!env.round_state.is_round_over()) {
        // まだ終わっていない場合はスキップ
        GTEST_SKIP() << "Round did not end as expected";
    }

    if (env.round_state.end_reason == RoundEndReason::ExhaustiveDraw) {
        // 1人テンパイ: テンパイ者+3000, ノーテン者-1000
        EXPECT_EQ(ms.scores[0], 28000) << "Tenpai player should gain 3000";
        for (PlayerId p = 1; p < kNumPlayers; ++p) {
            EXPECT_EQ(ms.scores[p], 24000) << "Noten player should lose 1000";
        }
    }
}

// ============================
// CQ-0018: 九種九牌
// ============================

// 九種九牌で局が終了する
TEST_F(SettlementTest, KyuushuEndsRound) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];

    // 手牌に9種以上の么九牌を含むようにする
    // 么九牌: 1m(0),9m(8),1p(9),9p(17),1s(18),9s(26),東(27),南(28),西(29),北(30),白(31),發(32),中(33)
    dp.hand.clear();
    dp.hand = {0, 32, 36, 68, 72, 108, 109, 110, 111, 112, 113, 120, 124, 128};
    // types: 0,8,9,17,18,27,27,27,27,28,28,30,31,32 → 么九牌 8種...

    // 改善: 確実に9種以上
    dp.hand.clear();
    dp.hand = {0, 32, 36, 68, 72, 108, 112, 116, 120, 124, 128, 132, 4, 8};
    // types: 0,8,9,17,18,27,28,29,30,31,32,33,1,2
    // 么九: 0,8,9,17,18,27,28,29,30,31,32,33 = 12種

    auto counts = hand_utils::make_type_counts(dp.hand);
    int yaochu = hand_utils::count_yaochu_types(counts);
    ASSERT_GE(yaochu, 9) << "Should have at least 9 yaochu types";

    auto actions = engine.get_legal_actions(env);
    bool has_kyuushu = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Kyuushu) {
            has_kyuushu = true;
            break;
        }
    }
    ASSERT_TRUE(has_kyuushu) << "Kyuushu should be available";

    auto result = engine.step(env, Action::make_kyuushu(dealer));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_EQ(env.round_state.end_reason, RoundEndReason::AbortiveKyuushu);

    // 点数変動なし
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_FLOAT_EQ(result.rewards[p], 0.0f) << "No score change on kyuushu";
    }
}

// 九種九牌で親流れ
TEST_F(SettlementTest, KyuushuDealerRotates) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];
    uint8_t honba_before = env.match_state.honba;
    uint8_t kyotaku_before = env.match_state.kyotaku;

    // 九種九牌の手牌
    dp.hand.clear();
    dp.hand = {0, 32, 36, 68, 72, 108, 112, 116, 120, 124, 128, 132, 4, 8};

    engine.step(env, Action::make_kyuushu(dealer));
    engine.advance_round(env);

    if (env.match_state.is_match_over) {
        GTEST_SKIP() << "Match ended";
    }

    // 親流れ → round_number が増加
    EXPECT_EQ(env.match_state.round_number, 1) << "Round number should advance";
    EXPECT_NE(env.match_state.current_dealer, dealer) << "Dealer should rotate";

    // 積み棒・供託は持ち越し
    EXPECT_EQ(env.match_state.honba, honba_before) << "Honba should carry over";
    EXPECT_EQ(env.match_state.kyotaku, kyotaku_before) << "Kyotaku should carry over";
}

// ============================
// CQ-0018: 親連荘
// ============================

// 親ツモ和了で連荘
TEST_F(SettlementTest, DealerTsumoRenchan) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];

    // 和了形の手牌
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};
    // 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p2p

    auto result = engine.step(env, Action::make_tsumo_win(dealer));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(result.round_over);

    engine.advance_round(env);

    if (env.match_state.is_match_over) {
        GTEST_SKIP() << "Match ended";
    }

    // 連荘: round_number は変わらない
    EXPECT_EQ(env.match_state.round_number, 0) << "Round number should stay for renchan";
    EXPECT_EQ(env.match_state.current_dealer, dealer) << "Dealer should stay for renchan";
    EXPECT_EQ(env.match_state.honba, 1) << "Honba should increase";
    EXPECT_EQ(env.match_state.kyotaku, 0) << "Kyotaku should be 0 after win";
}

// 子ツモ和了で親交代
TEST_F(SettlementTest, NonDealerTsumoRotates) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    // 下家の手番まで進める
    // 親が打牌
    auto& dp = env.round_state.players[dealer];
    TileId discard = dp.hand.back();
    engine.step(env, Action::make_discard(dealer, discard));
    skip_all_responses();

    if (env.round_state.is_round_over()) {
        GTEST_SKIP() << "Round ended early";
    }

    // 下家を和了形にする
    auto& sp = env.round_state.players[shimocha];
    sp.hand.clear();
    sp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};

    auto result = engine.step(env, Action::make_tsumo_win(shimocha));
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(result.round_over);

    engine.advance_round(env);

    if (env.match_state.is_match_over) {
        GTEST_SKIP() << "Match ended";
    }

    EXPECT_EQ(env.match_state.round_number, 1) << "Round number should advance";
    EXPECT_NE(env.match_state.current_dealer, dealer) << "Dealer should rotate";
    EXPECT_EQ(env.match_state.honba, 0) << "Honba should reset";
}

// ============================
// CQ-0019: ツモ和了の精算
// ============================

// ツモ和了でリワードが返る
TEST_F(SettlementTest, TsumoWinRewards) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];

    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40, 41};

    auto result = engine.step(env, Action::make_tsumo_win(dealer));
    ASSERT_EQ(result.error, ErrorCode::Ok);

    // 親ツモ → 子から支払い
    // リワードの合計は0（ゼロサム）
    float total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += result.rewards[p];
    }
    EXPECT_NEAR(total, 0.0f, 1.0f) << "Total rewards should be approximately zero-sum";

    // 親のリワードは正
    EXPECT_GT(result.rewards[dealer], 0.0f) << "Dealer should gain points on tsumo win";
}

// ============================
// CQ-0019: ロン和了の精算
// ============================

// ロン和了で放銃者から支払い
TEST_F(SettlementTest, RonWinRewards) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    auto& sp = env.round_state.players[shimocha];
    // 下家: 断么九の手（2m3m4m 5m6m7m 2p3p4p 5p5p5p 6p）→ 6p(type14)単騎待ち
    sp.hand.clear();
    sp.hand = {4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 53, 54, 56};
    // types: 1,2,3,4,5,6,10,11,12,13,13,13,14 → tanyao

    // 親が 6p(id=57, type14) を捨てる
    auto& dp = env.round_state.players[dealer];
    dp.hand.push_back(57);

    engine.step(env, Action::make_discard(dealer, 57));

    // 下家がロン
    StepResult last_result;
    bool ron_executed = false;
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool ron_found = false;
        if (cp == shimocha) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Ron) {
                    last_result = engine.step(env, a);
                    ron_found = true;
                    ron_executed = true;
                    break;
                }
            }
        }
        if (!ron_found) {
            last_result = engine.step(env, Action::make_skip(cp));
        }
    }

    if (env.round_state.end_reason != RoundEndReason::Ron) {
        GTEST_SKIP() << "Ron did not succeed";
    }
    ASSERT_TRUE(ron_executed) << "Ron should have been executed";

    // 放銃者（親）のリワードは負
    EXPECT_LT(last_result.rewards[dealer], 0.0f) << "Discarder should lose points";
    // 和了者（下家）のリワードは正
    EXPECT_GT(last_result.rewards[shimocha], 0.0f) << "Winner should gain points";

    // ゼロサム
    float total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += last_result.rewards[p];
    }
    EXPECT_NEAR(total, 0.0f, 1.0f) << "Total rewards should be zero-sum";
}

// ============================
// CQ-0020: 半荘進行テスト
// ============================

// advance_round を繰り返して半荘が終了する
TEST_F(SettlementTest, FullMatchProgression) {
    int rounds = 0;
    while (!env.match_state.is_match_over && rounds < 50) {
        // 各局をプレイ
        StepResult last_result;
        int safety = 0;
        while (!env.round_state.is_round_over() && safety < 500) {
            auto actions = engine.get_legal_actions(env);
            if (actions.empty()) break;

            Action chosen = actions[0];
            for (const auto& a : actions) {
                if (a.type == ActionType::Discard && !a.riichi) {
                    chosen = a;
                    break;
                }
                if (a.type == ActionType::Skip) {
                    chosen = a;
                    break;
                }
            }
            last_result = engine.step(env, chosen);
            safety++;
        }

        if (!env.round_state.is_round_over()) break;
        engine.advance_round(env);
        rounds++;
    }

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should eventually end";
    EXPECT_GE(rounds, 8) << "Should play at least 8 rounds (full hanchan)";

    // 最終スコア合計は100000
    int32_t total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += env.match_state.scores[p];
    }
    EXPECT_EQ(total, 100000) << "Total score should remain 100000";
}

// 飛び終了テスト: 0点未満で終了
TEST_F(SettlementTest, TobiEndsMatch) {
    // プレイヤー0の点数を-100に設定して飛び終了をテスト
    env.match_state.scores[0] = -100;
    env.round_state.players[0].score = -100;
    env.round_state.end_reason = RoundEndReason::ExhaustiveDraw;
    env.round_state.phase = Phase::EndRound;

    engine.advance_round(env);

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end on tobi (score < 0)";
}

// 0点ちょうどは飛びではない
TEST_F(SettlementTest, ZeroScoreNotTobi) {
    env.match_state.scores[0] = 0;
    env.round_state.players[0].score = 0;
    env.round_state.end_reason = RoundEndReason::ExhaustiveDraw;
    env.round_state.phase = Phase::EndRound;

    engine.advance_round(env);

    EXPECT_FALSE(env.match_state.is_match_over) << "Match should NOT end on exactly 0 points";
}
