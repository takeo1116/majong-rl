#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "core/environment_state.h"

using namespace mahjong;

// CQ-0036: オーラス終了・延長局遷移の条件分岐テスト
class OorasuTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    // オーラス状態（南4局）をセットアップするヘルパー
    // round_number=7, dealer=先頭親+7%4 のプレイヤー
    void setup_oorasu(RoundEndReason reason) {
        auto& rs = env.round_state;
        auto& ms = env.match_state;

        ms.round_number = 7;
        ms.current_dealer = (ms.first_dealer + 7 % kNumPlayers) % kNumPlayers;

        rs.round_number = 7;
        rs.dealer = ms.current_dealer;
        rs.end_reason = reason;
        rs.phase = Phase::EndRound;
    }
};

// 和了止め: オーラス親がトップで和了（ツモ）→ 終了
TEST_F(OorasuTest, AgariDomeTsumo) {
    setup_oorasu(RoundEndReason::Tsumo);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 親をトップにする
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 40000;
        } else {
            ms.scores[p] = 20000;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親のツモ和了
    rs.current_player = dealer;

    engine.advance_round(env);

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end on agari-dome (tsumo)";
}

// 和了止め: オーラス親がトップでロン → 終了
TEST_F(OorasuTest, AgariDomeRon) {
    setup_oorasu(RoundEndReason::Ron);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 親をトップにする
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 40000;
        } else {
            ms.scores[p] = 20000;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親がロン和了者に含まれるようにレスポンスコンテキストを設定
    auto& ctx = rs.response_context;
    ctx.discarder = (dealer + 1) % kNumPlayers;
    ctx.has_responded[dealer] = true;
    ctx.responses[dealer] = Action::make_ron(dealer, ctx.discarder);

    engine.advance_round(env);

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end on agari-dome (ron)";
}

// 和了止め不成立: 親が和了しているがトップではない → 連荘で継続
TEST_F(OorasuTest, AgariDomeNotTopContinues) {
    setup_oorasu(RoundEndReason::Tsumo);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 親をトップにしない（他のプレイヤーが高い）
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 20000;
        } else if (p == (dealer + 1) % kNumPlayers) {
            ms.scores[p] = 40000;
        } else {
            ms.scores[p] = 20000;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親のツモ和了（連荘になるがトップではない → 和了止めにならない）
    rs.current_player = dealer;

    engine.advance_round(env);

    // 連荘で続行（和了止めにならない）
    EXPECT_FALSE(env.match_state.is_match_over) << "Match should continue (dealer not top)";
}

// 聴牌止め: オーラス親がトップで流局テンパイ → 終了
TEST_F(OorasuTest, TenpaiDome) {
    setup_oorasu(RoundEndReason::ExhaustiveDraw);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 親をトップにする
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 40000;
        } else {
            ms.scores[p] = 20000;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親をテンパイにする（断么九テンパイ: 2m3m4m 5m6m7m 2p3p4p 5p5p5p 6p → 6p待ち）
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 53, 54, 56};

    engine.advance_round(env);

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end on tenpai-dome";
}

// 聴牌止め不成立: オーラス親ノーテンで流局 → 親流れ
TEST_F(OorasuTest, NotenDealerRotates) {
    setup_oorasu(RoundEndReason::ExhaustiveDraw);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 親をトップにする
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 40000;
        } else {
            ms.scores[p] = 20000;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親をノーテンにする（バラバラ手）
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};

    engine.advance_round(env);

    // 親ノーテン → 連荘なし → round_number が進む
    // round_number=8 になり、トップ≥30000 → 終了
    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end (dealer noten, top >= 30000)";
    EXPECT_EQ(env.match_state.round_number, 8) << "Round number should advance to 8";
}

// 延長局遷移: 南4局終了時、トップが30000未満 → 延長局へ
TEST_F(OorasuTest, ExtensionRoundTransition) {
    setup_oorasu(RoundEndReason::ExhaustiveDraw);

    auto& rs = env.round_state;
    auto& ms = env.match_state;
    PlayerId dealer = rs.dealer;

    // 全員30000未満にする（トップが29000）
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) {
            ms.scores[p] = 20000;
        } else if (p == (dealer + 1) % kNumPlayers) {
            ms.scores[p] = 29000;
        } else {
            ms.scores[p] = 25500;
        }
        rs.players[p].score = ms.scores[p];
    }

    // 親ノーテン → 連荘なし
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};

    engine.advance_round(env);

    EXPECT_FALSE(env.match_state.is_match_over) << "Match should NOT end (top < 30000)";
    EXPECT_TRUE(env.match_state.is_extra_round) << "Should enter extension round";
    EXPECT_EQ(env.match_state.round_number, 8) << "Round number should be 8";
}

// 延長局終了: 延長局実行後は必ず終了
TEST_F(OorasuTest, ExtensionRoundEnds) {
    auto& rs = env.round_state;
    auto& ms = env.match_state;

    // 延長局の状態をセットアップ
    ms.round_number = 8;
    ms.is_extra_round = true;
    ms.current_dealer = ms.first_dealer;

    rs.round_number = 8;
    rs.dealer = ms.current_dealer;
    rs.end_reason = RoundEndReason::ExhaustiveDraw;
    rs.phase = Phase::EndRound;

    // スコアを設定（全員30000未満のまま）
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        ms.scores[p] = 25000;
        rs.players[p].score = 25000;
    }

    // 親ノーテン
    rs.players[rs.dealer].hand.clear();
    rs.players[rs.dealer].hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};

    engine.advance_round(env);

    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end after extension round";
}

// 延長局中の連荘でも終了（延長局で親がツモ和了）
TEST_F(OorasuTest, ExtensionRoundRenchanAlsoEnds) {
    auto& rs = env.round_state;
    auto& ms = env.match_state;

    // 延長局の状態をセットアップ
    ms.round_number = 8;
    ms.is_extra_round = true;
    ms.current_dealer = ms.first_dealer;

    rs.round_number = 8;
    rs.dealer = ms.current_dealer;
    rs.end_reason = RoundEndReason::Tsumo;
    rs.current_player = rs.dealer;  // 親ツモ → 連荘
    rs.phase = Phase::EndRound;

    // スコアを設定
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        ms.scores[p] = 25000;
        rs.players[p].score = 25000;
    }

    engine.advance_round(env);

    // 延長局（is_extra_round && was_oorasu）なので、連荘でも終了
    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end even with renchan in extension";
}

// 飛び終了と0点継続の確認（settlement_test の補完）
TEST_F(OorasuTest, TobiVsZeroInOorasu) {
    setup_oorasu(RoundEndReason::ExhaustiveDraw);

    auto& rs = env.round_state;
    auto& ms = env.match_state;

    // プレイヤー0が0点（飛びではない）
    ms.scores[0] = 0;
    rs.players[0].score = 0;
    // 他プレイヤーで合計100000にする
    ms.scores[1] = 40000;
    ms.scores[2] = 35000;
    ms.scores[3] = 25000;
    rs.players[1].score = 40000;
    rs.players[2].score = 35000;
    rs.players[3].score = 25000;

    // 親ノーテン
    rs.players[rs.dealer].hand.clear();
    rs.players[rs.dealer].hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};

    engine.advance_round(env);

    // 0点は飛びではないのでまずは飛び判定を通過
    // 南4局終了 → トップ40000 ≥ 30000 → 通常終了
    EXPECT_TRUE(env.match_state.is_match_over) << "Match should end normally (top >= 30000)";

    // 飛びケース: スコア-100で再テスト
    EnvironmentState env2;
    engine.reset_match(env2, 42, static_cast<PlayerId>(0));
    env2.match_state.round_number = 7;
    env2.round_state.round_number = 7;
    env2.match_state.scores[0] = -100;
    env2.round_state.players[0].score = -100;
    env2.round_state.end_reason = RoundEndReason::ExhaustiveDraw;
    env2.round_state.phase = Phase::EndRound;

    engine.advance_round(env2);

    EXPECT_TRUE(env2.match_state.is_match_over) << "Match should end on tobi (score < 0)";
}
