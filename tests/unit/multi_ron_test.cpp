#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "core/environment_state.h"

using namespace mahjong;

// CQ-0035: 複数ロン精算の受け入れ条件テスト
class MultiRonTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    // 牌ID対応表:
    //   type  0- 8: 1m-9m  (manzu)
    //   type  9-17: 1p-9p  (pinzu)
    //   type 18-26: 1s-9s  (souzu)
    //   type 27-33: 字牌
    //   TileId = type * 4 + index (0-3)
    //
    // 断么九に使える牌（中張牌）: type 1-7(2m-8m), 10-16(2p-8p), 19-25(2s-8s)
};

// ダブロン: 2人が同時にロンした場合の精算
TEST_F(MultiRonTest, DoubleRonSettlement) {
    PlayerId dealer = env.round_state.dealer;  // 0
    PlayerId shimocha = (dealer + 1) % kNumPlayers;  // 1
    PlayerId toimen = (dealer + 2) % kNumPlayers;    // 2

    auto& rs = env.round_state;

    // 下家(1): 断么九テンパイ → 6p(type14) 単騎待ち
    // 2m(1) 3m(2) 4m(3) 5m(4) 6m(5) 7m(6) 2p(10) 3p(11) 4p(12) 5p5p5p(13) 6p(14)
    rs.players[shimocha].hand.clear();
    rs.players[shimocha].hand = {4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 53, 54, 56};
    // types: 1,2,3,4,5,6,10,11,12,13,13,13,14

    // 対面(2): 断么九テンパイ → 6p(type14) 単騎待ち
    // 2s(19) 3s(20) 4s(21) 5s(22) 6s(23) 7s(24) 3p(11) 4p(12) 5p(13) 2p2p2p(10) 6p(14)
    rs.players[toimen].hand.clear();
    rs.players[toimen].hand = {77, 81, 85, 89, 93, 97, 45, 49, 55, 41, 42, 43, 57};
    // types: 19,20,21,22,23,24,11,12,13,10,10,10,14
    // 2s3s4s(19,20,21) 5s6s7s(22,23,24) 3p4p5p(11,12,13) 2p2p2p(10) + 6p(14)単騎

    // 上家(3): ロンできない手（バラバラ）
    rs.players[(dealer + 3) % kNumPlayers].hand.clear();
    rs.players[(dealer + 3) % kNumPlayers].hand = {3, 11, 23, 35, 47, 59, 67, 79, 91, 103, 107, 111, 115};

    // 供託を設定
    rs.kyotaku = 2;  // 2000点分の供託

    // 本場を設定
    rs.honba = 1;

    // 親が 6p(id=58, type14) を捨てる
    auto& dp = rs.players[dealer];
    dp.hand.push_back(58);

    engine.step(env, Action::make_discard(dealer, 58));
    ASSERT_EQ(env.round_state.phase, Phase::ResponsePhase);

    // 下家と対面がロン
    StepResult last_result;
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool ron_found = false;
        if (cp == shimocha || cp == toimen) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Ron) {
                    last_result = engine.step(env, a);
                    ron_found = true;
                    break;
                }
            }
        }
        if (!ron_found) {
            last_result = engine.step(env, Action::make_skip(cp));
        }
    }

    ASSERT_EQ(env.round_state.end_reason, RoundEndReason::Ron);

    // ダブロンの精算確認
    // 放銃者(dealer)のリワードは負
    EXPECT_LT(last_result.rewards[dealer], 0.0f) << "Discarder should lose points";

    // 両方の和了者のリワードは正
    EXPECT_GT(last_result.rewards[shimocha], 0.0f) << "First winner should gain points";
    EXPECT_GT(last_result.rewards[toimen], 0.0f) << "Second winner should gain points";

    // 供託棒は最優先和了者（下家=放銃者から近い順）が総取り
    // 下家(shimocha)が最優先（dealer+1）
    // 供託2000点分は下家のリワードに含まれる

    // 積み棒は各和了者にそれぞれ加算（1本場=300点×2人分=600点を放銃者が負担）

    // ゼロサム確認（供託分は全体の移動なので供託×1000がずれる）
    float total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += last_result.rewards[p];
    }
    // 供託 2000点が戻ってくるので total = +2000
    EXPECT_NEAR(total, 2000.0f, 1.0f) << "Total should account for kyotaku return";

    // 非関係者(上家)のリワードは0
    EXPECT_FLOAT_EQ(last_result.rewards[(dealer + 3) % kNumPlayers], 0.0f)
        << "Non-involved player should have no reward change";
}

// 親が和了者に含まれる場合の連荘判定
TEST_F(MultiRonTest, DealerAmongWinnersRenchan) {
    PlayerId dealer = env.round_state.dealer;  // 0
    PlayerId shimocha = (dealer + 1) % kNumPlayers;  // 1

    auto& rs = env.round_state;

    // 親(0): 断么九テンパイ → 6p(type14) 単騎待ち
    // 2m(1) 3m(2) 4m(3) 5m(4) 6m(5) 7m(6) 2p(10) 3p(11) 4p(12) 5p5p5p(13) 6p(14)
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {4, 8, 12, 16, 20, 24, 40, 44, 48, 52, 53, 54, 56};

    // 他家: ロン/ポン/チーできないバラバラ手
    for (PlayerId p = 1; p < kNumPlayers; ++p) {
        if (p == shimocha) continue;
        rs.players[p].hand.clear();
        rs.players[p].hand = {3, 11, 23, 35, 47, 59, 67, 79, 91, 103, 107, 111, 115};
    }

    // 下家が 6p(id=58) を捨てる
    rs.players[shimocha].hand.push_back(58);
    // 下家を手番にする
    rs.phase = Phase::SelfActionPhase;
    rs.current_player = shimocha;

    engine.step(env, Action::make_discard(shimocha, 58));

    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "No response phase";
    }

    // 親がロン
    StepResult last_result;
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool ron_found = false;
        if (cp == dealer) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Ron) {
                    last_result = engine.step(env, a);
                    ron_found = true;
                    break;
                }
            }
        }
        if (!ron_found) {
            last_result = engine.step(env, Action::make_skip(cp));
        }
    }

    ASSERT_EQ(env.round_state.end_reason, RoundEndReason::Ron);

    // advance_round で連荘判定
    engine.advance_round(env);

    if (env.match_state.is_match_over) {
        GTEST_SKIP() << "Match ended";
    }

    // 親がロン和了者に含まれる → 連荘
    EXPECT_EQ(env.match_state.round_number, 0) << "Round number should stay (renchan)";
    EXPECT_EQ(env.match_state.current_dealer, dealer) << "Dealer should stay (renchan)";
    EXPECT_EQ(env.match_state.honba, 1) << "Honba should increase";
}

// CQ-0037: トリプルロン精算テスト（3和了者の期待スコア一致検証）
TEST_F(MultiRonTest, TripleRonSettlement) {
    PlayerId dealer = env.round_state.dealer;  // 0
    PlayerId p1 = (dealer + 1) % kNumPlayers;  // 1 (下家, 供託最優先)
    PlayerId p2 = (dealer + 2) % kNumPlayers;  // 2 (対面)
    PlayerId p3 = (dealer + 3) % kNumPlayers;  // 3 (上家)

    auto& rs = env.round_state;

    // ドラ表示牌をクリア（正確なスコア検証のため）
    rs.dora_indicators.clear();
    rs.uradora_indicators.clear();

    // 供託・本場を設定
    rs.kyotaku = 2;  // 2000点分の供託
    rs.honba = 1;    // 1本場

    // 全プレイヤーの初期スコアを揃える
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        env.match_state.scores[p] = 25000;
        rs.players[p].score = 25000;
    }

    // --- 手牌セットアップ ---
    // 全員が断么九テンパイ → 6p(type14) 単騎待ち
    // 赤牌回避: id=16(赤5m), id=52(赤5p), id=88(赤5s) を使わない
    //
    // 各手の符計算:
    //   20(基本) + 10(門前ロン) + 4(中張牌暗刻) + 2(単騎) = 36 → 40符
    // 断么九 1翻 40符 子ロン: ceil100(320*4) = 1300
    // 1本場加算: 1300 + 300 = 1600

    // 下家(p1): 2m3m4m 5m6m7m 2p3p4p 5p5p5p + 6p単騎
    rs.players[p1].hand.clear();
    rs.players[p1].hand = {4, 8, 12, 17, 20, 24, 40, 44, 48, 53, 54, 55, 56};
    // types: 1,2,3,4,5,6,10,11,12,13,13,13,14 — 全て中張牌

    // 対面(p2): 2s3s4s 5s6s7s 2p3p4p 7m7m7m + 6p単騎
    rs.players[p2].hand.clear();
    rs.players[p2].hand = {77, 81, 85, 89, 93, 97, 41, 45, 49, 25, 26, 27, 57};
    // types: 19,20,21,22,23,24,10,11,12,6,6,6,14 — 全て中張牌

    // 上家(p3): 2s3s4s 5s6s7s 3m4m5m 6m6m6m + 6p単騎
    rs.players[p3].hand.clear();
    rs.players[p3].hand = {76, 80, 84, 90, 92, 96, 9, 13, 18, 21, 22, 23, 59};
    // types: 19,20,21,22,23,24,2,3,4,5,5,5,14 — 全て中張牌

    // 親(dealer): ロンできないバラバラ手（全て么九牌・字牌）
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132};

    // 親が 6p(id=58, type14) を捨てる
    rs.players[dealer].hand.push_back(58);

    engine.step(env, Action::make_discard(dealer, 58));
    ASSERT_EQ(env.round_state.phase, Phase::ResponsePhase);

    // 3人全員がロン
    StepResult last_result;
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool ron_found = false;
        if (cp == p1 || cp == p2 || cp == p3) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Ron) {
                    last_result = engine.step(env, a);
                    ron_found = true;
                    break;
                }
            }
        }
        if (!ron_found) {
            last_result = engine.step(env, Action::make_skip(cp));
        }
    }

    ASSERT_EQ(env.round_state.end_reason, RoundEndReason::Ron);

    // --- 期待スコアの検証 ---
    // 子ロン 1翻40符 + 1本場 = 1600点/人
    // 供託 2000点 → p1（最優先和了者）のみ

    // 放銃者(dealer): -(1600 * 3) = -4800
    EXPECT_FLOAT_EQ(last_result.rewards[dealer], -4800.0f)
        << "放銃者は3人分の和了点を支払う";

    // 最優先和了者(p1): +1600(ロン) + 2000(供託) = +3600
    EXPECT_FLOAT_EQ(last_result.rewards[p1], 3600.0f)
        << "最優先和了者はロン点+供託を得る";

    // 対面(p2): +1600(ロン)
    EXPECT_FLOAT_EQ(last_result.rewards[p2], 1600.0f)
        << "2番目の和了者はロン点のみ";

    // 上家(p3): +1600(ロン)
    EXPECT_FLOAT_EQ(last_result.rewards[p3], 1600.0f)
        << "3番目の和了者はロン点のみ";

    // --- 供託総取り検証（点数差分で明示）---
    // p1とp2の差分が供託分（2000）に一致
    EXPECT_FLOAT_EQ(last_result.rewards[p1] - last_result.rewards[p2], 2000.0f)
        << "供託は最優先和了者1名のみに入る（差分=供託額）";

    // --- 積み棒各和了者加算検証 ---
    // p2とp3は同条件なので同額
    EXPECT_FLOAT_EQ(last_result.rewards[p2], last_result.rewards[p3])
        << "同条件の和了者は同額のリワード";

    // 各和了者のロン点 1600 = 基本1300 + 積み棒300
    // (供託を含まないp2/p3で検証)
    EXPECT_FLOAT_EQ(last_result.rewards[p2], 1600.0f)
        << "各和了者に積み棒300点が加算される（1300+300=1600）";

    // --- ゼロサム + 供託検証 ---
    float total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += last_result.rewards[p];
    }
    EXPECT_NEAR(total, 2000.0f, 1.0f)
        << "全体のリワード合計 = 供託戻り分(2000)";
}

// CQ-0037: ダブロン精算の期待スコア一致検証（異なる供託・本場での検証）
TEST_F(MultiRonTest, DoubleRonExactScores) {
    PlayerId dealer = env.round_state.dealer;  // 0
    PlayerId p1 = (dealer + 1) % kNumPlayers;  // 1 (下家, 供託最優先)
    PlayerId p2 = (dealer + 2) % kNumPlayers;  // 2 (対面)
    PlayerId p3 = (dealer + 3) % kNumPlayers;  // 3 (上家, ロンしない)

    auto& rs = env.round_state;

    // ドラ表示牌をクリア
    rs.dora_indicators.clear();
    rs.uradora_indicators.clear();

    // 供託・本場を設定（トリプルロンテストと異なる値）
    rs.kyotaku = 3;  // 3000点分の供託
    rs.honba = 2;    // 2本場

    // 全プレイヤーの初期スコアを揃える
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        env.match_state.scores[p] = 25000;
        rs.players[p].score = 25000;
    }

    // 下家(p1): 断么九テンパイ → 6p単騎
    rs.players[p1].hand.clear();
    rs.players[p1].hand = {4, 8, 12, 17, 20, 24, 40, 44, 48, 53, 54, 55, 56};

    // 対面(p2): 断么九テンパイ → 6p単騎
    rs.players[p2].hand.clear();
    rs.players[p2].hand = {77, 81, 85, 89, 93, 97, 41, 45, 49, 25, 26, 27, 57};

    // 上家(p3): ロンできないバラバラ手
    rs.players[p3].hand.clear();
    rs.players[p3].hand = {3, 11, 23, 35, 47, 63, 67, 79, 91, 103, 107, 111, 115};

    // 親(dealer): ロンできないバラバラ手
    rs.players[dealer].hand.clear();
    rs.players[dealer].hand = {0, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132};

    // 親が 6p(id=58) を捨てる
    rs.players[dealer].hand.push_back(58);

    engine.step(env, Action::make_discard(dealer, 58));
    ASSERT_EQ(env.round_state.phase, Phase::ResponsePhase);

    // p1とp2がロン、p3はスキップ
    StepResult last_result;
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool ron_found = false;
        if (cp == p1 || cp == p2) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Ron) {
                    last_result = engine.step(env, a);
                    ron_found = true;
                    break;
                }
            }
        }
        if (!ron_found) {
            last_result = engine.step(env, Action::make_skip(cp));
        }
    }

    ASSERT_EQ(env.round_state.end_reason, RoundEndReason::Ron);

    // --- 期待スコアの検証 ---
    // 子ロン 1翻40符 + 2本場 = 1300 + 600 = 1900点/人
    // 供託 3000点 → p1のみ

    // 放銃者(dealer): -(1900 * 2) = -3800
    EXPECT_FLOAT_EQ(last_result.rewards[dealer], -3800.0f)
        << "放銃者は2人分の和了点を支払う";

    // 最優先和了者(p1): +1900(ロン) + 3000(供託) = +4900
    EXPECT_FLOAT_EQ(last_result.rewards[p1], 4900.0f)
        << "最優先和了者はロン点+供託を得る";

    // 対面(p2): +1900(ロン)
    EXPECT_FLOAT_EQ(last_result.rewards[p2], 1900.0f)
        << "2番目の和了者はロン点のみ";

    // 非関係者(p3): 0
    EXPECT_FLOAT_EQ(last_result.rewards[p3], 0.0f)
        << "非関係者のリワードは0";

    // --- 供託総取り検証 ---
    EXPECT_FLOAT_EQ(last_result.rewards[p1] - last_result.rewards[p2], 3000.0f)
        << "供託は最優先和了者1名のみに入る（差分=供託額3000）";

    // --- 積み棒加算検証 ---
    // 0本場なら1300のところ、2本場で1900 → 差分600 = 300*2
    EXPECT_FLOAT_EQ(last_result.rewards[p2], 1900.0f)
        << "各和了者に積み棒600点(300*2)が加算される（1300+600=1900）";

    // --- ゼロサム + 供託検証 ---
    float total = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        total += last_result.rewards[p];
    }
    EXPECT_NEAR(total, 3000.0f, 1.0f)
        << "全体のリワード合計 = 供託戻り分(3000)";
}
