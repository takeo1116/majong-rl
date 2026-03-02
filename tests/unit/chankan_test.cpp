#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "core/environment_state.h"

using namespace mahjong;

class ChankanTest : public ::testing::Test {
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
};

// 加槓に対して他家がロンできる場合、槍槓応答フェーズが発生する
TEST_F(ChankanTest, ChankanRonSuccess) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    auto& dp = env.round_state.players[dealer];
    auto& sp = env.round_state.players[shimocha];

    // 親: ポン済み（5s type=22, id=88,89,90）+ 手牌に5s(id=91)を持つ
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 91};
    dp.melds.clear();
    dp.melds.push_back(Meld::make_pon(88, 89, 90, shimocha));
    dp.is_menzen = false;

    // 下家: 1m2m3m 4m5m6m 7m8m9m 4s6s 1p1p → 13枚、5s(type22)で嵌張和了
    // 4s=type21(id=84), 6s=type23(id=92), 1p=type9(id=37,38)
    sp.hand.clear();
    sp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 38, 84, 92};

    // 下家の手牌が 5s(type=22) を加えて和了形か確認
    auto counts = hand_utils::make_type_counts(sp.hand);
    counts[22]++;  // 5s
    ASSERT_TRUE(hand_utils::is_agari(counts)) << "Shimocha should be able to win with 5s";

    // 親が加槓を実行
    auto actions = engine.get_legal_actions(env);
    Action kakan_action{};
    bool found = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Kakan && a.tile == 91) {
            kakan_action = a;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Kakan action should be available";

    auto result = engine.step(env, kakan_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);

    // 槍槓応答フェーズに入る
    ASSERT_EQ(env.round_state.phase, Phase::ResponsePhase);
    ASSERT_TRUE(env.round_state.response_context.is_chankan_response);

    // 下家がロンを選択
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto resp_actions = engine.get_legal_actions(env);
        if (cp == shimocha) {
            bool ron_found = false;
            for (const auto& a : resp_actions) {
                if (a.type == ActionType::Ron) {
                    result = engine.step(env, a);
                    ron_found = true;
                    break;
                }
            }
            ASSERT_TRUE(ron_found) << "Shimocha should be able to ron (chankan)";
        } else {
            result = engine.step(env, Action::make_skip(cp));
        }
    }

    // ロンで局終了
    EXPECT_EQ(env.round_state.end_reason, RoundEndReason::Ron);
    EXPECT_TRUE(result.round_over);
}

// 槍槓応答で全員スキップした場合、嶺上ツモに進む
TEST_F(ChankanTest, ChankanSkipLeadsToRinshanDraw) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    auto& dp = env.round_state.players[dealer];

    // 親: ポン済み + 手牌に加槓牌
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 91};
    dp.melds.clear();
    dp.melds.push_back(Meld::make_pon(88, 89, 90, shimocha));
    dp.is_menzen = false;

    // 他家: 5s(type=22) で和了にならない手
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == dealer) continue;
        auto& player = env.round_state.players[p];
        auto c = hand_utils::make_type_counts(player.hand);
        c[22]++;
        if (hand_utils::is_agari(c)) {
            // 和了形になってしまう場合は手を変更
            player.hand.clear();
            player.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 49, 53, 57, 61};
        }
    }

    // 親が加槓を実行
    auto actions = engine.get_legal_actions(env);
    Action kakan_action{};
    bool found = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Kakan && a.tile == 91) {
            kakan_action = a;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Kakan action should be available";

    int hand_before = static_cast<int>(dp.hand.size());
    auto result = engine.step(env, kakan_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);

    // 槍槓応答なし → 直接嶺上ツモ（ResponsePhaseを経由しない）
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    EXPECT_EQ(env.round_state.current_player, dealer);
    // 加槓で1枚減、嶺上ツモで1枚増 → 元と同じ枚数
    EXPECT_EQ(static_cast<int>(dp.hand.size()), hand_before);
}

// 暗槓に対しては槍槓応答が発生しない
TEST_F(ChankanTest, NoChankanForAnkan) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    auto& dp = env.round_state.players[dealer];
    auto& sp = env.round_state.players[shimocha];

    // 親: 5s 4枚（id=88,89,90,91）を持つ
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 88, 89, 90, 91};

    // 下家: 5s(type22) で和了形になる手
    sp.hand.clear();
    sp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 38, 84, 92};
    auto counts = hand_utils::make_type_counts(sp.hand);
    counts[22]++;
    ASSERT_TRUE(hand_utils::is_agari(counts)) << "Shimocha would win with 5s";

    // 親が暗槓を実行
    auto actions = engine.get_legal_actions(env);
    Action ankan_action{};
    bool found = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Ankan) {
            ankan_action = a;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Ankan action should be available";

    auto result = engine.step(env, ankan_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);

    // 暗槓は槍槓応答なし → 直接嶺上ツモ
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    EXPECT_EQ(env.round_state.current_player, dealer);
}

// 槍槓応答フェーズではロンとスキップのみ提示される
TEST_F(ChankanTest, ChankanResponseOnlyRonAndSkip) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    auto& dp = env.round_state.players[dealer];
    auto& sp = env.round_state.players[shimocha];

    // 親: ポン済み + 加槓牌
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 91};
    dp.melds.clear();
    dp.melds.push_back(Meld::make_pon(88, 89, 90, shimocha));
    dp.is_menzen = false;

    // 下家: 5s(type22)でアガリ形
    sp.hand.clear();
    sp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 38, 84, 92};
    auto counts = hand_utils::make_type_counts(sp.hand);
    counts[22]++;
    ASSERT_TRUE(hand_utils::is_agari(counts));

    // 親が加槓
    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        if (a.type == ActionType::Kakan && a.tile == 91) {
            engine.step(env, a);
            break;
        }
    }

    ASSERT_EQ(env.round_state.phase, Phase::ResponsePhase);
    ASSERT_TRUE(env.round_state.response_context.is_chankan_response);

    // 応答者の合法手を確認
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto resp_actions = engine.get_legal_actions(env);

        // ロンかスキップのみ
        for (const auto& a : resp_actions) {
            EXPECT_TRUE(a.type == ActionType::Ron || a.type == ActionType::Skip)
                << "Chankan response should only have Ron or Skip, got: "
                << static_cast<int>(a.type);
        }

        engine.step(env, Action::make_skip(cp));
    }
}

// 立直後暗槓: 待ちが変わらない場合は許可
TEST_F(ChankanTest, RiichiAnkanWaitsUnchanged) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];

    // 手牌: 1m2m3m 4m5m6m 7m8m9m 5s5s5s 1p → 1p単騎待ち
    // 5sを4枚にして暗槓 → 1m2m3m 4m5m6m 7m8m9m + [5s暗槓] + 1p → 1p単騎待ち（変わらない）
    // リーチ手（13枚）: 0,4,8,12,16,20,24,28,32,88,89,90,36
    // ツモ: 91(5s) → 暗槓可能、待ちは 1p(type=9) 単騎のまま
    dp.hand.clear();
    dp.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 88, 89, 90, 36, 91};
    dp.is_riichi = true;
    dp.is_menzen = true;
    dp.score = 25000;

    // 待ち確認: 13枚（ツモ牌除く）
    auto counts_13 = hand_utils::make_type_counts(dp.hand);
    counts_13[91 / 4]--;  // ツモ牌を引く
    auto waits_before = hand_utils::get_waits(counts_13);
    ASSERT_FALSE(waits_before.empty()) << "Should be tenpai";

    // 暗槓後の10枚
    auto counts_after = hand_utils::make_type_counts(dp.hand);
    counts_after[22] -= 4;  // 5s type=22
    auto waits_after = hand_utils::get_waits(counts_after);

    ASSERT_EQ(waits_before, waits_after) << "Waits should not change";

    auto actions = engine.get_legal_actions(env);
    bool has_ankan = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Ankan) {
            has_ankan = true;
            break;
        }
    }
    EXPECT_TRUE(has_ankan) << "Riichi ankan with unchanged waits should be allowed";
}

// 立直後暗槓: 待ちが変わる場合は不許可
TEST_F(ChankanTest, RiichiAnkanWaitsChanged) {
    PlayerId dealer = env.round_state.dealer;
    auto& dp = env.round_state.players[dealer];

    // リーチ手（13枚）: 1m 2m2m2m 3m3m3m 4m4m 7p7p 5s5s
    // type0(1m)=1, type1(2m)=3, type2(3m)=3, type3(4m)=2, type15(7p)=2, type22(5s)=2
    // 分解: shuntsu(1m2m3m) + shuntsu(2m3m4m) + shuntsu(2m3m4m) + {7p7p + 5s5s}
    // → shanpon待ち: 7p or 5s
    // ツモ: 2m(id=7, type1) → 4枚揃い → 暗槓検討
    // 暗槓後（10枚）: type0=1, type2=3, type3=2, type15=2, type22=2
    // → テンパイにならないため待ちなし → 待ち変化 → 暗槓不可
    dp.hand.clear();
    dp.hand = {0, 4, 5, 6, 8, 9, 10, 12, 13, 60, 61, 88, 89, 7};
    dp.is_riichi = true;
    dp.is_menzen = true;
    dp.score = 25000;

    // 待ち確認
    auto counts_13 = hand_utils::make_type_counts(dp.hand);
    counts_13[7 / 4]--;  // ツモ牌(2m)を引く
    auto waits_before = hand_utils::get_waits(counts_13);
    ASSERT_FALSE(waits_before.empty()) << "Should be tenpai before ankan";

    auto counts_after = hand_utils::make_type_counts(dp.hand);
    counts_after[1] -= 4;  // 2m type=1
    auto waits_after = hand_utils::get_waits(counts_after);

    // 待ちが異なることを確認
    ASSERT_NE(waits_before, waits_after) << "Waits should differ for this test setup";

    auto actions = engine.get_legal_actions(env);
    bool has_ankan = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Ankan) {
            has_ankan = true;
            break;
        }
    }
    EXPECT_FALSE(has_ankan) << "Riichi ankan with changed waits should NOT be allowed";
}
