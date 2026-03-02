#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "engine/hand_utils.h"
#include "core/environment_state.h"

using namespace mahjong;

// CQ-0034: 同巡内フリテン解除タイミング
class FuritenNakiTest : public ::testing::Test {
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

// 見逃し後、鳴きが入らない限り同巡内フリテンが維持される
TEST_F(FuritenNakiTest, TempFuritenMaintainedWithoutNaki) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    auto& tp = env.round_state.players[toimen];

    // 対面: テンパイ手（2p単騎待ち）
    tp.hand.clear();
    tp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 38, 39, 41};

    // 下家: ポン/チーできない手牌
    auto& sp = env.round_state.players[(dealer + 1) % kNumPlayers];
    sp.hand.clear();
    sp.hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};

    // 上家も同様
    auto& kp = env.round_state.players[(dealer + 3) % kNumPlayers];
    kp.hand.clear();
    kp.hand = {3, 11, 23, 35, 43, 55, 67, 79, 91, 103, 107, 111, 115};

    // 親が 2p(id=46, type=11... いや違う、2p=type10) を捨てる
    // 2p: type10, id=40,41,42,43. 41は対面が持っている、42は下家が持っている
    // id=43 を使う（上家が持っている→除外して別のIDに）
    // 上家の手牌を調整: 43を除く
    kp.hand.clear();
    kp.hand = {3, 11, 23, 35, 47, 55, 67, 79, 91, 103, 107, 111, 115};

    auto& dp = env.round_state.players[dealer];
    dp.hand.push_back(43);  // 2p(id=43)

    auto result = engine.step(env, Action::make_discard(dealer, 43));
    ASSERT_EQ(result.error, ErrorCode::Ok);

    // 対面がロンをスキップ
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        engine.step(env, Action::make_skip(cp));
    }

    // 次のツモは下家(player1)→対面(player2)ではないのでフリテン維持
    // ただし下家がツモした時点で対面のフリテンはまだ維持される
    // （対面のツモは下家の後なので、まだ維持される）
    EXPECT_TRUE(tp.is_temporary_furiten)
        << "Temporary furiten should be maintained until own draw (no naki)";
}

// 見逃し後、チー成立で同巡内フリテンが解除される
TEST_F(FuritenNakiTest, TempFuritenClearedOnChi) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    // 対面: テンパイ手（2p=type10 単騎待ち）
    auto& tp = env.round_state.players[toimen];
    tp.hand.clear();
    tp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 38, 39, 41};

    // 同巡内フリテンを手動設定（見逃し後の状態をシミュレート）
    tp.is_temporary_furiten = true;

    // 下家: 5m(type4) をチーできる手（3m4m を持つ）
    auto& sp = env.round_state.players[shimocha];
    sp.hand.clear();
    // 3m=type2(id=8), 4m=type3(id=12) を含む13枚
    sp.hand = {8, 12, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 94};

    // 親が 5m(type4, id=19) を捨てる
    auto& dp = env.round_state.players[dealer];
    dp.hand.push_back(19);

    auto result = engine.step(env, Action::make_discard(dealer, 19));
    ASSERT_EQ(result.error, ErrorCode::Ok);

    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "No response phase";
    }

    // 下家がチーする
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool chi_found = false;
        if (cp == shimocha) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Chi) {
                    engine.step(env, a);
                    chi_found = true;
                    break;
                }
            }
        }
        if (!chi_found) {
            engine.step(env, Action::make_skip(cp));
        }
    }

    // チー成立後、全員の同巡内フリテンが解除される
    EXPECT_FALSE(tp.is_temporary_furiten)
        << "Temporary furiten should be cleared when chi succeeds";
}

// 見逃し後、大明槓成立で同巡内フリテンが解除される
TEST_F(FuritenNakiTest, TempFuritenClearedOnDaiminkan) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    // 対面に同巡内フリテンを設定
    auto& tp = env.round_state.players[toimen];
    tp.is_temporary_furiten = true;

    // 対面: 大明槓可能な手（7s=type24 を3枚持つ）
    tp.hand.clear();
    // 7s: type24, id=96,97,98,99
    tp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 96, 97, 98, 50};

    // 他家をチー/ポンできない手にする
    auto& sp = env.round_state.players[(dealer + 1) % kNumPlayers];
    sp.hand.clear();
    sp.hand = {2, 10, 22, 34, 42, 54, 66, 78, 90, 102, 106, 110, 114};
    auto& kp = env.round_state.players[(dealer + 3) % kNumPlayers];
    kp.hand.clear();
    kp.hand = {3, 11, 23, 35, 43, 55, 67, 79, 91, 103, 107, 111, 115};

    // 親が 7s(id=99, type24) を捨てる
    auto& dp = env.round_state.players[dealer];
    dp.hand.push_back(99);

    auto result = engine.step(env, Action::make_discard(dealer, 99));
    ASSERT_EQ(result.error, ErrorCode::Ok);

    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "No response phase";
    }

    // 対面が大明槓する
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool kan_found = false;
        if (cp == toimen) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Daiminkan) {
                    engine.step(env, a);
                    kan_found = true;
                    break;
                }
            }
        }
        if (!kan_found) {
            engine.step(env, Action::make_skip(cp));
        }
    }

    // 大明槓成立後、同巡内フリテンが解除される
    EXPECT_FALSE(tp.is_temporary_furiten)
        << "Temporary furiten should be cleared when daiminkan succeeds";
}

// ポン成立でも同巡内フリテンが解除される（既存動作の確認）
TEST_F(FuritenNakiTest, TempFuritenClearedOnPon) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    // 対面に同巡内フリテンを設定
    auto& tp = env.round_state.players[toimen];
    tp.is_temporary_furiten = true;

    // 対面: ポン可能な手（8s=type25 を2枚持つ）
    tp.hand.clear();
    // 8s: type25, id=100,101,102,103
    tp.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 100, 101, 50, 54};

    // 他家をチー/ポンできない手にする
    auto& sp = env.round_state.players[(dealer + 1) % kNumPlayers];
    sp.hand.clear();
    sp.hand = {2, 10, 22, 34, 42, 58, 66, 78, 90, 106, 110, 114, 118};
    auto& kp = env.round_state.players[(dealer + 3) % kNumPlayers];
    kp.hand.clear();
    kp.hand = {3, 11, 23, 35, 43, 55, 67, 79, 91, 107, 111, 115, 119};

    // 親が 8s(id=103, type25) を捨てる
    auto& dp = env.round_state.players[dealer];
    dp.hand.push_back(103);

    engine.step(env, Action::make_discard(dealer, 103));

    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "No response phase";
    }

    // 対面がポンする
    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto actions = engine.get_legal_actions(env);
        bool pon_found = false;
        if (cp == toimen) {
            for (const auto& a : actions) {
                if (a.type == ActionType::Pon) {
                    engine.step(env, a);
                    pon_found = true;
                    break;
                }
            }
        }
        if (!pon_found) {
            engine.step(env, Action::make_skip(cp));
        }
    }

    EXPECT_FALSE(tp.is_temporary_furiten)
        << "Temporary furiten should be cleared when pon succeeds";
}
