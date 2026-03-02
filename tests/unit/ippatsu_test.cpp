#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "core/environment_state.h"

using namespace mahjong;

class IppatsuTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    // 親にテンパイ形を設定して立直させるヘルパー
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5s5s → 14枚、5sを切ってテンパイ
    void setup_riichi_for_dealer() {
        PlayerId dealer = env.round_state.dealer;
        auto& player = env.round_state.players[dealer];
        player.hand.clear();
        player.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 84, 85};
        player.is_menzen = true;
        player.is_riichi = false;
        player.score = 25000;
    }

    // 立直打牌アクションを探して返す
    Action find_riichi_action() {
        auto actions = engine.get_legal_actions(env);
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && a.riichi) {
                return a;
            }
        }
        return Action::make_skip(0);  // fallback
    }

    // 全応答者にスキップさせて次のツモに進む
    void skip_all_responses(StepResult& result) {
        while (env.round_state.phase == Phase::ResponsePhase) {
            PlayerId cp = env.round_state.current_player;
            result = engine.step(env, Action::make_skip(cp));
        }
    }
};

// 立直直後の打牌で ippatsu = true が設定される
TEST_F(IppatsuTest, IppatsuSetOnRiichi) {
    setup_riichi_for_dealer();
    PlayerId dealer = env.round_state.dealer;
    auto& player = env.round_state.players[dealer];

    Action riichi_action = find_riichi_action();
    ASSERT_EQ(riichi_action.type, ActionType::Discard);
    ASSERT_TRUE(riichi_action.riichi);

    auto result = engine.step(env, riichi_action);
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(player.is_riichi);
    EXPECT_TRUE(player.ippatsu);
}

// 立直後、他家がスキップし続けて一巡回ったら ippatsu が消える
TEST_F(IppatsuTest, IppatsuExpiresAfterOneRound) {
    setup_riichi_for_dealer();
    PlayerId dealer = env.round_state.dealer;
    auto& player = env.round_state.players[dealer];

    // 立直宣言
    Action riichi_action = find_riichi_action();
    auto result = engine.step(env, riichi_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(player.ippatsu);

    // 応答フェーズ → 全員スキップ
    skip_all_responses(result);

    // 他家3人の手番を進める（ツモ→打牌→応答スキップ）
    for (int i = 0; i < 3; ++i) {
        if (env.round_state.is_round_over()) break;
        ASSERT_EQ(env.round_state.phase, Phase::SelfActionPhase);
        PlayerId cp = env.round_state.current_player;

        // 立直者ではない他家の手番
        // この時点では立直者の ippatsu はまだ有効
        if (cp == dealer) break;  // 立直者に戻ってきたら終了
        EXPECT_TRUE(player.ippatsu) << "ippatsu should remain during other players' turns";

        auto actions = engine.get_legal_actions(env);
        // 普通に打牌
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && !a.riichi) {
                result = engine.step(env, a);
                break;
            }
        }
        // 応答スキップ
        skip_all_responses(result);
    }

    // ここで立直者に戻ってきた（またはラウンド終了）
    if (env.round_state.is_round_over()) {
        GTEST_SKIP() << "Round ended before dealer's next turn";
    }

    ASSERT_EQ(env.round_state.current_player, dealer);
    // ツモした時点ではまだ ippatsu が有効（ツモ和了できる）
    EXPECT_TRUE(player.ippatsu);

    // 打牌する（ツモ和了せず、ツモ切り）
    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard) {
            result = engine.step(env, a);
            break;
        }
    }

    // 打牌後、ippatsu は消えている
    EXPECT_FALSE(player.ippatsu) << "ippatsu should expire after dealer's next discard";
}

// 鳴きが入ると全員の ippatsu が消える（既存挙動の維持確認）
TEST_F(IppatsuTest, IppatsuClearedByCall) {
    setup_riichi_for_dealer();
    PlayerId dealer = env.round_state.dealer;
    auto& dealer_player = env.round_state.players[dealer];

    // 下家にポン可能な手を作る
    PlayerId shimocha = (dealer + 1) % kNumPlayers;
    auto& shimocha_hand = env.round_state.players[shimocha].hand;
    shimocha_hand.clear();
    // 5s(type=22)のポン材を持たせる: 86, 87 + 残りは適当
    shimocha_hand = {86, 87, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};

    // 親の手牌に 5s(id=85) を確保
    auto& dealer_hand = env.round_state.players[dealer].hand;
    dealer_hand.clear();
    dealer_hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 84, 85};

    // 立直宣言（5sの85を切る）
    auto actions = engine.get_legal_actions(env);
    Action riichi_action{};
    bool found = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && a.riichi && a.tile == 85) {
            riichi_action = a;
            found = true;
            break;
        }
    }

    if (!found) {
        // 5s切りの立直がなければ任意の立直でOK
        riichi_action = find_riichi_action();
    }

    if (riichi_action.type != ActionType::Discard || !riichi_action.riichi) {
        GTEST_SKIP() << "Could not find riichi action";
    }

    auto result = engine.step(env, riichi_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(dealer_player.ippatsu);

    // 応答フェーズでポンを試みる
    if (env.round_state.phase != Phase::ResponsePhase) {
        GTEST_SKIP() << "Not in response phase";
    }

    while (env.round_state.phase == Phase::ResponsePhase) {
        PlayerId cp = env.round_state.current_player;
        auto resp_actions = engine.get_legal_actions(env);

        bool ponned = false;
        if (cp == shimocha) {
            for (const auto& a : resp_actions) {
                if (a.type == ActionType::Pon) {
                    result = engine.step(env, a);
                    ponned = true;
                    break;
                }
            }
        }
        if (!ponned) {
            result = engine.step(env, Action::make_skip(cp));
        }
    }

    // ポンが成立していれば ippatsu は消えている
    EXPECT_FALSE(dealer_player.ippatsu) << "ippatsu should be cleared by pon";
}

// 立直者がツモ和了する場面では ippatsu = true が残っている
TEST_F(IppatsuTest, IppatsuAvailableForTsumoWin) {
    setup_riichi_for_dealer();
    PlayerId dealer = env.round_state.dealer;
    auto& player = env.round_state.players[dealer];

    // 立直宣言
    Action riichi_action = find_riichi_action();
    auto result = engine.step(env, riichi_action);
    ASSERT_EQ(result.error, ErrorCode::Ok);
    ASSERT_TRUE(player.ippatsu);

    // 応答スキップ
    skip_all_responses(result);

    // 他家3人を進める
    for (int i = 0; i < 3; ++i) {
        if (env.round_state.is_round_over()) break;
        if (env.round_state.current_player == dealer) break;

        auto actions = engine.get_legal_actions(env);
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && !a.riichi) {
                result = engine.step(env, a);
                break;
            }
        }
        skip_all_responses(result);
    }

    if (env.round_state.is_round_over()) {
        GTEST_SKIP() << "Round ended before dealer's next turn";
    }

    // 立直者に戻ってきた。ツモした時点では ippatsu はまだ有効
    ASSERT_EQ(env.round_state.current_player, dealer);
    EXPECT_TRUE(player.ippatsu) << "ippatsu should still be true at draw time";

    // ツモ和了が合法手に含まれるか確認（和了形でないかもしれないが、ippatsu フラグ自体は残る）
    // フラグが残っていることが重要
}

// 二人同時立直で、各自の ippatsu が独立に管理される
TEST_F(IppatsuTest, MultipleRiichiIndependentIppatsu) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId next = (dealer + 1) % kNumPlayers;

    // 親にテンパイ手を設定
    auto& dealer_player = env.round_state.players[dealer];
    dealer_player.hand.clear();
    dealer_player.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 84, 85};
    dealer_player.is_menzen = true;
    dealer_player.is_riichi = false;
    dealer_player.score = 25000;

    // 親が立直
    Action riichi_action = find_riichi_action();
    ASSERT_TRUE(riichi_action.riichi);
    auto result = engine.step(env, riichi_action);
    ASSERT_TRUE(dealer_player.ippatsu);

    // 応答スキップ
    skip_all_responses(result);

    if (env.round_state.is_round_over()) {
        GTEST_SKIP() << "Round ended";
    }

    // 次の手番がnextかどうか確認
    if (env.round_state.current_player != next) {
        GTEST_SKIP() << "Unexpected player order";
    }

    // 下家にもテンパイ手を設定して立直させる
    auto& next_player = env.round_state.players[next];
    // 現在14枚持っているはず（ツモ済み）
    next_player.hand.clear();
    next_player.hand = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 88, 89};
    next_player.is_menzen = true;
    next_player.is_riichi = false;
    next_player.score = 25000;

    auto actions2 = engine.get_legal_actions(env);
    Action riichi2{};
    bool found2 = false;
    for (const auto& a : actions2) {
        if (a.type == ActionType::Discard && a.riichi) {
            riichi2 = a;
            found2 = true;
            break;
        }
    }
    if (!found2) {
        GTEST_SKIP() << "No riichi action for next player";
    }

    result = engine.step(env, riichi2);
    ASSERT_TRUE(next_player.ippatsu);
    // 親も立直中なので ippatsu は立直宣言打牌ではクリアされない
    // （next の打牌は立直宣言なので !action.riichi は false）
    // ただし、親の ippatsu は鳴きがない限り維持される

    // この時点で両者 ippatsu = true
    EXPECT_TRUE(dealer_player.ippatsu);
    EXPECT_TRUE(next_player.ippatsu);

    // 応答スキップ
    skip_all_responses(result);

    // 残り2人の手番を進める
    for (int i = 0; i < 2; ++i) {
        if (env.round_state.is_round_over()) break;
        PlayerId cp = env.round_state.current_player;
        if (cp == dealer || cp == next) break;

        auto acts = engine.get_legal_actions(env);
        for (const auto& a : acts) {
            if (a.type == ActionType::Discard && !a.riichi) {
                result = engine.step(env, a);
                break;
            }
        }
        skip_all_responses(result);
    }

    if (env.round_state.is_round_over()) {
        GTEST_SKIP() << "Round ended";
    }

    // 親に戻ってきた場合: ツモ時点では ippatsu は有効
    if (env.round_state.current_player == dealer) {
        EXPECT_TRUE(dealer_player.ippatsu);
        // 打牌すると親の ippatsu が消える
        auto acts = engine.get_legal_actions(env);
        for (const auto& a : acts) {
            if (a.type == ActionType::Discard) {
                result = engine.step(env, a);
                break;
            }
        }
        EXPECT_FALSE(dealer_player.ippatsu) << "Dealer ippatsu should expire";
        // 下家の ippatsu はまだ有効（次巡未到達）
        EXPECT_TRUE(next_player.ippatsu) << "Next player ippatsu should still be valid";
    }
}
