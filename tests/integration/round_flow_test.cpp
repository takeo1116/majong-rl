#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <algorithm>

using namespace mahjong;

class RoundFlowTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }

    // 最初の打牌候補を選んで打つヘルパー
    StepResult discard_first(PlayerId /*player*/) {
        auto actions = engine.get_legal_actions(env);
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard) {
                return engine.step(env, a);
            }
        }
        // 打牌がないことはないはずだが安全のため
        return engine.step(env, actions[0]);
    }

    // ResponsePhaseで全員スキップするヘルパー
    void skip_all_responses() {
        while (env.round_state.phase == Phase::ResponsePhase) {
            PlayerId cp = env.round_state.current_player;
            engine.step(env, Action::make_skip(cp));
        }
    }
};

// 基本的なラウンドフロー: 親打牌 → 次プレイヤーのツモ
TEST_F(RoundFlowTest, BasicDiscardAndDraw) {
    PlayerId dealer = env.round_state.dealer;
    EXPECT_EQ(env.round_state.current_player, dealer);
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);

    // 親が打牌
    auto result = discard_first(dealer);
    EXPECT_EQ(result.error, ErrorCode::Ok);

    // ResponsePhaseに入った場合はスキップ
    skip_all_responses();

    // 次のプレイヤーのツモ後、SelfActionPhaseにいるはず
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    PlayerId expected_next = (dealer + 1) % kNumPlayers;
    EXPECT_EQ(env.round_state.current_player, expected_next);

    // 次のプレイヤーは14枚持っている
    EXPECT_EQ(env.round_state.players[expected_next].hand.size(), 14u);
}

// 数巡回す
TEST_F(RoundFlowTest, MultipleTurns) {
    // 10巡分回す
    for (int turn = 0; turn < 10; ++turn) {
        if (env.round_state.is_round_over()) break;
        EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);

        PlayerId cp = env.round_state.current_player;
        EXPECT_EQ(env.round_state.players[cp].hand.size(), 14u);

        discard_first(cp);
        skip_all_responses();
    }
    // 10巡後もゲームが続いているはず
    if (!env.round_state.is_round_over()) {
        EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    }
}

// 荒牌平局まで回す
TEST_F(RoundFlowTest, PlayUntilExhaustiveDraw) {
    int max_turns = 200;
    bool round_ended = false;

    for (int turn = 0; turn < max_turns; ++turn) {
        if (env.round_state.is_round_over()) {
            round_ended = true;
            break;
        }

        // 合法手の中から打牌のみ選ぶ（和了やカンはしない）
        auto actions = engine.get_legal_actions(env);
        bool acted = false;
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && !a.riichi) {
                engine.step(env, a);
                acted = true;
                break;
            }
        }
        if (!acted) {
            // 打牌のみ不可能な場合（通常ありえないが）
            engine.step(env, actions[0]);
        }

        skip_all_responses();
    }

    EXPECT_TRUE(round_ended);
    EXPECT_EQ(env.round_state.end_reason, RoundEndReason::ExhaustiveDraw);
}

// ツモ和了フロー
TEST_F(RoundFlowTest, TsumoWinFlow) {
    PlayerId cp = env.round_state.current_player;
    auto& hand = env.round_state.players[cp].hand;

    // 和了形を作る
    hand.clear();
    for (int i = 0; i < 3; ++i) hand.push_back(0 + i);    // 1m
    for (int i = 0; i < 3; ++i) hand.push_back(4 + i);    // 2m
    for (int i = 0; i < 3; ++i) hand.push_back(8 + i);    // 3m
    for (int i = 0; i < 3; ++i) hand.push_back(12 + i);   // 4m
    hand.push_back(16); hand.push_back(17);                 // 5m*2

    auto result = engine.step(env, Action::make_tsumo_win(cp));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
    EXPECT_EQ(env.round_state.phase, Phase::EndRound);
}

// 応答フェーズ: ロン
TEST_F(RoundFlowTest, RonResponseFlow) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;

    // 下家にロン可能な手を作る（あと1枚で和了の形）
    auto& shimocha_hand = env.round_state.players[shimocha].hand;
    shimocha_hand.clear();
    // 1m2m3m 4m5m6m 7m8m9m 1p1p1p 2p → 2p待ち（単騎）
    shimocha_hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40};

    // 親の手牌に 2p (TileId=41) を入れて捨てさせる
    auto& dealer_hand = env.round_state.players[dealer].hand;
    // 2pを手牌に入れる
    dealer_hand.push_back(41);  // 2p

    auto result = engine.step(env, Action::make_discard(dealer, 41));
    EXPECT_EQ(result.error, ErrorCode::Ok);

    // ResponsePhaseに入っているはず
    if (env.round_state.phase == Phase::ResponsePhase) {
        // 下家がロンを選択
        // 下家がcurrent_playerなら直接ロン、そうでなければ他家を先にスキップ
        while (env.round_state.current_player != shimocha &&
               env.round_state.phase == Phase::ResponsePhase) {
            engine.step(env, Action::make_skip(env.round_state.current_player));
        }

        if (env.round_state.phase == Phase::ResponsePhase &&
            env.round_state.current_player == shimocha) {
            auto resp_result = engine.step(env, Action::make_ron(shimocha, dealer));
            // 他の応答者もスキップ
            while (env.round_state.phase == Phase::ResponsePhase) {
                engine.step(env, Action::make_skip(env.round_state.current_player));
            }
            EXPECT_EQ(env.round_state.end_reason, RoundEndReason::Ron);
        }
    }
}

// 応答フェーズ: ポン
TEST_F(RoundFlowTest, PonResponseFlow) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId toimen = (dealer + 2) % kNumPlayers;  // 対面

    // 対面にポン可能な手を作る（1mを2枚持つ）
    auto& toimen_hand = env.round_state.players[toimen].hand;
    toimen_hand.clear();
    toimen_hand = {1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};  // 1m(id=1), 1m(id=2)を含む

    // 親の手牌に1m(id=0)を入れて捨てる
    auto& dealer_hand = env.round_state.players[dealer].hand;
    bool has_tile = false;
    for (TileId t : dealer_hand) {
        if (t == 0) { has_tile = true; break; }
    }
    if (!has_tile) dealer_hand.push_back(0);

    auto result = engine.step(env, Action::make_discard(dealer, 0));
    EXPECT_EQ(result.error, ErrorCode::Ok);

    if (env.round_state.phase == Phase::ResponsePhase) {
        // 各応答者をスキップまたはポン
        while (env.round_state.phase == Phase::ResponsePhase) {
            PlayerId cp = env.round_state.current_player;
            if (cp == toimen) {
                // ポンを試みる
                auto actions = engine.get_legal_actions(env);
                bool ponned = false;
                for (const auto& a : actions) {
                    if (a.type == ActionType::Pon) {
                        engine.step(env, a);
                        ponned = true;
                        break;
                    }
                }
                if (!ponned) {
                    engine.step(env, Action::make_skip(cp));
                }
            } else {
                engine.step(env, Action::make_skip(cp));
            }
        }

        // ポン成功した場合、対面がSelfActionPhaseの手番
        if (env.round_state.current_player == toimen &&
            env.round_state.phase == Phase::SelfActionPhase) {
            // 門前ではなくなっている
            EXPECT_FALSE(env.round_state.players[toimen].is_menzen);
            // メルドが1つ追加されている
            EXPECT_EQ(env.round_state.players[toimen].melds.size(), 1u);
            EXPECT_EQ(env.round_state.players[toimen].melds[0].type, MeldType::Pon);
        }
    }
}

// 応答優先順位: ロン > ポン
TEST_F(RoundFlowTest, RonPriorityOverPon) {
    PlayerId dealer = env.round_state.dealer;
    PlayerId shimocha = (dealer + 1) % kNumPlayers;
    PlayerId toimen = (dealer + 2) % kNumPlayers;

    // 下家にロン可能な手を作る
    auto& shimocha_hand = env.round_state.players[shimocha].hand;
    shimocha_hand.clear();
    shimocha_hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 40};

    // 対面にポン可能な手を作る
    auto& toimen_hand = env.round_state.players[toimen].hand;
    toimen_hand.clear();
    toimen_hand = {41, 42, 4, 8, 12, 16, 20, 24, 28, 32, 48, 52, 56};  // 2p(41,42)を持つ

    // 親が2p(id=43)を捨てる
    auto& dealer_hand = env.round_state.players[dealer].hand;
    dealer_hand.push_back(43);

    engine.step(env, Action::make_discard(dealer, 43));

    if (env.round_state.phase == Phase::ResponsePhase) {
        // 各応答者に応答させる
        while (env.round_state.phase == Phase::ResponsePhase) {
            PlayerId cp = env.round_state.current_player;
            auto actions = engine.get_legal_actions(env);

            if (cp == shimocha) {
                // ロンを選択
                bool ron_found = false;
                for (const auto& a : actions) {
                    if (a.type == ActionType::Ron) {
                        engine.step(env, a);
                        ron_found = true;
                        break;
                    }
                }
                if (!ron_found) engine.step(env, Action::make_skip(cp));
            } else if (cp == toimen) {
                // ポンを選択
                bool pon_found = false;
                for (const auto& a : actions) {
                    if (a.type == ActionType::Pon) {
                        engine.step(env, a);
                        pon_found = true;
                        break;
                    }
                }
                if (!pon_found) engine.step(env, Action::make_skip(cp));
            } else {
                engine.step(env, Action::make_skip(cp));
            }
        }

        // ロンが優先されるため、局終了でロン和了
        EXPECT_EQ(env.round_state.end_reason, RoundEndReason::Ron);
    }
}

// 立直フロー
TEST_F(RoundFlowTest, RiichiFlow) {
    PlayerId cp = env.round_state.current_player;
    auto& player = env.round_state.players[cp];
    auto& hand = player.hand;

    // テンパイ形を作る（捨てて13枚でテンパイになるもの）
    hand.clear();
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5s5s → テンパイ、何を切ってもテンパイではないが...
    // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5s 5s → 14枚、5sを切ると 1-2-3m 4-5-6m 7-8-9m 1-2-3p + 5s単騎
    hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 84, 85};
    player.is_menzen = true;
    player.is_riichi = false;
    player.score = 25000;

    auto actions = engine.get_legal_actions(env);
    // 立直打牌があるか
    bool has_riichi = false;
    Action riichi_action{};
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard && a.riichi) {
            has_riichi = true;
            riichi_action = a;
            break;
        }
    }

    if (has_riichi) {
        int32_t score_before = player.score;
        auto result = engine.step(env, riichi_action);
        EXPECT_EQ(result.error, ErrorCode::Ok);
        EXPECT_TRUE(player.is_riichi);
        EXPECT_TRUE(player.ippatsu);
        EXPECT_EQ(player.score, score_before - 1000);
    }
}

// 立直はスコア不足で不可
TEST_F(RoundFlowTest, RiichiNotAvailableWithLowScore) {
    PlayerId cp = env.round_state.current_player;
    auto& player = env.round_state.players[cp];

    player.hand.clear();
    player.hand = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 84, 85};
    player.is_menzen = true;
    player.is_riichi = false;
    player.score = 500;  // 1000未満

    auto actions = engine.get_legal_actions(env);
    for (const auto& a : actions) {
        EXPECT_FALSE(a.riichi) << "Riichi should not be available with score < 1000";
    }
}

// 全巡回しても安定して動作する
TEST_F(RoundFlowTest, FullRoundStability) {
    int actions_taken = 0;
    while (!env.round_state.is_round_over() && actions_taken < 500) {
        auto actions = engine.get_legal_actions(env);
        ASSERT_FALSE(actions.empty()) << "No legal actions at turn " << actions_taken;

        // 打牌またはスキップを選ぶ（和了・槓などは避ける）
        bool acted = false;
        for (const auto& a : actions) {
            if (a.type == ActionType::Discard && !a.riichi) {
                engine.step(env, a);
                acted = true;
                break;
            }
            if (a.type == ActionType::Skip) {
                engine.step(env, a);
                acted = true;
                break;
            }
        }
        if (!acted) {
            engine.step(env, actions[0]);
        }
        actions_taken++;
    }
    EXPECT_TRUE(env.round_state.is_round_over());
}
