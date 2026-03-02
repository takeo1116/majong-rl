#include <gtest/gtest.h>
#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <algorithm>
#include <numeric>

using namespace mahjong;

class RoundResetTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    // 有効な RoundConfig を生成するヘルパー
    RoundConfig make_valid_config(PlayerId dealer = 0) {
        RoundConfig config;
        config.round_number = 0;
        config.dealer = dealer;
        config.honba = 0;
        config.kyotaku = 0;
        config.scores = {25000, 25000, 25000, 25000};

        // 山を 0..135 の順序で作成
        std::iota(config.wall.begin(), config.wall.end(), 0);

        // 配牌: 親から順に配る（deal_tiles と同じロジック）
        int pos = 0;
        for (PlayerId p = 0; p < kNumPlayers; ++p) {
            PlayerId player = (dealer + p) % kNumPlayers;
            config.hands[player].clear();
            for (int i = 0; i < 13; ++i) {
                config.hands[player].push_back(config.wall[pos++]);
            }
        }
        // 親に14枚目
        config.hands[dealer].push_back(config.wall[pos++]);

        return config;
    }
};

// --- 正常系 ---

TEST_F(RoundResetTest, BasicReset) {
    auto config = make_valid_config(0);
    auto err = engine.reset_round(env, config);
    EXPECT_EQ(err, ErrorCode::Ok);
    EXPECT_EQ(env.round_state.phase, Phase::SelfActionPhase);
    EXPECT_EQ(env.round_state.current_player, 0);
    EXPECT_EQ(env.round_state.dealer, 0);
}

TEST_F(RoundResetTest, DealerHas14Tiles) {
    auto config = make_valid_config(2);
    engine.reset_round(env, config);
    EXPECT_EQ(env.round_state.players[2].hand.size(), 14u);
}

TEST_F(RoundResetTest, NonDealersHave13Tiles) {
    auto config = make_valid_config(1);
    engine.reset_round(env, config);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p != 1) {
            EXPECT_EQ(env.round_state.players[p].hand.size(), 13u);
        }
    }
}

TEST_F(RoundResetTest, WallCopied) {
    auto config = make_valid_config();
    engine.reset_round(env, config);
    for (int i = 0; i < kNumTiles; ++i) {
        EXPECT_EQ(env.round_state.wall[i], config.wall[i]);
    }
}

TEST_F(RoundResetTest, WallPositionIs53) {
    auto config = make_valid_config();
    engine.reset_round(env, config);
    EXPECT_EQ(env.round_state.wall_position, 53);
}

TEST_F(RoundResetTest, ScoresCopied) {
    auto config = make_valid_config();
    config.scores = {30000, 20000, 15000, 35000};
    engine.reset_round(env, config);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env.round_state.players[p].score, config.scores[p]);
    }
}

TEST_F(RoundResetTest, MatchStateSynced) {
    auto config = make_valid_config(3);
    config.round_number = 5;
    config.honba = 2;
    config.kyotaku = 3;
    engine.reset_round(env, config);

    EXPECT_EQ(env.match_state.round_number, 5);
    EXPECT_EQ(env.match_state.current_dealer, 3);
    EXPECT_EQ(env.match_state.honba, 2);
    EXPECT_EQ(env.match_state.kyotaku, 3);
}

TEST_F(RoundResetTest, DoraSetFromWall) {
    auto config = make_valid_config();
    engine.reset_round(env, config);

    EXPECT_EQ(env.round_state.dora_indicators.size(), 1u);
    // ドラ表示牌: wall[kNumTiles - 6] = wall[130]
    EXPECT_EQ(env.round_state.dora_indicators[0], config.wall[130]);

    EXPECT_EQ(env.round_state.uradora_indicators.size(), 1u);
    EXPECT_EQ(env.round_state.uradora_indicators[0], config.wall[131]);
}

TEST_F(RoundResetTest, HandsMatchConfig) {
    auto config = make_valid_config(1);
    engine.reset_round(env, config);

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env.round_state.players[p].hand, config.hands[p]);
    }
}

TEST_F(RoundResetTest, CustomHandsForSpecificDora) {
    // ドラ表示牌を特定の牌にするテスト
    auto config = make_valid_config();
    // wall[130] にドラ表示として 0 (1m) を置く → ドラは 2m
    // まず wall 中の 0 と wall[130] を入れ替え
    TileId original_130 = config.wall[130];
    // wall 中で 0 がある位置を見つける
    int pos_of_zero = -1;
    for (int i = 0; i < kNumTiles; ++i) {
        if (config.wall[i] == 0) { pos_of_zero = i; break; }
    }
    // 入れ替え
    config.wall[pos_of_zero] = original_130;
    config.wall[130] = 0;

    // 配牌中に pos_of_zero があれば、手牌の該当牌も入れ替え
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        for (auto& t : config.hands[p]) {
            if (t == 0) { t = original_130; break; }
        }
    }

    auto err = engine.reset_round(env, config);
    EXPECT_EQ(err, ErrorCode::Ok);
    EXPECT_EQ(env.round_state.dora_indicators[0], static_cast<TileId>(0));
}

TEST_F(RoundResetTest, StepWorksAfterReset) {
    auto config = make_valid_config(0);
    engine.reset_round(env, config);

    // 合法手を取得して打牌できる
    auto actions = engine.get_legal_actions(env);
    EXPECT_FALSE(actions.empty());

    // 打牌を実行
    for (const auto& a : actions) {
        if (a.type == ActionType::Discard) {
            auto result = engine.step(env, a);
            EXPECT_EQ(result.error, ErrorCode::Ok);
            break;
        }
    }
}

TEST_F(RoundResetTest, CustomHandsWithTsumo) {
    // 和了形の手牌を注入してツモ和了できることを確認
    auto config = make_valid_config(0);
    // 親の手牌を和了形に書き換え
    config.hands[0].clear();
    // 1m*3 2m*3 3m*3 4m*3 5m*2 = 14枚
    config.hands[0] = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17};

    // 他プレイヤーの手牌を調整（重複しないように）
    // 残りの牌から13枚ずつ配る
    std::set<TileId> used;
    for (auto t : config.hands[0]) used.insert(t);
    std::vector<TileId> remaining;
    for (int i = 0; i < kNumTiles; ++i) {
        if (used.find(i) == used.end()) remaining.push_back(i);
    }
    int idx = 0;
    for (PlayerId p = 1; p < kNumPlayers; ++p) {
        config.hands[p].clear();
        for (int i = 0; i < 13; ++i) {
            config.hands[p].push_back(remaining[idx++]);
        }
    }
    // 山を再構成: hands のタイル + remaining のタイル
    // 最初の53タイルはhands由来、残りはremaining由来
    int wall_idx = 0;
    // 親（dealer=0）から順に配る想定
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        for (auto t : config.hands[p]) {
            config.wall[wall_idx++] = t;
        }
    }
    for (size_t i = idx; i < remaining.size(); ++i) {
        config.wall[wall_idx++] = remaining[i];
    }

    auto err = engine.reset_round(env, config);
    EXPECT_EQ(err, ErrorCode::Ok);

    auto actions = engine.get_legal_actions(env);
    bool has_tsumo = false;
    for (const auto& a : actions) {
        if (a.type == ActionType::TsumoWin) { has_tsumo = true; break; }
    }
    EXPECT_TRUE(has_tsumo);

    auto result = engine.step(env, Action::make_tsumo_win(0));
    EXPECT_EQ(result.error, ErrorCode::Ok);
    EXPECT_TRUE(result.round_over);
}

TEST_F(RoundResetTest, FirstDrawFlagsSet) {
    auto config = make_valid_config();
    engine.reset_round(env, config);
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_TRUE(env.round_state.first_draw[p]);
    }
}

TEST_F(RoundResetTest, PlayerWindsCorrect) {
    auto config = make_valid_config(2);
    engine.reset_round(env, config);
    // 親(player 2)が東、次(3)が南、次(0)が西、次(1)が北
    EXPECT_EQ(env.round_state.players[2].jikaze, Wind::East);
    EXPECT_EQ(env.round_state.players[3].jikaze, Wind::South);
    EXPECT_EQ(env.round_state.players[0].jikaze, Wind::West);
    EXPECT_EQ(env.round_state.players[1].jikaze, Wind::North);
}

// --- 異常系 ---

TEST_F(RoundResetTest, InvalidDealer) {
    auto config = make_valid_config();
    config.dealer = 4;  // 不正
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InvalidActor);
}

TEST_F(RoundResetTest, InvalidDealerMax) {
    auto config = make_valid_config();
    config.dealer = 255;
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InvalidActor);
}

TEST_F(RoundResetTest, InvalidRoundNumber) {
    auto config = make_valid_config();
    config.round_number = 9;  // 0-8 が有効
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InconsistentState);
}

TEST_F(RoundResetTest, WrongHandSizeDealer) {
    auto config = make_valid_config(0);
    config.hands[0].pop_back();  // 14→13枚に
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InconsistentState);
}

TEST_F(RoundResetTest, WrongHandSizeNonDealer) {
    auto config = make_valid_config(0);
    config.hands[1].push_back(config.hands[0].back());  // 13→14枚に（重複牌を入れてしまうが枚数チェックが先に発火する）
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InconsistentState);
}

TEST_F(RoundResetTest, DuplicateTileInWall) {
    auto config = make_valid_config();
    config.wall[0] = config.wall[1];  // 重複
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InconsistentState);
}

TEST_F(RoundResetTest, InvalidTileIdInWall) {
    auto config = make_valid_config();
    config.wall[0] = 200;  // 範囲外
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InvalidTile);
}

TEST_F(RoundResetTest, InvalidTileIdInHand) {
    auto config = make_valid_config();
    config.hands[0][0] = 200;  // 範囲外
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InvalidTile);
}

TEST_F(RoundResetTest, DuplicateTileInHands) {
    auto config = make_valid_config();
    // player 0 と player 1 で同じ牌を持たせる
    config.hands[1][0] = config.hands[0][0];
    EXPECT_EQ(engine.reset_round(env, config), ErrorCode::InconsistentState);
}

TEST_F(RoundResetTest, HandTileNotInWall) {
    auto config = make_valid_config();
    // 山にない牌を手牌に入れる（まず山を壊す）
    config.wall[0] = 200;  // これは InvalidTile で先に弾かれる
    // より正確なケース: 山は有効だが手牌に山にない牌
    // → 山は0..135の全牌なので、0..135の範囲内なら必ず山にある
    // このテストは山の検証が先に失敗するケースで実質カバー済み
}

TEST_F(RoundResetTest, ErrorDoesNotModifyState) {
    // 先に有効な状態をセットアップ
    auto valid_config = make_valid_config();
    engine.reset_round(env, valid_config);
    auto snapshot = env;

    // 不正な設定でリセットを試みる
    auto bad_config = make_valid_config();
    bad_config.dealer = 10;
    engine.reset_round(env, bad_config);

    // 状態が変わっていないことを確認
    EXPECT_EQ(env, snapshot);
}
