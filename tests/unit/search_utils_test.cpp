#include <gtest/gtest.h>
#include "rl/search_utils.h"
#include "engine/state_validator.h"
#include "engine/game_engine.h"
#include "core/environment_state.h"
#include <algorithm>
#include <set>

using namespace mahjong;

class SearchUtilsTest : public ::testing::Test {
protected:
    GameEngine engine;
    EnvironmentState env;

    void SetUp() override {
        engine.reset_match(env, 42, static_cast<PlayerId>(0));
    }
};

// ============================
// clone テスト
// ============================

TEST_F(SearchUtilsTest, CloneEquality) {
    auto cloned = search_utils::clone(env);
    EXPECT_EQ(cloned, env);
}

TEST_F(SearchUtilsTest, CloneIndependence) {
    auto cloned = search_utils::clone(env);

    // 元の env を進める
    PlayerId cp = env.round_state.current_player;
    TileId tile = env.round_state.players[cp].hand[0];
    engine.step(env, Action::make_discard(cp, tile));

    // clone 側は変わっていない
    EXPECT_NE(env.round_state.phase, cloned.round_state.phase);
    EXPECT_NE(env.round_state.players[cp].hand.size(),
              cloned.round_state.players[cp].hand.size());
}

TEST_F(SearchUtilsTest, CloneRngIndependence) {
    auto cloned = search_utils::clone(env);

    // 元の RNG を進める
    env.rng.next_int(100);

    // clone 側の RNG は影響を受けない
    EXPECT_NE(env.rng, cloned.rng);
}

// ============================
// get_hidden_tiles テスト
// ============================

TEST_F(SearchUtilsTest, HiddenTilesDoNotContainObserverHand) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);

    for (TileId t : env.round_state.players[observer].hand) {
        auto it = std::find(hidden.begin(), hidden.end(), t);
        EXPECT_EQ(it, hidden.end()) << "observer の手牌 " << static_cast<int>(t) << " が hidden に含まれている";
    }
}

TEST_F(SearchUtilsTest, HiddenTilesContainOtherHands) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        for (TileId t : env.round_state.players[p].hand) {
            auto it = std::find(hidden.begin(), hidden.end(), t);
            EXPECT_NE(it, hidden.end()) << "他家(player" << static_cast<int>(p)
                << ")の手牌 " << static_cast<int>(t) << " が hidden に含まれていない";
        }
    }
}

TEST_F(SearchUtilsTest, HiddenTilesContainUndrawnWallExcludingIndicators) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);
    const auto& rs = env.round_state;

    // 公開済みドラ表示牌・裏ドラ表示牌の wall インデックスを計算
    std::set<int> indicator_indices;
    for (size_t i = 0; i < rs.dora_indicators.size(); ++i) {
        indicator_indices.insert(kNumTiles - 6 - static_cast<int>(i) * 2);
        indicator_indices.insert(kNumTiles - 5 - static_cast<int>(i) * 2);
    }

    for (int i = rs.wall_position; i < kNumTiles; ++i) {
        if (indicator_indices.count(i) > 0) {
            // ドラ表示牌位置は hidden に含まれない
            auto it = std::find(hidden.begin(), hidden.end(), rs.wall[i]);
            EXPECT_EQ(it, hidden.end()) << "公開済みインジケータ wall[" << i << "] が hidden に含まれている";
        } else {
            auto it = std::find(hidden.begin(), hidden.end(), rs.wall[i]);
            EXPECT_NE(it, hidden.end()) << "山の牌 " << static_cast<int>(rs.wall[i]) << " (wall[" << i << "]) が hidden に含まれていない";
        }
    }
}

TEST_F(SearchUtilsTest, HiddenTilesCorrectCount) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);
    const auto& rs = env.round_state;

    // 他家手牌枚数 + 未ツモ山枚数 - 公開済みインジケータ枚数（山内分）
    int expected = 0;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        expected += rs.players[p].hand_count();
    }
    int undrawn = kNumTiles - rs.wall_position;
    // 公開済みドラ表示牌・裏ドラ表示牌のうち、未ツモ範囲内にあるものを除外
    std::set<int> indicator_indices;
    for (size_t i = 0; i < rs.dora_indicators.size(); ++i) {
        indicator_indices.insert(kNumTiles - 6 - static_cast<int>(i) * 2);
        indicator_indices.insert(kNumTiles - 5 - static_cast<int>(i) * 2);
    }
    int indicators_in_wall = 0;
    for (int idx : indicator_indices) {
        if (idx >= rs.wall_position) ++indicators_in_wall;
    }
    expected += undrawn - indicators_in_wall;
    EXPECT_EQ(static_cast<int>(hidden.size()), expected);
}

// ============================
// determinize テスト
// ============================

TEST_F(SearchUtilsTest, DeterminizePreservesOwnHand) {
    PlayerId observer = 0;
    auto original_hand = env.round_state.players[observer].hand;

    auto env_copy = search_utils::clone(env);
    bool ok = search_utils::determinize(env_copy, observer);
    EXPECT_TRUE(ok);

    EXPECT_EQ(env_copy.round_state.players[observer].hand, original_hand);
}

TEST_F(SearchUtilsTest, DeterminizePreservesPublicInfo) {
    PlayerId observer = 0;
    auto env_copy = search_utils::clone(env);
    bool ok = search_utils::determinize(env_copy, observer);
    EXPECT_TRUE(ok);

    // スコア
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_copy.round_state.players[p].score, env.round_state.players[p].score);
    }
    // ドラ
    EXPECT_EQ(env_copy.round_state.dora_indicators, env.round_state.dora_indicators);
    // 副露
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_copy.round_state.players[p].melds, env.round_state.players[p].melds);
    }
    // 河
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_copy.round_state.players[p].discards, env.round_state.players[p].discards);
    }
}

TEST_F(SearchUtilsTest, DeterminizeMaintains136Tiles) {
    auto env_copy = search_utils::clone(env);
    bool ok = search_utils::determinize(env_copy, 0);
    EXPECT_TRUE(ok);

    // 全136牌がユニーク
    std::set<TileId> all;
    const auto& rs = env_copy.round_state;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        for (TileId t : rs.players[p].hand) all.insert(t);
        for (const auto& m : rs.players[p].melds) {
            for (uint8_t i = 0; i < m.tile_count; ++i) all.insert(m.tiles[i]);
        }
        for (const auto& d : rs.players[p].discards) {
            if (!d.called) all.insert(d.tile);
        }
    }
    for (int i = rs.wall_position; i < kNumTiles; ++i) {
        all.insert(rs.wall[i]);
    }
    EXPECT_EQ(all.size(), static_cast<size_t>(kNumTiles));
}

TEST_F(SearchUtilsTest, DeterminizePassesValidator) {
    auto env_copy = search_utils::clone(env);
    bool ok = search_utils::determinize(env_copy, 0);
    EXPECT_TRUE(ok);

    auto vr = state_validator::validate(env_copy);
    EXPECT_TRUE(vr.valid) << "errors:";
    for (const auto& e : vr.errors) {
        std::cerr << "  " << e << std::endl;
    }
}

TEST_F(SearchUtilsTest, DeterminizePreservesHandCounts) {
    auto env_copy = search_utils::clone(env);
    // 各プレイヤーの手牌枚数を記録
    std::array<int, kNumPlayers> hand_counts;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        hand_counts[p] = env_copy.round_state.players[p].hand_count();
    }

    bool ok = search_utils::determinize(env_copy, 0);
    EXPECT_TRUE(ok);

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        EXPECT_EQ(env_copy.round_state.players[p].hand_count(), hand_counts[p]);
    }
}

// ============================
// CQ-0030: 公開情報牌の除外テスト
// ============================

TEST_F(SearchUtilsTest, HiddenTilesDoNotContainDoraIndicators) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);
    const auto& rs = env.round_state;

    // ドラ表示牌が hidden に含まれていないこと
    for (TileId indicator : rs.dora_indicators) {
        auto it = std::find(hidden.begin(), hidden.end(), indicator);
        EXPECT_EQ(it, hidden.end()) << "ドラ表示牌 " << static_cast<int>(indicator) << " が hidden に含まれている";
    }
}

TEST_F(SearchUtilsTest, HiddenTilesDoNotContainUradoraIndicators) {
    PlayerId observer = 0;
    auto hidden = search_utils::get_hidden_tiles(env, observer);
    const auto& rs = env.round_state;

    // 裏ドラ表示牌が hidden に含まれていないこと
    for (TileId indicator : rs.uradora_indicators) {
        auto it = std::find(hidden.begin(), hidden.end(), indicator);
        EXPECT_EQ(it, hidden.end()) << "裏ドラ表示牌 " << static_cast<int>(indicator) << " が hidden に含まれている";
    }
}

TEST_F(SearchUtilsTest, DeterminizePreservesDoraIndicators) {
    auto env_copy = search_utils::clone(env);
    auto original_dora = env_copy.round_state.dora_indicators;
    auto original_uradora = env_copy.round_state.uradora_indicators;

    bool ok = search_utils::determinize(env_copy, 0);
    EXPECT_TRUE(ok);

    // ドラ表示牌の牌IDが変わっていないこと
    EXPECT_EQ(env_copy.round_state.dora_indicators, original_dora);
    EXPECT_EQ(env_copy.round_state.uradora_indicators, original_uradora);

    // wall 配列上のドラ表示牌位置も保持されていること
    const auto& rs = env_copy.round_state;
    for (size_t i = 0; i < original_dora.size(); ++i) {
        int dora_idx = kNumTiles - 6 - static_cast<int>(i) * 2;
        int uradora_idx = kNumTiles - 5 - static_cast<int>(i) * 2;
        EXPECT_EQ(rs.wall[dora_idx], original_dora[i])
            << "wall[" << dora_idx << "] のドラ表示牌が破壊された";
        EXPECT_EQ(rs.wall[uradora_idx], original_uradora[i])
            << "wall[" << uradora_idx << "] の裏ドラ表示牌が破壊された";
    }
}
