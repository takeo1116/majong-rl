#include <gtest/gtest.h>
#include "core/environment_state.h"

using namespace mahjong;

// 同一 seed で同一初期状態が再現できる
TEST(EnvironmentTest, SeedReproducibility) {
    EnvironmentState env1;
    env1.reset(42);

    EnvironmentState env2;
    env2.reset(42);

    EXPECT_EQ(env1.match_state.first_dealer, env2.match_state.first_dealer);
    EXPECT_EQ(env1.match_state.current_dealer, env2.match_state.current_dealer);
    EXPECT_EQ(env1.run_mode, env2.run_mode);

    // 同一 seed から同一乱数列が得られる
    int v1 = env1.rng.next_int(100);
    int v2 = env2.rng.next_int(100);
    EXPECT_EQ(v1, v2);
}

// 異なる seed で異なる初期状態が得られる（確率的テスト）
TEST(EnvironmentTest, DifferentSeeds) {
    // 複数の seed で起家が常に同じになる確率は低い
    bool all_same = true;
    EnvironmentState env0;
    env0.reset(0);
    for (uint64_t s = 1; s < 10; ++s) {
        EnvironmentState env;
        env.reset(s);
        if (env.match_state.first_dealer != env0.match_state.first_dealer) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

// 起家指定でリセット
TEST(EnvironmentTest, ResetWithDealer) {
    EnvironmentState env;
    env.reset(42, 2);

    EXPECT_EQ(env.match_state.first_dealer, 2);
    EXPECT_EQ(env.match_state.current_dealer, 2);
    EXPECT_EQ(env.round_state.dealer, 2);
}

// 状態コピー後に RNG が独立する
TEST(EnvironmentTest, RngIndependenceAfterCopy) {
    EnvironmentState env1;
    env1.reset(42);

    // 何回か乱数を進める
    env1.rng.next_int(100);
    env1.rng.next_int(100);

    // コピーする
    EnvironmentState env2 = env1;

    // コピー後の乱数列は同一
    int v1a = env1.rng.next_int(100);
    int v2a = env2.rng.next_int(100);
    EXPECT_EQ(v1a, v2a);

    // env1 をさらに進めても env2 には影響しない
    env1.rng.next_int(100);
    env1.rng.next_int(100);

    // env1 は 2 回余分に進んでいるため、以降の乱数列は env2 と異なるはず
    int v1c = env1.rng.next_int(1000);
    int v2c = env2.rng.next_int(1000);
    EXPECT_NE(v1c, v2c);
}

// MatchState の値がコピーで独立する
TEST(EnvironmentTest, StateCopyIndependence) {
    EnvironmentState env1;
    env1.reset(42, 0);
    env1.match_state.scores = {30000, 20000, 25000, 25000};

    EnvironmentState env2 = env1;

    env2.match_state.scores[0] = 10000;
    EXPECT_EQ(env1.match_state.scores[0], 30000);
    EXPECT_EQ(env2.match_state.scores[0], 10000);
}

// RoundState の値がコピーで独立する
TEST(EnvironmentTest, RoundStateCopyIndependence) {
    EnvironmentState env1;
    env1.reset(42, 0);
    env1.round_state.turn_number = 5;
    env1.round_state.players[0].hand = {0, 4, 8, 12};

    EnvironmentState env2 = env1;

    env2.round_state.turn_number = 10;
    env2.round_state.players[0].hand.push_back(16);

    EXPECT_EQ(env1.round_state.turn_number, 5);
    EXPECT_EQ(env1.round_state.players[0].hand.size(), 4);
    EXPECT_EQ(env2.round_state.turn_number, 10);
    EXPECT_EQ(env2.round_state.players[0].hand.size(), 5);
}

// RunMode の設定
TEST(EnvironmentTest, RunModeDebug) {
    EnvironmentState env;
    env.reset(42, RunMode::Debug);
    EXPECT_EQ(env.run_mode, RunMode::Debug);
    EXPECT_TRUE(env.logging_enabled);
}

TEST(EnvironmentTest, RunModeFast) {
    EnvironmentState env;
    env.reset(42, RunMode::Fast);
    EXPECT_EQ(env.run_mode, RunMode::Fast);
    EXPECT_FALSE(env.logging_enabled);
}

// RNG のシャッフル
TEST(EnvironmentTest, RngShuffle) {
    Rng rng1;
    rng1.seed(42);

    std::array<int, 10> arr1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    rng1.shuffle(arr1.begin(), arr1.end());

    Rng rng2;
    rng2.seed(42);

    std::array<int, 10> arr2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    rng2.shuffle(arr2.begin(), arr2.end());

    // 同一 seed で同一シャッフル結果
    EXPECT_EQ(arr1, arr2);

    // シャッフルされている（元の順序と異なる）
    std::array<int, 10> orig = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_NE(arr1, orig);
}
