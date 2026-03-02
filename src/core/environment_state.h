#pragma once

#include "core/types.h"
#include "core/match_state.h"
#include "core/round_state.h"
#include <cstdint>
#include <random>

namespace mahjong {

// RNG ラッパー（値コピー可能）
// mt19937 は値コピーで内部状態が複製される
struct Rng {
    std::mt19937 engine;

    bool operator==(const Rng& other) const { return engine == other.engine; }

    // seed で初期化
    void seed(uint64_t s) { engine.seed(static_cast<std::mt19937::result_type>(s)); }

    // [0, n) の一様整数乱数を返す
    int next_int(int n) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(engine);
    }

    // 配列をシャッフルする
    template<typename RandomIt>
    void shuffle(RandomIt first, RandomIt last) {
        std::shuffle(first, last, engine);
    }
};

// 環境状態（値コピー可能）
// MatchState + 現在の RoundState + RNG + 実行モード
struct EnvironmentState {
    MatchState match_state;
    RoundState round_state;
    Rng rng;
    RunMode run_mode = RunMode::Debug;
    bool logging_enabled = true;

    // seed で環境を初期化する
    void reset(uint64_t seed, RunMode mode = RunMode::Debug);

    // 比較
    bool operator==(const EnvironmentState&) const = default;

    // seed + 起家指定で初期化する
    void reset(uint64_t seed, PlayerId first_dealer, RunMode mode = RunMode::Debug);
};

}  // namespace mahjong
