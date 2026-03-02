#include "engine/hand_utils.h"
#include <numeric>

namespace mahjong {
namespace hand_utils {

std::array<int, kNumTileTypes> make_type_counts(const std::vector<TileId>& hand) {
    std::array<int, kNumTileTypes> counts{};
    for (TileId id : hand) {
        counts[id / 4]++;
    }
    return counts;
}

// 再帰的に面子分解を試みる
// 最初の非ゼロ要素を見つけ、刻子または順子の先頭として消費する
bool can_decompose_mentsu(std::array<int, kNumTileTypes> counts) {
    // 最初の非ゼロ要素を探す
    int first = -1;
    for (int i = 0; i < kNumTileTypes; ++i) {
        if (counts[i] > 0) {
            first = i;
            break;
        }
    }
    if (first == -1) return true;  // 全てゼロ → 分解成功

    // 刻子として消費を試みる
    if (counts[first] >= 3) {
        counts[first] -= 3;
        if (can_decompose_mentsu(counts)) return true;
        counts[first] += 3;
    }

    // 順子として消費を試みる（数牌のみ、同スート内で連番）
    if (first < 27 && first % 9 <= 6) {
        if (counts[first + 1] >= 1 && counts[first + 2] >= 1) {
            counts[first]--;
            counts[first + 1]--;
            counts[first + 2]--;
            if (can_decompose_mentsu(counts)) return true;
            counts[first]++;
            counts[first + 1]++;
            counts[first + 2]++;
        }
    }

    return false;
}

bool is_agari(const std::array<int, kNumTileTypes>& counts) {
    // 合計が 3k + 2 であることを確認
    int total = 0;
    for (auto c : counts) total += c;
    if (total < 2 || total % 3 != 2) return false;

    // 各 TileType を雀頭として試す
    auto temp = counts;
    for (int j = 0; j < kNumTileTypes; ++j) {
        if (temp[j] < 2) continue;
        temp[j] -= 2;
        if (can_decompose_mentsu(temp)) {
            temp[j] += 2;
            return true;
        }
        temp[j] += 2;
    }
    return false;
}

bool is_tenpai(const std::array<int, kNumTileTypes>& counts) {
    auto temp = counts;
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (temp[t] >= 4) continue;  // 既に4枚あったら加えられない
        temp[t]++;
        if (is_agari(temp)) {
            temp[t]--;
            return true;
        }
        temp[t]--;
    }
    return false;
}

std::vector<TileType> get_waits(const std::array<int, kNumTileTypes>& counts) {
    std::vector<TileType> waits;
    auto temp = counts;
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (temp[t] >= 4) continue;
        temp[t]++;
        if (is_agari(temp)) {
            waits.push_back(static_cast<TileType>(t));
        }
        temp[t]--;
    }
    return waits;
}

int count_yaochu_types(const std::array<int, kNumTileTypes>& counts) {
    int count = 0;
    for (int i = 0; i < kNumTileTypes; ++i) {
        if (Tile::is_yaochu(static_cast<TileType>(i)) && counts[i] > 0) {
            count++;
        }
    }
    return count;
}

}  // namespace hand_utils
}  // namespace mahjong
