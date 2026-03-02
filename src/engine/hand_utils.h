#pragma once

#include "core/tile.h"
#include <array>
#include <vector>

namespace mahjong {
namespace hand_utils {

// 手牌の TileId 列から TileType 別カウント配列を作る
std::array<int, kNumTileTypes> make_type_counts(const std::vector<TileId>& hand);

// 面子分解チェック: 全てのカウントが面子（刻子 or 順子）に分解できるか
// 合計が 3 の倍数であること
bool can_decompose_mentsu(std::array<int, kNumTileTypes> counts);

// 和了形チェック: カウントが (面子 × N) + 雀頭 に分解できるか
// 合計が 3k + 2 であること
bool is_agari(const std::array<int, kNumTileTypes>& counts);

// テンパイチェック: 何らかの牌を加えると和了形になるか
bool is_tenpai(const std::array<int, kNumTileTypes>& counts);

// 待ち牌一覧を返す
std::vector<TileType> get_waits(const std::array<int, kNumTileTypes>& counts);

// 九種九牌の么九牌種類数をカウントする
int count_yaochu_types(const std::array<int, kNumTileTypes>& counts);

}  // namespace hand_utils
}  // namespace mahjong
