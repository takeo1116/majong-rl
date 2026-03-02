#pragma once

#include "core/tile.h"
#include "core/meld.h"
#include <array>
#include <vector>

namespace mahjong {

// 面子の種類
enum class MentsuKind : uint8_t {
    Shuntsu = 0,  // 順子
    Koutsu  = 1,  // 刻子
    Kantsu  = 2,  // 槓子
};

// 面子情報
struct MentsuInfo {
    MentsuKind kind;
    TileType first_type;  // 順子の場合は最小の TileType、刻子/槓子はその TileType
    bool is_open;         // 副露由来かどうか

    bool operator==(const MentsuInfo&) const = default;
};

// 和了の面子構成
struct AgariDecomposition {
    TileType jantai;                  // 雀頭の TileType
    std::vector<MentsuInfo> mentsu_list;  // 面子リスト（4つ）

    bool operator==(const AgariDecomposition&) const = default;
};

namespace agari {

// 閉じた手牌（カウント配列）と副露から、全ての面子構成を列挙する
// counts: 手牌の TileType 別カウント（雀頭+閉じた面子分、合計 3k+2）
// open_melds: 副露リスト
std::vector<AgariDecomposition> enumerate_decompositions(
    const std::array<int, kNumTileTypes>& counts,
    const std::vector<Meld>& open_melds);

}  // namespace agari
}  // namespace mahjong
