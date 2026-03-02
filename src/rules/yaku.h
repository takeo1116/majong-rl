#pragma once

#include "core/tile.h"
#include "rules/agari.h"
#include <array>
#include <string>
#include <vector>

namespace mahjong {

// 待ちの種類
enum class WaitType : uint8_t {
    Ryanmen  = 0,  // 両面
    Shanpon  = 1,  // 双碰
    Kanchan  = 2,  // 嵌張
    Penchan  = 3,  // 辺張
    Tanki    = 4,  // 単騎
};

// L2 役の種類
enum class YakuType : uint8_t {
    MenzenTsumo = 0,   // 門前清自摸和
    Riichi      = 1,   // 立直
    Ippatsu     = 2,   // 一発
    Tanyao      = 3,   // 断么九
    Yakuhai     = 4,   // 役牌（複数成立あり）
    Pinfu       = 5,   // 平和
    Iipeiko     = 6,   // 一盃口
    HaiteiTsumo = 7,   // 海底撈月
    HouteiRon   = 8,   // 河底撈魚
    RinshanKaihou = 9, // 嶺上開花
    Chankan     = 10,  // 槍槓
};

// 役判定結果（1つの役）
struct YakuResult {
    YakuType type;
    int han;  // 翻数

    bool operator==(const YakuResult&) const = default;
};

// 和了コンテキスト（役判定に必要な情報）
struct WinContext {
    TileType agari_tile;          // 和了牌の TileType
    bool is_tsumo;                // ツモ和了か
    bool is_menzen;               // 門前か
    bool is_riichi;               // 立直しているか
    bool is_ippatsu;              // 一発条件を満たすか
    bool is_rinshan;              // 嶺上開花か
    bool is_chankan;              // 槍槓か
    bool is_haitei;               // 海底か（最後のツモ）
    bool is_houtei;               // 河底か（最後の打牌へのロン）
    TileType bakaze;              // 場風の TileType
    TileType jikaze;              // 自風の TileType
    std::vector<TileId> all_tile_ids;       // 手牌+副露の全実牌ID
    std::vector<TileType> dora_indicators;  // ドラ表示牌
    std::vector<TileType> uradora_indicators; // 裏ドラ表示牌
};

// 1つの面子構成に対する役判定結果
struct YakuEvaluation {
    std::vector<YakuResult> yakus;  // 成立した役のリスト
    int total_han;                  // 翻合計（ドラ含む）
    int dora_count;                 // ドラ数
    int akadora_count;              // 赤ドラ数
    int uradora_count;              // 裏ドラ数
};

namespace yaku {

// 指定された面子構成+待ち形に対して役を判定する
YakuEvaluation evaluate_yaku(
    const AgariDecomposition& decomp,
    WaitType wait_type,
    const WinContext& ctx);

// 和了牌が特定の面子構成でどのような待ちになるかを列挙する
std::vector<WaitType> get_possible_waits(
    const AgariDecomposition& decomp,
    TileType agari_tile);

// ドラ数をカウントする
int count_dora(const std::vector<TileId>& tile_ids,
               const std::vector<TileType>& dora_indicators);

// 赤ドラ数をカウントする
int count_akadora(const std::vector<TileId>& tile_ids);

// 裏ドラ数をカウントする（立直時のみ呼ばれる）
int count_uradora(const std::vector<TileId>& tile_ids,
                  const std::vector<TileType>& uradora_indicators);

// YakuType の文字列表現
std::string to_string(YakuType type);

}  // namespace yaku
}  // namespace mahjong
