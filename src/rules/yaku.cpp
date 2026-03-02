#include "rules/yaku.h"
#include <algorithm>
#include <cassert>

namespace mahjong {
namespace yaku {

namespace {

// 断么九: 全ての牌が中張牌か
bool check_tanyao(const AgariDecomposition& decomp) {
    // 雀頭
    if (Tile::is_yaochu(decomp.jantai)) return false;
    // 面子
    for (const auto& m : decomp.mentsu_list) {
        if (m.kind == MentsuKind::Shuntsu) {
            // 順子: 1xx, xx9 は么九牌を含む
            if (Tile::is_yaochu(m.first_type)) return false;
            if (Tile::is_yaochu(static_cast<TileType>(m.first_type + 2))) return false;
        } else {
            // 刻子/槓子
            if (Tile::is_yaochu(m.first_type)) return false;
        }
    }
    return true;
}

// 役牌の判定: 三元牌、場風、自風の刻子/槓子をカウント
int count_yakuhai(const AgariDecomposition& decomp, TileType bakaze, TileType jikaze) {
    int count = 0;
    for (const auto& m : decomp.mentsu_list) {
        if (m.kind == MentsuKind::Shuntsu) continue;
        // 刻子 or 槓子
        if (Tile::is_sangenpai(m.first_type)) count++;
        if (m.first_type == bakaze) count++;
        if (m.first_type == jikaze) count++;
    }
    return count;
}

// 平和の判定
bool check_pinfu(const AgariDecomposition& decomp, WaitType wait_type,
                 TileType bakaze, TileType jikaze) {
    // 全面子が順子
    for (const auto& m : decomp.mentsu_list) {
        if (m.kind != MentsuKind::Shuntsu) return false;
    }
    // 雀頭が役牌でない
    if (Tile::is_sangenpai(decomp.jantai)) return false;
    if (decomp.jantai == bakaze) return false;
    if (decomp.jantai == jikaze) return false;
    // 両面待ち
    if (wait_type != WaitType::Ryanmen) return false;
    return true;
}

// 一盃口の判定: 同一順子が2組
int count_iipeiko(const AgariDecomposition& decomp) {
    int pairs = 0;
    std::vector<const MentsuInfo*> closed_shuntsu;
    for (const auto& m : decomp.mentsu_list) {
        if (!m.is_open && m.kind == MentsuKind::Shuntsu) {
            closed_shuntsu.push_back(&m);
        }
    }
    // 同一順子のペアをカウント
    std::vector<bool> used(closed_shuntsu.size(), false);
    for (size_t i = 0; i < closed_shuntsu.size(); ++i) {
        if (used[i]) continue;
        for (size_t j = i + 1; j < closed_shuntsu.size(); ++j) {
            if (used[j]) continue;
            if (closed_shuntsu[i]->first_type == closed_shuntsu[j]->first_type) {
                pairs++;
                used[i] = true;
                used[j] = true;
                break;
            }
        }
    }
    return pairs;
}

}  // anonymous namespace

YakuEvaluation evaluate_yaku(
    const AgariDecomposition& decomp,
    WaitType wait_type,
    const WinContext& ctx)
{
    YakuEvaluation eval{};
    eval.total_han = 0;

    // 門前ツモ
    if (ctx.is_tsumo && ctx.is_menzen) {
        eval.yakus.push_back({YakuType::MenzenTsumo, 1});
    }

    // 立直
    if (ctx.is_riichi) {
        eval.yakus.push_back({YakuType::Riichi, 1});
    }

    // 一発（立直前提）
    assert(!(ctx.is_ippatsu && !ctx.is_riichi) && "ippatsu without riichi is invalid");
    if (ctx.is_riichi && ctx.is_ippatsu) {
        eval.yakus.push_back({YakuType::Ippatsu, 1});
    }

    // 断么九
    if (check_tanyao(decomp)) {
        eval.yakus.push_back({YakuType::Tanyao, 1});
    }

    // 役牌
    int yakuhai_count = count_yakuhai(decomp, ctx.bakaze, ctx.jikaze);
    for (int i = 0; i < yakuhai_count; ++i) {
        eval.yakus.push_back({YakuType::Yakuhai, 1});
    }

    // 平和（門前限定）
    if (ctx.is_menzen && check_pinfu(decomp, wait_type, ctx.bakaze, ctx.jikaze)) {
        eval.yakus.push_back({YakuType::Pinfu, 1});
    }

    // 一盃口（門前限定）
    if (ctx.is_menzen) {
        int iipeiko = count_iipeiko(decomp);
        if (iipeiko >= 1) {
            eval.yakus.push_back({YakuType::Iipeiko, 1});
        }
    }

    // 海底撈月
    if (ctx.is_haitei && ctx.is_tsumo) {
        eval.yakus.push_back({YakuType::HaiteiTsumo, 1});
    }

    // 河底撈魚
    if (ctx.is_houtei && !ctx.is_tsumo) {
        eval.yakus.push_back({YakuType::HouteiRon, 1});
    }

    // 嶺上開花
    if (ctx.is_rinshan) {
        eval.yakus.push_back({YakuType::RinshanKaihou, 1});
    }

    // 槍槓
    if (ctx.is_chankan) {
        eval.yakus.push_back({YakuType::Chankan, 1});
    }

    // 役が1つもなければドラも加算しない
    if (eval.yakus.empty()) {
        eval.total_han = 0;
        return eval;
    }

    // 翻合計
    for (const auto& y : eval.yakus) {
        eval.total_han += y.han;
    }

    // ドラ
    eval.dora_count = count_dora(ctx.all_tile_ids, ctx.dora_indicators);
    eval.akadora_count = count_akadora(ctx.all_tile_ids);
    eval.total_han += eval.dora_count + eval.akadora_count;

    // 裏ドラ（立直時のみ）
    if (ctx.is_riichi) {
        eval.uradora_count = count_uradora(ctx.all_tile_ids, ctx.uradora_indicators);
        eval.total_han += eval.uradora_count;
    }

    return eval;
}

std::vector<WaitType> get_possible_waits(
    const AgariDecomposition& decomp,
    TileType agari_tile)
{
    std::vector<WaitType> waits;

    // 単騎: 雀頭が和了牌
    if (decomp.jantai == agari_tile) {
        waits.push_back(WaitType::Tanki);
    }

    // 面子を調べる
    for (const auto& m : decomp.mentsu_list) {
        if (m.is_open) continue;  // 副露面子は待ち判定の対象外

        if (m.kind == MentsuKind::Koutsu) {
            // 双碰: 刻子が和了牌
            if (m.first_type == agari_tile) {
                waits.push_back(WaitType::Shanpon);
            }
        } else if (m.kind == MentsuKind::Shuntsu) {
            TileType t0 = m.first_type;
            TileType t1 = t0 + 1;
            TileType t2 = t0 + 2;

            if (agari_tile == t0) {
                // 和了牌が順子の先頭 → 手牌に t1,t2 があった
                int num = Tile::number_of(t2);  // 順子末尾の数字
                if (num == 9) {
                    // 89+7 → 辺張（89は端、7でしか完成しない）
                    waits.push_back(WaitType::Penchan);
                } else {
                    // 例: 56+4 → 両面（56は4と7で完成）
                    waits.push_back(WaitType::Ryanmen);
                }
            } else if (agari_tile == t1) {
                // 和了牌が順子の中央 → 嵌張
                waits.push_back(WaitType::Kanchan);
            } else if (agari_tile == t2) {
                // 和了牌が順子の末尾 → 手牌に t0,t1 があった
                int num = Tile::number_of(t0);  // 順子先頭の数字
                if (num == 1) {
                    // 12+3 → 辺張（12は端、3でしか完成しない）
                    waits.push_back(WaitType::Penchan);
                } else {
                    // 例: 56+7 → 両面（56は4と7で完成）
                    waits.push_back(WaitType::Ryanmen);
                }
            }
        }
    }

    // 重複除去
    std::sort(waits.begin(), waits.end());
    waits.erase(std::unique(waits.begin(), waits.end()), waits.end());

    return waits;
}

int count_dora(const std::vector<TileId>& tile_ids,
               const std::vector<TileType>& dora_indicators)
{
    // ドラ表示牌 → ドラ牌の TileType を求める
    std::array<int, kNumTileTypes> dora_types{};
    for (TileType indicator : dora_indicators) {
        TileType dora_type = Tile::next_dora(indicator);
        dora_types[dora_type]++;
    }

    // 手牌の各牌がドラかカウント
    int count = 0;
    for (TileId id : tile_ids) {
        TileType t = id / 4;
        if (dora_types[t] > 0) {
            count += dora_types[t];
        }
    }
    return count;
}

int count_akadora(const std::vector<TileId>& tile_ids) {
    int count = 0;
    for (TileId id : tile_ids) {
        if (Tile::is_red_id(id)) count++;
    }
    return count;
}

int count_uradora(const std::vector<TileId>& tile_ids,
                  const std::vector<TileType>& uradora_indicators)
{
    return count_dora(tile_ids, uradora_indicators);
}

std::string to_string(YakuType type) {
    switch (type) {
        case YakuType::MenzenTsumo: return "門前清自摸和";
        case YakuType::Riichi:      return "立直";
        case YakuType::Ippatsu:     return "一発";
        case YakuType::Tanyao:      return "断么九";
        case YakuType::Yakuhai:     return "役牌";
        case YakuType::Pinfu:       return "平和";
        case YakuType::Iipeiko:     return "一盃口";
        case YakuType::HaiteiTsumo: return "海底撈月";
        case YakuType::HouteiRon:   return "河底撈魚";
        case YakuType::RinshanKaihou: return "嶺上開花";
        case YakuType::Chankan:     return "槍槓";
    }
    return "Unknown";
}

}  // namespace yaku
}  // namespace mahjong
