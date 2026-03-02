#include "rules/agari.h"
#include <algorithm>

namespace mahjong {
namespace agari {

namespace {

// 副露の Meld を MentsuInfo に変換する
MentsuInfo meld_to_mentsu(const Meld& meld) {
    MentsuInfo info{};
    info.is_open = true;

    switch (meld.type) {
        case MeldType::Chi:
            info.kind = MentsuKind::Shuntsu;
            // チーの構成牌から最小の TileType を求める
            {
                TileType min_type = 255;
                for (uint8_t i = 0; i < meld.tile_count; ++i) {
                    if (meld.tiles[i] != 255) {
                        TileType t = meld.tiles[i] / 4;
                        if (t < min_type) min_type = t;
                    }
                }
                info.first_type = min_type;
            }
            break;
        case MeldType::Pon:
            info.kind = MentsuKind::Koutsu;
            info.first_type = meld.tiles[0] / 4;
            break;
        case MeldType::Daiminkan:
        case MeldType::Kakan:
        case MeldType::Ankan:
            info.kind = MentsuKind::Kantsu;
            info.first_type = meld.tiles[0] / 4;
            // 暗槓は門前扱い
            if (meld.type == MeldType::Ankan) {
                info.is_open = false;
            }
            break;
    }
    return info;
}

// 再帰的に面子分解を列挙する
// counts: 残りの手牌カウント（雀頭は既に除かれている）
// current: 現在構築中の面子リスト
// results: 結果を格納するベクタ
void enumerate_mentsu_recursive(
    std::array<int, kNumTileTypes>& counts,
    std::vector<MentsuInfo>& current,
    std::vector<std::vector<MentsuInfo>>& results)
{
    // 最初の非ゼロ要素を探す
    int first = -1;
    for (int i = 0; i < kNumTileTypes; ++i) {
        if (counts[i] > 0) {
            first = i;
            break;
        }
    }
    if (first == -1) {
        // 全てゼロ → 分解成功
        results.push_back(current);
        return;
    }

    // 刻子として消費を試みる
    if (counts[first] >= 3) {
        counts[first] -= 3;
        current.push_back({MentsuKind::Koutsu, static_cast<TileType>(first), false});
        enumerate_mentsu_recursive(counts, current, results);
        current.pop_back();
        counts[first] += 3;
    }

    // 順子として消費を試みる（数牌のみ、同スート内で連番）
    if (first < 27 && first % 9 <= 6) {
        if (counts[first + 1] >= 1 && counts[first + 2] >= 1) {
            counts[first]--;
            counts[first + 1]--;
            counts[first + 2]--;
            current.push_back({MentsuKind::Shuntsu, static_cast<TileType>(first), false});
            enumerate_mentsu_recursive(counts, current, results);
            current.pop_back();
            counts[first]++;
            counts[first + 1]++;
            counts[first + 2]++;
        }
    }
}

}  // anonymous namespace

std::vector<AgariDecomposition> enumerate_decompositions(
    const std::array<int, kNumTileTypes>& counts,
    const std::vector<Meld>& open_melds)
{
    // 合計が 3k + 2 であることを確認
    int total = 0;
    for (auto c : counts) total += c;
    if (total < 2 || total % 3 != 2) return {};

    // 副露を MentsuInfo に変換
    std::vector<MentsuInfo> open_mentsu;
    for (const auto& meld : open_melds) {
        open_mentsu.push_back(meld_to_mentsu(meld));
    }

    std::vector<AgariDecomposition> result;
    auto temp = counts;

    // 各 TileType を雀頭として試す
    for (int j = 0; j < kNumTileTypes; ++j) {
        if (temp[j] < 2) continue;
        temp[j] -= 2;

        // 面子分解を列挙
        std::vector<MentsuInfo> current;
        std::vector<std::vector<MentsuInfo>> mentsu_results;
        enumerate_mentsu_recursive(temp, current, mentsu_results);

        for (auto& mentsu_list : mentsu_results) {
            AgariDecomposition decomp;
            decomp.jantai = static_cast<TileType>(j);
            decomp.mentsu_list = std::move(mentsu_list);
            // 副露由来の面子を追加
            for (const auto& om : open_mentsu) {
                decomp.mentsu_list.push_back(om);
            }
            result.push_back(std::move(decomp));
        }

        temp[j] += 2;
    }

    // 重複除去
    std::sort(result.begin(), result.end(), [](const AgariDecomposition& a, const AgariDecomposition& b) {
        if (a.jantai != b.jantai) return a.jantai < b.jantai;
        if (a.mentsu_list.size() != b.mentsu_list.size()) return a.mentsu_list.size() < b.mentsu_list.size();
        for (size_t i = 0; i < a.mentsu_list.size(); ++i) {
            if (a.mentsu_list[i].kind != b.mentsu_list[i].kind)
                return a.mentsu_list[i].kind < b.mentsu_list[i].kind;
            if (a.mentsu_list[i].first_type != b.mentsu_list[i].first_type)
                return a.mentsu_list[i].first_type < b.mentsu_list[i].first_type;
            if (a.mentsu_list[i].is_open != b.mentsu_list[i].is_open)
                return a.mentsu_list[i].is_open < b.mentsu_list[i].is_open;
        }
        return false;
    });
    result.erase(std::unique(result.begin(), result.end()), result.end());

    return result;
}

}  // namespace agari
}  // namespace mahjong
