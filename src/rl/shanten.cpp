/**
 * シャンテン数計算 (C++ 版)
 *
 * Python の baseline/shanten.py と同一ロジック。
 * 通常形・七対子・国士の3パターンの最小値を返す。
 */
#include "rl/shanten.h"

#include <algorithm>
#include <array>

namespace mahjong {

namespace {

// 么九牌インデックス (1m,9m,1p,9p,1s,9s,東,南,西,北,白,發,中)
constexpr int kTerminalsAndHonors[] = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};
constexpr int kNumTerminalsAndHonors = 13;

int kokushi_shanten(const std::array<int, 34>& counts) {
    int kinds = 0;
    bool has_pair = false;
    for (int i = 0; i < kNumTerminalsAndHonors; ++i) {
        int t = kTerminalsAndHonors[i];
        if (counts[t] > 0) {
            ++kinds;
            if (counts[t] >= 2) {
                has_pair = true;
            }
        }
    }
    return 13 - kinds - (has_pair ? 1 : 0);
}

int chiitoitsu_shanten(const std::array<int, 34>& counts) {
    int pairs = 0;
    for (int i = 0; i < 34; ++i) {
        if (counts[i] >= 2) {
            ++pairs;
        }
    }
    return 6 - pairs;
}

/// 残り牌から不完全面子 (対子・ターツ) を貪欲に数える
int count_partial(const std::array<int, 34>& counts) {
    std::array<int, 34> c = counts;
    int partial = 0;

    for (int suit = 0; suit < 3; ++suit) {
        int base = suit * 9;
        // 対子
        for (int i = 0; i < 9; ++i) {
            int t = base + i;
            if (c[t] >= 2) {
                c[t] -= 2;
                ++partial;
            }
        }
        // 両面ターツ
        for (int i = 0; i < 8; ++i) {
            int t = base + i;
            if (c[t] > 0 && c[t + 1] > 0) {
                --c[t];
                --c[t + 1];
                ++partial;
            }
        }
        // 嵌張ターツ
        for (int i = 0; i < 7; ++i) {
            int t = base + i;
            if (c[t] > 0 && c[t + 2] > 0) {
                --c[t];
                --c[t + 2];
                ++partial;
            }
        }
    }
    // 字牌の対子
    for (int t = 27; t < 34; ++t) {
        if (c[t] >= 2) {
            ++partial;
        }
    }
    return partial;
}

void remove_groups(
    std::array<int, 34>& counts,
    int pos,
    int mentsu,
    int jantai,
    int& best
) {
    // 枝刈り: 面子 4 以上はこれ以上不要
    if (mentsu >= 4) {
        int shanten = 8 - 2 * 4 - 0 - jantai;
        best = std::min(best, shanten);
        return;
    }

    // 次の非ゼロ位置を探す
    int idx = pos;
    while (idx < 34 && counts[idx] == 0) {
        ++idx;
    }

    if (idx >= 34) {
        int partial = count_partial(counts);
        int max_partial = 4 - mentsu;
        partial = std::min(partial, max_partial);
        int shanten = 8 - 2 * mentsu - partial - jantai;
        best = std::min(best, shanten);
        return;
    }

    // 枝刈り: 残りタイルで最大改善しても best 以下にならない
    int remaining_tiles = 0;
    for (int i = idx; i < 34; ++i) {
        remaining_tiles += counts[i];
    }
    int max_more_mentsu = remaining_tiles / 3;
    int max_total_mentsu = std::min(4, mentsu + max_more_mentsu);
    int lower_bound = 8 - 2 * max_total_mentsu - (4 - max_total_mentsu) - jantai;
    if (lower_bound >= best) {
        return;
    }

    // 刻子
    if (counts[idx] >= 3) {
        counts[idx] -= 3;
        remove_groups(counts, idx, mentsu + 1, jantai, best);
        counts[idx] += 3;
    }

    // 順子 (数牌のみ)
    int suit = idx / 9;
    int rel = idx % 9;
    if (suit < 3 && rel <= 6) {
        int base = suit * 9;
        if (counts[base + rel + 1] > 0 && counts[base + rel + 2] > 0) {
            counts[idx] -= 1;
            counts[base + rel + 1] -= 1;
            counts[base + rel + 2] -= 1;
            remove_groups(counts, idx, mentsu + 1, jantai, best);
            counts[idx] += 1;
            counts[base + rel + 1] += 1;
            counts[base + rel + 2] += 1;
        }
    }

    // この位置で面子を取らずに次へ
    remove_groups(counts, idx + 1, mentsu, jantai, best);
}

int regular_shanten(const std::array<int, 34>& counts) {
    int best = 8;
    std::array<int, 34> c = counts;

    // 雀頭なしで探索
    remove_groups(c, 0, 0, 0, best);

    // 各牌種を雀頭として取って探索
    for (int t = 0; t < 34; ++t) {
        if (c[t] >= 2) {
            c[t] -= 2;
            remove_groups(c, 0, 0, 1, best);
            c[t] += 2;
        }
    }

    return best;
}

}  // anonymous namespace

int compute_shanten(const std::array<int, kNumTileTypesShanten>& counts) {
    return std::min({
        regular_shanten(counts),
        chiitoitsu_shanten(counts),
        kokushi_shanten(counts),
    });
}

namespace {

/// 受け入れ枚数を計算する (shanten が下がる牌種の残り枚数合計)
int count_acceptance(std::array<int, 34>& counts, int shanten) {
    int total = 0;
    for (int t = 0; t < 34; ++t) {
        if (counts[t] >= 4) continue;
        counts[t] += 1;
        int new_sh = compute_shanten(counts);
        counts[t] -= 1;
        if (new_sh < shanten) {
            total += 4 - counts[t];  // 残り枚数の概算
        }
    }
    return total;
}

}  // anonymous namespace

DiscardAnalysis analyze_discards(
    const std::array<int, 34>& counts,
    const std::array<int, 34>& legal_mask)
{
    DiscardAnalysis result{};
    result.shanten_after.fill(-1);
    result.acceptance.fill(0);
    result.ukeire_norm.fill(0.0f);
    result.shanten_sign.fill(0.0f);

    int base_shanten = compute_shanten(counts);
    std::array<int, 34> c = counts;

    int max_acceptance = 0;

    for (int t = 0; t < 34; ++t) {
        if (c[t] < 1 || legal_mask[t] < 1) continue;

        c[t] -= 1;
        int sh_after = compute_shanten(c);
        result.shanten_after[t] = sh_after;

        int acc = count_acceptance(c, sh_after);
        result.acceptance[t] = acc;
        if (acc > max_acceptance) max_acceptance = acc;

        int delta = base_shanten - sh_after;
        if (delta > 0) {
            result.shanten_sign[t] = 1.0f;
        } else if (delta < 0) {
            result.shanten_sign[t] = -1.0f;
        }

        c[t] += 1;
    }

    // 正規化
    if (max_acceptance > 0) {
        for (int t = 0; t < 34; ++t) {
            result.ukeire_norm[t] =
                static_cast<float>(result.acceptance[t]) / static_cast<float>(max_acceptance);
        }
    }

    return result;
}

BestDiscardResult find_best_discard(
    const std::array<int, 34>& counts,
    const std::array<int, 34>& legal_mask)
{
    BestDiscardResult result{};
    result.best_shanten = 999;
    result.best_acceptance = -1;
    result.best_tile = -1;
    result.best_mask.fill(0);

    std::array<int, 34> c = counts;

    // 1パス目: 最良 (shanten, acceptance) を探す
    // shanten_after と acceptance を保持して2パス目で再利用
    std::array<int, 34> sh_after_arr;
    std::array<int, 34> acc_arr;
    sh_after_arr.fill(999);
    acc_arr.fill(-1);

    for (int t = 0; t < 34; ++t) {
        if (legal_mask[t] < 1 || c[t] <= 0) continue;
        c[t] -= 1;
        int sh = compute_shanten(c);
        int acc = count_acceptance(c, sh);
        sh_after_arr[t] = sh;
        acc_arr[t] = acc;
        if (sh < result.best_shanten ||
            (sh == result.best_shanten && acc > result.best_acceptance)) {
            result.best_shanten = sh;
            result.best_acceptance = acc;
        }
        c[t] += 1;
    }

    // 2パス目: 同率候補を収集 (再計算不要)
    for (int t = 0; t < 34; ++t) {
        if (sh_after_arr[t] == result.best_shanten &&
            acc_arr[t] == result.best_acceptance) {
            result.best_mask[t] = 1;
            if (result.best_tile == -1) {
                result.best_tile = t;
            }
        }
    }

    return result;
}

ShapeHint compute_shape_hint(const std::array<int, 34>& counts) {
    ShapeHint result{};
    result.chi.fill(0.0f);
    result.outside_wait.fill(0.0f);
    result.inside_wait.fill(0.0f);

    for (int suit = 0; suit < 3; ++suit) {
        int base = suit * 9;

        // 順子 (closed_chi): 中心牌 2-8 (index 1-7)
        for (int center = 1; center <= 7; ++center) {
            int idx = base + center;
            if (counts[idx - 1] >= 1 && counts[idx] >= 1 && counts[idx + 1] >= 1) {
                result.chi[suit * 7 + (center - 1)] = 1.0f;
            }
        }

        // 塔子 (closed_outside_wait): 隣接2牌ペア (12, 23, ..., 89)
        for (int pair_start = 0; pair_start < 8; ++pair_start) {
            int idx = base + pair_start;
            if (counts[idx] >= 1 && counts[idx + 1] >= 1) {
                result.outside_wait[suit * 8 + pair_start] = 1.0f;
            }
        }

        // 嵌張 (closed_inside_wait): 中心牌 2-8 (index 1-7), 間が空く
        for (int center = 1; center <= 7; ++center) {
            int idx = base + center;
            if (counts[idx - 1] >= 1 && counts[idx] < 1 && counts[idx + 1] >= 1) {
                result.inside_wait[suit * 7 + (center - 1)] = 1.0f;
            }
        }
    }

    return result;
}

}  // namespace mahjong
