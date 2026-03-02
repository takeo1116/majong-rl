#include "rules/score_calculator.h"
#include <algorithm>

namespace mahjong {
namespace score_calculator {

int ceil100(int value) {
    return ((value + 99) / 100) * 100;
}

int calculate_base_point(int han, int fu) {
    if (han >= 13) {
        // 数え役満
        return 8000;
    }
    if (han >= 11) {
        // 三倍満
        return 6000;
    }
    if (han >= 8) {
        // 倍満
        return 4000;
    }
    if (han >= 6) {
        // 跳満
        return 3000;
    }
    if (han >= 5) {
        // 満貫
        return 2000;
    }

    // 通常計算: 符 × 2^(翻+2)
    int base = fu * (1 << (han + 2));

    // 満貫超えは満貫に
    if (base >= 2000) {
        return 2000;
    }

    return base;
}

namespace {

// 支払い情報を計算する
PaymentInfo calc_payment(int base_point, bool is_dealer, bool is_tsumo, int honba) {
    PaymentInfo pay{};
    int honba_total = 300 * honba;

    if (is_tsumo) {
        if (is_dealer) {
            // 親ツモ: 各子から base*2 の切り上げ + 積み棒
            int per_child = ceil100(base_point * 2);
            pay.from_non_dealer = per_child + (honba_total / 3);
            pay.from_dealer = 0;
            pay.from_ron = 0;
        } else {
            // 子ツモ: 親から base*2, 子から base*1 の切り上げ + 積み棒
            pay.from_non_dealer = ceil100(base_point) + (honba_total / 3);
            pay.from_dealer = ceil100(base_point * 2) + (honba_total / 3);
            pay.from_ron = 0;
        }
    } else {
        // ロン
        if (is_dealer) {
            pay.from_ron = ceil100(base_point * 6) + honba_total;
        } else {
            pay.from_ron = ceil100(base_point * 4) + honba_total;
        }
        pay.from_non_dealer = 0;
        pay.from_dealer = 0;
    }

    return pay;
}

// 支払い総額を計算する（比較用）
int total_payment(const PaymentInfo& pay, bool is_dealer, bool is_tsumo) {
    if (is_tsumo) {
        if (is_dealer) {
            return pay.from_non_dealer * 3;
        } else {
            return pay.from_non_dealer * 2 + pay.from_dealer;
        }
    } else {
        return pay.from_ron;
    }
}

}  // anonymous namespace

ScoreResult calculate_win_score(
    const std::vector<AgariDecomposition>& decompositions,
    const WinContext& ctx,
    bool is_dealer,
    int honba)
{
    ScoreResult best{};
    best.valid = false;
    int best_total = -1;

    for (const auto& decomp : decompositions) {
        // この構成でありうる待ち形を取得
        auto possible_waits = yaku::get_possible_waits(decomp, ctx.agari_tile);
        if (possible_waits.empty()) continue;

        for (WaitType wt : possible_waits) {
            // 役判定
            auto eval = yaku::evaluate_yaku(decomp, wt, ctx);
            if (eval.yakus.empty()) continue;

            // 平和判定（符計算で必要）
            bool is_pinfu = false;
            for (const auto& y : eval.yakus) {
                if (y.type == YakuType::Pinfu) { is_pinfu = true; break; }
            }

            // 符計算
            auto fu_result = fu_calculator::calculate_fu(
                decomp, wt, ctx.agari_tile,
                ctx.is_tsumo, ctx.is_menzen, is_pinfu,
                ctx.bakaze, ctx.jikaze);

            // 基本点
            int base = calculate_base_point(eval.total_han, fu_result.total_fu);

            // 支払い情報
            auto pay = calc_payment(base, is_dealer, ctx.is_tsumo, honba);
            int total = total_payment(pay, is_dealer, ctx.is_tsumo);

            // 高点法: 支払い総額が最大のものを選ぶ
            if (total > best_total) {
                best_total = total;
                best.total_han = eval.total_han;
                best.fu = fu_result.total_fu;
                best.base_point = base;
                best.payment = pay;
                best.yakus = eval.yakus;
                best.dora_count = eval.dora_count;
                best.akadora_count = eval.akadora_count;
                best.uradora_count = eval.uradora_count;
                best.best_decomposition = decomp;
                best.best_wait_type = wt;
                best.valid = true;
            }
        }
    }

    return best;
}

}  // namespace score_calculator
}  // namespace mahjong
