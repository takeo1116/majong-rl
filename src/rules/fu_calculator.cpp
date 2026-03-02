#include "rules/fu_calculator.h"

namespace mahjong {
namespace fu_calculator {

namespace {

// 面子符を計算する
int calc_mentsu_fu(const MentsuInfo& m) {
    int fu = 0;
    switch (m.kind) {
        case MentsuKind::Shuntsu:
            fu = 0;
            break;
        case MentsuKind::Koutsu:
            // 中張牌: 2, 么九牌: 4
            fu = Tile::is_yaochu(m.first_type) ? 4 : 2;
            // 暗刻は2倍
            if (!m.is_open) fu *= 2;
            break;
        case MentsuKind::Kantsu:
            // 中張牌: 8, 么九牌: 16
            fu = Tile::is_yaochu(m.first_type) ? 16 : 8;
            // 暗槓は2倍
            if (!m.is_open) fu *= 2;
            break;
    }
    return fu;
}

// 雀頭符を計算する
int calc_jantai_fu(TileType jantai, TileType bakaze, TileType jikaze) {
    int fu = 0;
    if (Tile::is_sangenpai(jantai)) fu += 2;
    if (jantai == bakaze) fu += 2;
    if (jantai == jikaze) fu += 2;
    return fu;
}

// 待ち符を計算する
int calc_wait_fu(WaitType wait_type) {
    switch (wait_type) {
        case WaitType::Ryanmen:  return 0;
        case WaitType::Shanpon:  return 0;
        case WaitType::Kanchan:  return 2;
        case WaitType::Penchan:  return 2;
        case WaitType::Tanki:    return 2;
    }
    return 0;
}

// 10符単位切り上げ
int round_up_fu(int fu) {
    return ((fu + 9) / 10) * 10;
}

}  // anonymous namespace

FuResult calculate_fu(
    const AgariDecomposition& decomp,
    WaitType wait_type,
    [[maybe_unused]] TileType agari_tile,
    bool is_tsumo,
    bool is_menzen,
    bool is_pinfu,
    TileType bakaze,
    TileType jikaze)
{
    FuResult result{};

    // 平和ツモ: 固定20符
    if (is_pinfu && is_tsumo) {
        result.base_fu = 20;
        result.total_fu = 20;
        return result;
    }

    // 副底
    result.base_fu = 20;

    // 門前ロン加符
    if (is_menzen && !is_tsumo) {
        result.base_fu += 10;
    }

    // 面子符
    for (const auto& m : decomp.mentsu_list) {
        result.mentsu_fu += calc_mentsu_fu(m);
    }

    // 雀頭符
    result.jantai_fu = calc_jantai_fu(decomp.jantai, bakaze, jikaze);

    // 待ち符
    result.wait_fu = calc_wait_fu(wait_type);

    // ツモ符（平和以外のツモ）
    if (is_tsumo && !is_pinfu) {
        result.tsumo_fu = 2;
    }

    int raw = result.base_fu + result.mentsu_fu + result.jantai_fu
            + result.wait_fu + result.tsumo_fu;

    // 喰い平和形（符加算がゼロの非門前ロン）は30符
    if (raw == 20 && !is_menzen && !is_tsumo) {
        result.total_fu = 30;
    } else {
        result.total_fu = round_up_fu(raw);
    }

    return result;
}

}  // namespace fu_calculator
}  // namespace mahjong
