#include "rl/search_utils.h"
#include "engine/state_validator.h"
#include <algorithm>
#include <set>

namespace mahjong {
namespace search_utils {

EnvironmentState clone(const EnvironmentState& env) {
    // EnvironmentState は値コピー可能
    return env;
}

// ドラ表示牌・裏ドラ表示牌の wall インデックス集合を取得する
static std::set<int> get_indicator_wall_indices(const RoundState& rs) {
    std::set<int> indices;
    // ドラ表示牌: wall[130], wall[128], ... (kNumTiles-6, kNumTiles-8, ...)
    // 裏ドラ表示牌: wall[131], wall[129], ... (kNumTiles-5, kNumTiles-7, ...)
    // 開示済みの枚数 = dora_indicators.size()
    for (size_t i = 0; i < rs.dora_indicators.size(); ++i) {
        int dora_idx = kNumTiles - 6 - static_cast<int>(i) * 2;
        int uradora_idx = kNumTiles - 5 - static_cast<int>(i) * 2;
        indices.insert(dora_idx);
        indices.insert(uradora_idx);
    }
    return indices;
}

std::vector<TileId> get_hidden_tiles(const EnvironmentState& env, PlayerId observer) {
    const auto& rs = env.round_state;
    std::vector<TileId> hidden;

    // 他家手牌
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        for (TileId t : rs.players[p].hand) {
            hidden.push_back(t);
        }
    }

    // 公開済みドラ表示牌・裏ドラ表示牌の wall インデックスを除外
    auto indicator_indices = get_indicator_wall_indices(rs);

    // 未ツモの山（嶺上牌含む）のうち、公開済みインジケータ位置を除く
    for (int i = rs.wall_position; i < kNumTiles; ++i) {
        if (indicator_indices.count(i) == 0) {
            hidden.push_back(rs.wall[i]);
        }
    }

    return hidden;
}

bool determinize(EnvironmentState& env, PlayerId observer) {
    auto& rs = env.round_state;

    // 1. 非公開牌プールを取得
    auto pool = get_hidden_tiles(env, observer);

    // 2. RNG でプールをシャッフル
    env.rng.shuffle(pool.begin(), pool.end());

    // 公開済みインジケータ位置の集合
    auto indicator_indices = get_indicator_wall_indices(rs);

    // 3. プールから他家手牌と山を再配置
    size_t idx = 0;

    // 他家手牌を差し替え（枚数は維持）
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        size_t hand_size = rs.players[p].hand.size();
        for (size_t i = 0; i < hand_size; ++i) {
            if (idx >= pool.size()) return false;
            rs.players[p].hand[i] = pool[idx++];
        }
    }

    // 残りを山の未ツモ位置に配置（ドラ表示牌位置はスキップ）
    for (int i = rs.wall_position; i < kNumTiles; ++i) {
        if (indicator_indices.count(i) > 0) {
            // この位置はドラ/裏ドラ表示牌 → 変更しない
            continue;
        }
        if (idx >= pool.size()) return false;
        rs.wall[i] = pool[idx++];
    }

    // 全牌を使い切ったか確認
    if (idx != pool.size()) return false;

    // 4. 整合性チェック
    auto vr = state_validator::validate(env);
    return vr.valid;
}

}  // namespace search_utils
}  // namespace mahjong
