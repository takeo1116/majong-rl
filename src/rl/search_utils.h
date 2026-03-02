#pragma once

#include "core/tile.h"
#include "core/environment_state.h"
#include <vector>

namespace mahjong {
namespace search_utils {

// 完全状態コピー（RNG含む）— EnvironmentState の値コピー
EnvironmentState clone(const EnvironmentState& env);

// observer から見て非公開の牌ID一覧（他家手牌 + 未ツモ山）
std::vector<TileId> get_hidden_tiles(const EnvironmentState& env, PlayerId observer);

// 非公開牌をシャッフルして再配置（determinize）
// env.rng を使ってシャッフル。成功時 true。
bool determinize(EnvironmentState& env, PlayerId observer);

}  // namespace search_utils
}  // namespace mahjong
