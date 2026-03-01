#include "core/tile.h"
#include "core/types.h"
#include "core/environment_state.h"
#include <iostream>

int main() {
    // 牌一覧の表示テスト
    std::cout << "=== 麻雀エンジン サンプル ===" << std::endl;

    // 全牌の表示
    std::cout << "\n牌一覧:" << std::endl;
    const auto& tiles = mahjong::all_tiles();
    for (int i = 0; i < mahjong::kNumTileTypes; ++i) {
        std::cout << mahjong::Tile::type_to_string(i) << " ";
    }
    std::cout << std::endl;

    // 赤牌の確認
    std::cout << "\n赤牌:" << std::endl;
    for (int i = 0; i < mahjong::kNumTiles; ++i) {
        if (tiles[i].is_red) {
            std::cout << "  ID=" << i << " " << tiles[i].to_string() << std::endl;
        }
    }

    // 環境状態の初期化テスト
    std::cout << "\n環境初期化テスト:" << std::endl;
    mahjong::EnvironmentState env;
    env.reset(42);
    std::cout << "  起家: プレイヤー" << static_cast<int>(env.match_state.first_dealer) << std::endl;
    std::cout << "  実行モード: " << mahjong::to_string(env.run_mode) << std::endl;
    std::cout << "  フェーズ: " << mahjong::to_string(env.round_state.phase) << std::endl;

    std::cout << "\n初期化完了" << std::endl;
    return 0;
}
