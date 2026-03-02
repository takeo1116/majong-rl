#include "engine/state_validator.h"
#include "core/tile.h"
#include "core/types.h"
#include <array>
#include <sstream>

namespace mahjong {
namespace state_validator {

// 136牌の総数整合と牌ID重複チェック
static void check_tile_counts(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;
    std::array<bool, kNumTiles> seen = {};
    int total = 0;

    auto mark = [&](TileId id, const std::string& source) {
        if (id >= kNumTiles) {
            std::ostringstream oss;
            oss << "不正な牌ID " << static_cast<int>(id) << " (" << source << ")";
            result.add_error(oss.str());
            return;
        }
        if (seen[id]) {
            std::ostringstream oss;
            oss << "牌ID重複 " << static_cast<int>(id) << " (" << source << ")";
            result.add_error(oss.str());
            return;
        }
        seen[id] = true;
        ++total;
    };

    // 各プレイヤーの手牌
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        const auto& ps = rs.players[p];
        std::string player_str = "player" + std::to_string(p);
        for (TileId t : ps.hand) {
            mark(t, player_str + "手牌");
        }
        // 副露牌
        for (const auto& meld : ps.melds) {
            for (uint8_t i = 0; i < meld.tile_count; ++i) {
                mark(meld.tiles[i], player_str + "副露");
            }
        }
        // 捨て牌
        // called=true の牌は鳴かれた場合は副露に含まれるが、
        // ロン和了の場合は副露に含まれないため、副露に無ければカウントする
        for (const auto& d : ps.discards) {
            if (!d.called) {
                mark(d.tile, player_str + "河");
            } else {
                // called 牌が副露に存在するかチェック
                bool in_meld = false;
                for (PlayerId mp = 0; mp < kNumPlayers; ++mp) {
                    for (const auto& meld : rs.players[mp].melds) {
                        for (uint8_t mi = 0; mi < meld.tile_count; ++mi) {
                            if (meld.tiles[mi] == d.tile) {
                                in_meld = true;
                                break;
                            }
                        }
                        if (in_meld) break;
                    }
                    if (in_meld) break;
                }
                if (!in_meld) {
                    mark(d.tile, player_str + "河(ロン)");
                }
            }
        }
    }

    // 未ツモの山（wall_position 以降、嶺上ツモ済みの末尾を除く）
    // 嶺上牌は wall[kNumTiles-1] から逆順にツモされるため、
    // ツモ済み分を末尾から除外する
    int wall_end = kNumTiles - rs.rinshan_draw_count;
    for (int i = rs.wall_position; i < wall_end; ++i) {
        mark(rs.wall[i], "山");
    }

    if (total != kNumTiles) {
        std::ostringstream oss;
        oss << "牌総数不一致: " << total << " != " << kNumTiles;
        result.add_error(oss.str());
    }
}

// 山残枚数チェック
static void check_wall_position(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;
    if (rs.wall_position > kNumTiles) {
        std::ostringstream oss;
        oss << "wall_position 超過: " << static_cast<int>(rs.wall_position);
        result.add_error(oss.str());
    }
}

// 手牌枚数チェック（phase に応じた期待枚数との一致）
static void check_hand_counts(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;

    // 局終了後やマッチ終了後は手牌チェックをスキップ
    if (rs.phase == Phase::EndRound || rs.phase == Phase::EndMatch ||
        rs.phase == Phase::StartMatch || rs.phase == Phase::StartRound) {
        return;
    }

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        const auto& ps = rs.players[p];
        int meld_tiles = 0;
        for (const auto& m : ps.melds) {
            // チー/ポンは3枚分を占有、槓は4枚分を占有
            meld_tiles += m.tile_count;
        }
        // 手牌 + 副露牌は合計で 13 or 14 枚分に対応
        // 副露1つにつき手牌は3枚減る（槓は4枚減るが手牌外の1枚を加算）
        int expected_base = 13;

        if (rs.phase == Phase::SelfActionPhase && p == rs.current_player) {
            expected_base = 14;  // ツモ後の手牌は14枚ベース
        }

        int meld_count = ps.meld_count();
        int kan_count = 0;
        for (const auto& m : ps.melds) {
            if (m.type == MeldType::Daiminkan || m.type == MeldType::Ankan || m.type == MeldType::Kakan) {
                ++kan_count;
            }
        }
        // 期待手牌枚数 = expected_base - (meld_count * 3) + kan_count * 3 - kan_count * 4 + kan_count
        // = expected_base - meld_count * 3
        // 副露1回で手牌3枚消費。ただし槓は4枚消費だが、ツモ1枚追加で結局3枚減。
        // 実際は: チー/ポン = 手牌2枚 + 鳴牌1枚 = 手牌-2枚。大明槓 = 手牌3枚 + 鳴牌1枚 = 手牌-3枚。
        // 暗槓 = 手牌4枚 = 手牌-4枚+嶺上ツモ1枚 = 手牌-3枚。加槓 = 手牌1枚追加 = 手牌-1枚(ポン分を含めると-3枚)
        // 結局 meld_count * 3 を引くのが正しい。
        int expected_hand = expected_base - meld_count * 3;

        if (ps.hand_count() != expected_hand) {
            // ResponsePhase や ResolveResponsePhase では打牌後で13枚になるのが正常
            // その場合 current_player 以外も影響がある可能性があるので緩和
            if (rs.phase == Phase::ResponsePhase || rs.phase == Phase::ResolveResponsePhase ||
                rs.phase == Phase::ResolveWinPhase || rs.phase == Phase::ResolveDrawPhase ||
                rs.phase == Phase::DrawPhase) {
                // 全員 13 枚ベースが期待される
                int relaxed_expected = 13 - meld_count * 3;
                if (ps.hand_count() != relaxed_expected && ps.hand_count() != relaxed_expected + 1) {
                    std::ostringstream oss;
                    oss << "player" << static_cast<int>(p) << " 手牌枚数不正: "
                        << ps.hand_count() << " (期待: " << relaxed_expected << " or " << relaxed_expected + 1 << ")";
                    result.add_error(oss.str());
                }
            } else {
                std::ostringstream oss;
                oss << "player" << static_cast<int>(p) << " 手牌枚数不正: "
                    << ps.hand_count() << " (期待: " << expected_hand << ")";
                result.add_error(oss.str());
            }
        }
    }
}

// 副露構成チェック
static void check_meld_consistency(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;
    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        for (const auto& meld : rs.players[p].melds) {
            switch (meld.type) {
                case MeldType::Chi: {
                    if (meld.tile_count != 3) {
                        result.add_error("チーの構成牌数が3でない");
                        break;
                    }
                    // 3枚が連続する type であることを確認
                    std::array<TileType, 3> types;
                    for (int i = 0; i < 3; ++i) {
                        types[i] = Tile::from_id(meld.tiles[i]).type;
                    }
                    std::sort(types.begin(), types.end());
                    if (types[1] != types[0] + 1 || types[2] != types[0] + 2) {
                        result.add_error("チーの構成牌が連続していない");
                    }
                    if (Tile::is_jihai(types[0])) {
                        result.add_error("字牌でチーしている");
                    }
                    break;
                }
                case MeldType::Pon: {
                    if (meld.tile_count != 3) {
                        result.add_error("ポンの構成牌数が3でない");
                        break;
                    }
                    TileType t0 = Tile::from_id(meld.tiles[0]).type;
                    for (int i = 1; i < 3; ++i) {
                        if (Tile::from_id(meld.tiles[i]).type != t0) {
                            result.add_error("ポンの構成牌が同一種でない");
                            break;
                        }
                    }
                    break;
                }
                case MeldType::Daiminkan:
                case MeldType::Ankan:
                case MeldType::Kakan: {
                    if (meld.tile_count != 4) {
                        result.add_error("槓の構成牌数が4でない");
                        break;
                    }
                    TileType t0 = Tile::from_id(meld.tiles[0]).type;
                    for (int i = 1; i < 4; ++i) {
                        if (Tile::from_id(meld.tiles[i]).type != t0) {
                            result.add_error("槓の構成牌が同一種でない");
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
}

// ドラ表示牌数チェック
static void check_dora_indicators(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;
    if (rs.dora_indicators.size() != rs.uradora_indicators.size()) {
        std::ostringstream oss;
        oss << "ドラ表示牌数不一致: dora=" << rs.dora_indicators.size()
            << " uradora=" << rs.uradora_indicators.size();
        result.add_error(oss.str());
    }
}

// phase/手番の整合
static void check_phase_consistency(const EnvironmentState& env, ValidationResult& result) {
    const auto& rs = env.round_state;

    if (rs.current_player >= kNumPlayers) {
        std::ostringstream oss;
        oss << "current_player が不正: " << static_cast<int>(rs.current_player);
        result.add_error(oss.str());
    }

    // phase と end_reason の矛盾チェック
    if (rs.end_reason != RoundEndReason::None) {
        // 局が終了しているのに進行フェーズにいるのはおかしい
        if (rs.phase == Phase::DrawPhase || rs.phase == Phase::SelfActionPhase ||
            rs.phase == Phase::ResponsePhase) {
            std::ostringstream oss;
            oss << "局終了済み(end_reason=" << static_cast<int>(rs.end_reason)
                << ")なのに進行フェーズ " << static_cast<int>(rs.phase);
            result.add_error(oss.str());
        }
    }
}

ValidationResult validate(const EnvironmentState& env) {
    ValidationResult result;
    check_tile_counts(env, result);
    check_wall_position(env, result);
    check_hand_counts(env, result);
    check_meld_consistency(env, result);
    check_dora_indicators(env, result);
    check_phase_consistency(env, result);
    return result;
}

}  // namespace state_validator
}  // namespace mahjong
