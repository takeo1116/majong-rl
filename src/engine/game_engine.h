#pragma once

#include "core/tile.h"
#include "core/types.h"
#include "core/action.h"
#include "core/event.h"
#include "core/environment_state.h"
#include <vector>
#include <array>

namespace mahjong {

// 1局モードリセット用の設定構造体
struct RoundConfig {
    uint8_t round_number = 0;                     // 局番号（0=東1局, ..., 7=南4局, 8=延長局）
    PlayerId dealer = 0;                           // 親プレイヤーID
    uint8_t honba = 0;                             // 本場
    uint8_t kyotaku = 0;                           // 供託本数
    std::array<int32_t, kNumPlayers> scores = {25000, 25000, 25000, 25000};  // 各プレイヤーの持ち点
    std::array<TileId, kNumTiles> wall = {};        // 山（136枚、シャッフル済み）
    std::array<std::vector<TileId>, kNumPlayers> hands;  // 各プレイヤーの配牌
};

// step() の戻り値
struct StepResult {
    ErrorCode error = ErrorCode::Ok;
    bool round_over = false;
    bool match_over = false;
    std::array<float, kNumPlayers> rewards = {};
    std::vector<Event> events;
};

// ゲームエンジン
// 状態を持たない。すべての状態は EnvironmentState に保持される。
class GameEngine {
public:
    // 半荘リセット（起家をRNGで決定）
    void reset_match(EnvironmentState& env, uint64_t seed, RunMode mode = RunMode::Debug);

    // 半荘リセット（起家指定）
    void reset_match(EnvironmentState& env, uint64_t seed, PlayerId first_dealer, RunMode mode = RunMode::Debug);

    // 1局モードリセット（局面注入）
    ErrorCode reset_round(EnvironmentState& env, const RoundConfig& config, RunMode mode = RunMode::Debug);

    // 1アクションを実行する
    StepResult step(EnvironmentState& env, const Action& action);

    // 現在の合法手を列挙する
    std::vector<Action> get_legal_actions(const EnvironmentState& env) const;

    // 次の局へ進む / 半荘終了判定（局終了後に呼ぶ）
    void advance_round(EnvironmentState& env);

    // フェーズ遷移の検証
    static bool is_valid_transition(Phase from, Phase to);

    // フェーズごとの許可アクション種別
    static std::vector<ActionType> allowed_action_types(Phase phase);

private:
    // 局の初期化（山生成・配牌・ドラ設定）
    void init_round(EnvironmentState& env);
    void generate_wall(EnvironmentState& env);
    void deal_tiles(EnvironmentState& env);
    void setup_dora(EnvironmentState& env);

    // アクション処理
    StepResult process_self_action(EnvironmentState& env, const Action& action);
    StepResult process_response(EnvironmentState& env, const Action& action);

    // 応答解決
    void resolve_responses(EnvironmentState& env, StepResult& result);

    // ツモ処理
    void draw_tile(EnvironmentState& env, PlayerId player, StepResult& result);
    void draw_rinshan(EnvironmentState& env, PlayerId player, StepResult& result);

    // 次のプレイヤーのツモへ進む
    void advance_to_next_draw(EnvironmentState& env, StepResult& result);

    // 応答チェック
    bool has_any_response(const EnvironmentState& env, PlayerId discarder, TileId tile) const;
    void setup_response_phase(EnvironmentState& env, PlayerId discarder, TileId tile);
    PlayerId find_next_responder(const EnvironmentState& env) const;

    // 槍槓応答チェック（CQ-0016）
    bool has_chankan_response(const EnvironmentState& env, PlayerId kakan_player, TileId kakan_tile) const;
    void setup_chankan_response_phase(EnvironmentState& env, PlayerId kakan_player, TileId kakan_tile);

    // フリテン処理（CQ-0017）
    void update_furiten_on_discard(EnvironmentState& env, PlayerId discarder, TileId tile);

    // 局清算（CQ-0018, CQ-0019）
    void settle_round(EnvironmentState& env, StepResult& result);

    // 合法手列挙の内部
    std::vector<Action> get_self_actions(const EnvironmentState& env) const;
    std::vector<Action> get_response_actions(const EnvironmentState& env) const;
};

}  // namespace mahjong
