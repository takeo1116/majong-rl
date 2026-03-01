#pragma once

#include <cstdint>
#include <string>

namespace mahjong {

// 行動種別
enum class ActionType : uint8_t {
    Discard    = 0,  // 打牌（立直打牌は riichi フラグで表現）
    TsumoWin   = 1,  // ツモ和了
    Ron        = 2,  // ロン
    Chi        = 3,  // チー
    Pon        = 4,  // ポン
    Daiminkan  = 5,  // 大明槓
    Kakan      = 6,  // 加槓
    Ankan      = 7,  // 暗槓
    Skip       = 8,  // スキップ
    Kyuushu    = 9,  // 九種九牌宣言
};

// 副露種別
enum class MeldType : uint8_t {
    Chi       = 0,  // チー
    Pon       = 1,  // ポン
    Daiminkan = 2,  // 大明槓
    Kakan     = 3,  // 加槓
    Ankan     = 4,  // 暗槓
};

// フェーズ
enum class Phase : uint8_t {
    StartMatch          = 0,
    StartRound          = 1,
    DrawPhase           = 2,
    SelfActionPhase     = 3,
    ResponsePhase       = 4,
    ResolveResponsePhase = 5,
    ResolveWinPhase     = 6,
    ResolveDrawPhase    = 7,
    EndRound            = 8,
    EndMatch            = 9,
};

// エラーコード
enum class ErrorCode : uint8_t {
    Ok                = 0,
    IllegalAction     = 1,
    WrongPhase        = 2,
    InvalidTile       = 3,
    InvalidActor      = 4,
    InconsistentState = 5,
    UnknownError      = 6,
};

// イベント種別
enum class EventType : uint8_t {
    RoundStart      = 0,
    Deal            = 1,
    Draw            = 2,
    Discard         = 3,
    Riichi          = 4,
    Chi             = 5,
    Pon             = 6,
    Kan             = 7,
    DoraReveal      = 8,
    Ron             = 9,
    Tsumo           = 10,
    AbortiveDraw    = 11,
    ExhaustiveDraw  = 12,
    RoundEnd        = 13,
    MatchEnd        = 14,
};

// 実行モード
enum class RunMode : uint8_t {
    Debug = 0,
    Fast  = 1,
};

// 局終了理由
enum class RoundEndReason : uint8_t {
    None            = 0,
    Tsumo           = 1,  // ツモ和了
    Ron             = 2,  // ロン和了
    ExhaustiveDraw  = 3,  // 荒牌平局
    AbortiveKyuushu = 4,  // 九種九牌
};

// 各種文字列変換
std::string to_string(ActionType type);
std::string to_string(MeldType type);
std::string to_string(Phase phase);
std::string to_string(ErrorCode code);
std::string to_string(EventType type);
std::string to_string(RunMode mode);

}  // namespace mahjong
