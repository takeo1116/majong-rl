"""麻雀 RL パッケージ"""

from mahjong_rl._mahjong_core import (
    # 列挙型
    Suit,
    Wind,
    ActionType,
    MeldType,
    Phase,
    ErrorCode,
    EventType,
    RunMode,
    RoundEndReason,
    RewardPolicyType,
    # 構造体
    Tile,
    DiscardInfo,
    Meld,
    Action,
    Event,
    StepResult,
    PlayerState,
    RewardPolicyConfig,
    MatchState,
    RoundState,
    EnvironmentState,
    PartialObservation,
    FullObservation,
    # クラス
    GameEngine,
    # 関数
    make_partial_observation,
    make_full_observation,
    make_type_counts,
    is_agari,
    is_tenpai,
    get_waits,
    # 定数
    NUM_TILES,
    NUM_TILE_TYPES,
    NUM_PLAYERS,
)

__all__ = [
    "Suit", "Wind", "ActionType", "MeldType", "Phase", "ErrorCode",
    "EventType", "RunMode", "RoundEndReason", "RewardPolicyType",
    "Tile", "DiscardInfo", "Meld", "Action", "Event", "StepResult",
    "PlayerState", "RewardPolicyConfig", "MatchState", "RoundState",
    "EnvironmentState", "PartialObservation", "FullObservation",
    "GameEngine",
    "make_partial_observation", "make_full_observation",
    "make_type_counts", "is_agari", "is_tenpai", "get_waits",
    "NUM_TILES", "NUM_TILE_TYPES", "NUM_PLAYERS",
]
