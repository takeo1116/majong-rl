"""CQ-0256/0266: terminal / yaku の語彙定義 (single source of truth)"""

# CQ-0266: terminal class (5-class)
# policy にとって意味のある終局差に絞る
TERMINAL_CLASSES = [
    "win_menzen",        # 0: 自分が面前で和了
    "win_called",        # 1: 自分が副露して和了
    "draw_tenpai",       # 2: 自分が聴牌状態で流局
    "deal_in",           # 3: 自分が放銃して失点
    "other_non_dealin",  # 4: それ以外 (被ツモ/傍観/流局ノーテン/途中流局)
]
TERMINAL_CLASS_MAP = {name: i for i, name in enumerate(TERMINAL_CLASSES)}
NUM_TERMINAL_CLASSES = len(TERMINAL_CLASSES)

# yaku vocab (YakuType int → index)
# Stage02a 現在採用役
YAKU_VOCAB = [
    (0, "MenzenTsumo"),
    (1, "Riichi"),
    (2, "Ippatsu"),
    (3, "Tanyao"),
    (4, "Yakuhai"),
    (5, "Pinfu"),
    (6, "Iipeiko"),
    (7, "HaiteiTsumo"),
    (8, "HouteiRon"),
    (9, "RinshanKaihou"),
    (10, "Chankan"),
    (11, "Toitoi"),
    (12, "Ikkitsuukan"),
    (13, "SanshokuDoujun"),
]
YAKU_ID_TO_INDEX = {yid: i for i, (yid, _) in enumerate(YAKU_VOCAB)}
NUM_YAKU = len(YAKU_VOCAB)

# CQ-0266: 旧 8-class → 新 5-class mapping
_LABEL_REMAP = {
    "win_menzen": "win_menzen",
    "win_called": "win_called",
    "ryukyoku_tenpai": "draw_tenpai",
    "ron_loss": "deal_in",
    "tsumo_loss": "other_non_dealin",
    "ron_bystander": "other_non_dealin",
    "ryukyoku_noten": "other_non_dealin",
    "abortive_draw": "other_non_dealin",
    # 旧 "win" は後方互換
    "win": "win_menzen",
    # 新ラベル自身 (identity)
    "draw_tenpai": "draw_tenpai",
    "deal_in": "deal_in",
    "other_non_dealin": "other_non_dealin",
}


def terminal_label_to_class(label: str) -> int:
    """round_terminal_label → terminal class index

    旧 8-class ラベルも新 5-class にマップする。
    """
    mapped = _LABEL_REMAP.get(label, "other_non_dealin")
    return TERMINAL_CLASS_MAP[mapped]


def yaku_ids_to_multihot(yaku_ids: list[int]) -> list[float]:
    """eventual_win_yaku_ids → (NUM_YAKU,) multi-hot"""
    vec = [0.0] * NUM_YAKU
    for yid in yaku_ids:
        idx = YAKU_ID_TO_INDEX.get(yid)
        if idx is not None:
            vec[idx] = 1.0
    return vec
