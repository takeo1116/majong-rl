"""CQ-0222: Stage2 環境ラッパー（response decision を学習対象として返す）

Stage1Env の discard decision に加え、response phase の
Chi / Pon / Daiminkan / Skip を学習対象として返す。

decision_type:
  - "discard": 打牌決定（Stage1 と同じ）
  - "response": 応答決定（Chi/Pon/Daiminkan/Skip から選択）

自動処理:
  - Ron / Tsumo: 和了チャンスは自動実行
  - Riichi: テンパイ時の立直は自動
  - Ankan / Kakan: 自動スキップ
  - Kyuushu: 自動スキップ
"""
from __future__ import annotations

from enum import Enum

import numpy as np
from mahjong_rl._mahjong_core import (
    GameEngine, EnvironmentState, Action, ActionType, Phase,
    make_partial_observation, make_full_observation,
    RunMode, RewardPolicyConfig, ErrorCode, NUM_TILE_TYPES,
    RoundEndReason, EventType,
)
from mahjong_rl.legal_mask import (
    make_discard_mask_from_legal_actions,
    make_riichi_discard_mask_from_legal_actions,
    make_teacher_discard_mask_from_legal_actions,
)
from .response_candidate import (
    ResponseCandidate, extract_response_candidates, STAGE2A_LEARNABLE_TYPES,
)


def _is_red_tile_id(tile_id: int) -> bool:
    """CQ-0290: 赤牌 (5m=16, 5p=52, 5s=88) の TileId 判定"""
    return tile_id in (16, 52, 88)


class DecisionType(str, Enum):
    """決定の種類"""
    DISCARD = "discard"
    RESPONSE = "response"
    # CQ-0291 (batch 1): Riichi/NoRiichi の optional decision
    RIICHI_OPTIONAL = "riichi_optional"
    # CQ-0291 (batch 2): TsumoWin/Skip / Ron/Skip の和了 optional decision
    TSUMO_OPTIONAL = "tsumo_optional"
    RON_OPTIONAL = "ron_optional"
    # CQ-0291 (batch 3): Ankan/Skip / Kakan/Skip / Kyuushu/Skip の self-action
    # optional decision
    ANKAN_OPTIONAL = "ankan_optional"
    KAKAN_OPTIONAL = "kakan_optional"
    KYUUSHU_OPTIONAL = "kyuushu_optional"


class Stage2Env:
    """Stage2 環境: discard + response decision

    CQ-0291 batch 1: ``optional_riichi_enabled=True`` のとき、player が
    riichi 可能な tile_type を選んだ場合、即座に discard を実行せずに
    Riichi/NoRiichi の optional decision を発生させる。
    default ``False`` は既存挙動 (riichi が合法なら自動で riichi 打牌) を維持する。
    """

    def __init__(
        self,
        observation_mode: str = "full",
        reward_config: RewardPolicyConfig | None = None,
        run_mode: RunMode = RunMode.Fast,
        optional_riichi_enabled: bool = False,
        optional_tsumo_enabled: bool = False,
        optional_ron_enabled: bool = False,
        optional_ankan_enabled: bool = False,
        optional_kakan_enabled: bool = False,
        optional_kyuushu_enabled: bool = False,
    ):
        self._engine = GameEngine()
        self._env = EnvironmentState()
        self._observation_mode = observation_mode
        self._run_mode = run_mode
        if reward_config:
            self._env.reward_policy_config = reward_config
        self._current_player: int = 0
        self._done = False
        self._decision_type: DecisionType = DecisionType.DISCARD
        self._response_candidates: list[ResponseCandidate] = []
        self._round_end_events: list[dict] = []
        self._scores_at_round_start: list[int] = []
        self._last_round_outcome: dict | None = None
        self._round_outcomes_queue: list[dict] = []
        # CQ-0291 batch 1: Riichi optional decision を学習対象にするか
        # default False は既存自動 riichi 挙動と完全互換
        self._optional_riichi_enabled = bool(optional_riichi_enabled)
        # 現在の "Riichi optional" decision で deferred されている (NoRiichi, Riichi)
        # Action を記録する (decision_type=RIICHI_OPTIONAL の間だけ有効)
        self._pending_riichi_optional: tuple[Action, Action] | None = None
        # CQ-0291 batch 2: TsumoWin / Ron optional decision
        self._optional_tsumo_enabled = bool(optional_tsumo_enabled)
        self._optional_ron_enabled = bool(optional_ron_enabled)
        # CQ-0291 batch 3: Ankan / Kakan / Kyuushu optional decision
        self._optional_ankan_enabled = bool(optional_ankan_enabled)
        self._optional_kakan_enabled = bool(optional_kakan_enabled)
        self._optional_kyuushu_enabled = bool(optional_kyuushu_enabled)
        # CQ-0291 batch 2/3: SelfActionPhase で同じ optional を skip した後
        # 再度 auto_advance から提示しないようにするフラグ集合。
        # 値: {"tsumo", "ankan", "kakan", "kyuushu"}
        # 次に _execute_and_advance や reset が走るときに clear。
        self._optional_skipped_this_turn: set[str] = set()

    @property
    def action_space_size(self) -> int:
        """discard の action space (response は candidate 数で可変)"""
        return NUM_TILE_TYPES

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def observation_mode(self) -> str:
        return self._observation_mode

    @property
    def decision_type(self) -> DecisionType:
        return self._decision_type

    @property
    def response_candidates(self) -> list[ResponseCandidate]:
        """現在の response candidates (decision_type == RESPONSE のときのみ有効)"""
        return self._response_candidates

    @property
    def env_state(self) -> EnvironmentState:
        return self._env

    def reset(self, seed: int) -> tuple:
        """半荘をリセットし、最初の決定点まで進める

        Returns:
            (observation, info)
        """
        self._engine.reset_match(self._env, seed, self._run_mode)
        self._done = False
        self._decision_type = DecisionType.DISCARD
        self._response_candidates = []
        self._round_end_events = []
        self._scores_at_round_start = list(self._env.match_state.scores)
        # CQ-0291 batch 1/2/3: optional pending state をクリア
        self._pending_riichi_optional = None
        self._optional_skipped_this_turn = set()
        self._auto_advance()
        obs = self._make_observation()
        return obs, self._make_info()

    def step_discard(self, tile_type_action: int) -> tuple:
        """打牌アクション（TileType 0-33）を実行

        CQ-0291 (batch 1): ``optional_riichi_enabled=True`` かつ riichi/non-riichi
        の両方が同 tile_type で合法なときは、ここでは discard を実行せず
        ``DecisionType.RIICHI_OPTIONAL`` を発火し、agent に Riichi/NoRiichi
        を選ばせる。

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")
        if self._decision_type != DecisionType.DISCARD:
            raise RuntimeError(
                f"現在の decision_type は {self._decision_type}。"
                f"step_discard() は DISCARD 時のみ呼べます")

        if self._optional_riichi_enabled:
            legal_actions = self._engine.get_legal_actions(self._env)
            discard_actions = [a for a in legal_actions
                               if a.type == ActionType.Discard]
            riichi_optional = self._maybe_open_riichi_optional(
                tile_type_action, discard_actions)
            if riichi_optional is not None:
                # decision_type = RIICHI_OPTIONAL に切替え、step は実行しない
                return riichi_optional

        action = self._resolve_discard(tile_type_action)
        return self._execute_and_advance(action)

    # CQ-0291 batch 2/3: SelfActionPhase の optional decision_type → skip key
    # マッピング (Skip 選択時に "今の turn でこの optional は提示しない" 旨を
    # 記録するため)
    _SELF_ACTION_OPTIONAL_SKIP_KEYS = {
        DecisionType.TSUMO_OPTIONAL: "tsumo",
        DecisionType.ANKAN_OPTIONAL: "ankan",
        DecisionType.KAKAN_OPTIONAL: "kakan",
        DecisionType.KYUUSHU_OPTIONAL: "kyuushu",
    }

    def step_response(self, candidate_index: int) -> tuple:
        """response または optional candidate を index で選択して実行

        CQ-0291 (batch 1): ``DecisionType.RIICHI_OPTIONAL`` のときも本メソッドを
        使う。candidate_index=0 → NoRiichi、1 → Riichi (factory の出力順)。

        CQ-0291 (batch 2):
        - ``DecisionType.TSUMO_OPTIONAL`` のとき index=0 → TsumoWin、
          index=1 → Skip (engine step は走らせず DISCARD/次 optional に移行)
        - ``DecisionType.RON_OPTIONAL`` のとき index=0 → Ron、index=1 → Skip

        CQ-0291 (batch 3):
        - ``DecisionType.ANKAN_OPTIONAL`` / ``KAKAN_OPTIONAL`` /
          ``KYUUSHU_OPTIONAL`` のとき index=0 → primary action、
          index=1 → Skip (engine step は走らせず次 optional / DISCARD に移行)

        Args:
            candidate_index: response_candidates のインデックス

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")
        if self._decision_type not in (
                DecisionType.RESPONSE,
                DecisionType.RIICHI_OPTIONAL,
                DecisionType.TSUMO_OPTIONAL,
                DecisionType.RON_OPTIONAL,
                DecisionType.ANKAN_OPTIONAL,
                DecisionType.KAKAN_OPTIONAL,
                DecisionType.KYUUSHU_OPTIONAL):
            raise RuntimeError(
                f"現在の decision_type は {self._decision_type}。"
                f"step_response() は RESPONSE / *_OPTIONAL 時のみ呼べます")
        if not (0 <= candidate_index < len(self._response_candidates)):
            raise ValueError(
                f"candidate_index {candidate_index} は範囲外。"
                f"candidates 数: {len(self._response_candidates)}")

        # CQ-0291 batch 2/3: SelfActionPhase の optional の Skip は
        # engine step を実行せず、skipped 集合に記録した上で auto_advance を
        # 再呼出して次の decision (= 別 optional もしくは DISCARD) を探す。
        # CQ-0296: Skip 判定を candidate.action_type==OPTIONAL_SKIP_ACTION_TYPE
        # で行う。primary action 実行は ``candidate_index`` に対応する
        # candidate.action を使う (旧コードは [0] 決め打ちで、将来複数
        # primary candidate を提示したときに silent wrong action になる)。
        if self._decision_type in self._SELF_ACTION_OPTIONAL_SKIP_KEYS:
            from .response_candidate import OPTIONAL_SKIP_ACTION_TYPE
            selected = self._response_candidates[candidate_index]
            if selected.action_type == OPTIONAL_SKIP_ACTION_TYPE:
                skip_key = self._SELF_ACTION_OPTIONAL_SKIP_KEYS[
                    self._decision_type]
                self._optional_skipped_this_turn.add(skip_key)
                # 次の optional を探すため auto_advance を再呼出
                # (SelfActionPhase のまま、phase 移行は engine step が無い限り
                # 起きない)
                self._decision_type = DecisionType.DISCARD
                self._response_candidates = []
                rewards = self._auto_advance()
                obs = self._make_observation()
                return (obs, rewards, self._done, False, self._make_info())
            # primary action: 通常通り engine step
            # CQ-0296: candidate_index に対応する candidate.action を実行
            action = selected.action
            return self._execute_and_advance(action)

        action = self._response_candidates[candidate_index].action
        # CQ-0291: RIICHI_OPTIONAL から復帰するときは pending state を clear
        if self._decision_type == DecisionType.RIICHI_OPTIONAL:
            self._pending_riichi_optional = None
        return self._execute_and_advance(action)

    def get_legal_mask(self) -> np.ndarray:
        """34 種打牌の legal mask (discard decision 用)

        CQ-0292 (batch 2): ``optional_riichi_enabled=True`` のときは riichi
        打牌の有無に関わらず全 Discard tile_type を mask に含める。これに
        より policy が riichi/no-riichi を ``RIICHI_OPTIONAL`` decision で
        選べるようになる。default off では既存どおり、riichi 打牌が存在
        すれば riichi tile_type のみを mask に含める。
        """
        actions = self._engine.get_legal_actions(self._env)
        return make_discard_mask_from_legal_actions(
            actions,
            include_all_discards=self._optional_riichi_enabled,
        )

    def get_legal_discard_snapshot(self) -> tuple[np.ndarray, list]:
        """CQ-0271: legal mask と legal discard actions を同時に取得する

        CQ-0292 (batch 2): ``optional_riichi_enabled=True`` のときは
        ``get_legal_mask`` と同様に全 Discard tile_type を mask に含める。

        Returns:
            (legal_mask, discard_actions): policy 用 mask と concrete
                actions の snapshot
        """
        all_actions = self._engine.get_legal_actions(self._env)
        discard_actions = [a for a in all_actions
                           if a.type == ActionType.Discard]
        mask = make_discard_mask_from_legal_actions(
            all_actions,
            include_all_discards=self._optional_riichi_enabled,
        )
        return mask, discard_actions

    @staticmethod
    def get_teacher_discard_mask_from_snapshot(
        discard_actions,
    ) -> np.ndarray:
        """CQ-0294: rulebase teacher / baseline 用 discard mask を返す。

        ``optional_riichi_enabled`` の値に依らず、常に旧 auto-riichi
        優先 semantics (= riichi 打牌があるなら riichi tile_type のみ、
        無ければ通常 discard tile_type) で mask を作る。teacher / baseline
        は ``optional_riichi`` 有効時でも旧仕様を維持できる。

        ``discard_actions`` には ``get_legal_discard_snapshot`` で取得した
        Discard action リスト (riichi flag 付き) を渡す。policy 用 mask
        の superset として policy mask 上でも合法であることが保証される
        (= teacher mask は policy mask の subset)。
        """
        return make_teacher_discard_mask_from_legal_actions(discard_actions)

    def get_riichi_discard_mask(self) -> np.ndarray:
        """CQ-0294: ``feature_encoder.riichi_discard_mask`` feature 用の
        34 次元 flag を返す。

        engine の legal actions に含まれる ``Discard`` action のうち
        ``riichi=True`` の tile_type を 1.0 にする。立直不可局面では
        全 0。``optional_riichi_enabled`` の真偽とは独立に常に同じ式で
        計算する。
        """
        actions = self._engine.get_legal_actions(self._env)
        return make_riichi_discard_mask_from_legal_actions(actions)

    def step_discard_with_snapshot(
        self, tile_type: int, discard_actions: list,
    ) -> tuple:
        """CQ-0271: snapshot から concrete action を解決して実行する

        CQ-0291 (batch 1): ``optional_riichi_enabled=True`` かつ riichi/non-riichi
        の両方が同 tile_type で合法なときは、ここでは discard を実行せず
        ``DecisionType.RIICHI_OPTIONAL`` を発火する。

        Args:
            tile_type: policy/baseline が選択した牌種 (0-33)
            discard_actions: get_legal_discard_snapshot() で取得した discard actions

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")
        if self._decision_type != DecisionType.DISCARD:
            raise RuntimeError(
                f"現在の decision_type は {self._decision_type}。"
                f"step_discard_with_snapshot() は DISCARD 時のみ呼べます")

        if self._optional_riichi_enabled:
            riichi_optional = self._maybe_open_riichi_optional(
                tile_type, discard_actions)
            if riichi_optional is not None:
                return riichi_optional

        action = self._resolve_discard_from_snapshot(
            tile_type, discard_actions)
        return self._execute_and_advance(action)

    def _maybe_open_riichi_optional(
        self, tile_type: int, discard_actions: list,
    ) -> tuple | None:
        """CQ-0291 (batch 1): riichi/non-riichi の両方が同 tile_type で合法なら
        Riichi optional decision を発生させ obs を返す。そうでなければ None。

        engine の Discard 列挙は CQ-0290 で同 tile_type 内 通常 → 赤 の順。
        riichi flag は別軸なので、この helper では riichi/non-riichi の代表
        action を tile_id で sort key (`(not riichi, _is_red_tile_id(tile))`)
        に従って 1 つずつ拾う。
        """
        riichi_act = None
        non_riichi_act = None
        # CQ-0290 と同じ sort key で安定的に選ぶ
        same_type = [a for a in discard_actions if (a.tile // 4) == tile_type]
        if not same_type:
            return None
        same_type.sort(key=lambda a: (
            not a.riichi,                # riichi 先 → False(=riichi) が先
            _is_red_tile_id(a.tile),    # 通常牌先
        ))
        # 最初の riichi / non-riichi をそれぞれ拾う
        for a in same_type:
            if a.riichi and riichi_act is None:
                riichi_act = a
            elif (not a.riichi) and non_riichi_act is None:
                non_riichi_act = a
            if riichi_act is not None and non_riichi_act is not None:
                break
        if riichi_act is None or non_riichi_act is None:
            return None
        # decision_type を RIICHI_OPTIONAL に切替え
        self._pending_riichi_optional = (non_riichi_act, riichi_act)
        from .response_candidate import make_riichi_optional_candidates
        self._response_candidates = make_riichi_optional_candidates(
            non_riichi_act, riichi_act,
            current_player=self._env.round_state.current_player,
        )
        self._decision_type = DecisionType.RIICHI_OPTIONAL
        # discard 実行はまだしない
        obs = self._make_observation()
        rewards = np.zeros(4, dtype=np.float32)
        return obs, rewards, self._done, False, self._make_info()

    @staticmethod
    def _resolve_discard_from_snapshot(
        tile_type: int, discard_actions: list,
    ) -> 'Action':
        """CQ-0271: snapshot から tile_type に対応する concrete Action を解決する

        CQ-0290: 同一 tile_type 内で通常牌 → 赤牌の順に優先する
        (riichi 優先、次に通常牌優先)
        """
        candidates = [a for a in discard_actions if (a.tile // 4) == tile_type]
        if candidates:
            candidates.sort(key=lambda a: (
                not a.riichi,
                _is_red_tile_id(a.tile),
            ))
            return candidates[0]
        # 診断情報付きエラー
        action_info = [(a.tile // 4, a.riichi) for a in discard_actions]
        mask_types = sorted(set(a.tile // 4 for a in discard_actions))
        raise ValueError(
            f"CQ-0271 snapshot 不整合: tile_type={tile_type} は"
            f" snapshot 内の合法打牌に存在しません。\n"
            f"  legal discard tile_types: {mask_types}\n"
            f"  discard_actions: {action_info}\n"
            f"  snapshot size: {len(discard_actions)}"
        )

    def _execute_and_advance(self, action: Action) -> tuple:
        """action を実行し、次の決定点まで進める"""
        # CQ-0291 batch 2/3: 任意の engine step が走るときに optional skip
        # 集合を clear する (新しい turn / phase 移行のため)
        self._optional_skipped_this_turn = set()
        result = self._engine.step(self._env, action)
        if result.error != ErrorCode.Ok:
            raise RuntimeError(f"step エラー: {result.error}")

        step_rewards = np.array(result.rewards, dtype=np.float32)
        self._round_end_events = []

        if result.round_over:
            self._capture_round_end(result)
            if result.match_over:
                self._done = True
            else:
                self._engine.advance_round(self._env)
                self._scores_at_round_start = list(
                    self._env.match_state.scores)

        terminated = self._done
        if not terminated:
            auto_rewards = self._auto_advance()
            step_rewards = step_rewards + auto_rewards
            terminated = self._done

        obs = self._make_observation()
        return obs, step_rewards, terminated, False, self._make_info()

    def _auto_advance(self) -> np.ndarray:
        """非学習対象のアクションを自動実行し、次の決定点まで進める"""
        accumulated_rewards = np.zeros(4, dtype=np.float32)

        for _ in range(10000):
            if self._done:
                break

            phase = self._env.round_state.phase
            if phase in (Phase.EndRound, Phase.EndMatch):
                break

            legal_actions = self._engine.get_legal_actions(self._env)
            if not legal_actions:
                break

            if phase == Phase.SelfActionPhase:
                # CQ-0291 batch 2: TsumoWin optional unlock
                tsumo_action = next(
                    (a for a in legal_actions
                     if a.type == ActionType.TsumoWin), None)
                if (tsumo_action is not None
                        and self._optional_tsumo_enabled
                        and "tsumo" not in self._optional_skipped_this_turn):
                    from .response_candidate import (
                        make_tsumo_optional_candidates)
                    self._current_player = (
                        self._env.round_state.current_player)
                    self._decision_type = DecisionType.TSUMO_OPTIONAL
                    self._response_candidates = make_tsumo_optional_candidates(
                        tsumo_action, skip_action=None,
                        current_player=self._current_player)
                    return accumulated_rewards

                # CQ-0291 batch 3: Ankan / Kakan / Kyuushu optional unlock
                # 各 family は独立に skip 状態を保持し、self_action 内で
                # 順に提示する
                ankan_action = next(
                    (a for a in legal_actions
                     if a.type == ActionType.Ankan), None)
                if (ankan_action is not None
                        and self._optional_ankan_enabled
                        and "ankan" not in self._optional_skipped_this_turn):
                    from .response_candidate import (
                        make_ankan_optional_candidates)
                    self._current_player = (
                        self._env.round_state.current_player)
                    self._decision_type = DecisionType.ANKAN_OPTIONAL
                    self._response_candidates = make_ankan_optional_candidates(
                        ankan_action, skip_action=None,
                        current_player=self._current_player)
                    return accumulated_rewards

                kakan_action = next(
                    (a for a in legal_actions
                     if a.type == ActionType.Kakan), None)
                if (kakan_action is not None
                        and self._optional_kakan_enabled
                        and "kakan" not in self._optional_skipped_this_turn):
                    from .response_candidate import (
                        make_kakan_optional_candidates)
                    self._current_player = (
                        self._env.round_state.current_player)
                    self._decision_type = DecisionType.KAKAN_OPTIONAL
                    self._response_candidates = make_kakan_optional_candidates(
                        kakan_action, skip_action=None,
                        current_player=self._current_player)
                    return accumulated_rewards

                kyuushu_action = next(
                    (a for a in legal_actions
                     if a.type == ActionType.Kyuushu), None)
                if (kyuushu_action is not None
                        and self._optional_kyuushu_enabled
                        and "kyuushu" not in self._optional_skipped_this_turn):
                    from .response_candidate import (
                        make_kyuushu_optional_candidates)
                    self._current_player = (
                        self._env.round_state.current_player)
                    self._decision_type = DecisionType.KYUUSHU_OPTIONAL
                    self._response_candidates = make_kyuushu_optional_candidates(
                        kyuushu_action, skip_action=None,
                        current_player=self._current_player)
                    return accumulated_rewards

                auto_action = self._get_auto_self_action(legal_actions)
                if auto_action is None:
                    # 打牌決定が必要 → エージェントに返す
                    self._current_player = self._env.round_state.current_player
                    self._decision_type = DecisionType.DISCARD
                    self._response_candidates = []
                    return accumulated_rewards

            elif phase == Phase.ResponsePhase:
                # response decision を学習対象にするか判定
                responder = self._env.round_state.current_player
                candidates = extract_response_candidates(
                    legal_actions, responder, learnable_only=True)

                # CQ-0291 batch 2: Ron optional unlock
                ron_action = next(
                    (a for a in legal_actions
                     if a.type == ActionType.Ron), None)
                if ron_action is not None and self._optional_ron_enabled:
                    # 自動 Ron せず Ron/Skip optional decision を返す
                    skip_action = next(
                        (a for a in legal_actions
                         if a.type == ActionType.Skip),
                        None)
                    if skip_action is not None:
                        from .response_candidate import (
                            make_ron_optional_candidates)
                        self._current_player = responder
                        self._decision_type = DecisionType.RON_OPTIONAL
                        self._response_candidates = make_ron_optional_candidates(
                            ron_action, skip_action,
                            current_player=responder)
                        return accumulated_rewards

                # Ron は自動処理
                has_ron = ron_action is not None
                if has_ron:
                    auto_action = ron_action
                elif len(candidates) > 1:
                    # Skip 以外の選択肢がある → エージェントに返す
                    self._current_player = responder
                    self._decision_type = DecisionType.RESPONSE
                    self._response_candidates = candidates
                    return accumulated_rewards
                else:
                    # Skip のみ → 自動スキップ
                    auto_action = next(
                        (a for a in legal_actions if a.type == ActionType.Skip),
                        legal_actions[0])

            elif phase == Phase.DrawPhase:
                auto_action = legal_actions[0]
            else:
                auto_action = legal_actions[0]

            result = self._engine.step(self._env, auto_action)
            if result.error != ErrorCode.Ok:
                break

            accumulated_rewards += np.array(result.rewards, dtype=np.float32)

            if result.round_over:
                self._capture_round_end(result)
                if result.match_over:
                    self._done = True
                else:
                    self._engine.advance_round(self._env)
                    self._scores_at_round_start = list(
                        self._env.match_state.scores)

        return accumulated_rewards

    def _get_auto_self_action(self, legal_actions) -> Action | None:
        """SelfActionPhase で自動実行すべきアクションを返す"""
        # ツモ和了があれば自動実行 (TSUMO_OPTIONAL で Skip 済みなら除外)
        if "tsumo" not in self._optional_skipped_this_turn:
            for a in legal_actions:
                if a.type == ActionType.TsumoWin:
                    return a
        # 暗槓・加槓・九種九牌はスキップ
        discard_actions = [a for a in legal_actions
                           if a.type == ActionType.Discard]
        if discard_actions:
            return None  # エージェントに委ねる
        return legal_actions[0]

    def _resolve_discard(self, tile_type: int) -> Action:
        """TileType (0-33) から具体的な Discard Action を生成する

        注意: この方法は legal actions を再取得するため、
        snapshot と一致しない可能性がある。CQ-0271 以降は
        step_discard_with_snapshot() の使用を推奨。

        CQ-0290: 同一 tile_type 内で通常牌 → 赤牌の順に優先する
        """
        legal_actions = self._engine.get_legal_actions(self._env)
        candidates = [
            a for a in legal_actions
            if a.type == ActionType.Discard and (a.tile // 4) == tile_type
        ]
        if candidates:
            candidates.sort(key=lambda a: (
                not a.riichi,
                _is_red_tile_id(a.tile),
            ))
            return candidates[0]
        # CQ-0271: 診断情報付きエラー
        discard_types = sorted(set(
            a.tile // 4 for a in legal_actions if a.type == ActionType.Discard))
        rs = self._env.round_state
        raise ValueError(
            f"tile_type {tile_type} は合法打牌ではありません。\n"
            f"  legal discard tile_types: {discard_types}\n"
            f"  current_player: {rs.current_player}\n"
            f"  phase: {rs.phase}"
        )

    def _make_observation(self):
        if self._observation_mode == "full":
            return make_full_observation(self._env)
        return make_partial_observation(self._env, self._current_player)

    def _capture_round_end(self, result) -> None:
        rs = self._env.round_state
        ms = self._env.match_state
        end_reason = rs.end_reason
        scores_after = list(ms.scores)
        if end_reason == RoundEndReason.Tsumo:
            event_type = "tsumo"
        elif end_reason == RoundEndReason.Ron:
            event_type = "ron"
        else:
            event_type = "ryukyoku"
        winner_players: list[int] = []
        loser_player = -1
        if event_type == "ron":
            winner_players = [e.actor for e in result.events
                              if e.type == EventType.Ron]
            loser_player = rs.last_discarder
        elif event_type == "tsumo":
            deltas = [scores_after[i] - self._scores_at_round_start[i]
                      for i in range(4)]
            winner = max(range(4), key=lambda i: deltas[i])
            winner_players = [winner]

        # CQ-0227: round outcome summary from C++
        outcome = None
        try:
            from mahjong_rl._mahjong_core import get_round_outcome
            outcome = get_round_outcome(self._env)
        except (ImportError, Exception):
            pass

        self._round_end_events.append({
            "event_type": event_type,
            "winner_players": winner_players,
            "loser_player": loser_player,
            "round_id": rs.round_number,
            "outcome": outcome,
        })
        self._last_round_outcome = outcome
        if outcome is not None:
            self._round_outcomes_queue.append(outcome)

    def _make_info(self) -> dict:
        info = {
            "current_player": self._current_player,
            "phase": self._env.round_state.phase,
            "round_number": self._env.round_state.round_number,
            "scores": list(self._env.match_state.scores),
            "is_match_over": self._env.match_state.is_match_over,
            "decision_type": self._decision_type.value,
        }
        if self._decision_type == DecisionType.RESPONSE:
            info["num_response_candidates"] = len(self._response_candidates)
            info["response_candidate_types"] = [
                c.action_type.name for c in self._response_candidates
            ]
        if self._round_end_events:
            info["round_end_events"] = list(self._round_end_events)
        return info
