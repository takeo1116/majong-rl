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
from mahjong_rl.legal_mask import make_discard_mask_from_legal_actions
from .response_candidate import (
    ResponseCandidate, extract_response_candidates, STAGE2A_LEARNABLE_TYPES,
)


class DecisionType(str, Enum):
    """決定の種類"""
    DISCARD = "discard"
    RESPONSE = "response"


class Stage2Env:
    """Stage2 環境: discard + response decision"""

    def __init__(
        self,
        observation_mode: str = "full",
        reward_config: RewardPolicyConfig | None = None,
        run_mode: RunMode = RunMode.Fast,
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
        self._auto_advance()
        obs = self._make_observation()
        return obs, self._make_info()

    def step_discard(self, tile_type_action: int) -> tuple:
        """打牌アクション（TileType 0-33）を実行

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")
        if self._decision_type != DecisionType.DISCARD:
            raise RuntimeError(
                f"現在の decision_type は {self._decision_type}。"
                f"step_discard() は DISCARD 時のみ呼べます")

        action = self._resolve_discard(tile_type_action)
        return self._execute_and_advance(action)

    def step_response(self, candidate_index: int) -> tuple:
        """response candidate を index で選択して実行

        Args:
            candidate_index: response_candidates のインデックス

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")
        if self._decision_type != DecisionType.RESPONSE:
            raise RuntimeError(
                f"現在の decision_type は {self._decision_type}。"
                f"step_response() は RESPONSE 時のみ呼べます")
        if not (0 <= candidate_index < len(self._response_candidates)):
            raise ValueError(
                f"candidate_index {candidate_index} は範囲外。"
                f"candidates 数: {len(self._response_candidates)}")

        action = self._response_candidates[candidate_index].action
        return self._execute_and_advance(action)

    def get_legal_mask(self) -> np.ndarray:
        """34 種打牌の legal mask (discard decision 用)"""
        actions = self._engine.get_legal_actions(self._env)
        return make_discard_mask_from_legal_actions(actions)

    def _execute_and_advance(self, action: Action) -> tuple:
        """action を実行し、次の決定点まで進める"""
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

                # Ron は自動処理
                has_ron = any(a.type == ActionType.Ron for a in legal_actions)
                if has_ron:
                    auto_action = next(a for a in legal_actions
                                       if a.type == ActionType.Ron)
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
        # ツモ和了があれば自動実行
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
        """TileType (0-33) から具体的な Discard Action を生成する"""
        legal_actions = self._engine.get_legal_actions(self._env)
        # 立直打牌を優先（テンパイ時自動立直）
        for a in legal_actions:
            if a.type == ActionType.Discard and a.riichi and (a.tile // 4) == tile_type:
                return a
        for a in legal_actions:
            if a.type == ActionType.Discard and not a.riichi and (a.tile // 4) == tile_type:
                return a
        raise ValueError(
            f"tile_type {tile_type} は合法打牌ではありません")

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
