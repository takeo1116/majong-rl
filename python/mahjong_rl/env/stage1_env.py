"""Stage 1 DiscardOnly 環境ラッパー

行動空間: 34 牌種（TileType 0-33）
観測: PartialObservation or FullObservation（設定で切替可能）
自動処理: ツモ和了、ロン、立直、副露スキップ、九種九牌スキップ
"""
from __future__ import annotations

import numpy as np
from mahjong_rl._mahjong_core import (
    GameEngine, EnvironmentState, Action, ActionType, Phase,
    make_partial_observation, make_full_observation,
    RunMode, RewardPolicyConfig, ErrorCode, NUM_TILE_TYPES,
    RoundEndReason, EventType,
)
from mahjong_rl.legal_mask import make_discard_mask_from_legal_actions


class Stage1Env:
    """Stage 1 DiscardOnly 環境ラッパー"""

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
        # 局終了イベント蓄積用 (CQ-0105)
        self._round_end_events: list[dict] = []
        self._scores_at_round_start: list[int] = []

    @property
    def action_space_size(self) -> int:
        return NUM_TILE_TYPES

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def observation_mode(self) -> str:
        return self._observation_mode

    @property
    def env_state(self) -> EnvironmentState:
        """内部状態への直接アクセス（デバッグ・テスト用）"""
        return self._env

    def reset(self, seed: int) -> tuple:
        """半荘をリセットし、最初の打牌決定点まで進める

        Returns:
            (observation, info)
        """
        self._engine.reset_match(self._env, seed, self._run_mode)
        self._done = False
        self._round_end_events = []
        self._scores_at_round_start = list(self._env.match_state.scores)
        self._auto_advance()
        obs = self._make_observation()
        return obs, self._make_info()

    def step(self, tile_type_action: int) -> tuple:
        """打牌アクション（TileType 0-33）を実行

        Returns:
            (observation, rewards, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("環境は終了済み。reset() を呼んでください")

        action = self._resolve_discard(tile_type_action)
        result = self._engine.step(self._env, action)

        if result.error != ErrorCode.Ok:
            raise RuntimeError(f"step エラー: {result.error}")

        step_rewards = np.array(result.rewards, dtype=np.float32)

        # 局終了イベントを各 step 呼び出し単位でリセット
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
            # 非打牌フェーズを自動消化
            auto_rewards = self._auto_advance()
            step_rewards = step_rewards + auto_rewards
            terminated = self._done

        obs = self._make_observation()
        return obs, step_rewards, terminated, False, self._make_info()

    def get_legal_mask(self) -> np.ndarray:
        """34 種打牌の legal mask を返す"""
        actions = self._engine.get_legal_actions(self._env)
        return make_discard_mask_from_legal_actions(actions)

    def _auto_advance(self) -> np.ndarray:
        """非打牌アクションを自動実行し、次の打牌決定点まで進める

        Returns:
            自動消化中に蓄積された報酬
        """
        accumulated_rewards = np.zeros(4, dtype=np.float32)

        max_iterations = 10000
        for _ in range(max_iterations):
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
                    return accumulated_rewards
            elif phase == Phase.ResponsePhase:
                auto_action = self._get_auto_response(legal_actions)
            elif phase == Phase.DrawPhase:
                # DrawPhase は通常 engine が自動処理するが、念のため
                auto_action = legal_actions[0]
            else:
                # その他のフェーズでは最初の合法手を実行
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
        """SelfActionPhase で自動実行すべきアクションを返す
        打牌決定はエージェントに委ねるため None を返す
        """
        # ツモ和了があれば自動実行
        for a in legal_actions:
            if a.type == ActionType.TsumoWin:
                return a

        # 暗槓・加槓はスキップ（打牌のみ）
        # 九種九牌もスキップ
        discard_actions = [a for a in legal_actions
                           if a.type == ActionType.Discard]
        if discard_actions:
            return None  # エージェントに委ねる

        # 打牌がない場合（通常は到達しない）
        return legal_actions[0]

    def _get_auto_response(self, legal_actions) -> Action:
        """ResponsePhase で自動実行するアクションを返す"""
        # ロンがあれば自動実行
        for a in legal_actions:
            if a.type == ActionType.Ron:
                return a
        # それ以外は常にスキップ（副露なし）
        for a in legal_actions:
            if a.type == ActionType.Skip:
                return a
        return legal_actions[0]

    def _resolve_discard(self, tile_type: int) -> Action:
        """TileType (0-33) から具体的な Discard Action を生成する"""
        legal_actions = self._engine.get_legal_actions(self._env)

        # 立直打牌を優先（テンパイ時自動立直）
        for a in legal_actions:
            if a.type == ActionType.Discard and a.riichi and (a.tile // 4) == tile_type:
                return a

        # 通常打牌
        for a in legal_actions:
            if a.type == ActionType.Discard and not a.riichi and (a.tile // 4) == tile_type:
                return a

        raise ValueError(
            f"tile_type {tile_type} は合法打牌ではありません。"
            f"合法マスク: {self.get_legal_mask()}"
        )

    def make_dual_observation(self) -> tuple:
        """同一局面から Full / Partial 両方の Observation を返す

        蒸留実験の土台: teacher (Full) と student (Partial) に同一局面を渡す用途。

        Returns:
            (FullObservation, PartialObservation)
        """
        full_obs = make_full_observation(self._env)
        partial_obs = make_partial_observation(self._env, self._current_player)
        return full_obs, partial_obs

    def _make_observation(self):
        if self._observation_mode == "full":
            return make_full_observation(self._env)
        else:
            return make_partial_observation(self._env, self._current_player)

    def _capture_round_end(self, result) -> None:
        """局終了時の情報を蓄積する (CQ-0105)

        advance_round() の前に呼ぶこと（round_state がまだ当該局の情報を保持している間）。
        """
        rs = self._env.round_state
        ms = self._env.match_state
        end_reason = rs.end_reason
        scores_after = list(ms.scores)

        # event_type 判定
        if end_reason == RoundEndReason.Tsumo:
            event_type = "tsumo"
        elif end_reason == RoundEndReason.Ron:
            event_type = "ron"
        else:
            event_type = "ryukyoku"

        # winner_players 判定（multi-ron 対応）
        winner_players: list[int] = []
        loser_player = -1

        if event_type == "ron":
            # StepResult.events から Ron イベントを取得し、全勝者を列挙
            winner_players = [
                e.actor for e in result.events
                if e.type == EventType.Ron
            ]
            loser_player = rs.last_discarder
        elif event_type == "tsumo":
            # 点数変動の最大者を勝者とする
            deltas = [scores_after[i] - self._scores_at_round_start[i]
                      for i in range(4)]
            winner = max(range(4), key=lambda i: deltas[i])
            winner_players = [winner]

        self._round_end_events.append({
            "event_type": event_type,
            "winner_players": winner_players,
            "loser_player": loser_player,
            "round_id": rs.round_number,
        })

    def _make_info(self) -> dict:
        info = {
            "current_player": self._current_player,
            "phase": self._env.round_state.phase,
            "round_number": self._env.round_state.round_number,
            "scores": list(self._env.match_state.scores),
            "is_match_over": self._env.match_state.is_match_over,
        }
        # 局終了イベントがあれば付与 (CQ-0105)
        if self._round_end_events:
            info["round_end_events"] = list(self._round_end_events)
        return info
