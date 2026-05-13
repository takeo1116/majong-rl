"""CQ-0230: Stage2a deterministic evaluation

learned Stage2a policy を argmax で行動させ、
baseline 対戦相手と対戦して成績を計測する。
"""
from __future__ import annotations

import numpy as np
import torch

from mahjong_rl.env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import ResponseCandidate
from mahjong_rl.baseline.rule_based import RuleBasedBaseline
from mahjong_rl.baseline.call_policy import RuleBasedCallPolicy
from mahjong_rl.stage2a_selector import select_discard_argmax, select_optional_argmax
from mahjong_rl.stage2a_learner import _encode_candidates_batch
from mahjong_rl.call_shard import DecisionSample, CandidateRecord


class Stage2aEvaluator:
    """Stage2a deterministic evaluator

    fixed-seat: policy player = seat 0, baseline = seats 1-3
    """

    def __init__(
        self,
        model,
        encoder,
        observation_mode: str = "full",
        device=None,
        reward_config=None,
        optional_riichi_enabled: bool = False,
        optional_tsumo_enabled: bool = False,
        optional_ron_enabled: bool = False,
        optional_ankan_enabled: bool = False,
        optional_kakan_enabled: bool = False,
        optional_kyuushu_enabled: bool = False,
    ):
        self._model = model
        self._encoder = encoder
        self._obs_mode = observation_mode
        self._device = device or torch.device("cpu")
        self._reward_config = reward_config  # CQ-0276
        # CQ-0292: optional decision flags (default off で従来挙動を維持)
        self._optional_riichi_enabled = bool(optional_riichi_enabled)
        self._optional_tsumo_enabled = bool(optional_tsumo_enabled)
        self._optional_ron_enabled = bool(optional_ron_enabled)
        self._optional_ankan_enabled = bool(optional_ankan_enabled)
        self._optional_kakan_enabled = bool(optional_kakan_enabled)
        self._optional_kyuushu_enabled = bool(optional_kyuushu_enabled)
        self._model.to(self._device)
        self._model.eval()
        self._baseline = RuleBasedBaseline()

        # CQ-0244: preallocated inference buffers
        obs_dim = encoder.metadata().output_shape[0]
        self._feat_buf = torch.zeros(1, obs_dim, dtype=torch.float32, device=self._device)
        self._mask_buf = torch.zeros(1, 34, dtype=torch.float32, device=self._device)
        self._rc_buf = torch.zeros(1, 3, dtype=torch.float32, device=self._device)
        self._call_baseline = RuleBasedCallPolicy()

    def evaluate(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
        policy_seat: int = 0,
    ) -> dict:
        """fixed-seat evaluation

        Args:
            num_matches: 半荘数
            seed_start: 開始 seed
            policy_seat: policy が座る席 (0-3)

        Returns:
            eval metrics dict
        """
        env = Stage2Env(observation_mode=self._obs_mode,
                         reward_config=self._reward_config,
                         optional_riichi_enabled=self._optional_riichi_enabled,
                         optional_tsumo_enabled=self._optional_tsumo_enabled,
                         optional_ron_enabled=self._optional_ron_enabled,
                         optional_ankan_enabled=self._optional_ankan_enabled,
                         optional_kakan_enabled=self._optional_kakan_enabled,
                         optional_kyuushu_enabled=self._optional_kyuushu_enabled)
        scores: list[int] = []
        ranks: list[int] = []
        wins = 0
        deal_ins = 0
        total_rounds = 0

        for match_idx in range(num_matches):
            seed = seed_start + match_idx
            env.reset(seed)

            for _ in range(5000):
                player = env.current_player
                is_policy = (player == policy_seat)

                if env.decision_type == DecisionType.DISCARD:
                    # CQ-0271: snapshot-based discard
                    # CQ-0294: policy seat と baseline seat で mask を分離。
                    # baseline seat は teacher mask (riichi-only 優先) を
                    # 使うことで optional_riichi 有効時でも旧 auto-riichi
                    # 相当の挙動を維持する。
                    mask, discard_snap = env.get_legal_discard_snapshot()
                    if is_policy:
                        action = self._policy_discard(env, mask)
                    else:
                        teacher_mask = (
                            env.get_teacher_discard_mask_from_snapshot(
                                discard_snap))
                        hand_ids = list(
                            env.env_state.round_state.players[player].hand)
                        mc = len(env.env_state.round_state.players[player].melds)
                        action = self._baseline.select_discard(
                            hand_ids, teacher_mask, meld_count=mc)
                    _, _, terminated, _, info = env.step_discard_with_snapshot(
                        action, discard_snap)

                elif env.decision_type == DecisionType.RESPONSE:
                    candidates = env.response_candidates
                    if is_policy:
                        idx = self._policy_call(env, candidates, player)
                    else:
                        hand_ids = list(env.env_state.round_state.players[player].hand)
                        idx = self._call_baseline.select_action(candidates, hand_ids)
                    _, _, terminated, _, info = env.step_response(idx)

                elif env.decision_type in (
                    DecisionType.RIICHI_OPTIONAL,
                    DecisionType.TSUMO_OPTIONAL,
                    DecisionType.RON_OPTIONAL,
                    DecisionType.ANKAN_OPTIONAL,
                    DecisionType.KAKAN_OPTIONAL,
                    DecisionType.KYUUSHU_OPTIONAL,
                ):
                    # CQ-0292: optional decision を policy / baseline で選択
                    candidates = env.response_candidates
                    if is_policy:
                        idx = self._policy_call(env, candidates, player)
                    else:
                        idx = self._baseline_optional_index(env.decision_type)
                    _, _, terminated, _, info = env.step_response(idx)

                # round outcome
                for re in info.get("round_end_events", []):
                    total_rounds += 1
                    if policy_seat in re.get("winner_players", []):
                        wins += 1
                    if re.get("loser_player") == policy_seat:
                        deal_ins += 1

                if terminated:
                    break

            # match end scores
            final_scores = list(env.env_state.match_state.scores)
            scores.append(final_scores[policy_seat])
            # rank
            sorted_scores = sorted(final_scores, reverse=True)
            rank = sorted_scores.index(final_scores[policy_seat]) + 1
            ranks.append(rank)

        avg_rank = float(np.mean(ranks)) if ranks else 0.0
        avg_score = float(np.mean(scores)) if scores else 0.0
        win_rate = wins / total_rounds if total_rounds > 0 else 0.0
        deal_in_rate = deal_ins / total_rounds if total_rounds > 0 else 0.0

        return {
            "avg_rank": avg_rank,
            "avg_score": avg_score,
            "win_rate": win_rate,
            "deal_in_rate": deal_in_rate,
            "num_matches": num_matches,
            "total_rounds": total_rounds,
            "wins": wins,
            "deal_ins": deal_ins,
            "policy_seat": policy_seat,
            "eval_mode": "fixed_seat",
        }

    def evaluate_rotation(
        self,
        num_matches: int = 100,
        seed_start: int = 0,
    ) -> dict:
        """rotation eval: 各席で num_matches 半荘ずつ打つ"""
        all_ranks = []
        all_scores = []
        total_wins = 0
        total_deal_ins = 0
        total_rounds = 0

        for seat in range(4):
            result = self.evaluate(
                num_matches=num_matches,
                seed_start=seed_start + seat * num_matches,
                policy_seat=seat,
            )
            all_ranks.extend([result["avg_rank"]])
            all_scores.extend([result["avg_score"]])
            total_wins += result["wins"]
            total_deal_ins += result["deal_ins"]
            total_rounds += result["total_rounds"]

        return {
            "avg_rank": float(np.mean(all_ranks)),
            "avg_score": float(np.mean(all_scores)),
            "win_rate": total_wins / total_rounds if total_rounds > 0 else 0.0,
            "deal_in_rate": total_deal_ins / total_rounds if total_rounds > 0 else 0.0,
            "num_matches": num_matches * 4,
            "total_rounds": total_rounds,
            "eval_mode": "rotation",
        }

    @staticmethod
    def _baseline_optional_index(decision_type: DecisionType) -> int:
        """CQ-0292: optional decision baseline teacher index.

        - RIICHI_OPTIONAL: Riichi (idx=1)  既存自動 riichi
        - TSUMO_OPTIONAL / RON_OPTIONAL: Win (idx=0)  既存自動和了
        - ANKAN_OPTIONAL / KAKAN_OPTIONAL / KYUUSHU_OPTIONAL: Skip (idx=1)
          既存 Stage02a の自動 Skip 挙動
        """
        if decision_type == DecisionType.RIICHI_OPTIONAL:
            return 1
        if decision_type in (DecisionType.TSUMO_OPTIONAL,
                              DecisionType.RON_OPTIONAL):
            return 0
        return 1  # ankan / kakan / kyuushu → Skip

    def _policy_discard(self, env: Stage2Env, mask: np.ndarray) -> int:
        features = self._encode_env_obs(env, legal_mask=mask)
        if features.ndim > 1:
            features = features.flatten()
        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features))
            self._mask_buf[0].copy_(torch.from_numpy(mask))
            out = self._model.forward_discard(
                self._feat_buf, self._mask_buf, compute_value=False)
            action, _ = select_discard_argmax(out.discard_logits[0], self._mask_buf[0])
        return action

    def _encode_env_obs(
        self,
        env: Stage2Env,
        legal_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode current env state with mask-dependent auxiliary features.

        CQ-0297 follow-up: at ``ResponsePhase`` (RESPONSE / RON_OPTIONAL)
        ``env.get_legal_mask()`` returns an all-zero mask because no Discard
        actions are legal. Passing that mask to the encoder would silently
        change ``discard_ukeire_hint`` from "unrestricted normalized
        acceptance" (pre-CQ-0297) to "all-zero", which would break
        backward compatibility with response-branch features used by
        optional-off baselines. Detect the no-discard case and revert to
        ``legal_mask=None`` so the encoder preserves the legacy behavior.
        SelfActionPhase optional decisions retain the new mask-aware path.
        """
        obs = env._make_observation()
        if legal_mask is None:
            legal_mask = env.get_legal_mask()
            if legal_mask.sum() == 0:
                # ResponsePhase: no Discard legal; preserve pre-CQ-0297
                # encoder semantics (legal_mask=None) to avoid altering
                # response-branch features for optional-off baselines.
                legal_mask = None
        if getattr(self._encoder, "_riichi_discard_mask", False):
            r_mask = env.get_riichi_discard_mask()
            features = self._encoder.encode(
                obs, legal_mask=legal_mask, riichi_discard_mask=r_mask)
        else:
            features = self._encoder.encode(obs, legal_mask=legal_mask)
        if features.ndim > 1:
            features = features.flatten()
        return features

    def _policy_call(self, env: Stage2Env, candidates, player: int) -> int:
        features_flat = self._encode_env_obs(env)

        from mahjong_rl.candidate_encoding import encode_candidates_single
        cand_records = [
            CandidateRecord(
                action_type=(c.action_type.value
                             if hasattr(c.action_type, 'value')
                             else int(c.action_type)),
                tile_type=c.tile_type,
                target_rel_seat=c.target_rel_seat,
                consumed_tile_ids=c.consumed_tile_ids,
            )
            for c in candidates
        ]
        cand_feats, cand_mask = encode_candidates_single(cand_records)

        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features_flat))
            cand_feats = cand_feats.to(self._device)
            cand_mask = cand_mask.to(self._device)
            from mahjong_rl.stage2_selfplay_worker import make_response_context
            rc = make_response_context(env.env_state, player)
            self._rc_buf[0].copy_(torch.from_numpy(rc))
            out = self._model.forward_optional(
                self._feat_buf, cand_feats, cand_mask,
                response_context=self._rc_buf, compute_value=False)
            idx, _ = select_optional_argmax(out.optional_scores[0], cand_mask[0])
        return idx
