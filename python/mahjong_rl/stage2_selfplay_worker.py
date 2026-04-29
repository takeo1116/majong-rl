"""CQ-0224/0225/0226: Stage2a selfplay / imitation データ生成ワーカー

Stage2Env + RuleBasedCallPolicy + DecisionShardWriter を使って
discard + call decision のデータを生成する。

model が渡された場合は、各 decision で log_prob / value を推論して保存する。
reward は step 後の即時 reward (player 分) を使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mahjong_rl._mahjong_core import ActionType
from mahjong_rl.env import Stage2Env, DecisionType
from mahjong_rl.env.response_candidate import ResponseCandidate
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)
from mahjong_rl.baseline.call_policy import RuleBasedCallPolicy


def build_reward_policy_config(reward_cfg: dict | None):
    """CQ-0276: dict から RewardPolicyConfig を構築する

    Stage1 selfplay_worker と同じ責務を持つ。
    reward_cfg が None / 空のときは None を返す (Stage2Env default 維持)。
    """
    if not reward_cfg:
        return None
    from mahjong_rl import RewardPolicyConfig
    cfg = RewardPolicyConfig()
    cfg.point_delta_scale = float(reward_cfg.get("point_delta_scale", 1.0))
    return cfg


def make_response_context(env_state, current_player: int) -> np.ndarray:
    """response decision 用の response_context ベクトルを作る

    Returns:
        (3,) float32: [tile_type/34, rel_seat/4, menzen_flag]
    """
    rs = env_state.round_state
    rc = rs.response_context
    if not rc.active:
        return np.zeros(3, dtype=np.float32)
    tile_type = rc.discard_tile // 4 if rc.discard_tile < 255 else 0
    discarder = rc.discarder
    rel_seat = (discarder - current_player) % 4
    menzen = float(rs.players[current_player].is_menzen)
    return np.array([tile_type / 34.0, rel_seat / 4.0, menzen], dtype=np.float32)


class Stage2SelfPlayWorker:
    """Stage2a selfplay / imitation データ生成"""

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        observation_mode: str = "full",
        encoder=None,
        model=None,
        device=None,
        policy_ratio: float = 1.0,
        save_baseline_actions: bool = False,
    ):
        self._config = config
        self._output_dir = Path(output_dir)
        self._call_policy = RuleBasedCallPolicy()
        self._obs_mode = observation_mode
        self._encoder = encoder
        self._model = model
        self._device = device or torch.device("cpu")
        self._policy_ratio = policy_ratio
        self._save_baseline_actions = save_baseline_actions
        # CQ-0276: reward_config 伝播
        self._reward_config = build_reward_policy_config(
            config.get("reward") if isinstance(config, dict) else None)
        # CQ-0278: selfplay temperature を config から読む
        # runner._as_dict() 由来 (config["selfplay"]["temperature"]) と
        # flat / worker_config 由来 (config["temperature"]) の両方を許容
        self._temperature = 1.0
        if isinstance(config, dict):
            sp = config.get("selfplay")
            if isinstance(sp, dict) and "temperature" in sp:
                self._temperature = float(sp["temperature"])
            elif "temperature" in config:
                self._temperature = float(config["temperature"])
        if model is not None:
            self._model = model.to(self._device)
            self._model.eval()

        # CQ-0244: preallocated inference buffers
        if encoder is not None:
            obs_dim = encoder.metadata().output_shape[0]
        else:
            obs_dim = 10
        self._feat_buf = torch.zeros(1, obs_dim, dtype=torch.float32, device=self._device)
        self._mask_buf = torch.zeros(1, 34, dtype=torch.float32, device=self._device)
        self._rc_buf = torch.zeros(1, 3, dtype=torch.float32, device=self._device)

    def _encode_obs(self, obs, legal_mask=None) -> np.ndarray:
        if self._encoder is not None:
            features = self._encoder.encode(obs, legal_mask=legal_mask)
            # CQ-0244: encoder output は既に flat float32 → 不要なコピーを避ける
            if features.ndim > 1:
                features = features.flatten()
            return features
        return np.zeros(10, dtype=np.float32)

    def generate(
        self,
        num_matches: int,
        base_seed: int = 0,
        experiment_id: str = "",
        run_id: str = "",
        worker_id: str = "stage2_w0",
    ) -> dict:
        """指定数の半荘を自動プレイし、shard を書き出す"""
        writer = DecisionShardWriter(self._output_dir, max_samples=10000)
        env = Stage2Env(observation_mode=self._obs_mode,
                         reward_config=self._reward_config)
        from mahjong_rl.baseline.rule_based import RuleBasedBaseline
        baseline = RuleBasedBaseline()

        total_steps = 0
        discard_count = 0
        call_count = 0
        step_counter = 0
        num_rounds = 0
        tsumo_count = 0
        ron_count = 0
        ryukyoku_count = 0
        # per-player outcome (全席 actor)
        policy_wins = 0
        policy_deal_ins = 0
        policy_draws = 0
        policy_win_by_tsumo = 0
        policy_win_by_ron = 0

        for match_idx in range(num_matches):
            seed = base_seed + match_idx
            env.reset(seed)
            # CQ-0278: policy sampling 再現性のため torch RNG を match seed で固定
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            episode_id = f"match_{seed}"

            # CQ-0236: seat assignment
            seat_is_policy = self._assign_seats(seed)

            # pending: 直前 decision (reward backfill 用)
            pending: dict[int, DecisionSample] = {}
            # round_buffer: round 内の全 decision (future label backfill 用)
            round_buffer: list[DecisionSample] = []

            for _ in range(5000):
                player = env.current_player
                round_id = env.env_state.round_state.round_number

                if env.decision_type == DecisionType.DISCARD:
                    # CQ-0271: snapshot-based discard
                    mask, discard_snap = env.get_legal_discard_snapshot()
                    obs = env._make_observation()
                    features = self._encode_obs(obs, mask)
                    use_policy = seat_is_policy[player] and self._model is not None

                    if use_policy:
                        action, log_prob, value = self._policy_discard(
                            features, mask)
                        actor_type = "policy"
                    else:
                        hand_ids = list(env.env_state.round_state.players[player].hand)
                        mc = len(env.env_state.round_state.players[player].melds)
                        action = baseline.select_discard(hand_ids, mask, meld_count=mc)
                        log_prob, value = self._infer_discard(features, mask, action)
                        actor_type = "baseline"

                    # save: policy always, baseline if save_baseline or no model (imitation mode)
                    should_save = (actor_type == "policy"
                                   or self._save_baseline_actions
                                   or self._model is None)

                    if should_save:
                        # CQ-0239: teacher info for discard
                        hand_for_teacher = list(
                            env.env_state.round_state.players[player].hand)
                        mc_t = len(env.env_state.round_state.players[player].melds)
                        t_action, t_mask = baseline.select_discard_with_best_set(
                            hand_for_teacher, mask, meld_count=mc_t)
                        t_best = [i for i in range(34) if t_mask[i] > 0.5]

                        sample = DecisionSample(
                            decision_type="discard",
                            observation=features,
                            reward=0.0,
                            log_prob=log_prob,
                            value=value,
                            terminated=False, round_over=False,
                            action=action,
                            legal_mask=mask,
                            teacher_top1_index=t_action,
                            teacher_best_indices=t_best,
                            teacher_source="rule_based",
                            player_id=player,
                            episode_id=episode_id,
                            round_id=round_id,
                            step_id=step_counter,
                            actor_type=actor_type,
                            experiment_id=experiment_id,
                            run_id=run_id,
                            worker_id=worker_id,
                        )
                        if player in pending:
                            prev = pending.pop(player)
                            round_buffer.append(prev)
                        pending[player] = sample

                    step_counter += 1
                    discard_count += 1
                    _, rewards, terminated, _, _ = env.step_discard_with_snapshot(
                        action, discard_snap)
                    # CQ-0274: pending 中の全保存対象 player に reward を累積
                    self._accumulate_pending_rewards(
                        pending, rewards, terminated=terminated)

                elif env.decision_type == DecisionType.RESPONSE:
                    obs = env._make_observation()
                    features = self._encode_obs(obs)
                    candidates = env.response_candidates
                    use_policy = seat_is_policy[player] and self._model is not None

                    # CQ-0244: candidate record を1回だけ生成
                    cand_records = self._make_cand_records(candidates)
                    resp_ctx = make_response_context(env.env_state, player)

                    if use_policy:
                        idx, log_prob, value = self._policy_call(
                            features, candidates, resp_ctx=resp_ctx,
                            cand_records=cand_records)
                        actor_type = "policy"
                    else:
                        hand_ids = list(env.env_state.round_state.players[player].hand)
                        idx = self._call_policy.select_action(candidates, hand_ids)
                        log_prob, value = self._infer_call(
                            features, cand_records, idx, resp_ctx=resp_ctx)
                        actor_type = "baseline"
                    should_save = (actor_type == "policy"
                                   or self._save_baseline_actions
                                   or self._model is None)

                    if should_save:
                        # CQ-0239: teacher info for optional
                        hand_for_teacher = list(
                            env.env_state.round_state.players[player].hand)
                        t_idx = self._call_policy.select_action(
                            candidates, hand_for_teacher)

                        sample = DecisionSample(
                            decision_type="call",
                            observation=features,
                            reward=0.0,
                            log_prob=log_prob,
                            value=value,
                            terminated=False, round_over=False,
                            selected_candidate_index=idx,
                            candidate_count=len(candidates),
                            candidates=cand_records,
                            response_context=make_response_context(env.env_state, player),
                            teacher_top1_index=t_idx,
                            teacher_source="rule_based",
                            player_id=player,
                            episode_id=episode_id,
                            round_id=round_id,
                            step_id=step_counter,
                            actor_type=actor_type,
                            experiment_id=experiment_id,
                            run_id=run_id,
                            worker_id=worker_id,
                        )
                        if player in pending:
                            prev = pending.pop(player)
                            round_buffer.append(prev)
                        pending[player] = sample

                    step_counter += 1
                    call_count += 1
                    _, rewards, terminated, _, _ = env.step_response(idx)
                    # CQ-0274: pending 中の全保存対象 player に reward を累積
                    self._accumulate_pending_rewards(
                        pending, rewards, terminated=terminated)

                total_steps += 1

                # round end → drain outcome queue, backfill + flush
                while env._round_outcomes_queue:
                    outcome = env._round_outcomes_queue.pop(0)
                    num_rounds += 1
                    er = outcome.get("end_reason", 0)
                    winners = set(int(w) for w in outcome.get("winner_players", []))
                    loser = outcome.get("loser_player", -1)
                    # policy_* stats: policy 席のみ集計 (Stage1 parity)
                    policy_winners = [w for w in winners if seat_is_policy[w]]
                    if er == 1:  # Tsumo
                        tsumo_count += 1
                        policy_wins += len(policy_winners)
                        policy_win_by_tsumo += len(policy_winners)
                    elif er == 2:  # Ron
                        ron_count += 1
                        policy_wins += len(policy_winners)
                        policy_win_by_ron += len(policy_winners)
                        if isinstance(loser, int) and loser >= 0 and seat_is_policy[loser]:
                            policy_deal_ins += 1
                    elif er in (3, 4):  # draw
                        ryukyoku_count += 1
                        # Stage1 parity: policy 席が 1 つでもあれば 1 局 1 加算
                        if any(seat_is_policy[p] for p in range(4)):
                            policy_draws += 1
                    for p, s in pending.items():
                        s.round_over = True
                        round_buffer.append(s)
                    pending.clear()
                    for s in round_buffer:
                        self._backfill_single(s, outcome)
                    for s in round_buffer:
                        writer.add(s)
                    round_buffer.clear()
                env._last_round_outcome = None

                if terminated:
                    for p, s in pending.items():
                        s.terminated = True
                        writer.add(s)
                    pending.clear()
                    break

            # flush remaining
            for s in round_buffer:
                writer.add(s)
            for s in pending.values():
                writer.add(s)
            pending.clear()
            round_buffer.clear()

        writer.close()

        return {
            "num_matches": num_matches,
            "total_matches": num_matches,
            "total_steps": total_steps,
            "discard_count": discard_count,
            "call_count": call_count,
            "num_rounds": num_rounds,
            "tsumo_count": tsumo_count,
            "ron_count": ron_count,
            "ryukyoku_count": ryukyoku_count,
            "policy_wins": policy_wins,
            "policy_deal_ins": policy_deal_ins,
            "policy_draws": policy_draws,
            "policy_win_by_tsumo": policy_win_by_tsumo,
            "policy_win_by_ron": policy_win_by_ron,
        }

    def _assign_seats(self, seed: int) -> list[bool]:
        """CQ-0236: policy_ratio に基づいて席を割り当てる"""
        if self._policy_ratio >= 1.0:
            return [True, True, True, True]
        if self._policy_ratio <= 0.0:
            return [False, False, False, False]
        rng = np.random.RandomState(seed)
        return [rng.random() < self._policy_ratio for _ in range(4)]

    @staticmethod
    def _accumulate_pending_rewards(
        pending: dict, rewards, terminated: bool = False,
    ) -> None:
        """CQ-0274: pending 中の全保存対象 player の sample に reward を累積する

        Stage1 SelfPlayWorker と同じ same-player transition 累積の発想。
        action owner だけでなく、env step 中に発生した他家の得失点も
        全 pending sample に乗る。

        Args:
            pending: dict[player_id, DecisionSample]
            rewards: 4 player 分の reward 配列 (numpy / list / Tensor)
            terminated: match end フラグ
        """
        for p, sample in pending.items():
            if 0 <= p < len(rewards):
                sample.reward += float(rewards[p])
            if terminated:
                sample.terminated = True

    @staticmethod
    def _backfill_single(sample: DecisionSample, outcome: dict) -> None:
        """1 sample に future label を付ける"""
        end_reason = outcome.get("end_reason", 0)
        winner_players = set(int(w) for w in outcome.get("winner_players", []))
        loser_player = outcome.get("loser_player", -1)
        tenpai_set = set(int(p) for p in outcome.get("tenpai_players", []))
        wins = list(outcome.get("wins", []))
        win_by_player: dict[int, dict] = {int(w["winner"]): w for w in wins}

        is_tsumo = end_reason == 1
        is_ron = end_reason == 2
        is_draw = end_reason == 3
        is_abortive = end_reason == 4
        pid = sample.player_id

        # CQ-0266: 5-class terminal labels
        if pid in winner_players:
            wi = win_by_player.get(pid, {})
            is_menzen = wi.get("is_menzen", True)
            sample.round_terminal_label = "win_menzen" if is_menzen else "win_called"
            sample.eventual_win_yaku_ids = [int(y) for y in wi.get("yaku_ids", [])]
            sample.eventual_total_han = wi.get("total_han", -1)
            sample.eventual_fu = wi.get("fu", -1)
        elif is_ron and pid == loser_player:
            sample.round_terminal_label = "deal_in"
        elif is_draw and pid in tenpai_set:
            sample.round_terminal_label = "draw_tenpai"
        else:
            # 被ツモ / ロン傍観 / 流局ノーテン / 途中流局
            sample.round_terminal_label = "other_non_dealin"

    @staticmethod
    def _make_cand_records(candidates) -> list[CandidateRecord]:
        return [
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

    def _policy_discard(self, features: np.ndarray, mask: np.ndarray,
                        ) -> tuple[int, float, float]:
        """learned policy で discard action を選択"""
        from mahjong_rl.stage2a_selector import select_discard_sample
        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features))
            self._mask_buf[0].copy_(torch.from_numpy(mask))
            out = self._model.forward_discard(self._feat_buf, self._mask_buf)
            action, lp = select_discard_sample(
                out.discard_logits[0], self._mask_buf[0],
                temperature=self._temperature)
            val = out.values["round_delta"][0, 0].item()
        return action, lp, val

    def _policy_call(self, features: np.ndarray,
                     candidates,
                     resp_ctx: np.ndarray | None = None,
                     cand_records: list | None = None,
                     ) -> tuple[int, float, float]:
        """learned policy で call action を選択 (CQ-0245 fast path)"""
        from mahjong_rl.stage2a_selector import select_optional_sample
        from mahjong_rl.candidate_encoding import encode_candidates_single

        if cand_records is None:
            cand_records = self._make_cand_records(candidates)
        cand_feats, cand_mask = encode_candidates_single(cand_records)

        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features))
            cand_feats = cand_feats.to(self._device)
            cand_mask = cand_mask.to(self._device)
            rc_t = None
            if resp_ctx is not None:
                self._rc_buf[0].copy_(torch.from_numpy(resp_ctx))
                rc_t = self._rc_buf
            out = self._model.forward_optional(
                self._feat_buf, cand_feats, cand_mask, response_context=rc_t)
            idx, lp = select_optional_sample(
                out.optional_scores[0], cand_mask[0],
                temperature=self._temperature)
            val = out.values["round_delta"][0, 0].item()
        return idx, lp, val

    def _infer_discard(self, features: np.ndarray, mask: np.ndarray,
                       action: int) -> tuple[float, float]:
        """model があれば discard の log_prob / value を推論"""
        if self._model is None:
            return 0.0, 0.0
        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features))
            self._mask_buf[0].copy_(torch.from_numpy(mask))
            out = self._model.forward_discard(self._feat_buf, self._mask_buf)
            log_probs = torch.log_softmax(out.discard_logits, dim=-1)
            lp = log_probs[0, action].item()
            val = out.values["round_delta"][0, 0].item()
        return lp, val

    def _infer_call(self, features: np.ndarray,
                    cand_records: list[CandidateRecord],
                    selected_idx: int,
                    resp_ctx: np.ndarray | None = None,
                    ) -> tuple[float, float]:
        """model があれば call の log_prob / value を推論 (CQ-0245 fast path)"""
        if self._model is None:
            return 0.0, 0.0
        from mahjong_rl.candidate_encoding import encode_candidates_single

        cand_feats, cand_mask = encode_candidates_single(cand_records)

        with torch.inference_mode():
            self._feat_buf[0].copy_(torch.from_numpy(features))
            cand_feats = cand_feats.to(self._device)
            cand_mask = cand_mask.to(self._device)
            rc_t = None
            if resp_ctx is not None:
                self._rc_buf[0].copy_(torch.from_numpy(resp_ctx))
                rc_t = self._rc_buf
            out = self._model.forward_optional(
                self._feat_buf, cand_feats, cand_mask, response_context=rc_t)
            log_probs = torch.log_softmax(out.optional_scores, dim=-1)
            lp = log_probs[0, selected_idx].item()
            val = out.values["round_delta"][0, 0].item()
        return lp, val
