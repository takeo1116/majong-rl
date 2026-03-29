"""CQ-0224: RuleBasedCallPolicy テスト + Stage2a selfplay smoke テスト"""
import pytest
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl._mahjong_core import ActionType, NUM_TILE_TYPES
from mahjong_rl.env.response_candidate import ResponseCandidate
from mahjong_rl.baseline.call_policy import RuleBasedCallPolicy


def _make_candidate(action_type, tile_type=-1, rel_seat=-1):
    """テスト用の簡易 ResponseCandidate"""
    return ResponseCandidate(
        action=None,  # select_action は action フィールドを使わない
        action_type=action_type,
        tile_type=tile_type,
        target_rel_seat=rel_seat,
        consumed_tile_ids=(),
    )


def _make_chi_candidate(tile_type, consumed_types, rel_seat=3):
    """チー用 candidate (consumed_tile_ids は TileId 形式)"""
    consumed = tuple(t * 4 for t in consumed_types)  # TileType → TileId (簡易)
    return ResponseCandidate(
        action=None,
        action_type=ActionType.Chi,
        tile_type=tile_type,
        target_rel_seat=rel_seat,
        consumed_tile_ids=consumed,
    )


def _hand_ids_from_types(types_and_counts):
    """TileType/count ペアから TileId リストを生成"""
    ids = []
    for tt, count in types_and_counts:
        for i in range(count):
            ids.append(tt * 4 + i)
    return ids


class TestRuleBasedCallPolicy:
    """代表ケースの副露判断テスト"""

    def setup_method(self):
        self.policy = RuleBasedCallPolicy()

    def test_yakuhai_pon_accepted(self):
        """役牌ポンは常に鳴く"""
        # 手牌: 白白 + 何か
        hand = _hand_ids_from_types([
            (31, 2),  # 白白
            (0, 1), (1, 1), (2, 1),  # 123m
            (9, 1), (10, 1), (11, 1),  # 123p
            (18, 1), (19, 1), (20, 1),  # 123s
            (27, 2),  # 東東
        ])
        candidates = [
            _make_candidate(ActionType.Pon, tile_type=31, rel_seat=1),  # 白ポン
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 0, "役牌ポンが選択されるべき"

    def test_tanyao_pon(self):
        """喰いタン方向のポン"""
        # 手牌: 断么九のみ
        hand = _hand_ids_from_types([
            (1, 2), (2, 1), (3, 1),  # 2m2m3m4m
            (10, 2), (11, 1), (12, 1),  # 2p2p3p4p
            (19, 1), (20, 1), (21, 1),  # 2s3s4s
            (4, 2),  # 5m5m
        ])
        candidates = [
            _make_candidate(ActionType.Pon, tile_type=1, rel_seat=2),  # 2m ポン
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 0, "喰いタン方向のポンが選択されるべき"

    def test_toitoi_direction_pon(self):
        """対々和方向のポン"""
        # 手牌: 刻子候補多数
        hand = _hand_ids_from_types([
            (0, 3),  # 1m1m1m
            (9, 2),  # 1p1p
            (18, 2),  # 1s1s
            (27, 3),  # 東東東
            (28, 2),  # 南南
            (4, 1),   # 5m
        ])
        candidates = [
            _make_candidate(ActionType.Pon, tile_type=9, rel_seat=1),  # 1p ポン
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 0, "対々和方向のポンが選択されるべき"

    def test_skip_when_yaochu_pon_no_benefit(self):
        """么九牌ポンでメリットが薄い → Skip"""
        # 手牌: バラバラ
        hand = _hand_ids_from_types([
            (0, 2),  # 1m1m (么九)
            (3, 1), (5, 1), (7, 1),
            (12, 1), (14, 1), (16, 1),
            (21, 1), (23, 1), (25, 1),
            (28, 1), (29, 1), (30, 1),
        ])
        candidates = [
            _make_candidate(ActionType.Pon, tile_type=0, rel_seat=2),  # 1m ポン
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        # 方向がないのでスキップ寄り
        assert idx == 1, "メリットの薄いポンは Skip"

    def test_chi_ikkitsuukan_direction(self):
        """一気通貫方向のチー"""
        # 手牌: 萬子 456+789 のみ、123 が足りない
        hand = _hand_ids_from_types([
            (1, 1), (2, 1),  # 2m3m (1m が来ればチーで 123m 成立)
            (3, 1), (4, 1), (5, 1),  # 456m
            (6, 1), (7, 1), (8, 1),  # 789m
            (27, 2),  # 東東
            (9, 1), (10, 1), (11, 1),  # 123p
        ])
        candidates = [
            _make_chi_candidate(
                tile_type=0,  # 1m をチー
                consumed_types=[1, 2],  # 2m3m を消費
                rel_seat=3,
            ),
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 0, "一気通貫方向のチーが選択されるべき"

    def test_skip_only(self):
        """Skip のみなら 0 を返す"""
        hand = _hand_ids_from_types([(0, 3), (1, 3), (2, 3), (27, 2), (28, 2)])
        candidates = [_make_candidate(ActionType.Skip)]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 0

    def test_multiple_candidates_best_wins(self):
        """複数候補から最高スコアを選ぶ"""
        hand = _hand_ids_from_types([
            (31, 2),  # 白白
            (0, 1), (1, 1), (2, 1),
            (9, 1), (10, 1), (11, 1),
            (18, 1), (19, 1), (20, 1),
            (27, 2),
        ])
        candidates = [
            _make_chi_candidate(tile_type=3, consumed_types=[1, 2]),  # 3m チー
            _make_candidate(ActionType.Pon, tile_type=31, rel_seat=1),  # 白ポン
            _make_candidate(ActionType.Skip),
        ]
        idx = self.policy.select_action(candidates, hand)
        assert idx == 1, "役牌ポンがチーより優先"


class TestStage2aSelfplaySmoke:
    """Stage2a selfplay smoke テスト"""

    def test_stage2a_with_call_policy_completes(self):
        """Stage2Env + RuleBasedCallPolicy で半荘完走"""
        from mahjong_rl.env import Stage2Env, DecisionType

        policy = RuleBasedCallPolicy()
        env = Stage2Env()
        env.reset(42)

        for _ in range(5000):
            if env.decision_type == DecisionType.DISCARD:
                mask = env.get_legal_mask()
                action = int(np.argmax(mask))
                _, _, terminated, _, _ = env.step_discard(action)
            elif env.decision_type == DecisionType.RESPONSE:
                hand_ids = list(env.env_state.round_state.players[
                    env.current_player].hand)
                idx = policy.select_action(
                    env.response_candidates, hand_ids)
                _, _, terminated, _, _ = env.step_response(idx)
            if terminated:
                break
        assert terminated, "半荘が 5000 step で終わらなかった"

    def test_call_decisions_occur(self):
        """call decision が実際に発生し、Skip 以外も選ばれる"""
        from mahjong_rl.env import Stage2Env, DecisionType

        policy = RuleBasedCallPolicy()
        call_count = 0
        non_skip_count = 0

        for seed in range(10):
            env = Stage2Env()
            env.reset(seed)
            for _ in range(5000):
                if env.decision_type == DecisionType.DISCARD:
                    mask = env.get_legal_mask()
                    action = int(np.argmax(mask))
                    _, _, terminated, _, _ = env.step_discard(action)
                elif env.decision_type == DecisionType.RESPONSE:
                    call_count += 1
                    hand_ids = list(env.env_state.round_state.players[
                        env.current_player].hand)
                    idx = policy.select_action(
                        env.response_candidates, hand_ids)
                    cand = env.response_candidates[idx]
                    if cand.action_type != ActionType.Skip:
                        non_skip_count += 1
                    _, _, terminated, _, _ = env.step_response(idx)
                if terminated:
                    break

        assert call_count > 0, "call decision が1つも発生しなかった"
        # 10 seed で少なくとも1回は鳴きが発生するはず
        assert non_skip_count > 0, "10 seed で Skip 以外の鳴きがなかった"

    def test_call_decision_saved_to_shard(self):
        """call decision が shard に保存できる"""
        from mahjong_rl.env import Stage2Env, DecisionType
        from mahjong_rl.call_shard import (
            DecisionSample, CandidateRecord,
            DecisionShardWriter, DecisionShardReader,
        )
        import tempfile

        policy = RuleBasedCallPolicy()
        env = Stage2Env()

        with tempfile.TemporaryDirectory() as tmp:
            writer = DecisionShardWriter(tmp, max_samples=10000)
            step_counter = 0

            for seed in range(5):
                env.reset(seed)
                for _ in range(2000):
                    if env.decision_type == DecisionType.DISCARD:
                        mask = env.get_legal_mask()
                        action = int(np.argmax(mask))
                        obs = env._make_observation()
                        writer.add(DecisionSample(
                            decision_type="discard",
                            observation=np.zeros(10, dtype=np.float32),
                            reward=0.0, log_prob=0.0, value=0.0,
                            terminated=False, round_over=False,
                            action=action,
                            legal_mask=mask,
                            player_id=env.current_player,
                            episode_id=f"ep{seed}",
                            step_id=step_counter,
                            experiment_id="t", run_id="r", worker_id="w",
                        ))
                        step_counter += 1
                        _, _, terminated, _, _ = env.step_discard(action)
                    elif env.decision_type == DecisionType.RESPONSE:
                        hand_ids = list(env.env_state.round_state.players[
                            env.current_player].hand)
                        idx = policy.select_action(
                            env.response_candidates, hand_ids)
                        cands = [
                            CandidateRecord(
                                action_type=c.action_type.value
                                if hasattr(c.action_type, 'value')
                                else int(c.action_type),
                                tile_type=c.tile_type,
                                target_rel_seat=c.target_rel_seat,
                            )
                            for c in env.response_candidates
                        ]
                        writer.add(DecisionSample(
                            decision_type="call",
                            observation=np.zeros(10, dtype=np.float32),
                            reward=0.0, log_prob=0.0, value=0.0,
                            terminated=False, round_over=False,
                            selected_candidate_index=idx,
                            candidate_count=len(cands),
                            candidates=cands,
                            player_id=env.current_player,
                            episode_id=f"ep{seed}",
                            step_id=step_counter,
                            experiment_id="t", run_id="r", worker_id="w",
                        ))
                        step_counter += 1
                        _, _, terminated, _, _ = env.step_response(idx)
                    if terminated:
                        break
            writer.close()

            reader = DecisionShardReader(tmp)
            samples = reader.read_all()
            discard_count = sum(1 for s in samples
                                if s.decision_type == "discard")
            call_count = sum(1 for s in samples
                             if s.decision_type == "call")
            assert discard_count > 0
            assert call_count > 0
            # call sample の candidates が復元される
            call_samples = [s for s in samples if s.decision_type == "call"]
            for cs in call_samples:
                assert cs.candidate_count > 0
                assert len(cs.candidates) == cs.candidate_count
