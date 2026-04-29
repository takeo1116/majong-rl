"""CQ-0274: Stage2a selfplay reward backfill (cumulative transition reward)"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
from mahjong_rl.call_shard import DecisionSample, DecisionShardReader
from mahjong_rl.encoders import FlatFeatureEncoder


def _mk_pending_sample(player_id):
    return DecisionSample(
        decision_type="discard",
        observation=np.zeros(10, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        player_id=player_id, episode_id="ep0",
        step_id=0, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )


class TestAccumulatePendingRewards:
    """CQ-0274: _accumulate_pending_rewards helper の振る舞い"""

    def test_all_pending_get_reward(self):
        """全 pending sample に対応する reward が加算される"""
        pending = {
            0: _mk_pending_sample(0),
            1: _mk_pending_sample(1),
            2: _mk_pending_sample(2),
            3: _mk_pending_sample(3),
        }
        rewards = [1.0, 2.0, 3.0, 4.0]
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, rewards)
        assert pending[0].reward == pytest.approx(1.0)
        assert pending[1].reward == pytest.approx(2.0)
        assert pending[2].reward == pytest.approx(3.0)
        assert pending[3].reward == pytest.approx(4.0)

    def test_accumulation_not_overwrite(self):
        """複数回呼ぶと累積される (上書きされない)"""
        pending = {0: _mk_pending_sample(0)}
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, [1.0, 0, 0, 0])
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, [2.0, 0, 0, 0])
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, [3.0, 0, 0, 0])
        assert pending[0].reward == pytest.approx(6.0)

    def test_non_action_owner_pending_gets_reward(self):
        """action owner 以外の pending にも reward が入る"""
        # player 0 が action し、player 1 にも pending がある (前 decision 残り)
        pending = {
            0: _mk_pending_sample(0),
            1: _mk_pending_sample(1),
        }
        # 他家ツモで player 1 が失点したケースを模倣
        rewards = [-2000.0, 5000.0, -2000.0, -2000.0]
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, rewards)
        assert pending[0].reward == pytest.approx(-2000.0)
        assert pending[1].reward == pytest.approx(5000.0)

    def test_terminated_flag_set(self):
        """terminated=True のとき全 pending sample に立つ"""
        pending = {0: _mk_pending_sample(0), 1: _mk_pending_sample(1)}
        Stage2SelfPlayWorker._accumulate_pending_rewards(
            pending, [0, 0, 0, 0], terminated=True)
        assert pending[0].terminated is True
        assert pending[1].terminated is True

    def test_terminated_default_false(self):
        """terminated 未指定で False のまま"""
        pending = {0: _mk_pending_sample(0)}
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, [1.0, 0, 0, 0])
        assert pending[0].terminated is False

    def test_partial_pending(self):
        """pending が一部 player のみでも残りの reward は失われない (他に影響しない)"""
        pending = {2: _mk_pending_sample(2)}
        Stage2SelfPlayWorker._accumulate_pending_rewards(
            pending, [1.0, 2.0, 3.0, 4.0])
        assert pending[2].reward == pytest.approx(3.0)
        assert len(pending) == 1

    def test_empty_pending_no_crash(self):
        """pending が空でも crash しない"""
        pending = {}
        Stage2SelfPlayWorker._accumulate_pending_rewards(
            pending, [1.0, 2.0, 3.0, 4.0])
        assert pending == {}

    def test_rewards_as_numpy(self):
        """numpy array でも動く"""
        pending = {0: _mk_pending_sample(0)}
        rewards = np.array([1.5, 0, 0, 0], dtype=np.float32)
        Stage2SelfPlayWorker._accumulate_pending_rewards(pending, rewards)
        assert pending[0].reward == pytest.approx(1.5)


class TestSelfplayRewardSmoke:
    """CQ-0274: 実 selfplay で nonzero reward が記録される"""

    def test_selfplay_has_nonzero_rewards(self, tmp_path):
        """real selfplay で reward 0 でない sample が存在する"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=5, base_seed=42)
        samples = DecisionShardReader(tmp_path).read_all()
        assert len(samples) > 0
        # 少なくとも 1 sample に nonzero reward があるはず
        nonzero = sum(1 for s in samples if s.reward != 0.0)
        assert nonzero > 0, "全 sample の reward が 0 (CQ-0274 修正前の徴候)"

    def test_selfplay_reward_distribution_includes_negatives(self, tmp_path):
        """他家和了/被ツモ等で負の reward を持つ sample が存在する"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=20, base_seed=42)
        samples = DecisionShardReader(tmp_path).read_all()
        positives = sum(1 for s in samples if s.reward > 0)
        negatives = sum(1 for s in samples if s.reward < 0)
        # 20 match もあれば双方の符号が出るはず
        assert positives > 0
        assert negatives > 0
