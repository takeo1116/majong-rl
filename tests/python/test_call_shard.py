"""CQ-0223: DecisionSample shard round-trip テスト"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)


def _make_discard_sample(step: int) -> DecisionSample:
    return DecisionSample(
        decision_type="discard",
        observation=np.random.randn(100).astype(np.float32),
        reward=0.01 * step,
        log_prob=-0.5,
        value=0.1,
        terminated=(step == 9),
        round_over=False,
        player_id=step % 4,
        episode_id="ep0",
        round_id=0,
        step_id=step,
        action=step % 34,
        legal_mask=np.random.rand(34).astype(np.float32),
        experiment_id="test", run_id="run0", worker_id="w0",
    )


def _make_call_sample(step: int, num_cands: int = 3) -> DecisionSample:
    cands = []
    for j in range(num_cands):
        if j == num_cands - 1:
            cands.append(CandidateRecord(action_type=8, tile_type=-1,
                                          target_rel_seat=-1))
        else:
            cands.append(CandidateRecord(
                action_type=3 + j, tile_type=j * 2,
                target_rel_seat=1 + j,
                consumed_tile_ids=(j * 4, j * 4 + 1),
            ))
    return DecisionSample(
        decision_type="call",
        observation=np.random.randn(100).astype(np.float32),
        reward=0.0,
        log_prob=-1.0,
        value=0.0,
        terminated=False,
        round_over=False,
        player_id=1,
        episode_id="ep0",
        round_id=0,
        step_id=step,
        selected_candidate_index=0,
        candidate_count=num_cands,
        candidates=cands,
        experiment_id="test", run_id="run0", worker_id="w0",
    )


class TestDecisionShardRoundTrip:
    """discard / call の round-trip テスト"""

    def test_discard_roundtrip(self, tmp_path: Path):
        """discard sample の書き込み → 読み込みが一致"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        originals = [_make_discard_sample(i) for i in range(10)]
        for s in originals:
            writer.add(s)
        writer.close()

        reader = DecisionShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 10
        for orig, load in zip(originals, loaded):
            assert load.decision_type == "discard"
            assert load.action == orig.action
            assert load.player_id == orig.player_id
            np.testing.assert_array_almost_equal(
                load.observation, orig.observation)
            np.testing.assert_array_almost_equal(
                load.legal_mask, orig.legal_mask)

    def test_call_roundtrip(self, tmp_path: Path):
        """call sample の書き込み → 読み込みが一致"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        originals = [_make_call_sample(i, num_cands=4) for i in range(5)]
        for s in originals:
            writer.add(s)
        writer.close()

        reader = DecisionShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 5
        for orig, load in zip(originals, loaded):
            assert load.decision_type == "call"
            assert load.candidate_count == orig.candidate_count
            assert load.selected_candidate_index == orig.selected_candidate_index
            assert len(load.candidates) == len(orig.candidates)
            for oc, lc in zip(orig.candidates, load.candidates):
                assert lc.action_type == oc.action_type
                assert lc.tile_type == oc.tile_type
                assert lc.target_rel_seat == oc.target_rel_seat
                assert lc.consumed_tile_ids == oc.consumed_tile_ids

    def test_mixed_roundtrip(self, tmp_path: Path):
        """discard + call の混合 shard"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(_make_discard_sample(0))
        writer.add(_make_call_sample(1, num_cands=3))
        writer.add(_make_discard_sample(2))
        writer.add(_make_call_sample(3, num_cands=2))
        writer.close()

        reader = DecisionShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 4
        assert loaded[0].decision_type == "discard"
        assert loaded[1].decision_type == "call"
        assert loaded[2].decision_type == "discard"
        assert loaded[3].decision_type == "call"

    def test_candidate_order_preserved(self, tmp_path: Path):
        """candidate の順序と consumed_tile_ids が保存される"""
        cands = [
            CandidateRecord(action_type=4, tile_type=10, target_rel_seat=2,
                            consumed_tile_ids=(40, 41)),
            CandidateRecord(action_type=3, tile_type=5, target_rel_seat=3,
                            consumed_tile_ids=(20, 24)),
            CandidateRecord(action_type=8, tile_type=-1, target_rel_seat=-1),
        ]
        sample = DecisionSample(
            decision_type="call",
            observation=np.zeros(10, dtype=np.float32),
            reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=1,
            candidate_count=3, candidates=cands,
            experiment_id="t", run_id="r", worker_id="w",
            episode_id="e",
        )
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 1
        lc = loaded[0].candidates
        assert lc[0].action_type == 4  # Pon
        assert lc[0].consumed_tile_ids == (40, 41)
        assert lc[1].action_type == 3  # Chi
        assert lc[1].consumed_tile_ids == (20, 24)
        assert lc[2].action_type == 8  # Skip
        assert lc[2].consumed_tile_ids == ()

    def test_empty_candidates_discard(self, tmp_path: Path):
        """discard sample は candidates 空"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(_make_discard_sample(0))
        writer.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert loaded[0].candidates == []
        assert loaded[0].candidate_count == 0


class TestDecisionShardReaderNested:
    """DecisionShardReader の nested worker shard 対応テスト"""

    def test_flat_only(self, tmp_path: Path):
        """flat shard のみの構成で読める"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        for i in range(3):
            writer.add(_make_discard_sample(i))
        writer.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 3

    def test_worker_nested_only(self, tmp_path: Path):
        """worker_*/shard_*.parquet のみの構成で読める"""
        for w in range(2):
            w_dir = tmp_path / f"worker_{w}"
            writer = DecisionShardWriter(w_dir, max_samples=100)
            for i in range(3):
                writer.add(_make_discard_sample(w * 10 + i))
            writer.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 6

    def test_flat_and_nested_mixed(self, tmp_path: Path):
        """flat + worker nested の混合構成で重複なく読める"""
        # flat
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(_make_discard_sample(0))
        writer.close()
        # nested
        w_dir = tmp_path / "worker_0"
        writer2 = DecisionShardWriter(w_dir, max_samples=100)
        writer2.add(_make_discard_sample(1))
        writer2.add(_make_discard_sample(2))
        writer2.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 3

    def test_no_duplicate(self, tmp_path: Path):
        """flat と nested が同じファイル名でも重複しない"""
        w_dir = tmp_path / "worker_0"
        writer = DecisionShardWriter(w_dir, max_samples=100)
        for i in range(4):
            writer.add(_make_discard_sample(i))
        writer.close()
        loaded = DecisionShardReader(tmp_path).read_all()
        assert len(loaded) == 4
        # flat には shard がないはず
        flat_count = len(list(tmp_path.glob("shard_*.parquet")))
        assert flat_count == 0
