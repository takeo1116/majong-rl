"""テスト: shard.py — 学習サンプルの書き出し・読み込み"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.shard import (
    LearningSample, ShardWriter, ShardReader,
    ShardBackend, ParquetBackend, validate_metadata,
)


def _make_sample(obs_dim: int = 10, action: int = 0, reward: float = 1.0,
                 step_id: int = 0) -> LearningSample:
    """テスト用ダミーサンプル生成"""
    return LearningSample(
        observation=np.random.randn(obs_dim).astype(np.float32),
        legal_mask=np.random.rand(34).astype(np.float32),
        action=action,
        reward=reward,
        log_prob=-0.5,
        value=0.3,
        terminated=False,
        round_over=False,
        experiment_id="exp_001",
        run_id="run_001",
        worker_id="worker_0",
        shard_id="shard_0",
        model_version=1,
        generation=0,
        timestamp=1234567890.0,
        episode_id="ep_001",
        round_id=0,
        step_id=step_id,
        player_id=0,
    )


class TestShardWriteRead:
    """書き出し → 読み込み roundtrip テスト"""

    def test_roundtrip_single_sample(self, tmp_path: Path):
        """1サンプルの書き出し → 読み込みで内容一致"""
        sample = _make_sample(obs_dim=20, action=5, reward=2.0)

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 1

        s = loaded[0]
        np.testing.assert_array_almost_equal(s.observation, sample.observation)
        np.testing.assert_array_almost_equal(s.legal_mask, sample.legal_mask)
        assert s.action == sample.action
        assert s.reward == pytest.approx(sample.reward)
        assert s.log_prob == pytest.approx(sample.log_prob)
        assert s.value == pytest.approx(sample.value)
        assert s.terminated == sample.terminated
        assert s.round_over == sample.round_over
        assert s.experiment_id == sample.experiment_id
        assert s.run_id == sample.run_id
        assert s.worker_id == sample.worker_id
        assert s.model_version == sample.model_version
        assert s.episode_id == sample.episode_id
        assert s.step_id == sample.step_id
        assert s.player_id == sample.player_id

    def test_roundtrip_multiple_samples(self, tmp_path: Path):
        """複数サンプルの roundtrip"""
        n = 50
        writer = ShardWriter(tmp_path, max_samples=1000)
        for i in range(n):
            writer.add(_make_sample(obs_dim=15, action=i % 34, step_id=i))
        writer.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == n
        for i, s in enumerate(loaded):
            assert s.action == i % 34
            assert s.step_id == i

    def test_auto_flush_creates_multiple_files(self, tmp_path: Path):
        """max_samples 超過で複数 shard ファイルが生成される"""
        writer = ShardWriter(tmp_path, max_samples=10)
        for i in range(25):
            writer.add(_make_sample(step_id=i))
        writer.close()

        shards = sorted(tmp_path.glob("shard_*.parquet"))
        assert len(shards) == 3  # 10 + 10 + 5

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 25

    def test_empty_close(self, tmp_path: Path):
        """空バッファの close は安全"""
        writer = ShardWriter(tmp_path, max_samples=100)
        writer.close()

        shards = list(tmp_path.glob("shard_*.parquet"))
        assert len(shards) == 0

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 0


class TestShardTensors:
    """read_as_tensors のテスト"""

    def test_tensor_shapes(self, tmp_path: Path):
        """read_as_tensors の形状・型が正しい"""
        obs_dim = 20
        n = 30
        writer = ShardWriter(tmp_path, max_samples=1000)
        for i in range(n):
            writer.add(_make_sample(obs_dim=obs_dim, step_id=i))
        writer.close()

        reader = ShardReader(tmp_path)
        tensors = reader.read_as_tensors()

        assert tensors["observations"].shape == (n, obs_dim)
        assert tensors["observations"].dtype == np.float32
        assert tensors["legal_masks"].shape == (n, 34)
        assert tensors["legal_masks"].dtype == np.float32
        assert tensors["actions"].shape == (n,)
        assert tensors["actions"].dtype == np.int32
        assert tensors["rewards"].shape == (n,)
        assert tensors["rewards"].dtype == np.float32
        assert tensors["log_probs"].shape == (n,)
        assert tensors["log_probs"].dtype == np.float32
        assert tensors["values"].shape == (n,)
        assert tensors["values"].dtype == np.float32
        assert tensors["terminateds"].shape == (n,)
        assert tensors["terminateds"].dtype == bool

    def test_empty_tensors(self, tmp_path: Path):
        """空ディレクトリからの read_as_tensors"""
        reader = ShardReader(tmp_path)
        tensors = reader.read_as_tensors()

        assert tensors["observations"].shape[0] == 0
        assert tensors["actions"].shape == (0,)


class TestMetadataValidation:
    """メタデータ検証テスト"""

    def test_valid_sample_passes(self):
        """正しいメタデータのサンプルは検証を通る"""
        sample = _make_sample()
        validate_metadata(sample)  # 例外なし

    def test_empty_experiment_id_fails(self):
        """experiment_id が空だと ValueError"""
        sample = _make_sample()
        sample.experiment_id = ""
        with pytest.raises(ValueError, match="experiment_id"):
            validate_metadata(sample)

    def test_empty_run_id_fails(self):
        """run_id が空だと ValueError"""
        sample = _make_sample()
        sample.run_id = ""
        with pytest.raises(ValueError, match="run_id"):
            validate_metadata(sample)

    def test_empty_worker_id_fails(self):
        """worker_id が空だと ValueError"""
        sample = _make_sample()
        sample.worker_id = ""
        with pytest.raises(ValueError, match="worker_id"):
            validate_metadata(sample)

    def test_empty_episode_id_fails(self):
        """episode_id が空だと ValueError"""
        sample = _make_sample()
        sample.episode_id = ""
        with pytest.raises(ValueError, match="episode_id"):
            validate_metadata(sample)

    def test_negative_model_version_fails(self):
        """model_version が負だと ValueError"""
        sample = _make_sample()
        sample.model_version = -1
        with pytest.raises(ValueError, match="model_version"):
            validate_metadata(sample)

    def test_negative_step_id_fails(self):
        """step_id が負だと ValueError"""
        sample = _make_sample()
        sample.step_id = -1
        with pytest.raises(ValueError, match="step_id"):
            validate_metadata(sample)

    def test_writer_rejects_invalid_on_add(self, tmp_path: Path):
        """validate=True の ShardWriter は不正サンプルを add 時に拒否"""
        writer = ShardWriter(tmp_path, validate=True)
        bad_sample = _make_sample()
        bad_sample.experiment_id = ""
        with pytest.raises(ValueError):
            writer.add(bad_sample)

    def test_writer_allows_invalid_when_disabled(self, tmp_path: Path):
        """validate=False の ShardWriter は検証をスキップ"""
        writer = ShardWriter(tmp_path, validate=False)
        bad_sample = _make_sample()
        bad_sample.experiment_id = ""
        writer.add(bad_sample)  # 例外なし
        writer.close()


class TestShardBackendAbstraction:
    """ShardBackend 抽象層テスト"""

    def test_parquet_backend_is_shard_backend(self):
        """ParquetBackend は ShardBackend のサブクラス"""
        assert issubclass(ParquetBackend, ShardBackend)

    def test_parquet_backend_file_extension(self):
        """ParquetBackend の拡張子は .parquet"""
        backend = ParquetBackend()
        assert backend.file_extension() == ".parquet"

    def test_explicit_parquet_backend_roundtrip(self, tmp_path: Path):
        """明示的に ParquetBackend を指定して roundtrip"""
        backend = ParquetBackend()
        writer = ShardWriter(tmp_path, max_samples=100, backend=backend)
        writer.add(_make_sample(obs_dim=10, action=3))
        writer.close()

        reader = ShardReader(tmp_path, backend=backend)
        loaded = reader.read_all()
        assert len(loaded) == 1
        assert loaded[0].action == 3

    def test_shard_id_auto_assigned_at_add(self, tmp_path: Path):
        """shard_id が空の場合、add 時に writer が自動付与する"""
        sample = _make_sample()
        sample.shard_id = ""

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        # add 後にサンプル自体に shard_id が設定されている
        assert sample.shard_id == "shard_0000"

        writer.close()
        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert loaded[0].shard_id == "shard_0000"

    def test_shard_id_increments_across_flushes(self, tmp_path: Path):
        """auto-flush で shard_id が shard ごとに変わる"""
        writer = ShardWriter(tmp_path, max_samples=2)
        for i in range(5):
            s = _make_sample(step_id=i)
            s.shard_id = ""
            writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        # shard_0000: 2件, shard_0001: 2件, shard_0002: 1件
        assert loaded[0].shard_id == "shard_0000"
        assert loaded[1].shard_id == "shard_0000"
        assert loaded[2].shard_id == "shard_0001"
        assert loaded[3].shard_id == "shard_0001"
        assert loaded[4].shard_id == "shard_0002"

    def test_validate_metadata_rejects_empty_shard_id(self):
        """validate_metadata は shard_id 空を拒否する"""
        sample = _make_sample()
        sample.shard_id = ""
        with pytest.raises(ValueError, match="shard_id"):
            validate_metadata(sample)

    def test_writer_fills_shard_id_before_validation(self, tmp_path: Path):
        """ShardWriter は shard_id を付与してから検証するため、空でも add 可能"""
        sample = _make_sample()
        sample.shard_id = ""
        writer = ShardWriter(tmp_path, validate=True)
        writer.add(sample)  # 例外なし — writer が shard_id を付与済み
        writer.close()


@pytest.mark.smoke
class TestShardReaderSubdirectory:
    """ShardReader のサブディレクトリ対応テスト (CQ-0071)"""

    def test_read_from_worker_subdirectories(self, tmp_path: Path):
        """worker_*/shard_*.parquet 構造から読める"""
        for wid in range(2):
            worker_dir = tmp_path / f"worker_{wid}"
            writer = ShardWriter(worker_dir, max_samples=100)
            for i in range(3):
                writer.add(_make_sample(step_id=wid * 10 + i))
            writer.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 6

    def test_read_mixed_flat_and_nested(self, tmp_path: Path):
        """平坦 shard と nested shard が混在する場合に両方読める"""
        # 平坦
        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(_make_sample(step_id=0))
        writer.close()
        # nested
        nested_dir = tmp_path / "worker_0"
        writer2 = ShardWriter(nested_dir, max_samples=100)
        writer2.add(_make_sample(step_id=1))
        writer2.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 2

    def test_read_as_tensors_from_subdirectories(self, tmp_path: Path):
        """read_as_tensors もサブディレクトリから読める"""
        worker_dir = tmp_path / "worker_0"
        writer = ShardWriter(worker_dir, max_samples=100)
        for i in range(5):
            writer.add(_make_sample(step_id=i))
        writer.close()

        reader = ShardReader(tmp_path)
        tensors = reader.read_as_tensors()
        assert tensors["observations"].shape[0] == 5


@pytest.mark.smoke
class TestTeacherBestMask:
    """teacher_best_mask の shard roundtrip テスト (CQ-0126)"""

    def test_roundtrip_teacher_best_mask(self, tmp_path: Path):
        """teacher_best_mask の書き出し → 読み込みが一致する"""
        sample = _make_sample()
        sample.teacher_best_mask = np.zeros(34, dtype=np.float32)
        sample.teacher_best_mask[5] = 1.0
        sample.teacher_best_mask[10] = 1.0

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["teacher_best_masks"] is not None
        assert data["teacher_best_masks"].shape == (1, 34)
        assert data["teacher_best_masks"][0, 5] == 1.0
        assert data["teacher_best_masks"][0, 10] == 1.0

    def test_missing_teacher_best_mask_backward_compat(self, tmp_path: Path):
        """teacher_best_mask なしの旧 shard で None が返る"""
        sample = _make_sample()
        assert sample.teacher_best_mask is None

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["teacher_best_masks"] is None

    def test_filter_preserves_teacher_best_mask(self, tmp_path: Path):
        """filter_actor_type 適用時も teacher_best_masks が正しくフィルタされる"""
        writer = ShardWriter(tmp_path, max_samples=100)
        for i in range(4):
            s = _make_sample(step_id=i)
            s.actor_type = "baseline" if i % 2 == 0 else "policy"
            s.teacher_best_mask = np.zeros(34, dtype=np.float32)
            s.teacher_best_mask[i] = 1.0
            writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors(filter_actor_type="baseline")
        assert data["teacher_best_masks"] is not None
        assert data["teacher_best_masks"].shape[0] == 2
        assert data["teacher_best_masks"][0, 0] == 1.0
        assert data["teacher_best_masks"][1, 2] == 1.0

    def test_shard_info_all_have_mask(self, tmp_path: Path):
        """全 shard に teacher_best_mask がある場合の shard_info (CQ-0128)"""
        sample = _make_sample()
        sample.teacher_best_mask = np.zeros(34, dtype=np.float32)
        sample.teacher_best_mask[0] = 1.0

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        info = data["teacher_best_mask_shard_info"]
        assert info["available"] == 1
        assert info["total"] == 1

    def test_shard_info_none_have_mask(self, tmp_path: Path):
        """全 shard に teacher_best_mask がない場合の shard_info (CQ-0128)"""
        sample = _make_sample()
        assert sample.teacher_best_mask is None

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(sample)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        info = data["teacher_best_mask_shard_info"]
        assert info["available"] == 0
        assert info["total"] == 1
        assert data["teacher_best_masks"] is None

    def test_shard_info_mixed_shards(self, tmp_path: Path):
        """新旧混在 shard で shard_info が混在を検出する (CQ-0128)"""
        # shard_0000: teacher_best_mask あり
        writer1 = ShardWriter(tmp_path, max_samples=1)
        s1 = _make_sample(step_id=0)
        s1.teacher_best_mask = np.zeros(34, dtype=np.float32)
        s1.teacher_best_mask[0] = 1.0
        writer1.add(s1)
        writer1.close()

        # shard_0001: teacher_best_mask なし（旧形式）
        writer2 = ShardWriter(tmp_path, max_samples=1)
        writer2._shard_counter = 1  # shard_0001 として書き出す
        s2 = _make_sample(step_id=1)
        s2.shard_id = "shard_0001"
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        info = data["teacher_best_mask_shard_info"]
        assert info["available"] == 1
        assert info["total"] == 2
        # 混在時は teacher_best_masks は None（全 shard 揃わないため）
        assert data["teacher_best_masks"] is None


class TestShantenDeltaShard:
    """CQ-0145: shanten_delta の shard 読み書きテスト"""

    def test_shanten_delta_roundtrip(self, tmp_path: Path):
        """shanten_delta 付きサンプルの書き出し・読み込み"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.shanten_delta = 1.0
        s2 = _make_sample(step_id=1)
        s2.shanten_delta = -0.5
        s3 = _make_sample(step_id=2)
        s3.shanten_delta = 0.0
        writer.add(s1)
        writer.add(s2)
        writer.add(s3)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["shanten_deltas"] is not None
        np.testing.assert_allclose(
            data["shanten_deltas"], [1.0, -0.5, 0.0], atol=1e-6)

    def test_shanten_delta_none_returns_none(self, tmp_path: Path):
        """shanten_delta が全て None なら shanten_deltas は None"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s2 = _make_sample(step_id=1)
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["shanten_deltas"] is None

    def test_shanten_delta_mixed_fills_zero(self, tmp_path: Path):
        """shanten_delta が一部 None の場合、None は 0.0 で埋められる"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.shanten_delta = 2.0
        s2 = _make_sample(step_id=1)
        # s2.shanten_delta = None (default)
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["shanten_deltas"] is not None
        np.testing.assert_allclose(
            data["shanten_deltas"], [2.0, 0.0], atol=1e-6)

    def test_shanten_delta_filter_actor_type(self, tmp_path: Path):
        """filter_actor_type 時に shanten_deltas が正しくフィルタされる"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.shanten_delta = 1.0
        s1.actor_type = "policy"
        s2 = _make_sample(step_id=1)
        s2.shanten_delta = -1.0
        s2.actor_type = "baseline"
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors(filter_actor_type="policy")
        assert data["shanten_deltas"] is not None
        assert len(data["shanten_deltas"]) == 1
        np.testing.assert_allclose(data["shanten_deltas"], [1.0], atol=1e-6)

    def test_mixed_old_new_shard_nan_for_missing(self, tmp_path: Path):
        """CQ-0146: 旧 shard (shanten_delta なし) と新 shard の混在で欠落は NaN"""
        # shard 1: shanten_delta あり
        new_dir = tmp_path / "worker_0"
        new_dir.mkdir(parents=True, exist_ok=True)
        writer1 = ShardWriter(new_dir, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.shanten_delta = 1.5
        writer1.add(s1)
        writer1.close()

        # shard 2: shanten_delta なし（旧 shard 模擬）
        old_dir = tmp_path / "worker_1"
        old_dir.mkdir(parents=True, exist_ok=True)
        writer2 = ShardWriter(old_dir, max_samples=100)
        s2 = _make_sample(step_id=1)
        # shanten_delta = None → カラムなし
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["shanten_deltas"] is not None
        assert len(data["shanten_deltas"]) == 2
        # 新 shard のサンプルは値が保持される
        assert not np.isnan(data["shanten_deltas"][0])
        np.testing.assert_allclose(data["shanten_deltas"][0], 1.5, atol=1e-6)
        # 旧 shard のサンプルは NaN
        assert np.isnan(data["shanten_deltas"][1])

    def test_shanten_delta_shard_info(self, tmp_path: Path):
        """CQ-0146: shanten_delta_shard_info が正しく報告される"""
        # shard 1: shanten_delta あり
        new_dir = tmp_path / "worker_0"
        new_dir.mkdir(parents=True, exist_ok=True)
        writer1 = ShardWriter(new_dir, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.shanten_delta = 1.0
        writer1.add(s1)
        writer1.close()

        # shard 2: shanten_delta なし
        old_dir = tmp_path / "worker_1"
        old_dir.mkdir(parents=True, exist_ok=True)
        writer2 = ShardWriter(old_dir, max_samples=100)
        s2 = _make_sample(step_id=1)
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        info = data["shanten_delta_shard_info"]
        assert info["available"] == 1
        assert info["total"] == 2


class TestCurrentShantenShard:
    """CQ-0151: current_shanten の shard 読み書きテスト"""

    def test_current_shanten_roundtrip(self, tmp_path: Path):
        """current_shanten の書き込み→読み取りが一致する"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.current_shanten = 3
        s2 = _make_sample(step_id=1)
        s2.current_shanten = 0
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["current_shantens"] is not None
        assert len(data["current_shantens"]) == 2
        assert data["current_shantens"][0] == 3
        assert data["current_shantens"][1] == 0

    def test_current_shanten_none_returns_none(self, tmp_path: Path):
        """全サンプルが None のとき None が返る"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s = _make_sample(step_id=0)
        assert s.current_shanten is None
        writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["current_shantens"] is None


class TestTurnNumberShard:
    """CQ-0156: turn_number の shard 読み書きテスト"""

    def test_turn_number_roundtrip(self, tmp_path: Path):
        """turn_number の書き込み→読み取りが一致する"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.turn_number = 3
        s2 = _make_sample(step_id=1)
        s2.turn_number = 10
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["turn_numbers"] is not None
        assert len(data["turn_numbers"]) == 2
        assert data["turn_numbers"][0] == 3
        assert data["turn_numbers"][1] == 10

    def test_turn_number_none_returns_none(self, tmp_path: Path):
        """全サンプルが None のとき None が返る"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s = _make_sample(step_id=0)
        assert s.turn_number is None
        writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["turn_numbers"] is None


class TestRewardComponentShard:
    """CQ-0160: reward components の shard 読み書きテスト"""

    def test_reward_components_roundtrip(self, tmp_path: Path):
        """point_delta_reward / shanten_delta_reward の書き込み→読み取りが一致"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.point_delta_reward = 0.5
        s1.shanten_delta_reward = -0.1
        s2 = _make_sample(step_id=1)
        s2.point_delta_reward = 0.0
        s2.shanten_delta_reward = 0.3
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["point_delta_rewards"] is not None
        assert data["shanten_delta_rewards"] is not None
        np.testing.assert_allclose(
            data["point_delta_rewards"], [0.5, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            data["shanten_delta_rewards"], [-0.1, 0.3], atol=1e-6)

    def test_reward_components_none_returns_none(self, tmp_path: Path):
        """全サンプルが None のとき None が返る"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s = _make_sample(step_id=0)
        assert s.point_delta_reward is None
        assert s.shanten_delta_reward is None
        writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["point_delta_rewards"] is None
        assert data["shanten_delta_rewards"] is None

    def test_reward_components_mixed_old_new_shard(self, tmp_path: Path):
        """新旧 shard 混在で欠落は NaN"""
        # shard 1: reward components あり
        new_dir = tmp_path / "worker_0"
        new_dir.mkdir(parents=True, exist_ok=True)
        writer1 = ShardWriter(new_dir, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.point_delta_reward = 1.0
        s1.shanten_delta_reward = 0.5
        writer1.add(s1)
        writer1.close()

        # shard 2: reward components なし
        old_dir = tmp_path / "worker_1"
        old_dir.mkdir(parents=True, exist_ok=True)
        writer2 = ShardWriter(old_dir, max_samples=100)
        s2 = _make_sample(step_id=1)
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["point_delta_rewards"] is not None
        assert len(data["point_delta_rewards"]) == 2
        np.testing.assert_allclose(data["point_delta_rewards"][0], 1.0, atol=1e-6)
        assert np.isnan(data["point_delta_rewards"][1])
        assert data["shanten_delta_rewards"] is not None
        np.testing.assert_allclose(data["shanten_delta_rewards"][0], 0.5, atol=1e-6)
        assert np.isnan(data["shanten_delta_rewards"][1])


class TestPostRiichiDiscardShard:
    """CQ-0163: is_post_riichi_discard の shard 読み書きテスト"""

    def test_post_riichi_discard_roundtrip(self, tmp_path: Path):
        """is_post_riichi_discard の書き込み→読み取りが一致"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.is_post_riichi_discard = True
        s2 = _make_sample(step_id=1)
        s2.is_post_riichi_discard = False
        writer.add(s1)
        writer.add(s2)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["is_post_riichi_discards"] is not None
        assert data["is_post_riichi_discards"].dtype == np.bool_
        np.testing.assert_array_equal(
            data["is_post_riichi_discards"], [True, False])

    def test_post_riichi_discard_none_returns_none(self, tmp_path: Path):
        """全サンプルが None のとき None が返る"""
        writer = ShardWriter(tmp_path, max_samples=100)
        s = _make_sample(step_id=0)
        assert s.is_post_riichi_discard is None
        writer.add(s)
        writer.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["is_post_riichi_discards"] is None

    def test_post_riichi_discard_mixed_old_new_shard(self, tmp_path: Path):
        """新旧 shard 混在でカラム欠落時は None（安全フォールバック）"""
        # shard 1: is_post_riichi_discard あり
        new_dir = tmp_path / "worker_0"
        new_dir.mkdir(parents=True, exist_ok=True)
        writer1 = ShardWriter(new_dir, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.is_post_riichi_discard = True
        writer1.add(s1)
        writer1.close()

        # shard 2: is_post_riichi_discard なし（旧 shard）
        old_dir = tmp_path / "worker_1"
        old_dir.mkdir(parents=True, exist_ok=True)
        writer2 = ShardWriter(old_dir, max_samples=100)
        s2 = _make_sample(step_id=1)
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        # 旧 shard に -1 sentinel が混入するため、全 shard 揃い条件を満たさず None
        assert data["is_post_riichi_discards"] is None


class TestTeacherBestMaskFilterDeferred:
    """teacher_best_mask の filter 後判定テスト (CQ-0191)"""

    def test_mixed_shard_baseline_filter_returns_tbm(self, tmp_path: Path):
        """policy-only shard (tbm なし) + baseline shard (tbm あり) の mixed で、
        filter_actor_type="baseline" なら teacher_best_masks が返る"""
        # shard 1: policy のみ (tbm なし)
        dir1 = tmp_path / "worker_0"
        dir1.mkdir()
        writer1 = ShardWriter(dir1, max_samples=100)
        for i in range(3):
            s = _make_sample(step_id=i)
            s.actor_type = "policy"
            writer1.add(s)
        writer1.close()

        # shard 2: baseline のみ (tbm あり)
        dir2 = tmp_path / "worker_1"
        dir2.mkdir()
        writer2 = ShardWriter(dir2, max_samples=100)
        for i in range(3):
            s = _make_sample(step_id=10 + i)
            s.actor_type = "baseline"
            s.teacher_best_mask = np.zeros(34, dtype=np.float32)
            s.teacher_best_mask[i] = 1.0
            writer2.add(s)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors(filter_actor_type="baseline")
        assert data["teacher_best_masks"] is not None
        assert data["teacher_best_masks"].shape == (3, 34)
        assert data["teacher_best_masks"][0, 0] == 1.0

    def test_mixed_shard_no_filter_returns_none(self, tmp_path: Path):
        """mixed shard を filter なしで読むと tbm は None (後方互換)"""
        dir1 = tmp_path / "worker_0"
        dir1.mkdir()
        writer1 = ShardWriter(dir1, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.actor_type = "policy"
        writer1.add(s1)
        writer1.close()

        dir2 = tmp_path / "worker_1"
        dir2.mkdir()
        writer2 = ShardWriter(dir2, max_samples=100)
        s2 = _make_sample(step_id=1)
        s2.actor_type = "baseline"
        s2.teacher_best_mask = np.ones(34, dtype=np.float32)
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors()
        assert data["teacher_best_masks"] is None

    def test_baseline_cross_shard_tbm_gap_returns_none(self, tmp_path: Path):
        """baseline 行が shard 1 (tbm なし) と shard 2 (tbm あり) に分散 → None"""
        # shard 1: baseline without tbm
        dir1 = tmp_path / "worker_0"
        dir1.mkdir()
        writer1 = ShardWriter(dir1, max_samples=100)
        s1 = _make_sample(step_id=0)
        s1.actor_type = "baseline"
        writer1.add(s1)
        writer1.close()

        # shard 2: baseline with tbm
        dir2 = tmp_path / "worker_1"
        dir2.mkdir()
        writer2 = ShardWriter(dir2, max_samples=100)
        s2 = _make_sample(step_id=1)
        s2.actor_type = "baseline"
        s2.teacher_best_mask = np.ones(34, dtype=np.float32)
        writer2.add(s2)
        writer2.close()

        reader = ShardReader(tmp_path)
        data = reader.read_as_tensors(filter_actor_type="baseline")
        # shard 1 の baseline 行に tbm 欠落があるので None
        assert data["teacher_best_masks"] is None
