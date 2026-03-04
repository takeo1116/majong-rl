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
