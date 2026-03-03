"""CQ-0039: RL 基盤の統合テスト

- Shard IO roundtrip
- Full/Partial 切り替え
- Encoder 差し替え
- 再現性 (同一 seed + config → 同一 action 列)
- Self-play worker shard 生成
- Stage 1 固定ルール (副露なし)
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from mahjong_rl.env import Stage1Env
from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.shard import LearningSample, ShardWriter, ShardReader
from mahjong_rl.selfplay_worker import SelfPlayWorker
from mahjong_rl.learner import Learner


class TestShardIOIntegration:
    """Shard 書き出し → 読み込みの統合テスト"""

    def test_env_to_shard_roundtrip(self, tmp_path: Path):
        """env から特徴量抽出 → shard 書き出し → 読み込みで整合"""
        env = Stage1Env(observation_mode="full")
        encoder = FlatFeatureEncoder(observation_mode="full")
        obs, _ = env.reset(seed=42)

        features = encoder.encode(obs)
        mask = env.get_legal_mask()

        writer = ShardWriter(tmp_path, max_samples=100)
        writer.add(LearningSample(
            observation=features,
            legal_mask=mask,
            action=0,
            reward=0.0,
            log_prob=-1.0,
            value=0.0,
            terminated=False,
            round_over=False,
            experiment_id="test",
            run_id="run_0",
            worker_id="w0",
            episode_id="ep_0",
        ))
        writer.close()

        reader = ShardReader(tmp_path)
        loaded = reader.read_all()
        assert len(loaded) == 1
        np.testing.assert_array_almost_equal(loaded[0].observation, features)
        np.testing.assert_array_almost_equal(loaded[0].legal_mask, mask)


class TestObservationModeSwitching:
    """Full / Partial 切り替えテスト"""

    def test_full_mode_selfplay_to_shard(self, tmp_path: Path):
        """Full mode で self-play → shard → 読み込みが通る"""
        config = {
            "experiment": {"name": "test_full", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 1.0, "temperature": 1.0, "max_samples_per_shard": 10000},
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        worker = SelfPlayWorker(config=config, model=model, encoder=encoder,
                                output_dir=tmp_path / "shards")
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        tensors = reader.read_as_tensors()
        assert tensors["observations"].shape[0] > 0
        assert tensors["observations"].shape[1] == encoder.output_dim

    def test_partial_mode_selfplay_to_shard(self, tmp_path: Path):
        """Partial mode で self-play → shard → 読み込みが通る"""
        config = {
            "experiment": {"name": "test_partial", "observation_mode": "partial"},
            "selfplay": {"policy_ratio": 1.0, "temperature": 1.0, "max_samples_per_shard": 10000},
        }
        encoder = FlatFeatureEncoder(observation_mode="partial")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        worker = SelfPlayWorker(config=config, model=model, encoder=encoder,
                                output_dir=tmp_path / "shards")
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        tensors = reader.read_as_tensors()
        assert tensors["observations"].shape[0] > 0
        assert tensors["observations"].shape[1] == encoder.output_dim


class TestEncoderSwitching:
    """Encoder 差し替えテスト"""

    def test_flat_encoder_selfplay_to_learner(self, tmp_path: Path):
        """Flat encoder で self-play → learner が通る"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        # self-play
        sp_config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 1.0, "temperature": 1.0, "max_samples_per_shard": 10000},
        }
        shard_dir = tmp_path / "shards"
        worker = SelfPlayWorker(config=sp_config, model=model, encoder=encoder,
                                output_dir=shard_dir)
        worker.run(num_matches=1, seed_start=0)

        # learner
        train_config = {
            "training": {"lr": 1e-3, "batch_size": 32, "epochs": 1,
                         "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2},
        }
        learner = Learner(config=train_config, model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir, num_epochs=1)
        assert metrics["total_steps"] > 0

    def test_channel_encoder_selfplay_to_learner(self, tmp_path: Path):
        """Channel encoder で self-play → learner が通る"""
        encoder = ChannelTensorEncoder(observation_mode="full")
        # flatten して MLP に入れるので output_shape から計算
        shape = encoder.metadata().output_shape
        flat_dim = 1
        for d in shape:
            flat_dim *= d
        model = MLPPolicyValueModel(input_dim=flat_dim, hidden_dims=[16])

        sp_config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 1.0, "temperature": 1.0, "max_samples_per_shard": 10000},
        }
        shard_dir = tmp_path / "shards"
        worker = SelfPlayWorker(config=sp_config, model=model, encoder=encoder,
                                output_dir=shard_dir)
        worker.run(num_matches=1, seed_start=0)

        train_config = {
            "training": {"lr": 1e-3, "batch_size": 32, "epochs": 1,
                         "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2},
        }
        learner = Learner(config=train_config, model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir, num_epochs=1)
        assert metrics["total_steps"] > 0


class TestReproducibility:
    """同一 seed + 同一 config で再現性が取れるテスト"""

    def test_same_seed_same_actions(self):
        """同一 seed + 同一モデル → 同一 action 列"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])
        selector = ActionSelector(mode=SelectionMode.ARGMAX)

        actions_list = []
        for _ in range(2):
            env = Stage1Env(observation_mode="full")
            obs, _ = env.reset(seed=12345)
            torch.manual_seed(0)

            actions = []
            for _ in range(30):
                features = encoder.encode(obs)
                features_t = torch.from_numpy(features).unsqueeze(0)
                mask = env.get_legal_mask()
                mask_t = torch.from_numpy(mask).unsqueeze(0)

                with torch.no_grad():
                    output = model(features_t, mask_t)
                tile_type, _ = selector.select(output.logits[0], mask_t[0])
                actions.append(tile_type)

                obs, _, terminated, _, _ = env.step(tile_type)
                if terminated:
                    break

            actions_list.append(actions)

        assert actions_list[0] == actions_list[1]


class TestStage1Rules:
    """Stage 1 固定ルール: 副露が発生しないことの確認"""

    def test_no_calls_in_selfplay(self, tmp_path: Path):
        """self-play 中に副露が発生しない"""
        config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 0.5, "temperature": 1.0, "max_samples_per_shard": 10000},
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        worker = SelfPlayWorker(config=config, model=model, encoder=encoder,
                                output_dir=tmp_path / "shards")
        worker.run(num_matches=2, seed_start=42)

        # 全 shard を読んで action が 0-33 の範囲 (打牌のみ)
        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        for s in samples:
            assert 0 <= s.action < 34, f"不正な action: {s.action}"
