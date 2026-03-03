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


class TestStage1ExperimentIntegration:
    """Stage 1 実験経路の統合テスト (CQ-0047)"""

    def test_run_init_creates_directory(self, tmp_path: Path):
        """Stage 1 run 初期化でディレクトリ構造が作られる"""
        from mahjong_rl.experiment import ExperimentConfig, RunDirectory

        config = ExperimentConfig(
            experiment={"name": "integ_test", "stage": 1, "observation_mode": "full"},
            feature_encoder={"name": "FlatFeatureEncoder"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32, "epochs": 1},
            evaluation={"num_matches": 1},
        )
        run_dir = RunDirectory(base_dir=tmp_path).create(config)
        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "notes.md").exists()

    def test_baseline_teacher_data_saved(self, tmp_path: Path):
        """baseline 教師データが shard に保存される"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 0.5, "temperature": 1.0,
                          "max_samples_per_shard": 10000,
                          "save_baseline_actions": True},
        }
        shard_dir = tmp_path / "shards"
        worker = SelfPlayWorker(config=config, model=model, encoder=encoder,
                                output_dir=shard_dir)
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(shard_dir)
        tensors = reader.read_as_tensors()
        actor_types = tensors["actor_types"]
        has_baseline = any(a == "baseline" for a in actor_types)
        has_policy = any(a == "policy" for a in actor_types)
        assert has_baseline
        assert has_policy

    def test_imitation_filter_selects_baseline(self, tmp_path: Path):
        """imitation 用サンプル選別で baseline のみ抽出できる"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {"policy_ratio": 0.5, "temperature": 1.0,
                          "max_samples_per_shard": 10000,
                          "save_baseline_actions": True},
        }
        shard_dir = tmp_path / "shards"
        worker = SelfPlayWorker(config=config, model=model, encoder=encoder,
                                output_dir=shard_dir)
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(shard_dir)
        filtered = reader.read_as_tensors(filter_actor_type="baseline")
        assert filtered["observations"].shape[0] > 0
        assert all(a == "baseline" for a in filtered["actor_types"])

    def test_selfplay_learner_eval_pipeline(self, tmp_path: Path):
        """self-play → learner → eval の一連実行"""
        from mahjong_rl.experiment import ExperimentConfig
        from mahjong_rl.runner import Stage1Runner

        config = ExperimentConfig(
            experiment={"name": "pipeline_test", "stage": 1, "observation_mode": "full"},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert result["selfplay_stats"]["total_steps"] > 0
        assert result["train_metrics"]["total_steps"] > 0
        assert 1.0 <= result["eval_metrics"]["avg_rank"] <= 4.0

    def test_rotation_eval_in_pipeline(self, tmp_path: Path):
        """席ローテーション評価が動作する"""
        from mahjong_rl.evaluator import EvaluationRunner, RotationEvalResult

        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[16])

        runner = EvaluationRunner(model=model, encoder=encoder, observation_mode="full")
        result = runner.evaluate_rotation(num_matches=1, seed_start=42, seats=[0, 1])

        assert isinstance(result, RotationEvalResult)
        assert len(result.per_seat) == 2
        assert result.aggregate.num_matches == 2

    def test_reproducibility_across_runs(self, tmp_path: Path):
        """同一 config / seed で再現性がある"""
        from mahjong_rl.experiment import ExperimentConfig
        from mahjong_rl.runner import Stage1Runner

        def make_config():
            return ExperimentConfig(
                experiment={"name": "repro_test", "stage": 1, "observation_mode": "full"},
                feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
                model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                       "value_heads": ["round_delta"]},
                reward={"type": "point_delta"},
                selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                          "max_samples_per_shard": 10000, "seed_start": 42},
                training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                          "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                          "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                          "entropy_coef": 0.01, "max_grad_norm": 0.5},
                evaluation={"num_matches": 1, "seed_start": 100000},
            )

        results = []
        for i in range(2):
            torch.manual_seed(0)  # モデル初期化の乱数を固定
            config = make_config()
            runner = Stage1Runner(config=config, base_dir=tmp_path / f"run_{i}")
            result = runner.run()
            results.append(result)

        # 同一 seed + 同一モデル初期化なので self-play ステップ数は一致
        assert results[0]["selfplay_stats"]["total_steps"] == results[1]["selfplay_stats"]["total_steps"]
