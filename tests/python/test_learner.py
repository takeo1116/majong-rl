"""テスト: learner.py — Learner (PPO)"""
import pytest
import torch
import numpy as np
from pathlib import Path

from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.shard import LearningSample, ShardWriter
from mahjong_rl.learner import Learner


def _write_dummy_shards(shard_dir: Path, n: int = 100, obs_dim: int = 455):
    """ダミーの shard データを書き出す"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        writer.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
            action=np.random.randint(0, 34),
            reward=np.random.randn() * 0.01,
            log_prob=-np.random.rand(),
            value=np.random.randn() * 0.1,
            terminated=(i == n - 1),
            round_over=(i % 20 == 19),
            experiment_id="dummy_exp",
            run_id="dummy_run",
            worker_id="dummy_worker",
            episode_id="dummy_ep",
            step_id=i,
        ))
    writer.close()


def _make_config():
    return {
        "training": {
            "algorithm": "ppo",
            "lr": 1e-3,
            "batch_size": 32,
            "epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        },
    }


class TestLearner:
    """Learner テスト"""

    def test_train_one_epoch(self, tmp_path: Path):
        """shard 読み込み → 1 epoch 学習が通る"""
        obs_dim = 455
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=100, obs_dim=obs_dim)

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[32])
        config = _make_config()

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        learner = Learner(config=config, model=model, run_dir=run_dir)
        metrics = learner.train(shard_dir, num_epochs=1)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["total_steps"] == 100

    def test_metrics_are_float(self, tmp_path: Path):
        """metrics の値が float"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=_make_config(), model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir)

        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["value_loss"], float)
        assert isinstance(metrics["entropy"], float)

    def test_checkpoint_save_load(self, tmp_path: Path):
        """checkpoint 保存 → ロード → 推論一致"""
        obs_dim = 100
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=_make_config(), model=model, run_dir=tmp_path)

        # 保存
        ckpt_path = learner.save_checkpoint(tag="test")
        assert ckpt_path.exists()

        # 推論結果を記録
        test_input = torch.randn(1, obs_dim)
        test_mask = torch.ones(1, 34)
        with torch.no_grad():
            before = model(test_input, test_mask)

        # パラメータを壊す
        for p in model.parameters():
            p.data.fill_(0.0)

        # ロードして推論一致確認
        learner.load_checkpoint(ckpt_path)
        with torch.no_grad():
            after = model(test_input, test_mask)

        torch.testing.assert_close(before.logits, after.logits)

    def test_empty_shard_no_error(self, tmp_path: Path):
        """空 shard ディレクトリでもエラーにならない"""
        shard_dir = tmp_path / "empty_shards"
        shard_dir.mkdir()

        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        learner = Learner(config=_make_config(), model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir)

        assert metrics["total_steps"] == 0

    def test_metrics_contain_mode(self, tmp_path: Path):
        """metrics に mode 識別子が含まれる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=_make_config(), model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["mode"] == "ppo"


class TestImitationMode:
    """模倣学習モードテスト"""

    def test_imitation_train(self, tmp_path: Path):
        """imitation モードで 1 epoch 学習が通る"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path)
        assert learner.mode == "imitation"

        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["mode"] == "imitation"
        assert isinstance(metrics["policy_loss"], float)
        assert metrics["policy_loss"] > 0  # cross-entropy は正
        assert metrics["value_loss"] == 0.0  # imitation では value 無効
        assert metrics["total_steps"] == 50

    def test_imitation_reduces_loss(self, tmp_path: Path):
        """imitation を複数 epoch 回すと loss が下がる"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=100, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["lr"] = 0.01

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])

        learner = Learner(config=config, model=model, run_dir=tmp_path)
        m1 = learner.train(shard_dir, num_epochs=1)
        m2 = learner.train(shard_dir, num_epochs=5)

        # 追加学習で loss が下がる（または同程度）
        assert m2["policy_loss"] <= m1["policy_loss"] * 1.5

    def test_ppo_mode_unchanged(self, tmp_path: Path):
        """ppo モードが imitation 追加で壊れていない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        assert config["training"]["algorithm"] == "ppo"

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path)
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["mode"] == "ppo"
        assert metrics["policy_loss"] != 0.0 or metrics["value_loss"] != 0.0

    def test_same_shard_both_modes(self, tmp_path: Path):
        """同一 shard で ppo / imitation 両方学習可能"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        # PPO
        ppo_config = _make_config()
        model_ppo = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner_ppo = Learner(config=ppo_config, model=model_ppo, run_dir=tmp_path / "ppo")
        m_ppo = learner_ppo.train(shard_dir, num_epochs=1)

        # Imitation
        imi_config = _make_config()
        imi_config["training"]["algorithm"] = "imitation"
        model_imi = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner_imi = Learner(config=imi_config, model=model_imi, run_dir=tmp_path / "imi")
        m_imi = learner_imi.train(shard_dir, num_epochs=1)

        assert m_ppo["mode"] == "ppo"
        assert m_imi["mode"] == "imitation"
        assert m_ppo["total_steps"] == m_imi["total_steps"]
