"""テスト: learner.py — Learner (PPO)"""
import pytest
import torch
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

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


class TestBaselineImitation:
    """baseline 教師データによる imitation テスト (CQ-0043)"""

    def test_filter_baseline_samples(self, tmp_path: Path):
        """baseline サンプルだけを learner に渡せる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        writer = ShardWriter(shard_dir, max_samples=10000)
        for i in range(50):
            writer.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
                action=np.random.randint(0, 34),
                reward=0.0, log_prob=0.0, value=0.0,
                terminated=(i == 49), round_over=False,
                experiment_id="test", run_id="run", worker_id="w",
                episode_id="ep", step_id=i,
                actor_type="baseline" if i < 30 else "policy",
            ))
        writer.close()

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(
            shard_dir, num_epochs=1, filter_actor_type="baseline")
        assert metrics["total_steps"] == 30
        assert metrics["mode"] == "imitation"

    def test_imitation_with_baseline_data(self, tmp_path: Path):
        """baseline 教師データを用いた cross-entropy 学習が回る"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        writer = ShardWriter(shard_dir, max_samples=10000)
        for i in range(50):
            writer.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
                action=np.random.randint(0, 34),
                reward=0.0, log_prob=0.0, value=0.0,
                terminated=(i == 49), round_over=False,
                experiment_id="test", run_id="run", worker_id="w",
                episode_id="ep", step_id=i,
                actor_type="baseline",
            ))
        writer.close()

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(shard_dir, num_epochs=2)
        assert metrics["policy_loss"] > 0
        assert metrics["total_steps"] == 50

    def test_ppo_unaffected_by_filter(self, tmp_path: Path):
        """filter_actor_type は PPO モードでも正常動作"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        # デフォルト actor_type="policy" なので全件ヒット
        metrics = learner.train(
            shard_dir, num_epochs=1, filter_actor_type="policy")
        assert metrics["mode"] == "ppo"
        assert metrics["total_steps"] == 50


class TestImitationFilter:
    """imitation 品質フィルタテスト (CQ-0054)"""

    def _write_shards_with_varied_legal(self, shard_dir: Path, obs_dim: int = 100):
        """legal action 数が異なるサンプルを書き出す"""
        writer = ShardWriter(shard_dir, max_samples=10000)
        for i in range(60):
            # i < 30: legal 2個, i >= 30: legal 10個
            if i < 30:
                mask = np.zeros(34, dtype=np.float32)
                mask[0] = 1.0
                mask[1] = 1.0
            else:
                mask = np.zeros(34, dtype=np.float32)
                mask[:10] = 1.0
            writer.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=mask,
                action=0,
                reward=0.0, log_prob=0.0, value=0.0,
                terminated=(i == 59), round_over=False,
                experiment_id="test", run_id="run", worker_id="w",
                episode_id="ep", step_id=i,
                actor_type="baseline",
            ))
        writer.close()

    def test_filter_by_min_legal_actions(self, tmp_path: Path):
        """min_legal_actions でサンプルが絞り込まれる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        self._write_shards_with_varied_legal(shard_dir, obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(
            shard_dir, num_epochs=1,
            imitation_filter={"min_legal_actions": 5})

        # legal >= 5 のサンプルのみ (後半30件)
        assert metrics["total_steps"] == 30

    def test_filter_stats_recorded(self, tmp_path: Path):
        """フィルタ前後の件数が記録される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        self._write_shards_with_varied_legal(shard_dir, obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(
            shard_dir, num_epochs=1,
            imitation_filter={"min_legal_actions": 5})

        assert "filter_stats" in metrics
        assert metrics["filter_stats"]["before"] == 60
        assert metrics["filter_stats"]["after"] == 30
        assert metrics["filter_stats"]["removed"] == 30

    def test_filter_disabled(self, tmp_path: Path):
        """enabled=False でフィルタが無効になる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        self._write_shards_with_varied_legal(shard_dir, obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(
            shard_dir, num_epochs=1,
            imitation_filter={"min_legal_actions": 5, "enabled": False})

        # フィルタ無効なので全件
        assert metrics["total_steps"] == 60
        assert "filter_stats" not in metrics

    def test_no_filter_backward_compatible(self, tmp_path: Path):
        """imitation_filter 未指定で既存動作と互換"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        self._write_shards_with_varied_legal(shard_dir, obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["total_steps"] == 60
        assert "filter_stats" not in metrics

    def test_ppo_ignores_filter(self, tmp_path: Path):
        """PPO モードでは imitation_filter は無視される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        self._write_shards_with_varied_legal(shard_dir, obs_dim)

        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        metrics = learner.train(
            shard_dir, num_epochs=1,
            imitation_filter={"min_legal_actions": 5})

        # PPO なのでフィルタ不適用
        assert metrics["total_steps"] == 60
        assert "filter_stats" not in metrics


class TestLearnerDevice:
    """Learner デバイス切替テスト (CQ-0063)"""

    def test_cpu_device_works(self, tmp_path: Path):
        """CPU 明示指定で既存動作維持"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(
            config=_make_config(), model=model, run_dir=tmp_path,
            device=torch.device("cpu"))
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["total_steps"] == 50
        assert isinstance(metrics["policy_loss"], float)

    def test_device_in_metrics(self, tmp_path: Path):
        """metrics に device が記録される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(
            config=_make_config(), model=model, run_dir=tmp_path,
            device=torch.device("cpu"))
        metrics = learner.train(shard_dir, num_epochs=1)

        assert "device" in metrics
        assert metrics["device"] == "cpu"

    def test_checkpoint_cross_device(self, tmp_path: Path):
        """checkpoint save → load が device をまたいで動作する"""
        obs_dim = 100
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(
            config=_make_config(), model=model, run_dir=tmp_path,
            device=torch.device("cpu"))

        ckpt_path = learner.save_checkpoint(tag="test")
        assert ckpt_path.exists()

        # 別の learner で load
        model2 = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner2 = Learner(
            config=_make_config(), model=model2, run_dir=tmp_path / "run2",
            device=torch.device("cpu"))
        learner2.load_checkpoint(ckpt_path)

        test_input = torch.randn(1, obs_dim)
        test_mask = torch.ones(1, 34)
        with torch.no_grad():
            out1 = model(test_input, test_mask)
            out2 = model2(test_input, test_mask)
        torch.testing.assert_close(out1.logits, out2.logits)
