"""テスト: learner.py — Learner (PPO)"""
import pytest
import torch
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.shard import LearningSample, ShardReader, ShardWriter
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


def _write_shards_with_best_mask(
    shard_dir: Path, n: int, obs_dim: int,
    action: int = 0, tie_actions: list[int] | None = None,
):
    """teacher_best_mask 付きダミー shard を書き出す"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        mask = np.zeros(34, dtype=np.float32)
        mask[:10] = 1.0
        tbm = np.zeros(34, dtype=np.float32)
        tbm[action] = 1.0
        if tie_actions:
            for ta in tie_actions:
                tbm[ta] = 1.0
        writer.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=mask,
            action=action,
            reward=0.0,
            log_prob=0.0,
            value=0.0,
            terminated=(i == n - 1),
            round_over=False,
            experiment_id="test",
            run_id="run",
            worker_id="w0",
            episode_id="ep",
            step_id=i,
            actor_type="baseline",
            teacher_best_mask=tbm,
        ))
    writer.close()


class TestImitationReproMetrics:
    """imitation 教師再現メトリクスのテスト (CQ-0126)"""

    def test_metrics_present_with_best_mask(self, tmp_path: Path):
        """teacher_best_mask 付き shard で両指標が算出される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[32])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=5, filter_actor_type="baseline")

        assert "teacher_top1_match_rate" in metrics
        assert 0.0 <= metrics["teacher_top1_match_rate"] <= 1.0
        assert "teacher_best_set_hit_rate" in metrics
        assert 0.0 <= metrics["teacher_best_set_hit_rate"] <= 1.0

    def test_best_set_ge_top1(self, tmp_path: Path):
        """best_set_hit_rate >= top1_match_rate（定義上の整合）"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        # tie候補を追加: action=0 + tie=1
        _write_shards_with_best_mask(
            shard_dir, n=50, obs_dim=obs_dim, action=0, tie_actions=[1])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=3, filter_actor_type="baseline")

        assert metrics["teacher_best_set_hit_rate"] >= metrics["teacher_top1_match_rate"]

    def test_no_teacher_mask_top1_only(self, tmp_path: Path):
        """teacher_best_mask なし旧データで top1 のみ出力"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)

        assert "teacher_top1_match_rate" in metrics
        assert "teacher_best_set_hit_rate" not in metrics

    def test_teacher_best_set_status_available(self, tmp_path: Path):
        """teacher_best_mask 付きで status=available (CQ-0128)"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(shard_dir, n=20, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1, filter_actor_type="baseline")

        assert metrics["teacher_best_set_status"] == "available"
        assert "teacher_best_mask_shard_info" in metrics
        info = metrics["teacher_best_mask_shard_info"]
        assert info["available"] == info["total"]

    def test_teacher_best_set_status_missing(self, tmp_path: Path):
        """teacher_best_mask なしで status=missing (CQ-0128)"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=20, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["teacher_best_set_status"] == "missing"
        assert "teacher_best_mask_shard_info" in metrics
        info = metrics["teacher_best_mask_shard_info"]
        assert info["available"] == 0

    def test_teacher_best_set_status_mixed(self, tmp_path: Path):
        """新旧混在 shard で status=mixed (CQ-0128)"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"

        # shard_0000: teacher_best_mask あり
        writer1 = ShardWriter(shard_dir, max_samples=1)
        mask = np.zeros(34, dtype=np.float32)
        mask[:10] = 1.0
        tbm = np.zeros(34, dtype=np.float32)
        tbm[0] = 1.0
        writer1.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=mask,
            action=0, reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            experiment_id="test", run_id="run", worker_id="w0",
            episode_id="ep", step_id=0,
            teacher_best_mask=tbm,
        ))
        writer1.close()

        # shard_0001: teacher_best_mask なし（旧形式）
        writer2 = ShardWriter(shard_dir, max_samples=1)
        writer2._shard_counter = 1
        writer2.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=mask.copy(),
            action=0, reward=0.0, log_prob=0.0, value=0.0,
            terminated=True, round_over=False,
            experiment_id="test", run_id="run", worker_id="w0",
            shard_id="shard_0001",
            episode_id="ep", step_id=1,
        ))
        writer2.close()

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["teacher_best_set_status"] == "mixed"
        info = metrics["teacher_best_mask_shard_info"]
        assert info["available"] == 1
        assert info["total"] == 2


class TestImitationLossMode:
    """imitation loss mode 切替テスト (CQ-0131)"""

    def test_strict_top1_default(self, tmp_path: Path):
        """キー省略でデフォルト strict_top1 が使われる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(shard_dir, n=20, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        # imitation_loss_mode キーを設定しない（デフォルト）
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        assert learner.imitation_loss_mode == "strict_top1"
        metrics = learner.train(shard_dir, num_epochs=1, filter_actor_type="baseline")
        assert metrics["imitation_loss_mode"] == "strict_top1"
        assert metrics["policy_loss"] > 0

    def test_tie_aware_computes_loss(self, tmp_path: Path):
        """tie_aware_best_set で loss > 0 が計算される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(
            shard_dir, n=30, obs_dim=obs_dim, action=0, tie_actions=[1, 2])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=2, filter_actor_type="baseline")

        assert metrics["imitation_loss_mode"] == "tie_aware_best_set"
        assert metrics["policy_loss"] > 0
        assert metrics["num_updates"] > 0

    def test_tie_aware_increases_best_set_prob(self, tmp_path: Path):
        """tie-aware 学習後に best-set 確率が増加する"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        # best_set = {0, 1, 2}
        _write_shards_with_best_mask(
            shard_dir, n=50, obs_dim=obs_dim, action=0, tie_actions=[1, 2])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        config["training"]["lr"] = 1e-3
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        # 学習前の best-set 確率を計測
        reader = ShardReader(shard_dir)
        data = reader.read_as_tensors(filter_actor_type="baseline")
        obs_t = torch.from_numpy(data["observations"])
        masks_t = torch.from_numpy(data["legal_masks"])
        tbm_t = torch.from_numpy(data["teacher_best_masks"])

        model.eval()
        with torch.no_grad():
            out_before = model(obs_t, masks_t)
            probs_before = torch.softmax(out_before.logits, dim=-1)
            bsp_before = (probs_before * tbm_t).sum(dim=-1).mean().item()
        model.train()

        # 学習実行
        learner.train(shard_dir, num_epochs=10, filter_actor_type="baseline")

        # 学習後の best-set 確率を計測
        model.eval()
        with torch.no_grad():
            out_after = model(obs_t, masks_t)
            probs_after = torch.softmax(out_after.logits, dim=-1)
            bsp_after = (probs_after * tbm_t).sum(dim=-1).mean().item()

        assert bsp_after > bsp_before, (
            f"best-set prob should increase: {bsp_before:.4f} -> {bsp_after:.4f}")

    def test_missing_mask_tie_aware_raises(self, tmp_path: Path):
        """teacher_best_mask なしで tie_aware を使うと ValueError"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=20, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")

        with pytest.raises(ValueError, match="requires teacher_best_masks"):
            learner.train(shard_dir, num_epochs=1)

    def test_unknown_loss_mode_raises(self):
        """不明な loss mode で ValueError"""
        config = _make_config()
        config["training"]["imitation_loss_mode"] = "invalid_mode"
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="Unknown imitation_loss_mode"):
            Learner(config=config, model=model, run_dir=Path("/tmp/dummy"))

    def test_loss_mode_in_metrics(self, tmp_path: Path):
        """metrics に imitation_loss_mode が含まれる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(shard_dir, n=20, obs_dim=obs_dim)

        for mode in ["strict_top1", "tie_aware_best_set"]:
            config = _make_config()
            config["training"]["algorithm"] = "imitation"
            config["training"]["imitation_loss_mode"] = mode
            model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
            learner = Learner(config=config, model=model,
                              run_dir=tmp_path / f"run_{mode}")
            metrics = learner.train(shard_dir, num_epochs=1,
                                    filter_actor_type="baseline")
            assert metrics["imitation_loss_mode"] == mode


class TestTieAwareNumericalStability:
    """tie-aware loss の数値安定性テスト (CQ-0133)"""

    def test_loss_is_finite(self, tmp_path: Path):
        """tie-aware loss が NaN/inf にならない"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(
            shard_dir, n=30, obs_dim=obs_dim, action=0, tie_actions=[1])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        config["training"]["epochs"] = 5
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, filter_actor_type="baseline")

        assert np.isfinite(metrics["policy_loss"]), f"policy_loss is not finite: {metrics['policy_loss']}"
        assert np.isfinite(metrics["entropy"]), f"entropy is not finite: {metrics['entropy']}"

    def test_single_action_best_set(self, tmp_path: Path):
        """best-set が単一 action でも安定して計算される"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        # tie_actions なし = best_set は {action} のみ
        _write_shards_with_best_mask(
            shard_dir, n=20, obs_dim=obs_dim, action=0, tie_actions=None)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=3, filter_actor_type="baseline")

        assert np.isfinite(metrics["policy_loss"])
        assert metrics["policy_loss"] > 0

    def test_many_epochs_stable(self, tmp_path: Path):
        """多エポック学習後も loss が有限値"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(
            shard_dir, n=30, obs_dim=obs_dim, action=0, tie_actions=[1, 2])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        config["training"]["lr"] = 1e-2  # 大きめの学習率で安定性を検証
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=20, filter_actor_type="baseline")

        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["entropy"])


class TestPPODiagnosticStats:
    """PPO 診断統計テスト (CQ-0136)"""

    _DIAG_KEYS = [
        "advantage_mean", "advantage_std", "advantage_p50", "advantage_p90", "advantage_p99",
        "advantage_positive_ratio", "advantage_negative_ratio",
        "return_mean", "return_std", "return_p50", "return_p90", "return_p99",
        "old_value_mean", "old_value_std",
        "new_value_mean", "new_value_std",
        "value_error_mean", "value_error_std", "value_error_p50", "value_error_p90", "value_error_p99",
        "ratio_mean", "ratio_std", "ratio_p50", "ratio_p90", "ratio_p99",
        "clip_fraction",
    ]

    def _run_ppo(self, tmp_path: Path, n: int = 80, epochs: int = 2) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=n, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = epochs
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_ppo_diag_keys_present(self, tmp_path: Path):
        """PPO run で全診断キーが出力される"""
        metrics = self._run_ppo(tmp_path)
        assert "ppo_diag" in metrics
        diag = metrics["ppo_diag"]
        for key in self._DIAG_KEYS:
            assert key in diag, f"missing key: {key}"

    def test_ppo_diag_all_float(self, tmp_path: Path):
        """全診断値が float"""
        metrics = self._run_ppo(tmp_path)
        diag = metrics["ppo_diag"]
        for key in self._DIAG_KEYS:
            assert isinstance(diag[key], float), f"{key} is not float: {type(diag[key])}"

    def test_ppo_diag_no_nan_inf(self, tmp_path: Path):
        """NaN/inf が含まれない"""
        metrics = self._run_ppo(tmp_path)
        diag = metrics["ppo_diag"]
        for key in self._DIAG_KEYS:
            assert np.isfinite(diag[key]), f"{key} is not finite: {diag[key]}"

    def test_ratio_keys_in_unit_range(self, tmp_path: Path):
        """比率系キーが [0, 1] 範囲"""
        metrics = self._run_ppo(tmp_path)
        diag = metrics["ppo_diag"]
        for key in ["advantage_positive_ratio", "advantage_negative_ratio", "clip_fraction"]:
            assert 0.0 <= diag[key] <= 1.0, f"{key} out of range: {diag[key]}"

    def test_quantile_ordering(self, tmp_path: Path):
        """p50 <= p90 <= p99 の順序が成立"""
        metrics = self._run_ppo(tmp_path)
        diag = metrics["ppo_diag"]
        for prefix in ["advantage", "return", "value_error", "ratio"]:
            p50 = diag[f"{prefix}_p50"]
            p90 = diag[f"{prefix}_p90"]
            p99 = diag[f"{prefix}_p99"]
            assert p50 <= p90 + 1e-9, f"{prefix}: p50={p50} > p90={p90}"
            assert p90 <= p99 + 1e-9, f"{prefix}: p90={p90} > p99={p99}"

    def test_imitation_no_ppo_diag(self, tmp_path: Path):
        """imitation モードでは ppo_diag が出力されない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)
        assert "ppo_diag" not in metrics

    def test_ppo_diag_saved_to_train_metrics_json(self, tmp_path: Path):
        """train_metrics.json に ppo_diag が保存される"""
        import json
        metrics = self._run_ppo(tmp_path)
        json_path = tmp_path / "run" / "metrics" / "train_metrics.json"
        assert json_path.exists()
        with open(json_path) as f:
            saved = json.load(f)
        assert "ppo_diag" in saved
        assert saved["ppo_diag"]["clip_fraction"] == metrics["ppo_diag"]["clip_fraction"]


def _write_shards_with_shanten_delta(
    shard_dir: Path, n: int = 100, obs_dim: int = 100,
) -> None:
    """shanten_delta 付きダミー shard を書き出す"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        # 3群に分散: improve(+), same(0), worsen(-)
        if i % 3 == 0:
            sd = float(np.random.randint(1, 4))   # improve
        elif i % 3 == 1:
            sd = 0.0                                # same
        else:
            sd = float(-np.random.randint(1, 3))   # worsen
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
            shanten_delta=sd,
        ))
    writer.close()


class TestShantenConditionedDiag:
    """CQ-0145: shanten 変化別 advantage 診断統計テスト"""

    def _run_ppo_with_shanten(self, tmp_path: Path, n: int = 90) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_shanten_delta(shard_dir, n=n, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 1
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_shanten_diag_present(self, tmp_path: Path):
        """shanten_delta 付き shard で shanten_diag が出力される"""
        metrics = self._run_ppo_with_shanten(tmp_path)
        assert "ppo_diag" in metrics
        diag = metrics["ppo_diag"]
        assert "shanten_diag" in diag
        sd = diag["shanten_diag"]
        # CQ-0146: status / 件数情報
        assert sd["status"] == "complete"
        assert sd["unavailable_samples"] == 0
        for group in ("improve", "same", "worsen"):
            assert group in sd
            assert sd[group]["count"] > 0

    def test_shanten_diag_advantage_keys(self, tmp_path: Path):
        """各群の advantage に必要なキーが揃っている"""
        metrics = self._run_ppo_with_shanten(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        expected_adv_keys = {"mean", "std", "p50", "p90", "p99",
                             "positive_ratio", "negative_ratio"}
        expected_ret_keys = {"mean", "std", "p50", "p90", "p99"}
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            if entry["count"] == 0:
                continue
            assert expected_adv_keys <= set(entry["advantage"].keys())
            assert expected_ret_keys <= set(entry["return"].keys())
            assert expected_ret_keys <= set(entry["value_error"].keys())

    def test_shanten_diag_ratios_bounded(self, tmp_path: Path):
        """positive_ratio / negative_ratio が [0,1] 範囲"""
        metrics = self._run_ppo_with_shanten(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            if entry["count"] == 0:
                continue
            pr = entry["advantage"]["positive_ratio"]
            nr = entry["advantage"]["negative_ratio"]
            assert 0.0 <= pr <= 1.0
            assert 0.0 <= nr <= 1.0

    def test_shanten_diag_no_nan(self, tmp_path: Path):
        """全値が有限"""
        metrics = self._run_ppo_with_shanten(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            if entry["count"] == 0:
                continue
            for series in ("advantage", "return", "value_error"):
                for v in entry[series].values():
                    assert np.isfinite(v), f"{group}.{series} に非有限値: {v}"

    def test_shanten_diag_absent_without_shanten(self, tmp_path: Path):
        """shanten_delta なし shard では shanten_diag が出ない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 1
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert "ppo_diag" in metrics
        assert "shanten_diag" not in metrics["ppo_diag"]

    def test_mixed_old_new_shard_excludes_unavailable(self, tmp_path: Path):
        """CQ-0146: 旧/新 mixed shard で unavailable が same 群に入らない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"

        # 新 shard: shanten_delta あり (30 サンプル: 10 improve, 10 same, 10 worsen)
        new_dir = shard_dir / "worker_0"
        new_dir.mkdir(parents=True, exist_ok=True)
        writer1 = ShardWriter(new_dir, max_samples=10000)
        for i in range(30):
            if i < 10:
                sd = 1.0   # improve
            elif i < 20:
                sd = 0.0   # same
            else:
                sd = -1.0  # worsen
            writer1.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
                action=np.random.randint(0, 34),
                reward=np.random.randn() * 0.01,
                log_prob=-np.random.rand(),
                value=np.random.randn() * 0.1,
                terminated=(i == 29),
                round_over=(i % 10 == 9),
                experiment_id="dummy_exp",
                run_id="dummy_run",
                worker_id="dummy_worker",
                episode_id="dummy_ep",
                step_id=i,
                shanten_delta=sd,
            ))
        writer1.close()

        # 旧 shard: shanten_delta なし (20 サンプル)
        old_dir = shard_dir / "worker_1"
        old_dir.mkdir(parents=True, exist_ok=True)
        writer2 = ShardWriter(old_dir, max_samples=10000)
        for i in range(20):
            writer2.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
                action=np.random.randint(0, 34),
                reward=np.random.randn() * 0.01,
                log_prob=-np.random.rand(),
                value=np.random.randn() * 0.1,
                terminated=(i == 19),
                round_over=(i % 10 == 9),
                experiment_id="dummy_exp",
                run_id="dummy_run",
                worker_id="dummy_worker",
                episode_id="dummy_ep",
                step_id=30 + i,
            ))
        writer2.close()

        config = _make_config()
        config["training"]["epochs"] = 1
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)

        assert "ppo_diag" in metrics
        sd = metrics["ppo_diag"]["shanten_diag"]

        # status は partial（mixed）
        assert sd["status"] == "partial"
        assert sd["available_samples"] == 30
        assert sd["unavailable_samples"] == 20

        # same 群は真に delta==0 の 10 サンプルのみ（旧 shard の 20 が混入しない）
        assert sd["same"]["count"] == 10
        assert sd["improve"]["count"] == 10
        assert sd["worsen"]["count"] == 10


class TestJointImitation:
    """CQ-0150: imitation value warm start テスト"""

    def test_joint_imitation_off_value_loss_zero(self, tmp_path: Path):
        """off のとき value_loss=0.0"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        # imitation_value_warmstart 未設定（デフォルト off）

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)

        assert metrics["value_loss"] == 0.0
        assert metrics["imitation_value_warmstart"]["enabled"] is False

    def test_joint_imitation_on_value_loss_nonzero(self, tmp_path: Path):
        """on のとき value_loss > 0"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=50, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_value_warmstart"] = {
            "enabled": True,
            "coef": 0.5,
        }

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=2)

        assert metrics["value_loss"] > 0.0
        assert metrics["imitation_value_warmstart"]["enabled"] is True
        assert metrics["imitation_value_warmstart"]["coef"] == 0.5

    def test_joint_imitation_metrics_keys(self, tmp_path: Path):
        """metrics に imitation_value_warmstart が含まれる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=30, obs_dim=obs_dim)

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_value_warmstart"] = {
            "enabled": True,
            "coef": 0.25,
        }

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1)

        assert "imitation_value_warmstart" in metrics
        ivw = metrics["imitation_value_warmstart"]
        assert ivw["enabled"] is True
        assert ivw["coef"] == 0.25
        # policy_loss も出る
        assert "policy_loss" in metrics
        assert metrics["policy_loss"] > 0

    def test_joint_imitation_with_tie_aware(self, tmp_path: Path):
        """tie_aware_best_set mode でも joint imitation が動作する"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_best_mask(
            shard_dir, n=30, obs_dim=obs_dim, action=0, tie_actions=[1, 2])

        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["imitation_loss_mode"] = "tie_aware_best_set"
        config["training"]["imitation_value_warmstart"] = {
            "enabled": True,
            "coef": 0.5,
        }

        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir, num_epochs=1, filter_actor_type="baseline")

        assert metrics["imitation_loss_mode"] == "tie_aware_best_set"
        assert metrics["value_loss"] > 0.0
        assert metrics["policy_loss"] > 0.0


def _write_shards_with_turn_number(
    shard_dir: Path, n: int = 90, obs_dim: int = 100,
    include_reward_components: bool = False,
) -> None:
    """turn_number + shanten_delta 付きダミー shard を書き出す"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        # turn_number: 0-17 に分散
        tn = i % 18
        # shanten_delta: 3群に分散
        if i % 3 == 0:
            sd = float(np.random.randint(1, 4))
        elif i % 3 == 1:
            sd = 0.0
        else:
            sd = float(-np.random.randint(1, 3))
        # CQ-0160: reward components
        pdr = float(np.random.randn() * 0.01) if include_reward_components else None
        sdr_val = float(np.random.randn() * 0.005) if include_reward_components else None
        reward = (pdr + sdr_val) if include_reward_components else np.random.randn() * 0.01
        writer.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
            action=np.random.randint(0, 34),
            reward=reward,
            log_prob=-np.random.rand(),
            value=np.random.randn() * 0.1,
            terminated=(i == n - 1),
            round_over=(i % 20 == 19),
            experiment_id="dummy_exp",
            run_id="dummy_run",
            worker_id="dummy_worker",
            episode_id="dummy_ep",
            step_id=i,
            shanten_delta=sd,
            turn_number=tn,
            point_delta_reward=pdr,
            shanten_delta_reward=sdr_val,
        ))
    writer.close()


class TestShantenDiagValueBreakdown:
    """CQ-0155: shanten_diag に old_value/new_value/value_update_delta を追加"""

    def _run(self, tmp_path: Path) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_turn_number(shard_dir, n=90, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_old_value_in_groups(self, tmp_path: Path):
        """各群に old_value が存在し mean/std/p50/p90/p99 がある"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        expected_keys = {"mean", "std", "p50", "p90", "p99"}
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            assert "old_value" in entry
            assert expected_keys <= set(entry["old_value"].keys())

    def test_new_value_in_groups(self, tmp_path: Path):
        """各群に new_value が存在"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        expected_keys = {"mean", "std", "p50", "p90", "p99"}
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            assert "new_value" in entry
            assert expected_keys <= set(entry["new_value"].keys())

    def test_value_update_delta_in_groups(self, tmp_path: Path):
        """value_update_delta が存在し定義通り (new_value - old_value)"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            assert "value_update_delta" in entry
            # value_update_delta の全統計値が有限
            for v in entry["value_update_delta"].values():
                assert np.isfinite(v), f"{group}.value_update_delta に非有限値"


class TestTurnDiag:
    """CQ-0156: 巡目バケット別 learner 診断テスト"""

    def _run(self, tmp_path: Path) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_turn_number(shard_dir, n=90, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_turn_diag_buckets_exist(self, tmp_path: Path):
        """early/mid/late バケットが存在し count > 0"""
        metrics = self._run(tmp_path)
        assert "turn_diag" in metrics["ppo_diag"]
        td = metrics["ppo_diag"]["turn_diag"]
        for bucket in ("early", "mid", "late"):
            assert bucket in td
            assert td[bucket]["count"] > 0

    def test_turn_diag_stats_keys(self, tmp_path: Path):
        """各バケットに return/old_value/advantage 等が存在"""
        metrics = self._run(tmp_path)
        td = metrics["ppo_diag"]["turn_diag"]
        expected_stat_keys = {"mean", "std", "p50", "p90", "p99"}
        for bucket in ("early", "mid", "late"):
            entry = td[bucket]
            assert entry["count"] > 0
            for series in ("advantage", "return", "old_value", "value_error"):
                assert series in entry
                assert expected_stat_keys <= set(entry[series].keys())
            # new_value / value_update_delta も存在
            assert "new_value" in entry
            assert "value_update_delta" in entry

    def test_turn_diag_advantage_ratios(self, tmp_path: Path):
        """advantage に positive_ratio/negative_ratio がある"""
        metrics = self._run(tmp_path)
        td = metrics["ppo_diag"]["turn_diag"]
        for bucket in ("early", "mid", "late"):
            entry = td[bucket]
            assert entry["count"] > 0
            adv = entry["advantage"]
            assert "positive_ratio" in adv
            assert "negative_ratio" in adv
            assert 0.0 <= adv["positive_ratio"] <= 1.0
            assert 0.0 <= adv["negative_ratio"] <= 1.0


class TestRewardComponentDiag:
    """CQ-0160: shanten_diag に reward/point_delta_reward/shanten_delta_reward/delta_t を追加"""

    def _run(self, tmp_path: Path) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_turn_number(
            shard_dir, n=90, obs_dim=obs_dim, include_reward_components=True)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_reward_keys_in_groups(self, tmp_path: Path):
        """各群に reward/point_delta_reward/shanten_delta_reward/delta_t が存在"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        expected_stat_keys = {"mean", "std", "p50", "p90", "p99"}
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            for key in ("reward", "point_delta_reward", "shanten_delta_reward", "delta_t"):
                assert key in entry, f"{group} に {key} がない"
                assert expected_stat_keys <= set(entry[key].keys()), f"{group}.{key} の統計キーが不足"

    def test_reward_values_finite(self, tmp_path: Path):
        """全統計値が有限"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            for key in ("reward", "point_delta_reward", "shanten_delta_reward", "delta_t"):
                for stat_name, v in entry[key].items():
                    assert np.isfinite(v), f"{group}.{key}.{stat_name} が非有限値: {v}"

    def test_no_reward_components_no_keys(self, tmp_path: Path):
        """reward components がない場合は reward/delta_t キーが存在しない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        # include_reward_components=False (デフォルト)
        _write_shards_with_turn_number(shard_dir, n=90, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            # reward components がない → 新キーは出ない
            assert "point_delta_reward" not in entry
            assert "shanten_delta_reward" not in entry
            # reward と delta_t は rewards/terminateds から計算可能なので出る
            assert "reward" in entry
            assert "delta_t" in entry


def _write_shards_with_post_riichi(
    shard_dir: Path, n: int = 90, obs_dim: int = 100,
    post_riichi_ratio: float = 0.3,
) -> None:
    """is_post_riichi_discard 付きダミー shard を書き出す"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        # shanten_delta: 3群に分散
        if i % 3 == 0:
            sd = float(np.random.randint(1, 4))
        elif i % 3 == 1:
            sd = 0.0
        else:
            sd = float(-np.random.randint(1, 3))
        pdr = float(np.random.randn() * 0.01)
        sdr_val = float(np.random.randn() * 0.005)
        is_pr = (np.random.rand() < post_riichi_ratio)
        writer.add(LearningSample(
            observation=np.random.randn(obs_dim).astype(np.float32),
            legal_mask=(np.random.rand(34) > 0.5).astype(np.float32),
            action=np.random.randint(0, 34),
            reward=pdr + sdr_val,
            log_prob=-np.random.rand(),
            value=np.random.randn() * 0.1,
            terminated=(i == n - 1),
            round_over=(i % 20 == 19),
            experiment_id="dummy_exp",
            run_id="dummy_run",
            worker_id="dummy_worker",
            episode_id="dummy_ep",
            step_id=i,
            shanten_delta=sd,
            turn_number=i % 18,
            point_delta_reward=pdr,
            shanten_delta_reward=sdr_val,
            is_post_riichi_discard=is_pr,
        ))
    writer.close()


class TestPostRiichiDiag:
    """CQ-0163: shanten_diag に立直後打牌統計を追加"""

    def _run(self, tmp_path: Path) -> dict:
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_post_riichi(shard_dir, n=90, obs_dim=obs_dim, post_riichi_ratio=0.3)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        return learner.train(shard_dir)

    def test_post_riichi_count_in_groups(self, tmp_path: Path):
        """各群に post_riichi_discard_count / post_riichi_discard_ratio が存在"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert entry["count"] > 0
            assert "post_riichi_discard_count" in entry
            assert "post_riichi_discard_ratio" in entry
            assert 0 <= entry["post_riichi_discard_ratio"] <= 1.0

    def test_post_riichi_toplevel_stats(self, tmp_path: Path):
        """top-level に total/available post_riichi_discards が存在"""
        metrics = self._run(tmp_path)
        sd = metrics["ppo_diag"]["shanten_diag"]
        assert "total_post_riichi_discards" in sd
        assert "available_post_riichi_discards" in sd
        assert sd["total_post_riichi_discards"] > 0
        assert sd["available_post_riichi_discards"] > 0

    def test_no_post_riichi_flag_returns_none(self, tmp_path: Path):
        """is_post_riichi_discards がない場合は None"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_turn_number(shard_dir, n=90, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["epochs"] = 2
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        sd = metrics["ppo_diag"]["shanten_diag"]
        assert sd["total_post_riichi_discards"] is None
        assert sd["available_post_riichi_discards"] is None
        # 群にも post_riichi キーがない
        for group in ("improve", "same", "worsen"):
            entry = sd[group]
            assert "post_riichi_discard_count" not in entry


class TestPostRiichiExclusion:
    """CQ-0164: 立直後打牌の学習除外テスト"""

    def test_exclusion_off_keeps_all_samples(self, tmp_path: Path):
        """exclusion off では全サンプル使用"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_shards_with_post_riichi(shard_dir, n=90, obs_dim=obs_dim, post_riichi_ratio=0.5)
        config = _make_config()
        config["training"]["epochs"] = 1
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert metrics["total_steps"] == 90
        assert "post_riichi_exclusion" not in metrics

    def test_exclusion_on_reduces_samples_ppo(self, tmp_path: Path):
        """PPO: exclusion on で対象サンプル数が減る"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        np.random.seed(42)
        _write_shards_with_post_riichi(shard_dir, n=90, obs_dim=obs_dim, post_riichi_ratio=0.5)
        config = _make_config()
        config["training"]["epochs"] = 1
        config["training"]["exclude_post_riichi_discards"] = {"enabled": True}
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert metrics["total_steps"] < 90
        assert "post_riichi_exclusion" in metrics
        exc = metrics["post_riichi_exclusion"]
        assert exc["excluded_post_riichi_discards"] > 0
        assert exc["used_samples"] == metrics["total_steps"]

    def test_exclusion_on_reduces_samples_imitation(self, tmp_path: Path):
        """imitation: exclusion on で対象サンプル数が減る"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        np.random.seed(42)
        _write_shards_with_post_riichi(shard_dir, n=90, obs_dim=obs_dim, post_riichi_ratio=0.5)
        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["epochs"] = 1
        config["training"]["exclude_post_riichi_discards"] = {"enabled": True}
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert metrics["total_steps"] < 90
        assert "post_riichi_exclusion" in metrics
        exc = metrics["post_riichi_exclusion"]
        assert exc["excluded_post_riichi_discards"] > 0


class TestValueLossStabilization:
    """value_loss / advantage 安定化テスト (CQ-0176)"""

    def test_default_mse(self, tmp_path: Path):
        """デフォルトは mse で既存挙動維持"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert "ppo_diag" in metrics
        diag = metrics["ppo_diag"]
        assert diag["value_loss_type"] == "mse"
        assert diag["advantage_clip_fraction"] == 0.0
        assert "advantage_clip_value" not in diag

    def test_huber_loss(self, tmp_path: Path):
        """huber loss が有効になる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": 0.5}
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert metrics["ppo_diag"]["value_loss_type"] == "huber"

    def test_advantage_clip(self, tmp_path: Path):
        """advantage clip が有効になり診断値が出る"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": 1.0}
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert "advantage_abs_mean_before_clip" in diag
        assert "advantage_abs_mean_after_clip" in diag
        assert "advantage_clip_fraction" in diag
        assert "advantage_clip_value" in diag
        assert diag["advantage_clip_value"] == 1.0
        assert diag["advantage_abs_mean_after_clip"] <= diag["advantage_abs_mean_before_clip"] + 1e-6

    def test_combined_huber_and_clip(self, tmp_path: Path):
        """huber + advantage clip の併用"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": 1.0}
        config["training"]["advantage_stabilization"] = {"clip": 2.0}
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        diag = metrics["ppo_diag"]
        assert diag["value_loss_type"] == "huber"
        assert diag["advantage_clip_value"] == 2.0


class TestStabilizationValidation:
    """安定化設定の入力検証テスト (CQ-0177)"""

    def test_invalid_value_loss_type(self, tmp_path: Path):
        """不正な value_loss.type で ValueError"""
        config = _make_config()
        config["training"]["value_loss"] = {"type": "l1"}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="value_loss.type"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_advantage_clip_zero(self, tmp_path: Path):
        """advantage_stabilization.clip=0 で ValueError"""
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": 0}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_advantage_clip_negative(self, tmp_path: Path):
        """advantage_stabilization.clip=-1 で ValueError"""
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": -1.0}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_advantage_clip_string(self, tmp_path: Path):
        """advantage_stabilization.clip=文字列 で ValueError"""
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": "auto"}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_valid_mse(self, tmp_path: Path):
        """mse は正常に通る"""
        config = _make_config()
        config["training"]["value_loss"] = {"type": "mse"}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        assert learner._value_loss_type == "mse"

    def test_valid_huber(self, tmp_path: Path):
        """huber は正常に通る"""
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber"}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        assert learner._value_loss_type == "huber"

    def test_valid_clip_positive(self, tmp_path: Path):
        """正のclipは正常に通る"""
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": 3.0}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        assert learner._advantage_clip == 3.0


class TestStabilizationValidationExtended:
    """安定化設定の入力検証強化テスト (CQ-0178)"""

    def test_huber_delta_zero(self, tmp_path: Path):
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": 0}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="huber_delta"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_huber_delta_negative(self, tmp_path: Path):
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": -1.0}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="huber_delta"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_huber_delta_nan(self, tmp_path: Path):
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": float("nan")}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="huber_delta"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_clip_nan(self, tmp_path: Path):
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": float("nan")}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_clip_inf(self, tmp_path: Path):
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": float("inf")}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_clip_bool_true(self, tmp_path: Path):
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": True}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_clip_bool_false(self, tmp_path: Path):
        config = _make_config()
        config["training"]["advantage_stabilization"] = {"clip": False}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        with pytest.raises(ValueError, match="advantage_stabilization.clip"):
            Learner(config=config, model=model, run_dir=tmp_path / "run")

    def test_valid_huber_delta_positive(self, tmp_path: Path):
        config = _make_config()
        config["training"]["value_loss"] = {"type": "huber", "huber_delta": 0.5}
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        assert learner._huber_delta == 0.5


def _setup_anchor_run(tmp_path: Path, obs_dim: int = 100, n: int = 80):
    """anchor テスト用の shard + imitation checkpoint を準備する"""
    run_dir = tmp_path / "run"
    shard_dir = run_dir / "shards"
    _write_dummy_shards(shard_dir, n=n, obs_dim=obs_dim)
    # imitation checkpoint を作成
    model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "checkpoint_imitation.pt")
    return run_dir, shard_dir, obs_dim


class TestPolicyAnchor:
    """policy anchor (参照方策アンカー) テスト (CQ-0182)"""

    def test_anchor_disabled_default(self, tmp_path: Path):
        """デフォルト (enabled=false) で既存挙動維持"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert "ppo_diag" in metrics
        pa = metrics["ppo_diag"]["policy_anchor"]
        assert pa["enabled"] is False
        assert "anchor_loss_mean" not in pa

    def test_anchor_kl(self, tmp_path: Path):
        """type=kl で学習完走し診断キーが出る"""
        run_dir, shard_dir, obs_dim = _setup_anchor_run(tmp_path)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "kl", "coef": 0.1,
            "reference": "imitation_fixed",
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=run_dir)
        metrics = learner.train(shard_dir)
        assert "ppo_diag" in metrics
        pa = metrics["ppo_diag"]["policy_anchor"]
        assert pa["enabled"] is True
        assert pa["type"] == "kl"
        assert pa["coef"] == 0.1
        assert "anchor_loss_mean" in pa
        assert "anchor_kl_mean" in pa
        assert pa["anchor_loss_mean"] >= 0

    def test_anchor_bc(self, tmp_path: Path):
        """type=bc で学習完走し診断キーが出る"""
        run_dir, shard_dir, obs_dim = _setup_anchor_run(tmp_path)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "bc", "coef": 0.05,
            "reference": "imitation_fixed",
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=run_dir)
        metrics = learner.train(shard_dir)
        pa = metrics["ppo_diag"]["policy_anchor"]
        assert pa["enabled"] is True
        assert pa["type"] == "bc"
        assert "anchor_ce_mean" in pa

    def test_missing_checkpoint_error(self, tmp_path: Path):
        """checkpoint_imitation.pt 欠落時に PPO 学習で FileNotFoundError (CQ-0183)"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "kl", "coef": 0.1,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        # __init__ は成功する (遅延ロード)
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        # train() 時に PPO が参照モデルをロードしようとして失敗
        with pytest.raises(FileNotFoundError, match="checkpoint_imitation"):
            learner.train(shard_dir)

    def test_imitation_with_anchor_config_ok(self, tmp_path: Path):
        """CQ-0183: imitation フェーズで anchor 設定があっても例外にならない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        config["training"]["algorithm"] = "imitation"
        config["training"]["epochs"] = 1
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "kl", "coef": 0.1,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        # imitation では checkpoint_imitation.pt が不在でも問題ない
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        # imitation は ppo_diag を出さないので anchor 診断もない
        assert metrics.get("mode") == "imitation"
        assert "ppo_diag" not in metrics

    def test_invalid_type(self, tmp_path: Path):
        """不正な type で ValueError"""
        run_dir, _, obs_dim = _setup_anchor_run(tmp_path)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "l2", "coef": 0.1,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        with pytest.raises(ValueError, match="policy_anchor.type"):
            Learner(config=config, model=model, run_dir=run_dir)

    def test_invalid_coef(self, tmp_path: Path):
        """coef=0 で ValueError"""
        run_dir, _, obs_dim = _setup_anchor_run(tmp_path)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "kl", "coef": 0,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        with pytest.raises(ValueError, match="policy_anchor.coef"):
            Learner(config=config, model=model, run_dir=run_dir)

    def test_invalid_reference(self, tmp_path: Path):
        """不正な reference で ValueError"""
        run_dir, _, obs_dim = _setup_anchor_run(tmp_path)
        config = _make_config()
        config["training"]["policy_anchor"] = {
            "enabled": True, "type": "kl", "coef": 0.1,
            "reference": "latest",
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        with pytest.raises(ValueError, match="policy_anchor.reference"):
            Learner(config=config, model=model, run_dir=run_dir)


def _write_mixed_shards(shard_dir: Path, n: int = 80, obs_dim: int = 100,
                         post_riichi_ratio: float = 0.3):
    """policy/baseline 混合 shard (post_riichi 付き)"""
    writer = ShardWriter(shard_dir, max_samples=10000)
    for i in range(n):
        is_bl = (i % 4 == 0)  # 25% baseline
        is_pr = (i < int(n * post_riichi_ratio))
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
            actor_type="baseline" if is_bl else "policy",
            is_post_riichi_discard=is_pr,
        ))
    writer.close()


class TestMixedPPOActorSync:
    """mixed PPO の actor_types 同期テスト (CQ-0194)"""

    def test_mixed_ppo_with_exclude_post_riichi(self, tmp_path: Path):
        """exclude_post_riichi + mixed PPO で actor_types がズレない"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_mixed_shards(shard_dir, n=80, obs_dim=obs_dim, post_riichi_ratio=0.3)
        config = _make_config()
        config["training"]["exclude_post_riichi_discards"] = {"enabled": True}
        config["training"]["rule_mix_learner"] = {
            "ppo_mode": "mixed",
            "baseline_sample_weight": 0.5,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        assert "ppo_diag" in metrics
        mp = metrics["ppo_diag"]["mixed_ppo"]
        # 件数合計 == total_steps
        assert mp["num_policy_samples"] + mp["num_baseline_samples"] == metrics["total_steps"]

    def test_separated_default_backward_compat(self, tmp_path: Path):
        """separated (default) で既存挙動維持"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)
        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        mp = metrics["ppo_diag"]["mixed_ppo"]
        assert mp["mixed_ppo_enabled"] is False


class TestTeacherAgreement:
    """teacher_agreement 診断テスト (CQ-0199)"""

    def test_with_baseline_samples(self, tmp_path: Path):
        """baseline サンプルありで teacher_agreement が出力される"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_mixed_shards(shard_dir, n=80, obs_dim=obs_dim, post_riichi_ratio=0.0)
        config = _make_config()
        config["training"]["rule_mix_learner"] = {
            "ppo_mode": "mixed", "baseline_sample_weight": 1.0,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        ta = metrics["ppo_diag"]["teacher_agreement"]
        assert ta["enabled"] is True
        assert ta["num_baseline_samples"] > 0
        assert 0.0 <= ta["action_match_rate_before"] <= 1.0
        assert 0.0 <= ta["action_match_rate_after"] <= 1.0

    def test_no_baseline_samples(self, tmp_path: Path):
        """baseline 0件で skip になる"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_dummy_shards(shard_dir, n=80, obs_dim=obs_dim)  # actor_type 未設定 = policy-like
        config = _make_config()
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        ta = metrics["ppo_diag"]["teacher_agreement"]
        assert ta["enabled"] is False

    def test_no_teacher_best_masks(self, tmp_path: Path):
        """teacher_best_masks なしで best_set_hit_rate が null"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_mixed_shards(shard_dir, n=80, obs_dim=obs_dim, post_riichi_ratio=0.0)
        config = _make_config()
        config["training"]["rule_mix_learner"] = {
            "ppo_mode": "mixed", "baseline_sample_weight": 1.0,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        ta = metrics["ppo_diag"]["teacher_agreement"]
        assert ta["enabled"] is True
        assert ta["num_best_set_samples"] == 0
        assert ta["best_set_hit_rate_before"] is None
        assert ta["best_set_hit_rate_after"] is None

    def test_with_teacher_best_masks(self, tmp_path: Path):
        """CQ-0201: teacher_best_masks ありで best_set_hit_rate が正常算出"""
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        # baseline + teacher_best_mask 付き shard を作成
        writer = ShardWriter(shard_dir, max_samples=10000)
        for i in range(80):
            is_bl = (i % 4 == 0)
            tbm = None
            if is_bl:
                tbm = np.zeros(34, dtype=np.float32)
                tbm[i % 34] = 1.0  # best action
                tbm[(i + 1) % 34] = 1.0  # 2nd best
            writer.add(LearningSample(
                observation=np.random.randn(obs_dim).astype(np.float32),
                legal_mask=(np.random.rand(34) > 0.3).astype(np.float32),
                action=i % 34,
                reward=np.random.randn() * 0.01,
                log_prob=-np.random.rand(),
                value=np.random.randn() * 0.1,
                terminated=(i == 79),
                round_over=(i % 20 == 19),
                experiment_id="dummy", run_id="dummy", worker_id="w0",
                episode_id="ep0", step_id=i,
                actor_type="baseline" if is_bl else "policy",
                teacher_best_mask=tbm,
            ))
        writer.close()

        config = _make_config()
        config["training"]["rule_mix_learner"] = {
            "ppo_mode": "mixed", "baseline_sample_weight": 1.0,
        }
        model = MLPPolicyValueModel(input_dim=obs_dim, hidden_dims=[16])
        learner = Learner(config=config, model=model, run_dir=tmp_path / "run")
        metrics = learner.train(shard_dir)
        ta = metrics["ppo_diag"]["teacher_agreement"]
        assert ta["enabled"] is True
        assert ta["num_best_set_samples"] > 0
        assert ta["best_set_hit_rate_before"] is not None
        assert 0.0 <= ta["best_set_hit_rate_before"] <= 1.0
        assert ta["best_set_hit_rate_after"] is not None
        assert 0.0 <= ta["best_set_hit_rate_after"] <= 1.0
