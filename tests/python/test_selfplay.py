"""テスト: selfplay_worker.py — Self-Play Worker"""
import pytest
import torch
import numpy as np
from pathlib import Path

from mahjong_rl.encoders import FlatFeatureEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.action_selector import ActionSelector, SelectionMode
from mahjong_rl.shard import ShardReader
from mahjong_rl.selfplay_worker import SelfPlayWorker


def _make_config(observation_mode="full", policy_ratio=0.5):
    """テスト用設定 dict"""
    return {
        "experiment": {"name": "test", "stage": 1, "observation_mode": observation_mode},
        "selfplay": {
            "policy_ratio": policy_ratio,
            "baseline_ratio": 1.0 - policy_ratio,
            "temperature": 1.0,
            "max_samples_per_shard": 10000,
        },
    }


def _make_model(encoder):
    """テスト用モデル"""
    return MLPPolicyValueModel(input_dim=encoder.output_dim, hidden_dims=[32])


@pytest.mark.slow
class TestSelfPlayWorker:
    """SelfPlayWorker テスト"""

    def test_generates_shard_files(self, tmp_path: Path):
        """self-play で shard ファイルが生成される"""
        config = _make_config(observation_mode="full", policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config,
            model=model,
            encoder=encoder,
            output_dir=tmp_path / "shards",
            worker_id="test_worker",
        )
        stats = worker.run(num_matches=2, seed_start=42)

        assert stats["num_matches"] == 2
        assert stats["total_steps"] > 0

        # shard ファイルが存在する
        shards = list((tmp_path / "shards").glob("shard_*.parquet"))
        assert len(shards) >= 1

    def test_shard_readable(self, tmp_path: Path):
        """生成された shard を読み込めて中身がある"""
        config = _make_config(observation_mode="full", policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config,
            model=model,
            encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=100)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) > 0

        # サンプルの基本フィールド検証
        s = samples[0]
        assert s.observation.dtype.name == "float32"
        assert s.legal_mask.shape == (34,)
        assert 0 <= s.action < 34
        assert s.experiment_id == "test"

    def test_stats_dict(self, tmp_path: Path):
        """統計 dict の内容確認"""
        config = _make_config(policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=1, seed_start=0)

        assert "num_matches" in stats
        assert "total_steps" in stats
        assert "total_rounds" in stats
        assert stats["total_rounds"] >= 1

    def test_policy_ratio_all_baseline(self, tmp_path: Path):
        """policy_ratio=0 で全席ベースラインのとき、サンプルは0件"""
        config = _make_config(policy_ratio=0.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) == 0


@pytest.mark.slow
class TestSampleTemporalAlignment:
    """サンプル時点整合テスト"""

    def test_observation_matches_action_decision_point(self, tmp_path: Path):
        """保存観測から再推論した action が保存 action と整合する"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        selector = ActionSelector(mode=SelectionMode.ARGMAX)

        # argmax 選択で deterministic にするため temperature=1e-10
        config = {
            "experiment": {"name": "test", "observation_mode": "full"},
            "selfplay": {
                "policy_ratio": 1.0,
                "temperature": 1e-10,
                "max_samples_per_shard": 10000,
            },
        }

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) > 0

        # 保存観測 + 保存 mask から再推論して action が一致するか確認
        matches = 0
        for s in samples[:20]:  # 先頭20サンプルで検証
            obs_t = torch.from_numpy(s.observation).unsqueeze(0)
            mask_t = torch.from_numpy(s.legal_mask).unsqueeze(0)

            with torch.no_grad():
                output = model(obs_t, mask_t)
            re_action, _ = selector.select(output.logits[0], mask_t[0])

            if re_action == s.action:
                matches += 1

        # 低 temperature での argmax 一致は高い率で期待できる
        assert matches >= len(samples[:20]) * 0.8

    def test_step_id_is_consecutive(self, tmp_path: Path):
        """step_id がサンプル順に連番"""
        config = _make_config(observation_mode="full", policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=0)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) > 0

        step_ids = [s.step_id for s in samples]
        expected = list(range(len(samples)))
        assert step_ids == expected

    def test_legal_mask_matches_observation(self, tmp_path: Path):
        """保存された legal_mask が observation 時点の合法手と整合"""
        config = _make_config(observation_mode="full", policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()

        for s in samples:
            # action は legal_mask で合法な位置に対応する
            assert s.legal_mask[s.action] > 0.5, (
                f"action {s.action} が legal_mask で非合法"
            )


@pytest.mark.slow
class TestBaselineTeacherData:
    """baseline 教師データ保存テスト (CQ-0042)"""

    def test_save_baseline_actions(self, tmp_path: Path):
        """save_baseline_actions=True で baseline サンプルが保存される"""
        config = _make_config(policy_ratio=0.5)
        config["selfplay"]["save_baseline_actions"] = True
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) > 0

        actor_types = {s.actor_type for s in samples}
        assert "baseline" in actor_types
        assert "policy" in actor_types

    def test_baseline_not_saved_by_default(self, tmp_path: Path):
        """デフォルトでは baseline サンプルは保存されない"""
        config = _make_config(policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        for s in samples:
            assert s.actor_type == "policy"

    def test_baseline_identifiable_in_shard(self, tmp_path: Path):
        """baseline サンプルを actor_type で識別できる"""
        config = _make_config(policy_ratio=0.0)  # 全席 baseline
        config["selfplay"]["save_baseline_actions"] = True
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        samples = reader.read_all()
        assert len(samples) > 0
        for s in samples:
            assert s.actor_type == "baseline"

    def test_actor_type_in_tensors(self, tmp_path: Path):
        """read_as_tensors でも actor_types が取れる"""
        config = _make_config(policy_ratio=0.5)
        config["selfplay"]["save_baseline_actions"] = True
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=1, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        tensors = reader.read_as_tensors()
        assert "actor_types" in tensors
        assert set(tensors["actor_types"]).issubset({"policy", "baseline"})


@pytest.mark.smoke
class TestRoundResultsAndStats:
    """round_results.jsonl / 局結果集計 smoke テスト (CQ-0107)"""

    def test_round_results_jsonl_generated(self, tmp_path: Path):
        """self-play 実行後に round_results.jsonl が生成される"""
        import json

        config = _make_config(observation_mode="full", policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        output_dir = tmp_path / "shards"
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=output_dir, worker_id="w0",
        )
        worker.run(num_matches=1, seed_start=42)

        jsonl_path = output_dir / "round_results.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) >= 1

        row = json.loads(lines[0])
        # 必須フィールド確認
        for key in ["event_type", "winner_players", "loser_player",
                     "is_policy_win", "is_policy_deal_in", "is_draw",
                     "round_id", "episode_id", "worker_id", "seed"]:
            assert key in row, f"round_results.jsonl に {key} がない"

        assert row["event_type"] in ("tsumo", "ron", "ryukyoku")
        assert isinstance(row["winner_players"], list)

    def test_stats_has_round_stat_keys(self, tmp_path: Path):
        """worker stats に局結果集計キーが含まれる"""
        config = _make_config(observation_mode="full", policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=1, seed_start=42)

        expected_keys = [
            "num_rounds", "tsumo_count", "ron_count", "ryukyoku_count",
            "policy_wins", "policy_deal_ins", "policy_draws",
            "policy_win_by_tsumo", "policy_win_by_ron",
        ]
        for key in expected_keys:
            assert key in stats, f"stats に {key} がない"
            assert isinstance(stats[key], int)

        # num_rounds は少なくとも 1 以上
        assert stats["num_rounds"] >= 1
        # 合計は整合する
        assert (stats["tsumo_count"] + stats["ron_count"]
                + stats["ryukyoku_count"]) == stats["num_rounds"]


@pytest.mark.smoke
class TestMultiRonStats:
    """multi-ron 集計ロジックテスト (CQ-0108)"""

    def test_policy_wins_counts_each_policy_winner(self, tmp_path: Path):
        """multi-ron で policy 席が複数勝者に含まれる場合、
        policy_wins は勝者人数分カウントされる"""
        config = _make_config(observation_mode="full", policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        # _round_results を直接設定して _compute_round_stats をテスト
        worker._round_results = [
            # ダブロン: policy 席 0, 2 が勝者
            {
                "event_type": "ron",
                "winner_players": [0, 2],
                "loser_player": 1,
                "is_policy_win": True,
                "is_policy_deal_in": False,
                "is_draw": False,
                "policy_winner_players": [0, 2],
                "round_id": 0,
                "episode_id": "ep_0",
                "worker_id": "w0",
                "seed": 0,
            },
            # シングルロン: policy 席 3 が勝者
            {
                "event_type": "ron",
                "winner_players": [3],
                "loser_player": 1,
                "is_policy_win": True,
                "is_policy_deal_in": False,
                "is_draw": False,
                "policy_winner_players": [3],
                "round_id": 1,
                "episode_id": "ep_0",
                "worker_id": "w0",
                "seed": 0,
            },
            # ツモ: policy 席 1
            {
                "event_type": "tsumo",
                "winner_players": [1],
                "loser_player": -1,
                "is_policy_win": True,
                "is_policy_deal_in": False,
                "is_draw": False,
                "policy_winner_players": [1],
                "round_id": 2,
                "episode_id": "ep_0",
                "worker_id": "w0",
                "seed": 0,
            },
        ]

        stats = worker._compute_round_stats()

        # ダブロンで 2 + シングルロンで 1 + ツモで 1 = 4
        assert stats["policy_wins"] == 4
        assert stats["policy_win_by_ron"] == 3  # ダブロン 2 + シングル 1
        assert stats["policy_win_by_tsumo"] == 1
        assert stats["ron_count"] == 2
        assert stats["tsumo_count"] == 1
        assert stats["num_rounds"] == 3

    def test_mixed_policy_baseline_multi_ron(self, tmp_path: Path):
        """multi-ron で policy/baseline 混合の場合、policy 席のみカウント"""
        config = _make_config(observation_mode="full", policy_ratio=0.5)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker._round_results = [
            # ダブロン: 席 0 (policy), 席 2 (baseline) が勝者
            {
                "event_type": "ron",
                "winner_players": [0, 2],
                "loser_player": 1,
                "is_policy_win": True,
                "is_policy_deal_in": False,
                "is_draw": False,
                "policy_winner_players": [0],  # 席 0 のみ policy
                "round_id": 0,
                "episode_id": "ep_0",
                "worker_id": "w0",
                "seed": 0,
            },
        ]

        stats = worker._compute_round_stats()

        # policy 勝者は 1 人のみ
        assert stats["policy_wins"] == 1
        assert stats["policy_win_by_ron"] == 1


@pytest.mark.slow
class TestSelfPlayDevice:
    """SelfPlayWorker デバイス切替テスト (CQ-0064)"""

    def test_cpu_device_works(self, tmp_path: Path):
        """CPU 明示指定で既存動作維持"""
        config = _make_config()
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
            inference_device=torch.device("cpu"),
        )
        stats = worker.run(num_matches=1, seed_start=42)
        assert stats["total_steps"] > 0
        assert stats["inference_device"] == "cpu"
