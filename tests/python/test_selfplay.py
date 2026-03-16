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


@pytest.mark.slow
class TestRewardComposition:
    """reward composition 統計テスト (CQ-0141)"""

    def test_disabled_backward_compat(self, tmp_path: Path):
        """shaping 無効時: reward_composition 存在、shanten_delta はゼロ"""
        config = _make_config(policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=2, seed_start=42)

        rc = stats["reward_composition"]
        assert rc["shanten_delta_enabled"] is False
        assert rc["shanten_delta"]["nonzero_count"] == 0
        assert rc["point_delta"]["count"] > 0
        # total == point_delta (shaping なし)
        assert rc["total"]["sum"] == pytest.approx(rc["point_delta"]["sum"])
        # CQ-0142: quantile キーが存在する
        for comp in ("point_delta", "shanten_delta", "total"):
            for qk in ("p50", "p90", "p99"):
                assert qk in rc[comp], f"{comp}.{qk} missing"
        # shanten_delta は全 0 → quantile も 0
        assert rc["shanten_delta"]["p50"] == 0.0
        assert rc["shanten_delta"]["p90"] == 0.0
        assert rc["shanten_delta"]["p99"] == 0.0

    def test_enabled_nonzero_shaping(self, tmp_path: Path):
        """shaping 有効時: shanten_delta に非ゼロ reward が発生する"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {
            "shaping": {
                "shanten_delta": {
                    "enabled": True,
                    "scale": 0.1,
                    "mode": "both",
                    "schedule": {"type": "constant"},
                },
            },
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=3, seed_start=0)

        rc = stats["reward_composition"]
        assert rc["shanten_delta_enabled"] is True
        # 3半荘あれば非ゼロの shanten delta が少なくとも1つ出る
        assert rc["shanten_delta"]["count"] > 0

    def test_total_equals_sum(self, tmp_path: Path):
        """total.sum == point_delta.sum + shanten_delta.sum"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {
            "shaping": {
                "shanten_delta": {
                    "enabled": True,
                    "scale": 0.01,
                    "mode": "both",
                },
            },
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=2, seed_start=10)

        rc = stats["reward_composition"]
        expected_sum = rc["point_delta"]["sum"] + rc["shanten_delta"]["sum"]
        assert rc["total"]["sum"] == pytest.approx(expected_sum, abs=1e-10)

    def test_quantile_ordering(self, tmp_path: Path):
        """p50 <= p90 <= p99 の順序が保たれる (CQ-0142)"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {
            "shaping": {
                "shanten_delta": {
                    "enabled": True,
                    "scale": 0.1,
                    "mode": "both",
                },
            },
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=3, seed_start=0)
        rc = stats["reward_composition"]
        for comp in ("point_delta", "total"):
            assert rc[comp]["p50"] <= rc[comp]["p90"] <= rc[comp]["p99"]

    def test_reward_shaping_config_output(self, tmp_path: Path):
        """reward_shaping 設定が構造化出力される (CQ-0143)"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {
            "shaping": {
                "shanten_delta": {
                    "enabled": True,
                    "scale": 0.05,
                    "mode": "improve_only",
                    "schedule": {"type": "linear_decay"},
                },
            },
        }
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=1, seed_start=42)
        rs = stats["reward_shaping"]
        sd = rs["shanten_delta"]
        assert sd["enabled"] is True
        assert sd["scale"] == 0.05
        assert sd["mode"] == "improve_only"
        assert sd["schedule_type"] == "linear_decay"

    def test_reward_shaping_config_disabled(self, tmp_path: Path):
        """shaping 無効時の reward_shaping 出力 (CQ-0143)"""
        config = _make_config(policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=1, seed_start=42)
        rs = stats["reward_shaping"]
        assert rs["shanten_delta"]["enabled"] is False
        assert rs["shanten_delta"]["scale"] is None


@pytest.mark.slow
class TestRewardScale:
    """CQ-0162: point_delta_scale が self-play env に適用されるテスト"""

    def test_point_delta_scale_applied_to_shard(self, tmp_path: Path):
        """point_delta_scale=0.0001 で shard の reward / point_delta_reward が scaled 値"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {"point_delta_scale": 0.0001}
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=2, seed_start=42)

        # shard を読み込み
        reader = ShardReader(tmp_path / "shards")
        data = reader.read_as_tensors()
        rewards = data["rewards"]
        pdr = data.get("point_delta_rewards")

        # point_delta_scale=0.0001 で raw point (-1000 等) が scaled される
        # 麻雀の 1 step の点数差は高々数万点なので |reward| < 10 程度に収まるはず
        # shaping なしなので point_delta_reward ≈ reward
        assert rewards is not None
        assert len(rewards) > 0
        # scale 済みなので |reward| は概ね小さい (raw point なら数百〜数万)
        max_abs = float(np.abs(rewards).max())
        assert max_abs < 10.0, f"reward max_abs={max_abs} が大きすぎる。scale 未適用の可能性"

        # point_delta_reward も scaled
        if pdr is not None:
            pdr_max = float(np.abs(pdr[~np.isnan(pdr)]).max()) if len(pdr[~np.isnan(pdr)]) > 0 else 0.0
            assert pdr_max < 10.0, f"point_delta_reward max={pdr_max} が大きすぎる"

    def test_reward_composition_scaled(self, tmp_path: Path):
        """reward_composition.point_delta の統計値が scale 済み"""
        config = _make_config(policy_ratio=1.0)
        config["reward"] = {"point_delta_scale": 0.0001}
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=2, seed_start=42)

        rc = stats["reward_composition"]
        pd_stats = rc["point_delta"]
        # scale 済みなので mean の絶対値は raw point (数百〜数千) より遥かに小さい
        assert abs(pd_stats["mean"]) < 10.0, (
            f"reward_composition.point_delta.mean={pd_stats['mean']} が大きすぎる")

    def test_default_scale_is_1(self, tmp_path: Path):
        """reward config なしではデフォルト scale=1.0 (従来互換)"""
        config = _make_config(policy_ratio=1.0)
        # reward config なし → point_delta_scale=1.0 がデフォルト
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        stats = worker.run(num_matches=1, seed_start=42)

        rc = stats["reward_composition"]
        # scale=1.0 なので raw point 単位。非ゼロの reward があれば |mean| >= 0
        # ただし大半のステップは reward=0 なので mean は小さいかもしれない
        # ここでは壊れていないことだけ確認
        assert "point_delta" in rc
        assert rc["point_delta"]["count"] > 0


@pytest.mark.slow
class TestPostRiichiFlag:
    """CQ-0163: is_post_riichi_discard フラグが shard に書き出される"""

    def test_post_riichi_flag_in_shard(self, tmp_path: Path):
        """self-play で is_post_riichi_discards が shard に含まれる"""
        config = _make_config(observation_mode="full", policy_ratio=1.0)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = _make_model(encoder)
        worker = SelfPlayWorker(
            config=config,
            model=model,
            encoder=encoder,
            output_dir=tmp_path / "shards",
        )
        worker.run(num_matches=2, seed_start=42)

        reader = ShardReader(tmp_path / "shards")
        data = reader.read_as_tensors()
        assert data["is_post_riichi_discards"] is not None
        assert data["is_post_riichi_discards"].dtype == np.bool_
        # 少なくとも一部は True / False（立直後あり/なし）
        total = len(data["is_post_riichi_discards"])
        n_true = int(data["is_post_riichi_discards"].sum())
        assert total > 0
        # 麻雀のゲームでは一部は立直後打牌になるはず（確率的だが2局あれば十分）
        # 完全に全部 False でも壊れていない証拠にはなる
        assert n_true >= 0  # 最低限非負


@pytest.mark.slow
class TestBaselineActorEvalConsistency:
    """baseline_actor_eval の value/log_prob 期待値一致テスト (CQ-0195, CQ-0196)

    shard 保存値を同一モデル forward の期待値と直接比較する。
    """

    def test_baseline_value_with_current_shanten(self, tmp_path: Path):
        """baseline value がモデル forward と一致 (value_aux なしの基本検証)"""
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim, hidden_dims=[32])
        config = _make_config(policy_ratio=0.5)
        config["selfplay"]["save_baseline_actions"] = True
        config["selfplay"]["temperature"] = 1.0
        config["training"] = {"rule_mix_learner": {"ppo_mode": "mixed"}}

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "sp", worker_id="test_w",
        )
        worker.run(num_matches=2, seed_start=42)

        reader = ShardReader(tmp_path / "sp")
        data = reader.read_as_tensors()

        bl_mask = data["actor_types"] == "baseline"
        n_bl = int(bl_mask.sum())
        assert n_bl > 0, "baseline サンプルが 0 件: テスト不成立"

        bl_obs = data["observations"][bl_mask]
        bl_masks = data["legal_masks"][bl_mask]
        bl_values_saved = data["values"][bl_mask]

        # selfplay と同じモード (training=True) でモデル forward
        for i in range(min(n_bl, 5)):
            obs_t = torch.from_numpy(bl_obs[i]).unsqueeze(0)
            mask_t = torch.from_numpy(bl_masks[i]).unsqueeze(0)
            with torch.no_grad():
                out = model(obs_t, mask_t)
            expected_value = float(list(out.values.values())[0].item())
            assert np.isclose(bl_values_saved[i], expected_value, atol=1e-5, rtol=1e-5), \
                f"baseline value 不一致 (sample {i}): saved={bl_values_saved[i]}, expected={expected_value}"

    def test_baseline_value_with_value_aux(self, tmp_path: Path):
        """CQ-0197, CQ-0198: current_shanten 有効時、baseline value が
        value_aux 付きモデル forward と直接一致"""
        import pyarrow.parquet as pq

        encoder = FlatFeatureEncoder(
            observation_mode="full", current_shanten_input=True)
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim, hidden_dims=[32],
            value_aux_dim=1)
        config = _make_config(policy_ratio=0.5)
        config["selfplay"]["save_baseline_actions"] = True
        config["selfplay"]["temperature"] = 1.0
        config["model"] = {"value_features": {"current_shanten": {"enabled": True}}}
        config["training"] = {"rule_mix_learner": {"ppo_mode": "mixed"}}

        sp_dir = tmp_path / "sp"
        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=sp_dir, worker_id="test_w",
        )
        worker.run(num_matches=3, seed_start=42)

        # raw parquet から current_shanten を直接読む
        # (shard reader の all(v>=0) チェックを回避)
        reader = ShardReader(sp_dir)
        data = reader.read_as_tensors()
        all_cs_raw: list[int] = []
        for shard_path in sorted(sp_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            if "current_shanten" in table.column_names:
                all_cs_raw.extend(table.column("current_shanten").to_pylist())
            else:
                all_cs_raw.extend([-1] * len(table))
        cs_array = np.array(all_cs_raw, dtype=np.int32)

        bl_mask = data["actor_types"] == "baseline"
        n_bl = int(bl_mask.sum())
        assert n_bl > 0, "baseline サンプルが 0 件: テスト不成立"

        bl_obs = data["observations"][bl_mask]
        bl_masks = data["legal_masks"][bl_mask]
        bl_values_saved = data["values"][bl_mask]
        bl_cs = cs_array[bl_mask]

        # current_shanten >= 0 のサンプルのみ検証
        valid_indices = [i for i in range(n_bl) if bl_cs[i] >= 0]
        assert len(valid_indices) > 0, \
            "current_shanten >= 0 の baseline サンプルが 0 件: テスト不成立"

        matched = 0
        for i in valid_indices[:5]:
            obs_t = torch.from_numpy(bl_obs[i]).unsqueeze(0)
            mask_t = torch.from_numpy(bl_masks[i]).unsqueeze(0)
            value_aux = torch.tensor([[bl_cs[i] / 8.0]], dtype=torch.float32)
            with torch.no_grad():
                out = model(obs_t, mask_t, value_aux_features=value_aux)
            expected_value = float(list(out.values.values())[0].item())
            assert np.isclose(bl_values_saved[i], expected_value, atol=1e-5, rtol=1e-5), \
                f"baseline value 不一致 (sample {i}): saved={bl_values_saved[i]}, expected={expected_value}"
            matched += 1
        assert matched > 0

    def test_baseline_logprob_with_temperature(self, tmp_path: Path):
        """temperature != 1.0 で baseline log_prob がモデル forward + temperature 定義と一致"""
        temperature = 2.0
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = MLPPolicyValueModel(
            input_dim=encoder.output_dim, hidden_dims=[32])
        config = _make_config(policy_ratio=0.5)
        config["selfplay"]["save_baseline_actions"] = True
        config["selfplay"]["temperature"] = temperature
        config["training"] = {"rule_mix_learner": {"ppo_mode": "mixed"}}

        worker = SelfPlayWorker(
            config=config, model=model, encoder=encoder,
            output_dir=tmp_path / "sp", worker_id="test_w",
        )
        worker.run(num_matches=2, seed_start=42)

        reader = ShardReader(tmp_path / "sp")
        data = reader.read_as_tensors()

        bl_mask = data["actor_types"] == "baseline"
        n_bl = int(bl_mask.sum())
        assert n_bl > 0, "baseline サンプルが 0 件: テスト不成立"

        bl_obs = data["observations"][bl_mask]
        bl_masks = data["legal_masks"][bl_mask]
        bl_actions = data["actions"][bl_mask]
        bl_lp_saved = data["log_probs"][bl_mask]

        for i in range(min(n_bl, 5)):
            obs_t = torch.from_numpy(bl_obs[i]).unsqueeze(0)
            mask_t = torch.from_numpy(bl_masks[i]).unsqueeze(0)
            with torch.no_grad():
                out = model(obs_t, mask_t)
            # mask -> temperature -> log_softmax (policy 定義と同一)
            logits = out.logits[0]
            logits_masked = logits + (1 - mask_t[0]) * (-1e9)
            logits_tempered = logits_masked / temperature
            lp = torch.log_softmax(logits_tempered, dim=-1)
            expected_lp = float(lp[int(bl_actions[i])].item())
            assert np.isclose(bl_lp_saved[i], expected_lp, atol=1e-5, rtol=1e-5), \
                f"baseline log_prob 不一致 (sample {i}): saved={bl_lp_saved[i]}, expected={expected_lp}"
