"""CQ-0258: semantic_eval の unit / integration テスト"""
import pytest
import numpy as np
import torch
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.semantic_eval import (
    evaluate_semantic_heads,
    _compute_terminal_metrics,
    _compute_yaku_metrics,
    _compute_terminal_label_conditioned_confidence,
    _compute_yaku_label_conditioned_confidence,
    _compute_deal_in_risk,
    _deal_in_binary_summary,
    format_summary,
)
from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)


# ========== helpers ==========

def _write_labeled_shards(shard_dir, obs_dim=50, n_discard=30, n_call=20):
    """terminal_label + yaku_ids 付き shard"""
    writer = DecisionShardWriter(shard_dir, max_samples=10000)
    labels = ["win_menzen", "win_called", "draw_tenpai", "deal_in",
              "other_non_dealin"]
    for i in range(n_discard):
        mask = np.random.rand(34).astype(np.float32)
        mask[mask < 0.3] = 0
        mask[mask > 0] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        label = labels[i % len(labels)]
        yaku_ids = [3, 5] if label.startswith("win") else []
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=(i == n_discard - 1), round_over=False,
            action=int(np.argmax(mask)), legal_mask=mask,
            player_id=i % 4, episode_id="ep0", step_id=i,
            round_terminal_label=label,
            eventual_win_yaku_ids=yaku_ids,
            experiment_id="test", run_id="r0", worker_id="w0",
        ))
    for i in range(n_call):
        n_cands = np.random.randint(2, 5)
        cands = []
        for j in range(n_cands):
            if j == n_cands - 1:
                cands.append(CandidateRecord(action_type=8))
            else:
                cands.append(CandidateRecord(
                    action_type=np.random.choice([3, 4, 5]),
                    tile_type=np.random.randint(0, 34),
                    target_rel_seat=np.random.randint(1, 4),
                    consumed_tile_ids=(np.random.randint(0, 136),
                                       np.random.randint(0, 136)),
                ))
        label = labels[(n_discard + i) % len(labels)]
        yaku_ids = [0, 1] if label.startswith("win") else []
        writer.add(DecisionSample(
            decision_type="call",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=0.0, log_prob=-0.5, value=0.0,
            terminated=False, round_over=False,
            selected_candidate_index=0, candidate_count=n_cands,
            candidates=cands,
            player_id=1, episode_id="ep0", step_id=n_discard + i,
            round_terminal_label=label,
            eventual_win_yaku_ids=yaku_ids,
            response_context=np.array([0.1, 0.25, 1.0], dtype=np.float32),
            experiment_id="test", run_id="r0", worker_id="w0",
        ))
    writer.close()


def _make_semantic_model(obs_dim=50):
    return Stage2aModel(
        input_dim=obs_dim, discard_hidden_dims=[32],
        optional_hidden_dims=[32], value_hidden_dims=[32],
        semantic_aux_config={"enabled": True, "policy_projection_dim": 8},
    )


# ========== unit tests ==========

class TestTerminalMetrics:
    """terminal_head metrics 計算"""

    def test_perfect_prediction(self):
        targets = np.array([0, 1, 2, 3, 4])
        preds = np.array([0, 1, 2, 3, 4])
        result = _compute_terminal_metrics(targets, preds)
        assert result["accuracy"] == 1.0
        assert result["support"] == 5
        assert len(result["class_metrics"]) == NUM_TERMINAL_CLASSES
        assert len(result["class_names"]) == NUM_TERMINAL_CLASSES
        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (NUM_TERMINAL_CLASSES, NUM_TERMINAL_CLASSES)
        assert np.diag(cm).sum() == 5

    def test_all_wrong(self):
        targets = np.array([0, 0, 0])
        preds = np.array([1, 2, 3])
        result = _compute_terminal_metrics(targets, preds)
        assert result["accuracy"] == 0.0
        # class 0: recall=0 (3 targets, 0 correct)
        assert result["class_metrics"][0]["recall"] == 0.0
        assert result["class_metrics"][0]["support"] == 3

    def test_class_names_match_vocab(self):
        targets = np.array([0])
        preds = np.array([0])
        result = _compute_terminal_metrics(targets, preds)
        from mahjong_rl.outcome_vocab import TERMINAL_CLASSES
        assert result["class_names"] == list(TERMINAL_CLASSES)


class TestYakuMetrics:
    """yaku_head metrics 計算"""

    def test_perfect_winner_only(self):
        targets = np.zeros((2, NUM_YAKU))
        targets[0, 0] = 1.0  # MenzenTsumo
        targets[1, 3] = 1.0  # Tanyao
        targets[1, 5] = 1.0  # Pinfu
        preds = targets.copy()
        is_winner = np.array([1.0, 1.0])
        result = _compute_yaku_metrics(targets, preds, is_winner)
        assert result["micro_precision"] == 1.0
        assert result["micro_recall"] == 1.0
        assert result["micro_f1"] == 1.0
        assert result["exact_match_rate"] == 1.0
        assert result["num_winner"] == 2

    def test_non_winner_excluded(self):
        """non-winner は yaku 指標から除外される"""
        targets = np.zeros((3, NUM_YAKU))
        targets[0, 0] = 1.0
        targets[2, 3] = 1.0
        preds = np.zeros((3, NUM_YAKU))
        preds[0, 0] = 1.0
        preds[1, :5] = 1.0  # non-winner: should be ignored
        preds[2, 3] = 1.0
        is_winner = np.array([1.0, 0.0, 1.0])
        result = _compute_yaku_metrics(targets, preds, is_winner)
        assert result["num_winner"] == 2
        assert result["micro_precision"] == 1.0
        assert result["micro_recall"] == 1.0

    def test_no_winner(self):
        targets = np.zeros((5, NUM_YAKU))
        preds = np.zeros((5, NUM_YAKU))
        is_winner = np.zeros(5)
        result = _compute_yaku_metrics(targets, preds, is_winner)
        assert result["num_winner"] == 0
        assert result["micro_f1"] == 0.0

    def test_yaku_names_match_vocab(self):
        targets = np.zeros((1, NUM_YAKU))
        preds = np.zeros((1, NUM_YAKU))
        is_winner = np.array([1.0])
        result = _compute_yaku_metrics(targets, preds, is_winner)
        from mahjong_rl.outcome_vocab import YAKU_VOCAB
        assert result["yaku_names"] == [n for _, n in YAKU_VOCAB]


# ========== integration tests ==========

class TestEvaluateSemanticHeads:
    """evaluate_semantic_heads integration"""

    def test_full_eval(self, tmp_path):
        """semantic_aux enabled model + shard で結果が出る"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)

        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model,
            shard_dir=shard_dir,
            device=torch.device("cpu"),
            batch_size=16,
            label="test",
            checkpoint_path="test.pt",
        )

        assert result["label"] == "test"
        assert result["checkpoint"] == "test.pt"
        assert result["num_samples"] == 50  # 30 + 20
        assert result["num_winner_samples"] > 0

        # terminal
        t = result["terminal"]
        assert 0.0 <= t["accuracy"] <= 1.0
        assert len(t["class_metrics"]) == NUM_TERMINAL_CLASSES
        assert len(t["confusion_matrix"]) == NUM_TERMINAL_CLASSES
        assert t["support"] == 50

        # yaku
        y = result["yaku"]
        assert 0.0 <= y["micro_precision"] <= 1.0
        assert 0.0 <= y["micro_recall"] <= 1.0
        assert 0.0 <= y["micro_f1"] <= 1.0
        assert len(y["per_yaku_metrics"]) == NUM_YAKU
        assert y["num_winner"] > 0

    def test_json_serializable(self, tmp_path):
        """結果が JSON シリアライズ可能"""
        import json
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)

        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        # should not raise
        text = json.dumps(result, indent=2)
        parsed = json.loads(text)
        assert "terminal" in parsed
        assert "yaku" in parsed

    def test_format_summary(self, tmp_path):
        """format_summary が crash しない"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)

        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
            label="test_summary",
        )
        summary = format_summary(result)
        assert "terminal_head" in summary
        assert "yaku_head" in summary
        assert "test_summary" in summary


class TestSemanticAuxDisabledError:
    """semantic_aux disabled model で明確に fail する"""

    def test_disabled_raises(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32],
        )
        with pytest.raises(ValueError, match="semantic_aux が無効"):
            evaluate_semantic_heads(
                model=model, shard_dir=shard_dir,
                device=torch.device("cpu"),
            )


# ========== CQ-0262: label-conditioned confidence ==========

class TestTerminalLabelConditionedConfidence:
    """terminal label-conditioned confidence"""

    def test_perfect_confidence(self):
        """perfect prediction: mean_p=1, top1_hit_rate=1"""
        targets = np.array([0, 1, 2])
        probs = np.eye(NUM_TERMINAL_CLASSES)[:3]  # one-hot
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        assert len(result) == NUM_TERMINAL_CLASSES
        for c in range(3):
            assert result[c]["support"] > 0
            assert result[c]["mean_p"] == 1.0
            assert result[c]["top1_hit_rate"] == 1.0
            assert result[c]["mean_rank"] == 1.0

    def test_low_confidence(self):
        """model が uniform に近い → mean_p ~ 1/8"""
        targets = np.array([0, 0, 0])
        probs = np.ones((3, NUM_TERMINAL_CLASSES)) / NUM_TERMINAL_CLASSES
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        r0 = result[0]
        assert r0["support"] == 3
        assert abs(r0["mean_p"] - 1.0 / NUM_TERMINAL_CLASSES) < 0.01
        assert r0["top1_hit_rate"] <= 1.0  # may or may not be 0

    def test_wrong_top1_confusers(self):
        """actual=1 だが top-1 で 4 に負ける → confuser に class 4"""
        targets = np.array([1, 1, 1])
        probs = np.zeros((3, NUM_TERMINAL_CLASSES))
        probs[:, 4] = 0.5  # other_non_dealin が最高
        probs[:, 1] = 0.3  # win_called は 2 番目
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        r1 = result[1]
        assert r1["support"] == 3
        assert r1["top1_hit_rate"] == 0.0
        assert len(r1["top1_confusers"]) > 0
        assert r1["top1_confusers"][0]["class"] == "other_non_dealin"
        assert r1["top1_confusers"][0]["count"] == 3

    def test_percentiles(self):
        """p50/p90/p99 の値域"""
        targets = np.array([0] * 100)
        probs = np.zeros((100, NUM_TERMINAL_CLASSES))
        probs[:, 0] = np.linspace(0.1, 0.9, 100)
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        r0 = result[0]
        assert 0.0 < r0["p50"] < 1.0
        assert r0["p50"] < r0["p90"]
        assert r0["p90"] <= r0["p99"]

    def test_zero_support_class(self):
        """support=0 の class は mean_p=0"""
        targets = np.array([0, 0])
        probs = np.ones((2, NUM_TERMINAL_CLASSES)) / NUM_TERMINAL_CLASSES
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        # class 1 has support=0
        assert result[1]["support"] == 0
        assert result[1]["mean_p"] == 0.0

    def test_top3_hit(self):
        """actual=0 で p(0) が 3 位以内"""
        targets = np.array([0, 0])
        probs = np.zeros((2, NUM_TERMINAL_CLASSES))
        probs[0, 0] = 0.3   # 1st
        probs[0, 1] = 0.2
        probs[1, 0] = 0.05  # low, but others even lower
        probs[1, 2] = 0.4
        probs[1, 3] = 0.3
        probs[1, 4] = 0.2
        result = _compute_terminal_label_conditioned_confidence(targets, probs)
        r0 = result[0]
        # sample 0: rank=1, sample 1: rank>3 (4th or worse)
        assert 0.0 < r0["top3_hit_rate"] <= 1.0


class TestYakuLabelConditionedConfidence:
    """yaku positive-conditioned confidence"""

    def test_perfect_yaku_confidence(self):
        """perfect prediction: mean_p=1, threshold=1"""
        targets = np.zeros((3, NUM_YAKU))
        targets[0, 0] = 1.0  # MenzenTsumo
        targets[1, 3] = 1.0  # Tanyao
        targets[2, 0] = 1.0
        probs = np.zeros((3, NUM_YAKU))
        probs[0, 0] = 0.99
        probs[1, 3] = 0.99
        probs[2, 0] = 0.99
        is_winner = np.array([1.0, 1.0, 1.0])
        result = _compute_yaku_label_conditioned_confidence(targets, probs, is_winner)
        r0 = result[0]  # MenzenTsumo
        assert r0["support"] == 2
        assert r0["mean_p"] >= 0.99
        assert r0["threshold_hit_rate_0p5"] == 1.0

    def test_low_yaku_confidence(self):
        """model が見ていない yaku → mean_p ~ 0"""
        targets = np.zeros((3, NUM_YAKU))
        targets[0, 5] = 1.0  # Pinfu
        targets[1, 5] = 1.0
        probs = np.full((3, NUM_YAKU), 0.01)
        is_winner = np.array([1.0, 1.0, 0.0])
        result = _compute_yaku_label_conditioned_confidence(targets, probs, is_winner)
        r5 = result[5]  # Pinfu
        assert r5["support"] == 2
        assert r5["mean_p"] < 0.1
        assert r5["threshold_hit_rate_0p5"] == 0.0

    def test_non_winner_excluded_from_yaku_confidence(self):
        """non-winner は positive-conditioned から除外"""
        targets = np.zeros((3, NUM_YAKU))
        targets[0, 0] = 1.0
        targets[1, 0] = 1.0  # non-winner: should be excluded
        targets[2, 0] = 1.0
        probs = np.full((3, NUM_YAKU), 0.8)
        is_winner = np.array([1.0, 0.0, 1.0])
        result = _compute_yaku_label_conditioned_confidence(targets, probs, is_winner)
        r0 = result[0]
        assert r0["support"] == 2  # not 3

    def test_zero_support_yaku(self):
        """support=0 の yaku"""
        targets = np.zeros((2, NUM_YAKU))
        probs = np.zeros((2, NUM_YAKU))
        is_winner = np.array([1.0, 1.0])
        result = _compute_yaku_label_conditioned_confidence(targets, probs, is_winner)
        for r in result:
            assert r["support"] == 0
            assert r["mean_p"] == 0.0

    def test_threshold_0p2(self):
        """threshold@0.2 と @0.5 が分かれるケース"""
        targets = np.zeros((4, NUM_YAKU))
        targets[:, 3] = 1.0  # Tanyao
        probs = np.zeros((4, NUM_YAKU))
        probs[0, 3] = 0.6   # > 0.5
        probs[1, 3] = 0.3   # > 0.2 but < 0.5
        probs[2, 3] = 0.1   # < 0.2
        probs[3, 3] = 0.25  # > 0.2 but < 0.5
        is_winner = np.array([1.0, 1.0, 1.0, 1.0])
        result = _compute_yaku_label_conditioned_confidence(targets, probs, is_winner)
        r3 = result[3]  # Tanyao
        assert r3["threshold_hit_rate_0p5"] == 0.25  # 1/4
        assert r3["threshold_hit_rate_0p2"] == 0.75  # 3/4


class TestIntegrationLabelConditioned:
    """integration: JSON に新 field が含まれ、summary が出る"""

    def test_full_eval_has_confidence(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)

        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        # terminal
        t = result["terminal"]
        assert "label_conditioned_confidence" in t
        assert len(t["label_conditioned_confidence"]) == NUM_TERMINAL_CLASSES
        for lc in t["label_conditioned_confidence"]:
            assert "mean_p" in lc
            assert "top1_hit_rate" in lc
            assert "top1_confusers" in lc

        # yaku
        y = result["yaku"]
        assert "label_conditioned_confidence" in y
        assert len(y["label_conditioned_confidence"]) == NUM_YAKU
        for yc in y["label_conditioned_confidence"]:
            assert "mean_p" in yc
            assert "threshold_hit_rate_0p5" in yc

    def test_json_serializable_with_confidence(self, tmp_path):
        import json
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)
        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        text = json.dumps(result, indent=2)
        parsed = json.loads(text)
        assert "label_conditioned_confidence" in parsed["terminal"]
        assert "label_conditioned_confidence" in parsed["yaku"]

    def test_summary_includes_confidence(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)
        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        summary = format_summary(result)
        assert "terminal confidence" in summary
        assert "yaku confidence" in summary
        assert "mean_p" in summary


# ========== CQ-0269: deal_in risk ==========

class TestDealInBinarySummary:
    """deal_in binary risk summary"""

    def test_basic_summary(self):
        """positive/negative mean_p が出る"""
        from mahjong_rl.outcome_vocab import TERMINAL_CLASS_MAP
        deal_in_idx = TERMINAL_CLASS_MAP["deal_in"]
        n = 10
        targets = np.array([deal_in_idx] * 3 + [0] * 7)
        p = np.zeros(n)
        p[:3] = 0.8  # positive → high
        p[3:] = 0.1  # negative → low
        is_pos = targets == deal_in_idx
        result = _deal_in_binary_summary(p, is_pos, "test")
        assert result["support_pos"] == 3
        assert result["support_neg"] == 7
        assert result["mean_p_pos"] > result["mean_p_neg"]

    def test_roc_auc_present(self):
        """roc_auc / pr_auc が出る"""
        targets = np.array([3, 3, 0, 0, 1])  # deal_in=3
        p = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
        is_pos = targets == 3
        result = _deal_in_binary_summary(p, is_pos, "test")
        assert result["roc_auc"] is not None
        assert result["pr_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["pr_auc"] <= 1.0

    def test_no_positive_safe(self):
        """positive=0 でも crash しない"""
        is_pos = np.array([False, False, False])
        p = np.array([0.1, 0.2, 0.3])
        result = _deal_in_binary_summary(p, is_pos, "test")
        assert result["support_pos"] == 0
        assert result["mean_p_pos"] is None
        assert result["roc_auc"] is None

    def test_no_negative_safe(self):
        """negative=0 でも crash しない"""
        is_pos = np.array([True, True])
        p = np.array([0.8, 0.9])
        result = _deal_in_binary_summary(p, is_pos, "test")
        assert result["support_neg"] == 0
        assert result["roc_auc"] is None


class TestDealInRiskFull:
    """_compute_deal_in_risk integration"""

    def test_overall_present(self):
        """overall が出る"""
        from mahjong_rl.outcome_vocab import TERMINAL_CLASS_MAP
        deal_in_idx = TERMINAL_CLASS_MAP["deal_in"]
        n = 20
        targets = np.random.choice(5, n)
        probs = np.random.rand(n, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        result = _compute_deal_in_risk(targets, probs, None, None)
        assert "overall" in result
        assert result["overall"]["support_pos"] >= 0

    def test_subsets_with_features(self):
        """subset diagnostics が features ありで出る"""
        from mahjong_rl.outcome_vocab import TERMINAL_CLASS_MAP
        n = 100
        targets = np.random.choice(5, n)
        probs = np.random.rand(n, 5)
        probs = probs / probs.sum(axis=1, keepdims=True)
        tenpai = np.random.choice([0.0, 1.0], n)
        remaining = np.random.rand(n)
        result = _compute_deal_in_risk(targets, probs, tenpai, remaining)
        assert "subsets" in result
        assert "late_and_noten" in result["subsets"]
        assert "early_and_tenpai" in result["subsets"]

    def test_subsets_without_features(self):
        """features なしでも crash しない"""
        n = 10
        targets = np.random.choice(5, n)
        probs = np.random.rand(n, 5)
        result = _compute_deal_in_risk(targets, probs, None, None)
        assert "overall" in result
        assert result["subsets"] == {}


class TestDealInIntegration:
    """integration: deal_in_risk が JSON / summary に出る"""

    def test_deal_in_risk_in_result(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)
        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        assert "deal_in_risk" in result
        dr = result["deal_in_risk"]
        assert "overall" in dr

    def test_deal_in_risk_json_serializable(self, tmp_path):
        import json
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)
        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        text = json.dumps(result, indent=2)
        parsed = json.loads(text)
        assert "deal_in_risk" in parsed

    def test_summary_has_deal_in_risk(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_labeled_shards(shard_dir, obs_dim=obs_dim)
        model = _make_semantic_model(obs_dim)
        result = evaluate_semantic_heads(
            model=model, shard_dir=shard_dir,
            device=torch.device("cpu"), batch_size=16,
        )
        summary = format_summary(result)
        assert "deal_in risk" in summary
