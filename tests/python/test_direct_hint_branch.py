"""CQ-0265: Stage2a direct hint branch テスト"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.models.stage2a_model import Stage2aModel, Stage2aOutput
from mahjong_rl.encoders.flat_encoder import FlatFeatureEncoder


# ========== helpers ==========

def _make_model_with_hints(input_dim=100, hint_ranges=None, semantic=False):
    sa = {"enabled": True, "policy_projection_dim": 8} if semantic else None
    return Stage2aModel(
        input_dim=input_dim,
        discard_hidden_dims=[32],
        optional_hidden_dims=[32],
        value_hidden_dims=[32],
        direct_hint_ranges=hint_ranges,
        semantic_aux_config=sa,
    )


# ========== model tests ==========

class TestDirectHintForwardDiscard:
    """direct hint source 付き model が forward_discard できる"""

    def test_forward_runs(self):
        """basic forward with direct hints"""
        hint_ranges = {
            "shanten_hint": (80, 114),      # 34 dim
            "discard_ukeire_hint": (114, 148),  # 34 dim
        }
        model = _make_model_with_hints(input_dim=148, hint_ranges=hint_ranges)
        x = torch.randn(2, 148)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)

    def test_hint_changes_discard_logits(self):
        """hint 部分を変えると discard logits が変わる"""
        hint_ranges = {
            "shanten_hint": (80, 114),
            "discard_ukeire_hint": (114, 148),
        }
        model = _make_model_with_hints(input_dim=148, hint_ranges=hint_ranges)
        model.eval()

        x1 = torch.randn(1, 148)
        x2 = x1.clone()
        # hint 部分だけ変更
        x2[0, 80:148] = torch.randn(68)
        m = torch.ones(1, 34)

        with torch.no_grad():
            out1 = model.forward_discard(x1, m)
            out2 = model.forward_discard(x2, m)
        assert not torch.allclose(out1.discard_logits, out2.discard_logits)

    def test_hint_does_not_affect_optional(self):
        """hint 部分を変えても optional scores は同じ"""
        hint_ranges = {
            "shanten_hint": (80, 114),
            "discard_ukeire_hint": (114, 148),
        }
        model = _make_model_with_hints(input_dim=148, hint_ranges=hint_ranges)
        model.eval()

        x1 = torch.randn(1, 148)
        x2 = x1.clone()
        x2[0, 80:148] = torch.randn(68)  # hint only

        cf = torch.zeros(1, 3, 6, dtype=torch.long)
        cm = torch.ones(1, 3)

        with torch.no_grad():
            o1 = model.forward_optional(x1, cf, cm)
            o2 = model.forward_optional(x2, cf, cm)
        assert torch.allclose(o1.optional_scores, o2.optional_scores)

    def test_hint_does_not_affect_semantic(self):
        """hint 部分を変えても semantic logits は同じ"""
        hint_ranges = {
            "shanten_hint": (80, 114),
            "discard_ukeire_hint": (114, 148),
        }
        model = _make_model_with_hints(
            input_dim=148, hint_ranges=hint_ranges, semantic=True)
        model.eval()

        x1 = torch.randn(1, 148)
        x2 = x1.clone()
        x2[0, 80:148] = torch.randn(68)
        m = torch.ones(1, 34)

        with torch.no_grad():
            o1 = model.forward_discard(x1, m)
            o2 = model.forward_discard(x2, m)
        # semantic terminal/yaku logits should be the same
        assert torch.allclose(
            o1.semantic["terminal_logits"], o2.semantic["terminal_logits"])
        assert torch.allclose(
            o1.semantic["yaku_logits"], o2.semantic["yaku_logits"])

    def test_legal_mask_still_works(self):
        """legal mask で非合法手が masked out される"""
        hint_ranges = {"shanten_hint": (80, 114)}
        model = _make_model_with_hints(input_dim=114, hint_ranges=hint_ranges)
        x = torch.randn(1, 114)
        m = torch.zeros(1, 34)
        m[0, 5] = 1.0  # only tile 5 legal
        out = model.forward_discard(x, m)
        # tile 5 should have highest logit
        assert out.discard_logits[0].argmax().item() == 5


class TestNoHintFallback:
    """hint_ranges=None のとき従来どおり"""

    def test_no_hints_works(self):
        model = _make_model_with_hints(input_dim=100, hint_ranges=None)
        x = torch.randn(2, 100)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)

    def test_empty_hints_works(self):
        model = _make_model_with_hints(input_dim=100, hint_ranges={})
        x = torch.randn(2, 100)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)


# ========== validation ==========

class TestStage2aHintValidation:
    """Stage2a で shanten_hint / discard_ukeire_hint が必須"""

    def test_shanten_hint_false_error(self):
        """shanten_hint=false で validation error"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path

        config = ExperimentConfig()
        config.experiment = {
            "name": "test", "stage": "stage2a",
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": False,
            "discard_ukeire_hint": True,
        }
        config.model = {}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("shanten_hint" in e for e in errors)

    def test_ukeire_hint_false_error(self):
        """discard_ukeire_hint=false で validation error"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path

        config = ExperimentConfig()
        config.experiment = {
            "name": "test", "stage": "stage2a",
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": True,
            "discard_ukeire_hint": False,
        }
        config.model = {}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("discard_ukeire_hint" in e for e in errors)

    def test_both_true_no_error(self):
        """両方 true なら OK"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path

        config = ExperimentConfig()
        config.experiment = {
            "name": "test", "stage": "stage2a",
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": True,
            "discard_ukeire_hint": True,
        }
        config.model = {}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert not any("shanten_hint" in e for e in errors)
        assert not any("discard_ukeire_hint" in e for e in errors)

    def test_stage1_not_affected(self):
        """Stage1 では hint false でも OK"""
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path

        config = ExperimentConfig()
        config.experiment = {
            "name": "test", "stage": 1,
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": False,
            "discard_ukeire_hint": False,
        }
        config.model = {"name": "MLPPolicyValueModel", "hidden_dims": [32]}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert not any("shanten_hint" in e for e in errors)


# ========== integration with real encoder ==========

class TestEncoderModelIntegration:
    """real encoder + Stage2aModel の integration"""

    def test_full_encoder_with_model(self):
        """full encoder (hint有効) → Stage2aModel が動く"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
            current_shanten_input=True,
        )
        meta = enc.metadata()
        input_dim = meta.output_shape[0]
        hint_ranges = {s: meta.feature_ranges[s]
                       for s in ("shanten_hint", "discard_ukeire_hint")
                       if s in meta.feature_ranges}
        model = Stage2aModel(
            input_dim=input_dim,
            discard_hidden_dims=[32],
            optional_hidden_dims=[32],
            direct_hint_ranges=hint_ranges,
        )
        x = torch.randn(2, input_dim)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)

    def test_full_encoder_model_semantic(self):
        """semantic_aux 有効で動く"""
        enc = FlatFeatureEncoder(
            observation_mode="full",
            shanten_hint=True,
            discard_ukeire_hint=True,
        )
        meta = enc.metadata()
        input_dim = meta.output_shape[0]
        hint_ranges = {s: meta.feature_ranges[s]
                       for s in ("shanten_hint", "discard_ukeire_hint")
                       if s in meta.feature_ranges}
        model = Stage2aModel(
            input_dim=input_dim,
            discard_hidden_dims=[32],
            optional_hidden_dims=[32],
            value_hidden_dims=[32],
            semantic_aux_config={"enabled": True, "policy_projection_dim": 8},
            direct_hint_ranges=hint_ranges,
        )
        x = torch.randn(2, input_dim)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)
        assert out.semantic is not None


# ========== CQ-0265 仕上げ: validation dict + context gate ==========

class TestValidationDictForm:
    """dict 形式 {enabled: False} でも validation error になる"""

    def test_shanten_hint_dict_false(self):
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path
        config = ExperimentConfig()
        config.experiment = {
            "name": "t", "stage": "stage2a",
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": {"enabled": False},
            "discard_ukeire_hint": True,
        }
        config.model = {}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("shanten_hint" in e for e in errors)

    def test_ukeire_hint_dict_false(self):
        from mahjong_rl.runner import Stage1Runner
        from mahjong_rl.experiment import ExperimentConfig
        from pathlib import Path
        config = ExperimentConfig()
        config.experiment = {
            "name": "t", "stage": "stage2a",
            "observation_mode": "full", "global_seed": 42,
            "phases": ["selfplay"],
        }
        config.feature_encoder = {
            "shanten_hint": True,
            "discard_ukeire_hint": {"enabled": False},
        }
        config.model = {}
        config.selfplay = {"num_matches": 1}
        config.training = {"algorithm": "ppo"}
        config.evaluation = {}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("discard_ukeire_hint" in e for e in errors)


class TestContextGate:
    """context gate が forward を壊さない & 効果が調整可能"""

    def test_forward_with_gate(self):
        """gate 付き forward が動く"""
        hint_ranges = {"shanten_hint": (80, 114)}
        model = _make_model_with_hints(input_dim=114, hint_ranges=hint_ranges)
        x = torch.randn(2, 114)
        m = torch.ones(2, 34)
        out = model.forward_discard(x, m)
        assert out.discard_logits.shape == (2, 34)
        # context gate が存在する
        assert hasattr(model, "_context_gate")

    def test_gate_zero_suppresses_hints(self):
        """gate bias を大きな負値にすると hint が効かなくなる"""
        hint_ranges = {"shanten_hint": (80, 114)}
        model = _make_model_with_hints(input_dim=114, hint_ranges=hint_ranges)
        model.eval()

        # gate bias を -100 に設定 → sigmoid(-100) ≈ 0
        with torch.no_grad():
            model._context_gate.bias.fill_(-100.0)
            model._context_gate.weight.fill_(0.0)

        x1 = torch.randn(1, 114)
        x2 = x1.clone()
        x2[0, 80:114] = torch.randn(34)  # hint だけ変更
        m = torch.ones(1, 34)
        with torch.no_grad():
            o1 = model.forward_discard(x1, m)
            o2 = model.forward_discard(x2, m)
        # gate ≈ 0 なので hint の変化が logits にほぼ影響しない
        assert torch.allclose(o1.discard_logits, o2.discard_logits, atol=1e-4)

    def test_gate_one_lets_hints_through(self):
        """gate bias を大きな正値にすると hint が効く"""
        hint_ranges = {"shanten_hint": (80, 114)}
        model = _make_model_with_hints(input_dim=114, hint_ranges=hint_ranges)
        model.eval()

        # gate bias を +100 → sigmoid(+100) ≈ 1
        with torch.no_grad():
            model._context_gate.bias.fill_(100.0)
            model._context_gate.weight.fill_(0.0)

        x1 = torch.randn(1, 114)
        x2 = x1.clone()
        x2[0, 80:114] = torch.randn(34)
        m = torch.ones(1, 34)
        with torch.no_grad():
            o1 = model.forward_discard(x1, m)
            o2 = model.forward_discard(x2, m)
        # gate ≈ 1 なので hint が logits に影響する
        assert not torch.allclose(o1.discard_logits, o2.discard_logits)
