"""CQ-0031: DiscardPolicy Model テスト"""
import pytest
import torch
import numpy as np

pytestmark = pytest.mark.smoke

from mahjong_rl.models import (
    DiscardPolicyModel, PolicyValueOutput, MLPPolicyValueModel,
)


@pytest.fixture
def model():
    return MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32])


@pytest.fixture
def model_multi_heads():
    return MLPPolicyValueModel(
        input_dim=353,
        hidden_dims=[64, 32],
        value_heads=["round_delta", "match_total"],
    )


@pytest.fixture
def batch_data():
    """バッチ入力データ"""
    batch_size = 4
    features = torch.randn(batch_size, 353)
    # 各サンプルで異なる合法手マスク
    legal_mask = torch.zeros(batch_size, 34)
    for b in range(batch_size):
        # ランダムに5〜10個の合法手
        n_legal = torch.randint(5, 11, (1,)).item()
        indices = torch.randperm(34)[:n_legal]
        legal_mask[b, indices] = 1.0
    return features, legal_mask


class TestModelOutput:
    """出力形状テスト"""

    def test_output_shape(self, model, batch_data):
        features, legal_mask = batch_data
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)

    def test_output_is_policy_value(self, model, batch_data):
        features, legal_mask = batch_data
        output = model(features, legal_mask)
        assert isinstance(output, PolicyValueOutput)
        assert isinstance(output.logits, torch.Tensor)
        assert isinstance(output.values, dict)

    def test_value_head_shape(self, model, batch_data):
        features, legal_mask = batch_data
        output = model(features, legal_mask)
        assert "round_delta" in output.values
        assert output.values["round_delta"].shape == (4, 1)

    def test_single_sample(self, model):
        features = torch.randn(1, 353)
        legal_mask = torch.zeros(1, 34)
        legal_mask[0, [0, 5, 10, 20]] = 1.0
        output = model(features, legal_mask)
        assert output.logits.shape == (1, 34)


class TestLegalMask:
    """Legal mask 適用テスト"""

    def test_illegal_actions_masked(self, model):
        features = torch.randn(1, 353)
        legal_mask = torch.zeros(1, 34)
        legal_mask[0, [3, 7, 15]] = 1.0
        output = model(features, legal_mask)

        logits = output.logits[0]
        # 非合法手のロジットは非常に小さい
        for i in range(34):
            if legal_mask[0, i] == 0.0:
                assert logits[i].item() < -1e8

    def test_legal_actions_not_masked(self, model):
        features = torch.randn(1, 353)
        legal_mask = torch.zeros(1, 34)
        legal_mask[0, [3, 7, 15]] = 1.0
        output = model(features, legal_mask)

        logits = output.logits[0]
        # 合法手のロジットは有限値
        for i in [3, 7, 15]:
            assert logits[i].item() > -1e8

    def test_softmax_concentrates_on_legal(self, model):
        features = torch.randn(1, 353)
        legal_mask = torch.zeros(1, 34)
        legal_mask[0, [0, 5, 10]] = 1.0
        output = model(features, legal_mask)

        probs = torch.softmax(output.logits[0], dim=0)
        # 合法手の確率合計 ≈ 1.0
        legal_prob_sum = probs[[0, 5, 10]].sum().item()
        assert legal_prob_sum > 0.999


class TestMultipleValueHeads:
    """複数バリューヘッドテスト"""

    def test_multiple_heads_present(self, model_multi_heads, batch_data):
        features, legal_mask = batch_data
        output = model_multi_heads(features, legal_mask)
        assert "round_delta" in output.values
        assert "match_total" in output.values

    def test_multiple_heads_shape(self, model_multi_heads, batch_data):
        features, legal_mask = batch_data
        output = model_multi_heads(features, legal_mask)
        assert output.values["round_delta"].shape == (4, 1)
        assert output.values["match_total"].shape == (4, 1)

    def test_value_head_names(self, model_multi_heads):
        assert model_multi_heads.value_head_names == ["round_delta", "match_total"]


class TestGradient:
    """勾配テスト"""

    def test_policy_gradient_flows(self, model):
        features = torch.randn(1, 353, requires_grad=True)
        legal_mask = torch.zeros(1, 34)
        legal_mask[0, [0, 5, 10]] = 1.0
        output = model(features, legal_mask)

        # ポリシーロスの勾配が流れる
        log_probs = torch.log_softmax(output.logits, dim=-1)
        loss = -log_probs[0, 0]  # 牌種0を選んだときのログ確率
        loss.backward()

        assert features.grad is not None
        assert not torch.all(features.grad == 0)

    def test_value_gradient_flows(self, model):
        features = torch.randn(1, 353, requires_grad=True)
        legal_mask = torch.ones(1, 34)
        output = model(features, legal_mask)

        value_loss = output.values["round_delta"].mean()
        value_loss.backward()

        assert features.grad is not None
        assert not torch.all(features.grad == 0)

    def test_model_params_updated(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        features = torch.randn(2, 353)
        legal_mask = torch.ones(2, 34)

        output = model(features, legal_mask)
        loss = output.logits.sum() + output.values["round_delta"].sum()

        # パラメータの初期値を記録
        initial_params = {
            name: p.clone() for name, p in model.named_parameters()
        }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # パラメータが更新されている
        any_changed = False
        for name, p in model.named_parameters():
            if not torch.equal(p, initial_params[name]):
                any_changed = True
                break
        assert any_changed


class TestIsDiscardPolicyModel:
    """抽象クラスの継承テスト"""

    def test_isinstance(self, model):
        assert isinstance(model, DiscardPolicyModel)
        assert isinstance(model, torch.nn.Module)


class TestValueAuxFeatures:
    """CQ-0151: value head 専用補助特徴テスト"""

    def test_value_aux_off_shape_unchanged(self):
        """value_aux_dim=0 で従来通りの出力 shape"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32], value_aux_dim=0)
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)

    def test_value_aux_on_value_shape(self):
        """value_aux_dim=1 で value output shape は (batch, 1) のまま"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32], value_aux_dim=1)
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        value_aux = torch.randn(4, 1)
        output = model(features, legal_mask, value_aux_features=value_aux)
        assert output.values["round_delta"].shape == (4, 1)

    def test_value_aux_policy_unchanged(self):
        """value_aux_dim=1 でも policy logits shape は (batch, 34) のまま"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32], value_aux_dim=1)
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        value_aux = torch.randn(4, 1)
        output = model(features, legal_mask, value_aux_features=value_aux)
        assert output.logits.shape == (4, 34)


class TestValueAuxZeroPad:
    """CQ-0153: value_aux_dim>0 / value_aux_features=None の zero pad テスト"""

    def test_zero_pad_when_aux_none(self):
        """value_aux_dim=1, value_aux_features=None で forward が通り shape が正しい"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32], value_aux_dim=1)
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        # value_aux_features=None でも crash しない
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)

    def test_zero_pad_vs_actual_policy_same(self):
        """zero pad と actual features で logits が同一（policy 不変の確認）"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32], value_aux_dim=1)
        model.eval()
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        value_aux = torch.randn(4, 1)
        with torch.no_grad():
            out_none = model(features, legal_mask)
            out_aux = model(features, legal_mask, value_aux_features=value_aux)
        # policy logits は value_aux_features に依存しない
        assert torch.allclose(out_none.logits, out_aux.logits)


class TestTowerStructure:
    """CQ-0157: task-specific tower 構造テスト"""

    def test_baseline_no_tower(self):
        """tower off で従来出力 shape と一致"""
        model = MLPPolicyValueModel(input_dim=353, hidden_dims=[64, 32])
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)
        # tower 属性は None
        assert model.policy_tower is None
        assert model.value_tower is None

    def test_policy_tower_only(self):
        """policy tower on, value tower off"""
        model = MLPPolicyValueModel(
            input_dim=353, hidden_dims=[64, 32],
            policy_tower_config={"enabled": True, "hidden_dim": 16},
        )
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)
        assert model.policy_tower is not None
        assert model.value_tower is None

    def test_value_tower_only(self):
        """value tower on, policy tower off"""
        model = MLPPolicyValueModel(
            input_dim=353, hidden_dims=[64, 32],
            value_tower_config={"enabled": True, "hidden_dim": 16},
        )
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)
        assert model.policy_tower is None
        assert model.value_tower is not None

    def test_dual_towers(self):
        """both towers on"""
        model = MLPPolicyValueModel(
            input_dim=353, hidden_dims=[64, 32],
            policy_tower_config={"enabled": True, "hidden_dim": 16},
            value_tower_config={"enabled": True, "hidden_dim": 16},
        )
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        output = model(features, legal_mask)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)
        assert model.policy_tower is not None
        assert model.value_tower is not None

    def test_tower_with_current_shanten(self):
        """dual towers + value_aux_dim=1 で shape 正しい"""
        model = MLPPolicyValueModel(
            input_dim=353, hidden_dims=[64, 32],
            value_aux_dim=1,
            policy_tower_config={"enabled": True, "hidden_dim": 16},
            value_tower_config={"enabled": True, "hidden_dim": 16},
        )
        features = torch.randn(4, 353)
        legal_mask = torch.ones(4, 34)
        value_aux = torch.randn(4, 1)
        # aux あり
        output = model(features, legal_mask, value_aux_features=value_aux)
        assert output.logits.shape == (4, 34)
        assert output.values["round_delta"].shape == (4, 1)
        # aux なし (zero pad)
        output_none = model(features, legal_mask)
        assert output_none.logits.shape == (4, 34)
        assert output_none.values["round_delta"].shape == (4, 1)


class TestPolicyDirectHints:
    """CQ-0203: policy direct hints branch テスト"""

    def test_disabled_backward_compat(self):
        """enabled=false で既存と同形状・同動作"""
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[32])
        features = torch.randn(2, 100)
        mask = torch.ones(2, 34)
        out = model(features, mask)
        assert out.logits.shape == (2, 34)
        assert out.values["round_delta"].shape == (2, 1)

    def test_enabled_forward(self):
        """enabled=true で forward が通る"""
        # input_dim=100, shanten_hint at [80,114], ukeire at [114,148]
        # → total 148 but model sees 148 as input_dim
        # hint_ranges: 2 sources of 34 each = 68 extracted
        # trunk gets 148 - 68 = 80
        pdh_cfg = {
            "enabled": True,
            "sources": ["shanten_hint", "discard_ukeire_hint"],
            "local_hidden_dim": 8,
            "tile_embedding_dim": 4,
            "context_gate": {"enabled": True},
        }
        ranges = {
            "shanten_hint": (80, 114),
            "discard_ukeire_hint": (114, 148),
        }
        model = MLPPolicyValueModel(
            input_dim=148,
            hidden_dims=[32],
            policy_direct_hints_config=pdh_cfg,
            direct_hint_ranges=ranges,
        )
        features = torch.randn(2, 148)
        mask = torch.ones(2, 34)
        out = model(features, mask)
        assert out.logits.shape == (2, 34)
        assert out.values["round_delta"].shape == (2, 1)

    def test_hint_changes_policy_not_value(self):
        """direct hint を変えると policy は変化するが value は変化しない"""
        pdh_cfg = {
            "enabled": True,
            "sources": ["shanten_hint"],
            "local_hidden_dim": 8,
            "tile_embedding_dim": 4,
            "context_gate": {"enabled": False},
        }
        ranges = {"shanten_hint": (80, 114)}
        model = MLPPolicyValueModel(
            input_dim=114,
            hidden_dims=[32],
            policy_direct_hints_config=pdh_cfg,
            direct_hint_ranges=ranges,
        )
        model.eval()

        base = torch.randn(1, 114)
        mask = torch.ones(1, 34)

        # hint を変えた版
        modified = base.clone()
        modified[0, 80:114] = torch.randn(34)

        with torch.no_grad():
            out1 = model(base, mask)
            out2 = model(modified, mask)

        # value は同じ（trunk 入力の global 部分 [0:80] は同一）
        v1 = out1.values["round_delta"]
        v2 = out2.values["round_delta"]
        assert torch.allclose(v1, v2, atol=1e-6), "value は変化しないべき"

        # policy logits は異なる
        assert not torch.allclose(out1.logits, out2.logits, atol=1e-6), \
            "hint を変えたら policy は変化するべき"

    def test_context_gate_shapes(self):
        """context gate 有効時の shape 確認"""
        pdh_cfg = {
            "enabled": True,
            "sources": ["shanten_hint"],
            "local_hidden_dim": 8,
            "tile_embedding_dim": 4,
            "context_gate": {"enabled": True},
        }
        ranges = {"shanten_hint": (80, 114)}
        model = MLPPolicyValueModel(
            input_dim=114,
            hidden_dims=[32],
            policy_direct_hints_config=pdh_cfg,
            direct_hint_ranges=ranges,
        )
        assert model._context_gate is not None
        assert model._context_gate.in_features == 32
        assert model._context_gate.out_features == 34

    def test_split_features_correctness(self):
        """_split_features が global と hint を正しく分離する"""
        pdh_cfg = {
            "enabled": True,
            "sources": ["shanten_hint", "discard_ukeire_hint"],
            "local_hidden_dim": 8,
            "tile_embedding_dim": 4,
        }
        ranges = {
            "shanten_hint": (80, 114),
            "discard_ukeire_hint": (114, 148),
        }
        model = MLPPolicyValueModel(
            input_dim=148,
            hidden_dims=[32],
            policy_direct_hints_config=pdh_cfg,
            direct_hint_ranges=ranges,
        )
        features = torch.arange(148, dtype=torch.float32).unsqueeze(0)
        g, h = model._split_features(features)
        # global: 0..79 (80 dims)
        assert g.shape == (1, 80)
        assert g[0, 0].item() == 0.0
        assert g[0, 79].item() == 79.0
        # hint: [1, 34, 2]
        assert h.shape == (1, 34, 2)
        # h[:,:,0] = shanten_hint (80..113)
        assert h[0, 0, 0].item() == 80.0
        # h[:,:,1] = discard_ukeire_hint (114..147)
        assert h[0, 0, 1].item() == 114.0
