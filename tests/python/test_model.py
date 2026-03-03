"""CQ-0031: DiscardPolicy Model テスト"""
import pytest
import torch
import numpy as np

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
