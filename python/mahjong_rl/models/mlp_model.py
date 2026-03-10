"""MLPPolicyValueModel: MLP ベースの打牌ポリシー + バリューモデル"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .base import DiscardPolicyModel, PolicyValueOutput

_LARGE_NEGATIVE = -1e9


class MLPPolicyValueModel(DiscardPolicyModel):
    """MLP ベースのポリシー・バリューモデル

    入力: フラット特徴量 (batch, input_dim)
    出力: 34種打牌ロジット + バリューヘッド

    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 隠れ層の次元数リスト
        value_heads: バリューヘッドの名前リスト
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        value_heads: Sequence[str] = ("round_delta",),
        value_aux_dim: int = 0,
        policy_tower_config: dict | None = None,
        value_tower_config: dict | None = None,
    ):
        super().__init__()
        self._value_head_names = list(value_heads)
        self._value_aux_dim = value_aux_dim

        # 共有トランク
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        trunk_out_dim = in_dim

        # CQ-0157: policy tower (optional)
        policy_head_in = trunk_out_dim
        if policy_tower_config is not None and policy_tower_config.get("enabled", False):
            pt_hidden = policy_tower_config.get("hidden_dim", trunk_out_dim)
            self.policy_tower: nn.Module | None = nn.Sequential(
                nn.Linear(trunk_out_dim, pt_hidden),
                nn.ReLU(),
            )
            policy_head_in = pt_hidden
        else:
            self.policy_tower = None

        # ポリシーヘッド
        self.policy_head = nn.Linear(policy_head_in, 34)

        # CQ-0157: value tower (optional)
        value_tower_in = trunk_out_dim + value_aux_dim
        if value_tower_config is not None and value_tower_config.get("enabled", False):
            vt_hidden = value_tower_config.get("hidden_dim", trunk_out_dim)
            self.value_tower: nn.Module | None = nn.Sequential(
                nn.Linear(value_tower_in, vt_hidden),
                nn.ReLU(),
            )
            value_head_in = vt_hidden
        else:
            self.value_tower = None
            value_head_in = value_tower_in

        # バリューヘッド群 (CQ-0151: value_aux_dim > 0 なら trunk + aux を入力)
        self.value_heads_modules = nn.ModuleDict({
            name: nn.Linear(value_head_in, 1) for name in value_heads
        })

    def forward(self, features: Tensor, legal_mask: Tensor,
                value_aux_features: Tensor | None = None) -> PolicyValueOutput:
        h = self.trunk(features)

        # CQ-0157: policy tower (optional)
        h_policy = self.policy_tower(h) if self.policy_tower is not None else h
        logits = self.policy_head(h_policy)
        logits = logits + (1.0 - legal_mask) * _LARGE_NEGATIVE

        # バリューヘッド (CQ-0151: value_aux_features があれば concat)
        # CQ-0153: aux 未渡し時は zero pad で value head 入力次元を合わせる
        if self._value_aux_dim > 0:
            if value_aux_features is not None:
                h_value = torch.cat([h, value_aux_features], dim=-1)
            else:
                zeros = torch.zeros(
                    h.size(0), self._value_aux_dim, device=h.device, dtype=h.dtype)
                h_value = torch.cat([h, zeros], dim=-1)
        else:
            h_value = h
        # CQ-0157: value tower (optional)
        h_value = self.value_tower(h_value) if self.value_tower is not None else h_value
        values = {
            name: head(h_value) for name, head in self.value_heads_modules.items()
        }

        return PolicyValueOutput(logits=logits, values=values)

    @property
    def value_head_names(self) -> list[str]:
        return self._value_head_names
