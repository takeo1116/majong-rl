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
    ):
        super().__init__()
        self._value_head_names = list(value_heads)

        # 共有トランク
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # ポリシーヘッド
        self.policy_head = nn.Linear(in_dim, 34)

        # バリューヘッド群
        self.value_heads_modules = nn.ModuleDict({
            name: nn.Linear(in_dim, 1) for name in value_heads
        })

    def forward(self, features: Tensor, legal_mask: Tensor) -> PolicyValueOutput:
        h = self.trunk(features)

        # ポリシーロジット + legal mask 適用
        logits = self.policy_head(h)
        logits = logits + (1.0 - legal_mask) * _LARGE_NEGATIVE

        # バリューヘッド
        values = {
            name: head(h) for name, head in self.value_heads_modules.items()
        }

        return PolicyValueOutput(logits=logits, values=values)

    @property
    def value_head_names(self) -> list[str]:
        return self._value_head_names
