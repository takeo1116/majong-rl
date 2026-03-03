"""DiscardPolicyModel 抽象基底クラス"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


class PolicyValueOutput(NamedTuple):
    """ポリシー + バリューの出力"""
    logits: Tensor        # (batch, 34) 打牌ロジット
    values: dict[str, Tensor]  # {"round_delta": (batch, 1), ...}


class DiscardPolicyModel(ABC, nn.Module):
    """打牌ポリシーモデルの抽象基底クラス

    入力: 特徴量テンソル + 合法手マスク
    出力: 34種打牌ロジット + バリューヘッド
    """

    @abstractmethod
    def forward(self, features: Tensor, legal_mask: Tensor) -> PolicyValueOutput:
        """
        Args:
            features: (batch, *input_shape) 特徴量テンソル
            legal_mask: (batch, 34) 合法手マスク (1=合法, 0=非合法)

        Returns:
            PolicyValueOutput: logits (batch, 34), values dict
        """
        ...

    @property
    @abstractmethod
    def value_head_names(self) -> list[str]:
        """バリューヘッドの名前リスト"""
        ...
