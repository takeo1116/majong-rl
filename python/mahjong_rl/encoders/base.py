"""FeatureEncoder 抽象基底クラス"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np

from mahjong_rl._mahjong_core import PartialObservation, FullObservation

Observation = Union[PartialObservation, FullObservation]


@dataclass
class EncoderMetadata:
    """エンコーダの出力メタデータ"""
    output_shape: tuple[int, ...]
    dtype: np.dtype
    observation_mode: str  # "full", "partial", "both"
    name: str
    description: str = ""


class FeatureEncoder(ABC):
    """特徴量エンコーダの抽象基底クラス

    Observation をモデル入力用の numpy 配列に変換する。
    Model は FeatureEncoder の出力に依存するが、Observation には直接依存しない。
    """

    @abstractmethod
    def encode(self, obs: Observation) -> np.ndarray:
        """Observation を特徴量ベクトル/テンソルに変換する"""
        ...

    @abstractmethod
    def metadata(self) -> EncoderMetadata:
        """出力形状・型・メタ情報を返す"""
        ...

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.metadata().output_shape

    @property
    def output_dim(self) -> int:
        """フラットな出力次元（MLP 向け）"""
        shape = self.output_shape
        result = 1
        for s in shape:
            result *= s
        return result
