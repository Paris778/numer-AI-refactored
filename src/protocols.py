from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl


class Transformer(Protocol):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame: ...


class Predictor(Protocol):
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FeaturePipeline:
    transformers: list[Transformer] = field(default_factory=list)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        transformed = df
        for transformer in self.transformers:
            transformed = transformer.transform(transformed)
        return transformed


class IdentityTransformer:
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
