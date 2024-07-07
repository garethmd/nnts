from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Transformation(ABC):
    @abstractmethod
    def transform(self, data: Any, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def fit(self, data, *args, **kwargs) -> Transformation:
        pass

    @abstractmethod
    def inverse_transform(self, data, *args, **kwargs) -> Any:
        pass
