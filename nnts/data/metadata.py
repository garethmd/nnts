import json
import os

from pydantic import BaseModel, PositiveInt


class Metadata(BaseModel):
    """Class for storing dataset metadata"""

    filename: str
    dataset: str
    context_length: int
    prediction_length: int
    freq: str
    seasonality: int


def load(
    dataset: str,
    path: str = None,
) -> Metadata:
    # Get the directory of the current script
    with open(path) as f:
        data = json.load(f)
    return Metadata(**data[dataset])
