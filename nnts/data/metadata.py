from pydantic import BaseModel, PositiveInt


class Metadata(BaseModel):
    """Class for storing dataset metadata"""

    path: str
    dataset: str
    context_length: int
    prediction_length: int
    freq: str
    seasonality: int
