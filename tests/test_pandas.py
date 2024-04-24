import pandas as pd

import nnts.data
import nnts.pandas


def test_load_data_should_return_dataframe():
    # Arrange

    # Act
    data, *_ = nnts.pandas.read_tsf("tests/artifacts/hospital_dataset.tsf")

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert len(data.columns) > 0
    assert len(data.index) > 0
