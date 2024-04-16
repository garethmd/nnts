import pandas as pd
import pytest
import torch

import nnts.torch.data.preprocessing as preprocessing


def test_masked_mean_abs_scaling_returns_mean_of_non_masked_values():
    # Arrange
    seq = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    eps = 1

    # Act
    result = preprocessing.masked_mean_abs_scaling(seq, mask, eps)

    # Assert
    assert torch.allclose(result, torch.tensor([[2], [4.5]]))


def test_masked_mean_abs_scaling_not_less_than_one():
    # Arrange
    seq = torch.tensor([[0, 0, 0], [0, 0, 6]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    eps = 1

    # Act
    result = preprocessing.masked_mean_abs_scaling(seq, mask, eps)

    # Assert
    assert torch.allclose(result, torch.tensor([[1.0], [1.0]]))


def test_masked_mean_abs_scaling_should_calculate_without_mask():
    # Arrange
    seq = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Act
    result = preprocessing.masked_mean_abs_scaling(seq)

    # Assert
    assert torch.allclose(result, torch.tensor([[2.0], [5.0]]))


def test_masked_mean_abs_scaling_should_calculate_correct_with_batches():
    # Arrange
    seq = torch.tensor(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]]
    )  # 1,2,3 4,5,6 7,8,9 10,11,12

    # Act
    result = preprocessing.masked_mean_abs_scaling(seq)

    # Assert
    print(result)
    assert torch.allclose(result, torch.tensor([[[2.0, 5.0]], [[8.0, 11.0]]]))


def test_masked_mean_abs_scaling_should_calculate_correct_with_batches_and_2d_mask():
    # Arrange
    seq = torch.tensor(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]]
    )  # 1,2,3 4,5,6 7,8 10,11
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    # Act
    result = preprocessing.masked_mean_abs_scaling(seq, mask)

    # Assert
    assert torch.allclose(result, torch.tensor([[[2.0, 5.0]], [[7.5, 10.5]]]))


def test_should_raise_error_if_mask_and_seq_different_shapes():
    # Arrange
    seq = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[[1, 1, 1], [1, 1, 0]]])

    # Act / Assert
    with pytest.raises(ValueError):
        preprocessing.masked_mean_abs_scaling(seq, mask)


def test_should_fit_standard_scalar():
    # Arrange
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 5, 6, 7],
            "ds": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "unique_id": ["T1", "T1", "T2", "T2"],
        }
    )
    scaler = preprocessing.StandardScaler()

    # Act
    result = scaler.fit(data)

    # Assert
    assert result.mean.equals(pd.Series({"a": 2.5, "b": 5.5}))
    assert result.std.equals(
        pd.Series({"a": 1.2909944487358056, "b": 1.2909944487358056})
    )


def test_should_transform_standard_scalar():
    # Arrange
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 5, 6, 7],
            "ds": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "unique_id": ["T1", "T1", "T2", "T2"],
        }
    )
    scaler = preprocessing.StandardScaler(
        mean=pd.Series({"a": 2.5, "b": 5.5}),
        std=pd.Series({"a": 1.0, "b": 1.0}),
    )

    # Act
    result = scaler.transform(data)

    # a -1.5 -0.5 0.5 1.5
    # b  -1.5 -0.5 0.5 1.5
    print(result)
    # Assert
    assert result.equals(
        pd.DataFrame(
            {
                "a": [-1.5, -0.5, 0.5, 1.5],
                "b": [-1.5, -0.5, 0.5, 1.5],
                "ds": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
                "unique_id": ["T1", "T1", "T2", "T2"],
            }
        )
    )


def test_should_inverse_transform_standard_scalar():
    # Arrange
    data = pd.DataFrame(
        {
            "a": [-1.5, -0.5, 0.5, 1.5],
            "b": [-1.5, -0.5, 0.5, 1.5],
            "ds": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "unique_id": ["T1", "T1", "T2", "T2"],
        }
    )
    scaler = preprocessing.StandardScaler(
        mean=pd.Series({"a": 2.5, "b": 5.5}),
        std=pd.Series({"a": 1.0, "b": 1.0}),
    )

    # Act
    result = scaler.inverse_transform(data)
    # Assert
    assert result.equals(
        pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 5.0, 6.0, 7.0],
                "ds": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
                "unique_id": ["T1", "T1", "T2", "T2"],
            }
        )
    )
