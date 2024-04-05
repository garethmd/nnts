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
