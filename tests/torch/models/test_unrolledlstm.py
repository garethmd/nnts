import torch


def test_should_append_new_timestep():
    context_length = 15
    scaled_features = 2
    t = 0

    # X history
    X = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0, -2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                [3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
                [5.0, 4.0, 3.0, 2.0, 0.0, 0.0],
                [6.0, 5.0, 4.0, 3.0, 0.0, 0.0],
                [7.0, 6.0, 5.0, 4.0, 0.0, 0.0],
                [8.0, 7.0, 6.0, 5.0, 0.0, 0.0],
                [9.0, 8.0, 7.0, 6.0, 0.0, 0.0],
                [10.0, 9.0, 8.0, 7.0, 0.0, 0.0],
                [11.0, 10.0, 9.0, 8.0, 0.0, 0.0],
                [12.0, 11.0, 10.0, 9.0, 0.0, 0.0],
                [13.0, 12.0, 11.0, 10.0, 0.0, 0.0],
                [14.0, 13.0, 12.0, 11.0, 0.0, 0.0],
                [15.0, 14.0, 13.0, 12.0, 0.0, 0.0],
                [16.0, 15.0, 14.0, 13.0, 21.0, 22.0],
                [17.0, 16.0, 15.0, 14.0, 0.0, 0.0],
                [18.0, 17.0, 16.0, 15.0, 0.0, 0.0],
                [19.0, 18.0, 17.0, 16.0, 0.0, 0.0],
                [20.0, 19.0, 18.0, 17.0, 0.0, 0.0],
            ]
        ]
    )

    out = torch.tensor([[[16]]])

    new_timestep = torch.cat(
        [
            out[:, -1:, :],
            X[
                :,
                context_length + t - 1 : context_length + t,
                : -scaled_features - 1,
            ],
        ],
        dim=2,
    )

    assert new_timestep.allclose(torch.tensor([[[16.0, 15.0, 14.0, 13.0]]]))

    X[:, context_length + t : context_length + t + 1, :-scaled_features] = new_timestep
    assert X[:, context_length + t : context_length + t + 1, :].allclose(
        torch.tensor([[[16.0, 15.0, 14.0, 13.0, 21.0, 22.0]]])
    )
