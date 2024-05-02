import nnts.loggers


def test_should_log_info():
    # Arrange
    run = nnts.loggers.ProjectRun(
        nnts.loggers.PrintHandler,
        project="fake_project",
        name="test",
        config={"config": 1},
    )
    data = {"data": 2}
    # Act
    run.log(data)
    # Assert
    assert run.static_data == {
        "config": 1,
        "data": 2,
    }
