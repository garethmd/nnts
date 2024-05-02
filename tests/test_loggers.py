import json
from enum import Enum, auto

import numpy as np
import pytest

import nnts.events
import nnts.loggers


def test_should_log_info_on_printrun():
    # Arrange
    run = nnts.loggers.PrintRun(
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


def test_should_log_info_on_localfilerun():
    # Arrange
    run = nnts.loggers.LocalFileRun(
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


class FakeHandler(nnts.loggers.Handler):
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.data = {}

    def handle(self, data: dict) -> None:
        self.data = data


def test_should_finish_and_handle_summary():
    # Arrange
    run = nnts.loggers.LocalFileRun(
        project="fake_project",
        name="test",
        config={"config": 1},
        Handler=FakeHandler,
    )
    # Act
    data = {"data": 2}
    run.log(data)
    run.finish()
    # Assert
    assert list(run.handler.data.keys()) == ["config", "data", "run_time"]


class TestEpochEventMixin:
    def test_should_configure_events(self):
        # Arrange
        mixin = nnts.loggers.EpochEventMixin()
        evts = nnts.events.EventManager()
        # Act
        mixin.configure(evts)
        # Assert
        assert len(evts.listeners) == 3


@pytest.fixture
def test_data():
    return {"key1": "value1", "key2": ["value2", "value3"]}


@pytest.fixture
def handler(tmp_path):
    return nnts.loggers.JsonFileHandler(tmp_path, "test_file")


def test_should_handle_creates_file(handler, test_data):
    # Call handle method
    handler.handle(test_data)

    # Assert that the file is created in the expected location
    expected_file_path = handler.path / handler.filename
    assert expected_file_path.exists()


def test_should_handle_writes_correct_data(handler, test_data):
    handler.handle(test_data)
    expected_file_path = handler.path / handler.filename
    file_contents = expected_file_path.read_text()
    saved_data = json.loads(file_contents)

    # Assert that the saved data matches the test data
    assert saved_data == test_data


def test_should_convert_np_float_with_np_float():
    obj = np.float32(3.14)
    assert nnts.loggers.convert_np_float(obj) == float(obj)


class MockEnum(Enum):
    VALUE = auto()


def test_should_convert_np_float_with_enum():
    obj = MockEnum.VALUE
    assert nnts.loggers.convert_np_float(obj) == obj.value


def test_should_convert_np_float_with_invalid_type():
    with pytest.raises(TypeError):
        obj = "not a numpy float or enum"
        nnts.loggers.convert_np_float(obj)
