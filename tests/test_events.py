import pytest

import nnts.events


class MockListener(nnts.events.Listener):
    def __init__(self):
        self.event_type = None
        self.event_data = None

    def notify(self, event_data):
        self.event_data = event_data


class MockEvent:
    pass


class MockAltEvent:
    pass


def test_should_add_listener():
    # Arrange
    event_manager = nnts.events.EventManager()
    listener = MockListener()
    event_type = MockEvent
    # Act
    event_manager.add_listener(event_type, listener)
    # Assert
    assert event_manager.listeners[event_type] == [listener]


def test_should_remove_listener():
    # Arrange
    event_manager = nnts.events.EventManager()
    listener = MockListener()
    event_type = MockEvent
    event_manager.add_listener(event_type, listener)
    assert event_manager.listeners[event_type] == [listener]

    # Act
    event_manager.remove_listener(event_type, listener)
    # Assert
    assert event_manager.listeners[event_type] == []


def test_should_notify_listener():
    # Arrange
    event_manager = nnts.events.EventManager()
    listener = MockListener()
    event_type = MockEvent
    event_manager.add_listener(event_type, listener)
    event = MockEvent()
    # Act
    event_manager.notify(event)
    # Assert
    assert listener.event_data == event


def test_should_not_notify_listener_given_different_events_type():
    # Arrange
    event_manager = nnts.events.EventManager()
    listener = MockListener()
    event_type = MockEvent
    event_manager.add_listener(event_type, listener)
    event = MockAltEvent()
    # Act
    event_manager.notify(event)
    # Assert
    assert listener.event_data is None
