from abc import ABC, abstractmethod


class Listener(ABC):
    @abstractmethod
    def notify(self, event):
        raise NotImplementedError


class EventManager:
    def __init__(self):
        self.listeners = {}

    def add_listener(self, event_type, listener):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)

    def remove_listener(self, event_type, listener):
        if event_type in self.listeners:
            self.listeners[event_type].remove(listener)

    def notify(self, event):
        event_type = event.__class__
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                listener.notify(event)
