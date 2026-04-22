import heapq
import itertools
from typing import Callable, Any

class EventHandle:
    __slots__ = ["canceled", "event"]
    def __init__(self) -> None:
        self.canceled = False
        self.event = None

    def cancel(self) -> None:
        self.canceled = True
        if self.event is not None:
            self.event.args = ()
            self.event.kwargs = {}
            self.event.callback = None
            self.event = None

class Event:
    __slots__ = ["time", "id", "handle", "callback", "args", "kwargs"]
    def __init__(self, time: float, eid: int, handle: EventHandle, callback: Callable, args: tuple, kwargs: dict) -> None:
        self.time, self.id, self.handle, self.callback, self.args, self.kwargs = time, eid, handle, callback, args, kwargs
    def __lt__(self, other: "Event") -> bool:
        if self.time != other.time: return self.time < other.time
        return self.id < other.id

class EventLoop:
    def __init__(self) -> None:
        self.now = 0.0
        self._queue = []
        self._counter = itertools.count()
        self.crashed = False

    def schedule(self, delay_ms: float, callback: Callable, *args: Any, **kwargs: Any) -> EventHandle:
        handle = EventHandle()
        ev = Event(self.now + delay_ms, next(self._counter), handle, callback, args, kwargs)
        handle.event = ev 
        heapq.heappush(self._queue, ev)
        return handle

    def run(self, max_time_ms: float) -> None:
        while self._queue and self._queue[0].time <= max_time_ms:
            if len(self._queue) > 10_000_000:
                print("   💥 [EventLoop] Queue depth exceeded 10M! Deadlock circuit breaker triggered!")
                self.crashed = True
                break

            ev = heapq.heappop(self._queue)
            if ev.handle.canceled:
                ev.handle.event = None 
                continue
                
            self.now = ev.time
            if ev.callback is not None:
                ev.callback(*ev.args, **ev.kwargs)
                
            ev.handle.event = None 

        self.now = max_time_ms