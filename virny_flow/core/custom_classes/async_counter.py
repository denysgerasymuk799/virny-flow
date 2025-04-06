import asyncio


class AsyncCounter:
    def __init__(self):
        self._value = 0
        self._lock = asyncio.Lock()

    async def increment(self, n=1):
        async with self._lock:
            self._value += n

    async def get_value(self):
        async with self._lock:
            return self._value
