"""Alpine Meadow exception class."""


class AMException(Exception):

    def __init__(self, message):
        self._message = message
        super().__init__(message)

    @property
    def message(self):
        return self._message
