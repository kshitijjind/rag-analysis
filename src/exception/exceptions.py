class BadRequest(Exception):
    def __init__(self, message="Bad request"):
        self.message = message
        super().__init__(self.message)


class TimeoutError(Exception):
    def __init__(self, message="Request timed out"):
        self.message = message
        super().__init__(self.message)