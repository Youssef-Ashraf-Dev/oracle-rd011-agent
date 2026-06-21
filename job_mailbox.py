import threading

class JobMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self.done = False
        self.status = None          # "completed" | "waiting_approval" | "error"
        self.result = None          # final state values
        self.error_message = None

    def set_result(self, status, result, error=None):
        with self._lock:
            self.status = status
            self.result = result
            self.error_message = error
            self.done = True

    def get_and_clear(self):
        """Return a dict with job data if done, else None. Consume the result once."""
        with self._lock:
            if not self.done:
                return None
            data = {
                "status": self.status,
                "result": self.result,
                "error_message": self.error_message,
            }
            self.done = False   # ready for next job
            return data

# One global instance per app process
mailbox = JobMailbox()