import threading

class JobMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self.done = False
        self.status = None
        self.result = None
        self.error_message = None
        # Progress tracking
        self.progress = 0.0        # 0.0 – 1.0
        self.progress_label = ""

    def set_result(self, status, result, error=None):
        with self._lock:
            self.status = status
            self.result = result
            self.error_message = error
            self.done = True
            # Automatically set progress to 100% when finished
            self.progress = 1.0
            self.progress_label = "Completed"

    def set_progress(self, progress: float, label: str):
        """Called from the background thread to update the progress bar."""
        with self._lock:
            self.progress = max(0.0, min(1.0, progress))
            self.progress_label = label

    def get_and_clear(self):
        """Return the final result if done, otherwise None (consumed once)."""
        with self._lock:
            if not self.done:
                return None
            data = {
                "status": self.status,
                "result": self.result,
                "error_message": self.error_message,
            }
            self.done = False
            return data

    def get_progress(self):
        """Return current (progress, label) without clearing anything."""
        with self._lock:
            return self.progress, self.progress_label

# Global instance used by the app
mailbox = JobMailbox()