from collections import deque
from datetime import datetime, timezone

class DTNQueue:
    """Small in-memory store-and-forward queue for the UI demo."""
    def __init__(self):
        self._queue = deque()
    def store(self, payload, node):
        if not payload:
            raise ValueError("payload must not be empty")
        item = {"payload": str(payload), "node": str(node), "stored_at": datetime.now(timezone.utc).isoformat()}
        self._queue.append(item)
        return item
    def forward(self):
        return self._queue.popleft() if self._queue else None
    def __len__(self):
        return len(self._queue)
