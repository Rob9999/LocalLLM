from enum import Enum


class TrainingStatus(Enum):
    NONE = "None"
    STARTED = "Started"
    PENDING = "Pending"
    CANCELLATION_IS_REQUESTED = "CancellationIsRequested"
    STOPPED = "Stopped"
    FAILED = "Failed"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
