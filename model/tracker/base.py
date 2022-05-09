from abc import ABCMeta, abstractmethod

class BaseTracker(metaclass=ABCMeta):
    """Base class for tracker."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _update(self, *args, **kwargs):
        """update tracking object every call."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device):
        """Define the device where the computation will be performed."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._update(*args, **kwargs)
