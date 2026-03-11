

class BaseScheduler:

    def __init__(self, coef_init: float):
        self.coef_init = coef_init
        self.coef = coef_init
        self.epoch = 0

    def step(self, epoch: int):
        """Update internal coef based on epoch"""
        self.epoch = epoch
        raise NotImplementedError

    def get(self) -> float:
        """Return current coef"""
        return self.coef
