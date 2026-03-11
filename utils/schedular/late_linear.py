from utils.schedular.base import BaseScheduler


class LateLinearDecaySchedular(BaseScheduler):
    def __init__(
        self,
        coef_init: float,
        coef_min: float,
        start_epoch: int,
        total_epochs: int,
    ):
        super().__init__(coef_init)
        self.coef_min = coef_min
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs

    def step(self, epoch: int):
        self.epoch = epoch

        if epoch < self.start_epoch:
            self.coef = self.coef_init
            return

        progress = (epoch - self.start_epoch) / max(
            1, self.total_epochs - self.start_epoch
        )
        progress = min(progress, 1.0)

        self.coef = self.coef_init * (
                1 - progress * (1 - self.coef_min)
        )