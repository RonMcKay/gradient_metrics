import torch


class GradientMetric(object):
    def __call__(self, grad: torch.Tensor) -> None:
        self._collect(grad)

    def _collect(self, grad: torch.Tensor) -> None:
        raise NotImplementedError

    @property
    def data(self) -> torch.Tensor:
        return self._get_metric().view(-1)

    def _get_metric(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class PNorm(GradientMetric):
    def __init__(self, p: int = 1, eps: float = 1e-16) -> None:
        super().__init__()

        if not 0 < p < float("inf"):
            raise ValueError(f"p has to be in (0, inf), got {p}")
        self.p = float(p)

        if not eps > 0:
            raise ValueError(f"eps has to be greater than zero, got {eps}")
        self.eps = eps

        self.metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self.metric_buffer.device != grad.device:
            self.metric_buffer = self.metric_buffer.to(grad.device)

        self.metric_buffer = torch.pow(
            self.metric_buffer.pow(self.p) + grad.view(-1).abs().pow(self.p).sum(),
            1.0 / self.p,
        )

    def _get_metric(self) -> torch.Tensor:
        return self.metric_buffer

    def reset(self) -> None:
        self.metric_buffer = torch.tensor(0.0)


class Min(GradientMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self.metric_buffer.device != grad.device:
            self.metric_buffer = self.metric_buffer.to(grad.device)
        self.metric_buffer = torch.min(self.metric_buffer, grad.min())

    def _get_metric(self) -> torch.Tensor:
        return self.metric_buffer

    def reset(self) -> None:
        self.metric_buffer = torch.tensor(float("inf"))


class Max(GradientMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self.metric_buffer.device != grad.device:
            self.metric_buffer = self.metric_buffer.to(grad.device)
        self.metric_buffer = torch.max(self.metric_buffer, grad.max())

    def _get_metric(self) -> torch.Tensor:
        return self.metric_buffer

    def reset(self) -> None:
        self.metric_buffer = torch.tensor(float("-inf"))


class Mean(GradientMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric_buffer: torch.Tensor
        self.count: int
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self.metric_buffer.device != grad.device:
            self.metric_buffer = self.metric_buffer.to(grad.device)

        old_mean = self.metric_buffer.detach().clone()
        self.count += grad.view(-1).shape[0]
        self.metric_buffer = (
            self.metric_buffer + torch.sum(grad.view(-1) - old_mean) / self.count
        )

    def _get_metric(self) -> torch.Tensor:
        return self.metric_buffer

    def reset(self) -> None:
        self.metric_buffer = torch.tensor(0.0)
        self.count = 0


class MeanStd(GradientMetric):
    """
    This uses Welford's online algorithm for mean and variance computation
    to reduce GPU memory usage.
    """

    def __init__(self, return_mean: bool = True, eps: float = 1e-16) -> None:
        super().__init__()

        if not eps > 0:
            raise ValueError(f"eps has to be greater than zero, got {eps}")
        self.eps = eps

        self.return_mean = return_mean

        self._mean: torch.Tensor
        self._m2: torch.Tensor
        self._count: int
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._m2.device != grad.device:
            self._m2 = self._m2.to(grad.device)
            self._mean = self._mean.to(grad.device)

        # do a batch update according to Welford's algorithm

        self._count += grad.view(-1).shape[0]
        # gradient computation graph of mean is still available through
        # mean in the following line so we can detach this one
        old_mean = self._mean.detach().clone()
        self._mean = self._mean + torch.sum((grad.view(-1) - old_mean) / self._count)
        self._m2 = self._m2 + torch.sum(
            grad.view(-1).pow(2)
            - grad.view(-1) * (old_mean + self._mean)
            + old_mean * self._mean
        )

    def _get_metric(self) -> torch.Tensor:
        if self._count > 1:
            return torch.cat(
                (
                    self._mean.view(-1)
                    if self.return_mean
                    else torch.empty((0,), device=self._mean.device),
                    torch.sqrt(self._m2 / (self._count - 1) + self.eps**2).view(-1),
                )
            )
        else:
            return torch.cat(
                (
                    self._mean.view(-1)
                    if self.return_mean
                    else torch.empty((0,), device=self._mean.device),
                    torch.tensor(self.eps, device=self._mean.device).view(-1),
                )
            )

    def reset(self) -> None:
        """Initializes/resets the buffer

        | The following values are set:
        | mean = 0
        | m2 = 0
        | count = 0
        """
        self._mean = torch.tensor(0.0)
        self._m2 = torch.tensor(0.0)
        self._count = 0
