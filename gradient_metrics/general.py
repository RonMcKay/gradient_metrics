from typing import List, Sequence, Union

from gradient_metrics.metrics import GradientMetric
import torch
import torch.nn as nn


class GradientMetricCollector(object):
    def __init__(
        self,
        metrics: Union[Sequence[GradientMetric], GradientMetric],
    ) -> None:
        """Helper class for computing gradients.

        Args:
            metrics (sequence of GradientMetric or GradientMetric):
                A list of gradient metrics.

        Raises:
            ValueError: If the list of metrics is empty.
        """

        self.metrics = (
            tuple(metrics) if isinstance(metrics, (list, tuple)) else (metrics,)
        )
        self.target_layers: List[Union[nn.Module, torch.Tensor]] = []

        # collect all parameters for zeroing gradients
        t_layers = set()
        for m in self.metrics:
            t_layers.update(m.target_layers)
        self.target_layers = list(t_layers)

        if len(self.metrics) == 0:
            raise ValueError("No metrics specified!")

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        """Computes gradient metrics per sample.

        Args:
            loss (torch.Tensor): A loss tensor to compute the gradients on. This should
                have a shape of ``(N,)`` with ``N`` being the number of samples.

        Raises:
            ValueError: If the loss does not require a gradient
            ValueError: If the loss does not have a shape of ``(N,)``

        Returns:
            torch.Tensor: Gradient metrics per sample with a shape of ``(N,dim)``.
        """
        if not loss.requires_grad:
            raise ValueError(
                "'loss' should require grad in order to extract gradient metrics."
            )
        if len(loss.shape) != 1:
            raise ValueError(f"'loss' should have shape [N,] but found {loss.shape}")

        self.reset()
        metrics = []

        for sample_loss in loss:
            sample_loss.backward(retain_graph=True)

            metrics.append(self.data)
            self.reset()
            self.zero_grad()

        return torch.stack(metrics).to(loss.device)

    def reset(self) -> None:
        """Resets all gradient metric instances to their default values."""
        for m in self.metrics:
            m.reset()

    def zero_grad(self) -> None:
        for t in self.target_layers:
            if isinstance(t, torch.Tensor):
                # This part is taken from `torch.nn.Module.zero_grad`
                if t.grad is not None:
                    if t.grad.grad_fn is not None:
                        t.grad.detach_()
                    else:
                        t.grad.requires_grad_(False)
                    t.grad.zero_()
            else:
                t.zero_grad()

    @property
    def data(self) -> torch.Tensor:
        """Holds the metric data.

        Returns:
            torch.Tensor:
                The metric values.
                All metrics are read out of the ``GradientMetric`` instances and
                concatenated. The output shape is ``(dim,)``.
        """
        metric_data = []
        for m in self.metrics:
            metric_data.append(m.data)

        return torch.cat(metric_data)

    @property
    def dim(self) -> int:
        """Number of gradient metrics per sample.

        This is useful if you want to build a meta model based on the retrieved
        gradient metrics and need to now the input shape per sample.

        Returns:
            int: The number of gradient metrics per sample.
        """
        return self.data.shape[0]
