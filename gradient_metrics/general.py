from typing import List, Sequence, Type, Union

from gradient_metrics.metrics import GradientMetric
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class GradientMetricCollector(object):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        metrics: Union[Sequence[Type[GradientMetric]], Type[GradientMetric]],
    ) -> None:
        """Helper class for computing gradients.

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            metrics (Type[GradientMetric] or sequence of Type[GradientMetric]):
                A list of metric types to use. Each parameter from ``target_layers``
                will register all of these metrics as backward hooks.

        Raises:
            ValueError: If the list of metrics is empty.
        """

        self.metric_collection: List[GradientMetric] = []
        self.metric_handles: List[RemovableHandle] = []

        self.target_layers = (
            (target_layers,)
            if isinstance(target_layers, (nn.Module, torch.Tensor))
            else tuple(target_layers)
        )

        self.metrics = (
            tuple(metrics) if isinstance(metrics, (list, tuple)) else (metrics,)
        )

        if len(self.metrics) == 0:
            raise ValueError("No metrics specified!")

        self._register_metrics()

    def __call__(self, loss: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """Computes gradient metrics per sample.

        Args:
            loss (torch.Tensor): A loss tensor to compute the gradients on. This should
                have a shape of ``(N,)`` with ``N`` being the number of samples.
            create_graph (bool, optional): If true, creates a backward graph when
                computing the gradients. This could be utilized for second order
                derivatives on the gradient metrics themselfs. Defaults to False.

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
            sample_loss.backward(retain_graph=True, create_graph=create_graph)

            metrics.append(self.data)
            self.reset()
            self.zero_grad()

        return torch.stack(metrics).to(loss.device)

    def __del__(self) -> None:
        for h in self.metric_handles:
            h.remove()

    def reset(self) -> None:
        """Resets all gradient metric instances to their default values."""
        for m in self.metric_collection:
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
        metrics = []
        for m in self.metric_collection:
            metrics.append(m.data)

        return torch.cat(metrics)

    @property
    def dim(self) -> int:
        """Number of gradient metrics per sample.

        This is useful if you want to build a meta model based on the retrieved
        gradient metrics and need to now the input shape per sample.

        Returns:
            int: The number of gradient metrics per sample.
        """
        return self.data.shape[0]

    def _register_metrics(self) -> None:
        for t in self.target_layers:
            if isinstance(t, torch.Tensor):
                for m in self.metrics:
                    current_metric = m()
                    self.metric_handles.append(t.register_hook(current_metric))
                    self.metric_collection.append(current_metric)
            else:
                for m in self.metrics:
                    current_metric = m()
                    self.metric_collection.append(current_metric)
                    for param in t.parameters():
                        self.metric_handles.append(param.register_hook(current_metric))
