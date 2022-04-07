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
        metrics = []
        for m in self.metric_collection:
            metrics.append(m.data)

        return torch.cat(metrics)

    @property
    def dim(self) -> int:
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
