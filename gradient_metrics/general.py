from typing import List, Sequence, Type, Union

from gradient_metrics.metrics import GradientMetric
import torch
import torch.nn as nn


class GradientMetricCollector(object):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        metrics: Union[Sequence[Type[GradientMetric]], Type[GradientMetric]],
    ) -> None:
        self.target_layers = (
            (target_layers,)
            if isinstance(target_layers, (nn.Module, torch.Tensor))
            else tuple(target_layers)
        )

        self.metrics = (
            tuple(metrics) if isinstance(metrics, (list, tuple)) else (metrics,)
        )

        self.metric_collection: List[GradientMetric] = []

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

            metrics.append(self.get_metrics())

        return torch.stack(metrics).to(loss.device)

    def get_metrics(self, keep_buffer: bool = False) -> torch.Tensor:
        metrics = []
        for m in self.metric_collection:
            metrics.append(m.data)
            if not keep_buffer:
                m.reset()

        return torch.cat(metrics)

    def reset(self) -> None:
        for m in self.metric_collection:
            m.reset()

    @property
    def dim(self) -> int:
        return self.get_metrics(keep_buffer=True).shape[0]

    def _register_metrics(self) -> None:
        for t in self.target_layers:
            if isinstance(t, torch.Tensor):
                for m in self.metrics:
                    current_metric = m()
                    t.register_hook(current_metric)  # type: ignore
                    self.metric_collection.append(current_metric)
            else:
                for m in self.metrics:
                    current_metric = m()
                    self.metric_collection.append(current_metric)
                    for param in t.parameters():
                        param.register_hook(current_metric)  # type: ignore
