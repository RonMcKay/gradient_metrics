from collections import defaultdict
from typing import Dict, List, Union

import torch

from gradient_metrics.metrics import GradientMetric


class GradientMetricCollector(object):
    def __init__(
        self,
        metrics: Union[List[GradientMetric], GradientMetric],
    ) -> None:
        """Helper class for computing gradients.

        Args:
            metrics (sequence of GradientMetric or GradientMetric):
                A list of gradient metrics.

        Raises:
            ValueError: If the list of metrics is empty.
        """

        if isinstance(metrics, list):
            for m in metrics:
                if not isinstance(m, GradientMetric):
                    raise ValueError(
                        "'metrics' needs to be a list of 'GradientMetric' instances. "
                        f"Found {m.__class__.__name__}"
                    )
            if len(metrics) == 0:
                raise ValueError("List of metrics is empty!")
        elif not isinstance(metrics, GradientMetric):
            raise ValueError(
                "'metrics' needs to be either a 'GradientMetric' instance or "
                "a list of 'GradientMetric' instances. "
                f"Found '{type(metrics)}'"
            )

        self._metrics = [metrics] if isinstance(metrics, GradientMetric) else metrics

        # Build mapping from parameters to gradient metrics to be able to map gradients
        # to the correct metric.
        # Also build a list of all parameters to compute the gradients on
        _params_set = set()
        self._param_metrics_map: Dict[
            Union[torch.nn.parameter.Parameter, torch.Tensor], List[GradientMetric]
        ] = defaultdict(list)
        for metric in self._metrics:
            _params_set.update(metric.target_parameters)
            for param in metric.target_parameters:
                self._param_metrics_map[param].append(metric)

        self._params: List[torch.Tensor] = list(_params_set)

    def __call__(self, loss: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        """Computes gradient metrics per sample.

        Args:
            loss (torch.Tensor): A loss tensor to compute the gradients on. This should
                have a shape of ``(N,)`` with ``N`` being the number of samples.
            retain_graph (bool): If True, retains the graph of the supplied loss.
                Default False.

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
        gradient_metrics = []

        for i, sample_loss in enumerate(loss):
            gradients = torch.autograd.grad(
                sample_loss,
                self._params,
                retain_graph=retain_graph if i == (len(loss) - 1) else True,
            )

            for param, grad in zip(self._params, gradients):
                for metric in self._param_metrics_map[param]:
                    metric(grad)

            gradient_metrics.append(self.data)
            self.reset()

        return torch.stack(gradient_metrics)

    def reset(self) -> None:
        """Resets all gradient metric instances to their default values."""
        for m in self._metrics:
            m.reset()

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
        for m in self._metrics:
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
