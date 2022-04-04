from functools import partial

from gradient_metrics import GradientMetricCollector, __version__
from gradient_metrics.metrics import Max, MeanStd, Min
import pytest
import torch


def test_version():
    assert __version__ == "0.1.0"


def test_gradientmetriccollector():
    parameter = torch.ones((3,), requires_grad=True)
    eps = 1e-16
    metric_collector = GradientMetricCollector(
        target_layers=parameter, metrics=[Max, partial(MeanStd, eps=eps), Min]
    )

    # Raise ValueError if loss does not requires gradient
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1,), requires_grad=False))

    # Raise ValueError if loss does not have expected shape
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1, 1), requires_grad=True))

    loss = (parameter * torch.full(((2, 3)), 1.0)).sum(1)

    metrics = metric_collector(loss)

    assert torch.all(metrics[0] == torch.tensor([1.0, 1.0, eps, 1.0]))
    assert metric_collector.dim == 4

    metric_collector = GradientMetricCollector(
        target_layers=parameter, metrics=partial(MeanStd, eps=eps)
    )

    samples = torch.tensor([-1.0, 0.0, 1.0]).view(1, -1).repeat((2, 1))

    loss = (parameter * samples).sum(1)
    metrics = metric_collector(loss)

    assert torch.all(metrics[0] == torch.tensor([0.0, 1.0]))
    assert metric_collector.dim == 2

    linear_layer = torch.nn.Linear(3, 3, bias=True)
    metric_collector = GradientMetricCollector(linear_layer, Max)
