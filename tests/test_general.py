from functools import partial
import gc
from pathlib import Path

import gradient_metrics
from gradient_metrics import GradientMetricCollector
from gradient_metrics.metrics import Max, MeanStd, Min
import pytest
import toml
import torch


def test_versions_are_in_sync():
    """
    Checks if the pyproject.toml and gradient_metrics.__init__.py __version__
    are in sync.
    """

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    pyproject_version = pyproject["tool"]["poetry"]["version"]

    package_init_version = gradient_metrics.__version__

    assert package_init_version == pyproject_version


def test_gradientmetriccollector_inputs():
    parameter = torch.ones((3,), requires_grad=True)
    eps = 1e-16

    with pytest.raises(ValueError):
        metric_collector = GradientMetricCollector(target_layers=parameter, metrics=[])

    metric_collector = GradientMetricCollector(
        target_layers=parameter, metrics=[Max, partial(MeanStd, eps=eps), Min]
    )

    # Raise ValueError if loss does not requires gradient
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1,), requires_grad=False))

    # Raise ValueError if loss does not have expected shape
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1, 1), requires_grad=True))

    # Check if backward hooks get removed on deletion
    # Garbage collector needs to be invoked explicitely
    del metric_collector
    gc.collect()
    assert len(parameter._backward_hooks) == 0


def test_gradientmetriccollector():
    parameter = torch.ones((3,), requires_grad=True)
    eps = 1e-16
    metric_collector = GradientMetricCollector(
        target_layers=parameter, metrics=[Max, partial(MeanStd, eps=eps), Min]
    )

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
    linear_layer.weight.data.fill_(torch.tensor(2.0))
    linear_layer.bias.data.zero_()
    metric_collector = GradientMetricCollector(linear_layer, [Max, Min])
    assert metric_collector.dim == 2

    loss = linear_layer(torch.ones((1, 3))).sum(1)
    metrics = metric_collector(loss)
    assert torch.all(metrics[0] == torch.tensor([1.0, 1.0]))
