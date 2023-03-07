from pathlib import Path

import pytest
import toml
import torch

import gradient_metrics
from gradient_metrics import GradientMetricCollector
from gradient_metrics.metrics import Max, MeanStd, Min


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
        GradientMetricCollector(metrics=[])

    with pytest.raises(ValueError):
        GradientMetricCollector(metrics=(Max(parameter),))

    with pytest.raises(ValueError):
        GradientMetricCollector(metrics=[Max(parameter), "test"])

    metric_collector = GradientMetricCollector(
        metrics=[Max(parameter), MeanStd(parameter, eps=eps), Min(parameter)]
    )

    # Raise ValueError if loss does not requires gradient
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1,), requires_grad=False))

    # Raise ValueError if loss does not have expected shape
    with pytest.raises(ValueError):
        metric_collector(torch.ones((1, 1), requires_grad=True))


def test_gradientmetriccollector():
    parameter = torch.ones((3,), requires_grad=True)
    eps = 1e-16
    metric_collector = GradientMetricCollector(
        metrics=[Max(parameter), MeanStd(parameter, eps=eps), Min(parameter)]
    )

    # _params should only contain the parameter defined above once
    assert len(metric_collector._params) == 1

    # parameter should have three metrics assigned
    assert len(metric_collector._param_metrics_map[parameter]) == 3

    loss = (parameter * torch.full(((2, 3)), 1.0)).sum(1)

    metrics = metric_collector(loss, retain_graph=True)

    # We should be able to compute the gradient metrics again if we retain the graph
    metrics = metric_collector(loss)

    assert torch.all(metrics[0] == torch.tensor([1.0, 1.0, eps, 1.0]))
    assert metric_collector.dim == 4

    metric_collector = GradientMetricCollector(metrics=MeanStd(parameter, eps=eps))

    samples = torch.tensor([-1.0, 0.0, 1.0]).view(1, -1).repeat((2, 1))

    loss = (parameter * samples).sum(1)
    metrics = metric_collector(loss)

    assert torch.all(metrics[0] == torch.tensor([0.0, 1.0]))
    assert metric_collector.dim == 2

    linear_layer = torch.nn.Linear(3, 3, bias=True)
    linear_layer.weight.data.fill_(torch.tensor(2.0))
    linear_layer.bias.data.zero_()
    metric_collector = GradientMetricCollector([Max(linear_layer), Min(linear_layer)])
    assert metric_collector.dim == 2

    loss = linear_layer(torch.ones((1, 3))).sum(1)
    metrics = metric_collector(loss)
    assert torch.all(metrics[0] == torch.tensor([1.0, 1.0]))

    # Test if the returned gradient metrics are on the same device as loss
    parameter = torch.ones((3,), requires_grad=True, device="meta")
    metric_collector = GradientMetricCollector(
        metrics=[Max(parameter), MeanStd(parameter, eps=eps), Min(parameter)]
    )
    loss = (parameter * torch.full(((2, 3)), 1.0, device=parameter.device)).sum(1)
    metrics = metric_collector(loss)
    assert loss.device == metrics.device
