from gradient_metrics.metrics import GradientMetric, Max, Mean, MeanStd, Min, PNorm
import pytest
import torch


def test_baseclass():
    class TestMetric(GradientMetric):
        pass

    metric = TestMetric()

    with pytest.raises(NotImplementedError):
        metric._collect(torch.ones((1,)))

    with pytest.raises(NotImplementedError):
        metric._get_metric()

    with pytest.raises(NotImplementedError):
        metric.reset()


def test_max():
    metric = Max()
    parameter = torch.ones((1,), requires_grad=True)
    parameter.register_hook(metric)

    loss = parameter * torch.ones((1,))
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = parameter * torch.tensor(0.5)
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(2.0)


def test_min():
    metric = Min()
    parameter = torch.ones((1,), requires_grad=True)
    parameter.register_hook(metric)

    loss = parameter * torch.ones((1,))
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    metric.reset()
    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(2.0)

    loss = parameter * torch.tensor(0.0)
    loss.backward()
    assert metric.data == torch.tensor(0.0)


def test_pnorm():
    with pytest.raises(ValueError):
        PNorm(p=float("inf"))

    with pytest.raises(ValueError):
        PNorm(eps=-1.0)

    with pytest.raises(ValueError):
        PNorm(eps=0.0)

    metric = PNorm()
    parameter = torch.ones((3,), requires_grad=True)
    parameter.register_hook(metric)

    loss = (parameter * torch.ones_like(parameter)).sum()
    loss.backward()
    assert metric.data == torch.tensor(3.0)


def test_mean():
    metric = Mean()
    parameter = torch.ones((3,), requires_grad=True)
    parameter.register_hook(metric)

    loss = (parameter * torch.ones_like(parameter)).sum()
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = (parameter * torch.tensor([-1.0, 0.0, 1.0])).sum()
    loss.backward()
    assert metric.data == torch.tensor(0.5)
    metric.reset()

    loss = (parameter * torch.tensor([-1.0, 0.0, 1.0])).sum()
    loss.backward()
    assert metric.data == torch.tensor(0.0)


def test_meanstd():
    with pytest.raises(ValueError):
        MeanStd(eps=-1.0)

    eps = 1e-16
    metric = MeanStd(eps=eps)
    parameter = torch.ones((3,), requires_grad=True)
    handle = parameter.register_hook(metric)

    loss = (parameter * torch.ones_like(parameter)).sum()
    loss.backward()
    assert torch.all(metric.data == torch.tensor([1.0, eps]))
    metric.reset()
    handle.remove()

    parameter = torch.ones((1,), requires_grad=True)
    handle = parameter.register_hook(metric)

    loss = parameter * torch.tensor(0.0)
    loss.backward()
    assert torch.all(metric.data == torch.tensor([0.0, eps]))
    handle.remove()

    metric = MeanStd(return_mean=False, eps=eps)
    parameter.register_hook(metric)

    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(eps)
