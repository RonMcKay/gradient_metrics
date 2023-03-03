import gc

import pytest
import torch

from gradient_metrics.metrics import GradientMetric, Max, Mean, MeanStd, Min, PNorm


def test_baseclass():
    class TestMetric(GradientMetric):
        pass

    metric = TestMetric([])

    with pytest.raises(NotImplementedError):
        metric._collect(torch.ones((1,)))

    with pytest.raises(NotImplementedError):
        metric._get_metric()

    with pytest.raises(NotImplementedError):
        metric.reset()


@pytest.mark.parametrize(
    ["metric_class", "kwargs", "error_type"],
    [
        (PNorm, dict(p=0), ValueError),
        (PNorm, dict(p=-1.0), ValueError),
        (PNorm, dict(p="test"), ValueError),
        (MeanStd, dict(eps=-1.0), ValueError),
        (MeanStd, dict(eps=0.0), ValueError),
    ],
)
def test_inputs(metric_class, kwargs, error_type):
    with pytest.raises(error_type):
        metric_class([], **kwargs)


def test_max():
    parameter = torch.ones((1,), requires_grad=True)
    metric = Max(parameter)

    loss = parameter * torch.ones((1,))
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = parameter * torch.tensor(0.5)
    loss.backward()
    assert metric.data == torch.tensor(1.0)

    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(2.0)

    metric = Max(parameter, grad_transform=lambda grad: torch.tensor(-1.0))
    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(-1.0)

    parameter = torch.ones((1,), requires_grad=True, device="meta")
    metric = Max(parameter)
    loss = parameter * torch.tensor(2.0, device="meta")
    metric(autograd.grad(loss, parameter))
    assert metric.data.device == torch.device("meta")


def test_min():
    parameter = torch.ones((1,), requires_grad=True)
    metric = Min(parameter)

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

    metric = Min(parameter, grad_transform=lambda grad: torch.tensor(100.0))
    loss = parameter * torch.tensor(0.0)
    loss.backward()
    assert metric.data == torch.tensor(100.0)

    parameter = torch.ones((1,), requires_grad=True, device="meta")
    metric = Min(parameter)
    loss = parameter * torch.tensor(0.0, device="meta")
    metric(autograd.grad(loss, parameter))
    assert metric.data.device == torch.device("meta")


def test_pnorm():
    parameter = torch.ones((3,), requires_grad=True)
    metric = PNorm(parameter)

    loss = (parameter * torch.ones_like(parameter)).sum()
    loss.backward()
    assert metric.data == torch.tensor(3.0)

    metric = PNorm(parameter, grad_transform=lambda grad: torch.tensor(100.0))
    loss = (parameter * torch.tensor(0.0)).sum()
    loss.backward()
    assert metric.data == torch.tensor(100.0)

    metric = PNorm(parameter, p=float("inf"))
    loss = (parameter * torch.tensor(-2.0)).sum()
    loss.backward()
    assert metric.data == torch.tensor(2.0)

    metric = PNorm(parameter, p=float("inf"))
    loss = (parameter * torch.tensor(2.0)).sum()
    loss.backward()
    assert metric.data == torch.tensor(2.0)

    metric = PNorm(parameter, grad_transform=lambda grad: torch.tensor([100, 100]), p=2)
    loss = (parameter * torch.tensor(1.0)).sum()
    loss.backward()
    assert metric.data == torch.tensor(20000**0.5)

    parameter = torch.ones((3,), requires_grad=True, device="meta")
    metric = PNorm(parameter)
    loss = (parameter * torch.tensor(2.0, device="meta")).sum()
    metric(autograd.grad(loss, parameter))
    assert metric.data.device == torch.device("meta")


def test_mean():
    parameter = torch.ones((3,), requires_grad=True)
    metric = Mean(parameter)

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

    metric = Mean(parameter, grad_transform=lambda grad: torch.tensor(1))
    loss = (parameter * torch.tensor(0.0)).sum()
    loss.backward()
    assert metric.data == torch.tensor(1)

    parameter = torch.ones((3,), requires_grad=True, device="meta")
    metric = Mean(parameter)
    loss = (parameter * torch.tensor([-1.0, 0.0, 1.0], device="meta")).sum()
    metric(autograd.grad(loss, parameter))
    assert metric.data.device == torch.device("meta")


def test_meanstd():
    eps = 1e-16
    parameter = torch.ones((3,), requires_grad=True)
    metric = MeanStd(parameter, eps=eps)

    loss = (parameter * torch.ones_like(parameter)).sum()
    loss.backward()
    assert torch.all(metric.data == torch.tensor([1.0, eps]))
    metric.reset()
    del metric
    gc.collect()

    parameter = torch.ones((1,), requires_grad=True)
    metric = MeanStd(parameter, eps=eps)

    loss = parameter * torch.tensor(0.0)
    loss.backward()
    assert torch.all(metric.data == torch.tensor([0.0, eps]))
    del metric
    gc.collect()

    metric = MeanStd(parameter, return_mean=False, eps=eps)

    loss = parameter * torch.tensor(2.0)
    loss.backward()
    assert metric.data == torch.tensor(eps)

    metric = MeanStd(parameter, grad_transform=lambda grad: torch.tensor(1))
    loss = (parameter * torch.tensor(0.0)).sum()
    loss.backward()
    assert torch.all(metric.data == torch.tensor([1, eps]))

    parameter = torch.ones((1,), requires_grad=True, device="meta")
    metric = MeanStd(parameter, eps=eps)
    loss = parameter * torch.tensor(2.0, device="meta")
    metric(autograd.grad(loss, parameter))
    assert metric.data.device == torch.device("meta")
