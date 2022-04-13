from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class GradientMetric(object):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """This is the base class for all gradient metrics

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
        """
        self.hook_handles: List[RemovableHandle] = []
        self.target_layers = (
            (target_layers,)
            if isinstance(target_layers, (nn.Module, torch.Tensor))
            else tuple(target_layers)
        )
        self.grad_transform = grad_transform
        self._register()

    def __call__(self, grad: torch.Tensor) -> None:
        """A gradient metric instance is registered as a backward hook on parameters.
        This is going to be called when the associated parameter is part of a backward
        call.

        Args:
            grad (torch.Tensor): The gradient of the associated parameter. On this the
                metric is going to be computed.
        """
        if self.grad_transform is not None:
            grad = self.grad_transform(grad)
        self._collect(grad)

    def __del__(self) -> None:
        for hook in self.hook_handles:
            hook.remove()

    def _collect(self, grad: torch.Tensor) -> None:
        """This method has to be implemented by every GradientMetric subclass.
        It will be called on the gradient supplied to the instance via the
        backward hook.

        Args:
            grad (torch.Tensor): The gradient of the associated parameter.

        Raises:
            NotImplementedError: Raises an error if not implemented by sub classes.
        """
        raise NotImplementedError

    @property
    def data(self) -> torch.Tensor:
        """Holds the metric data.

        Returns:
            torch.Tensor: The metric value stored in the buffer.
        """
        return self._get_metric().view(-1)

    def _get_metric(self) -> torch.Tensor:
        """This method should return the metric values stored in buffer.
        It will be used by the data property and has to be implemented by
        all sub classes.

        Raises:
            NotImplementedError: Raises an error if not implemented by sub classes.

        Returns:
            torch.Tensor: The metric value stored in the buffer.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the metric values to a default value.
        Has to be implemented by all sub classes.

        Raises:
            NotImplementedError: Raises an error if not implemented by sub classes.
        """
        raise NotImplementedError

    def _register(self) -> None:
        for t in self.target_layers:
            if isinstance(t, torch.Tensor):
                self.hook_handles.append(t.register_hook(self))
            else:
                for param in t.parameters():
                    self.hook_handles.append(param.register_hook(self))


class PNorm(GradientMetric):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        p: int = 1,
        eps: float = 1e-16,
    ) -> None:
        r"""Computes the p-norm over the flattened gradients.

        .. math::
            (\sum_{i=1}^n |x_i|^p)^{\frac{1}{p}}

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
            p (int, optional): Power of the norm. Defaults to 1 (absolute-value norm).
            eps (float, optional): Small epsilon for gradients with very small vector
                norms which would otherwise result in a possible division by zero in
                the second order derivatives. Defaults to 1e-16.

        Raises:
            ValueError: If p is smaller or equal to zero.
            ValueError: If eps is smaller or equal to zero.
        """
        super().__init__(target_layers=target_layers, grad_transform=grad_transform)

        if not 0 < p < float("inf"):
            raise ValueError(f"p has to be in (0, inf), got {p}")
        self.p = float(p)

        if not eps > 0:
            raise ValueError(f"eps has to be greater than zero, got {eps}")
        self.eps = eps

        self._metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._metric_buffer.device != grad.device:
            self._metric_buffer = self._metric_buffer.to(grad.device)

        self._metric_buffer = torch.pow(
            self._metric_buffer.pow(self.p) + grad.view(-1).abs().pow(self.p).sum(),
            1.0 / self.p,
        )

    def _get_metric(self) -> torch.Tensor:
        return self._metric_buffer

    def reset(self) -> None:
        """Initializes/resets the buffer to 0"""
        self._metric_buffer = torch.tensor(0.0)


class Min(GradientMetric):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Computes the minimum over the gradients.

        The minimum between the currently saved buffer and the supplied gradients is
        computed on each call, overwriting the buffer with the result.

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
        """
        super().__init__(target_layers=target_layers, grad_transform=grad_transform)
        self._metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._metric_buffer.device != grad.device:
            self._metric_buffer = self._metric_buffer.to(grad.device)
        self._metric_buffer = torch.min(self._metric_buffer, grad.min())

    def _get_metric(self) -> torch.Tensor:
        return self._metric_buffer

    def reset(self) -> None:
        r"""Initializes/resets the buffer to :math:`\infty`"""
        self._metric_buffer = torch.tensor(float("inf"))


class Max(GradientMetric):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Computes the maximum over the gradients.

        The maximum between the currently saved buffer and the supplied gradients is
        computed on each call, saving the result in the buffer.

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
        """
        super().__init__(target_layers=target_layers, grad_transform=grad_transform)
        self._metric_buffer: torch.Tensor
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._metric_buffer.device != grad.device:
            self._metric_buffer = self._metric_buffer.to(grad.device)
        self._metric_buffer = torch.max(self._metric_buffer, grad.max())

    def _get_metric(self) -> torch.Tensor:
        return self._metric_buffer

    def reset(self) -> None:
        r"""Initializes/resets the buffer to :math:`-\infty`"""
        self._metric_buffer = torch.tensor(float("-inf"))


class Mean(GradientMetric):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Computes the mean of the supplied gradients.

        The buffer always holds the mean of all previously supplied gradients.
        This exists besides :class:`~gradient_metrics.metrics.MeanStd` to reduce
        computation cost if you do not want to computed the standard deviation.

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
        """
        super().__init__(target_layers=target_layers, grad_transform=grad_transform)
        self._metric_buffer: torch.Tensor
        self._count: int
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._metric_buffer.device != grad.device:
            self._metric_buffer = self._metric_buffer.to(grad.device)

        old_mean = self._metric_buffer.detach().clone()
        self._count += grad.view(-1).shape[0]
        self._metric_buffer = (
            self._metric_buffer + torch.sum(grad.view(-1) - old_mean) / self._count
        )

    def _get_metric(self) -> torch.Tensor:
        return self._metric_buffer

    def reset(self) -> None:
        """Initializes/resets the buffer and counter to 0"""
        self._metric_buffer = torch.tensor(0.0)
        self._count = 0


class MeanStd(GradientMetric):
    def __init__(
        self,
        target_layers: Union[
            Sequence[Union[nn.Module, torch.Tensor]], nn.Module, torch.Tensor
        ],
        grad_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_mean: bool = True,
        eps: float = 1e-16,
    ) -> None:
        """Computes Mean and Standard Deviation.

        This uses `Welford's online algorithm <https://doi.org/10.2307%2F1266577>`_
        for mean and variance computation to reduce memory usage.

        If there was only a single gradient entry, the returned standard deviation
        is equal to ``eps``. This is also the lower bound for the standard deviation.

        Args:
            target_layers (torch.nn.Module, torch.Tensor or sequence of them): Layers
                or tensors on which the metrics will be registered as backward hooks.
                For ``torch.nn.Module`` instances a single metric instance will be
                registered to all parameters returned by
                ``torch.nn.Module.parameters()``, thus computing the metric over all
                parameters of the Module.
            grad_transform (Optional[Callable[[torch.Tensor], torch.Tensor]], optional):
                A callable which accepts a ``torch.Tensor`` as input and returns a
                ``torch.Tensor``. The callable is applied to the gradient before it is
                handed over to the ``_collect`` method of a ``GradientMetric`` instance.
            return_mean (bool, optional): Whether to return the mean or not.
                Defaults to True.
            eps (float, optional): Small epsilon for gradients with very small standard
                deviation which would otherwise result in a possible division by zero in
                the second order derivatives. Defaults to 1e-16.

        Raises:
            ValueError: If eps is smaller or equal to zero.
        """
        super().__init__(target_layers=target_layers, grad_transform=grad_transform)

        if not eps > 0:
            raise ValueError(f"eps has to be greater than zero, got {eps}")
        self.eps = eps

        self.return_mean = return_mean

        self._mean: torch.Tensor
        self._m2: torch.Tensor
        self._count: int
        self.reset()

    def _collect(self, grad: torch.Tensor) -> None:
        if self._m2.device != grad.device:
            self._m2 = self._m2.to(grad.device)
            self._mean = self._mean.to(grad.device)

        # do a batch update according to Welford's algorithm

        self._count += grad.view(-1).shape[0]
        # gradient computation graph of mean is still available through
        # mean in the following line so we can detach this one
        old_mean = self._mean.detach().clone()
        self._mean = self._mean + torch.sum((grad.view(-1) - old_mean) / self._count)
        self._m2 = self._m2 + torch.sum(
            grad.view(-1).pow(2)
            - grad.view(-1) * (old_mean + self._mean)
            + old_mean * self._mean
        )

    def _get_metric(self) -> torch.Tensor:
        if self._count > 1:
            return torch.cat(
                (
                    self._mean.view(-1)
                    if self.return_mean
                    else torch.empty((0,), device=self._mean.device),
                    torch.sqrt(self._m2 / (self._count - 1) + self.eps**2).view(-1),
                )
            )
        else:
            return torch.cat(
                (
                    self._mean.view(-1)
                    if self.return_mean
                    else torch.empty((0,), device=self._mean.device),
                    torch.tensor(self.eps, device=self._mean.device).view(-1),
                )
            )

    def reset(self) -> None:
        """Initializes/resets the buffers."""
        self._mean = torch.tensor(0.0)
        self._m2 = torch.tensor(0.0)
        self._count = 0
