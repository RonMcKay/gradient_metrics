<div align="center">

[![PyPI](https://img.shields.io/pypi/v/gradient-metrics)](https://pypi.org/project/gradient-metrics/) ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/ronmckay/gradient_metrics/Publish%20to%20PyPI/main) [![](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black) [![PyPI - License](https://img.shields.io/pypi/l/gradient-metrics)](https://github.com/RonMcKay/gradient_metrics/blob/main/LICENSE) ![PyPI - Downloads](https://img.shields.io/pypi/dm/gradient-metrics)

</div>

This package implements utilities for computing gradient metrics for measuring uncertainties in neural networks based on the paper "[Classification Uncertainty of Deep Neural Networks Based on Gradient Information, Oberdiek et al., 2018][1]".  
An application of this can also be found in "[On the Importance of Gradients for Detecting Distributional Shifts in the Wild, Huang et al., 2021][2]"

Documentation and examples can be found on [GitHub pages](https://ronmckay.github.io/gradient_metrics/).

# Installation

```
pip install gradient-metrics
```

# Usage

Example of computing the maximum, minimum, mean and standard deviation of gradient entries as in [Classification Uncertainty of Deep Neural Networks Based on Gradient Information][1]:

```python
from gradient_metrics import GradientMetricCollector
from gradient_metrics.metrics import Max, Min, MeanStd
import torch.nn.functional as tfunc

# Initialize a network
mynet = MyNeuralNetwork()

# Initialize the GradientMetricCollector
mcollector = GradientMetricCollector(
    [
        Max(mynet),
        Min(mynet),
        MeanStd(mynet),
    ]
)

# Predict your data
out = mynet(x)

# Construct pseudo labels
y_pred = out.argmax(1).clone().detach()

# Construct the sample wise loss for backpropagation
sample_loss = tfunc.cross_entropy(out, y_pred, reduction="none")

# Compute the gradient metrics
metrics = mcollector(sample_loss)
```

----

Example of computing the L1-Norm from [On the Importance of Gradients for Detecting Distributional Shifts in the Wild][2]:

```python
from gradient_metrics import GradientMetricCollector
from gradient_metrics.metrics import PNorm
import torch
import torch.nn.functional as tfunc

# Initialize a network
mynet = MyNeuralNetwork()

# Initialize the GradientMetricCollector
mcollector = GradientMetricCollector(PNorm(mynet))

# Predict your data
out = mynet(x)

# Construct the sample wise loss for backpropagation
sample_loss = torch.log(tfunc.softmax(out, dim=1)).mean(1).neg()

# Compute the gradient metrics
metrics = mcollector(sample_loss)
```

# Citing

```txt
@inproceedings{OberdiekRG18,  
  author    = {Philipp Oberdiek and  
               Matthias Rottmann and  
               Hanno Gottschalk},  
  editor    = {Luca Pancioni and  
               Friedhelm Schwenker and  
               Edmondo Trentin},  
  title     = {Classification Uncertainty of Deep Neural Networks Based on Gradient  
               Information},  
  booktitle = {Artificial Neural Networks in Pattern Recognition - 8th {IAPR} {TC3}  
               Workshop, {ANNPR} 2018, Siena, Italy, September 19-21, 2018, Proceedings},  
  series    = {Lecture Notes in Computer Science},  
  volume    = {11081},  
  pages     = {113--125},  
  publisher = {Springer},  
  year      = {2018},  
  url       = { https://doi.org/10.1007/978-3-319-99978-4_9 },  
  doi       = { 10.1007/978-3-319-99978-4\_9 },  
}
```

[1]: https://arxiv.org/abs/1805.08440 "Classification Uncertainty of Deep Neural Networks Based on Gradient Information, Oberdiek et al., 2018"
[2]: https://proceedings.neurips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html "On the Importance of Gradients for Detecting Distributional Shifts in the Wild, Huang et al., 2021"
