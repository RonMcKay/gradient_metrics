from gradient_metrics import GradientMetricCollector
from gradient_metrics.metrics import Max, Min, PNorm
import torch
import torch.nn as nn


# Define some model
class MyNet(nn.Module):
    def __init__(self, image_size=32) -> None:
        """This is a model which predicts one of 10 classes.

        Args:
            image_size (int, optional): Input size. Should be a power of 2.
                Defaults to 32.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(image_size // 4) ** 2 * 64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):
        out = self.features(x)
        return self.fc(out)


if __name__ == "__main__":
    # Create some dummy data
    x = torch.randn((10, 3, 32, 32))

    # Initialize the model
    net = MyNet()

    # Initialize the GradientMetricCollector
    # In this example we want to gather metrics from the whole model,
    # the feature extractor and the fully connected part
    # We use Max, Min and PNorm with p=2
    mcollector = GradientMetricCollector(
        [
            # Extract metrics from the whole network
            Max(net),
            Min(net),
            PNorm(net, p=2),
            # Extract metrics from the feature extraction part
            Max(net.features),
            Min(net.features),
            PNorm(net.features, p=2),
            # Extract metrics from the fully connected part
            Max(net.fc),
            Min(net.fc),
            PNorm(net.fc, p=2),
        ]
    )

    # predict the dummy data
    out = net(x)

    # create pseudo labels
    y_pred = out.argmax(1).clone().detach()

    # Compute a sample-wise loss with the pseudo labels
    # For this we use the binary-cross-entropy loss function
    crit = nn.CrossEntropyLoss(reduction="none")
    loss = crit(out, y_pred)

    # gather gradient metrics
    grad_metrics = mcollector(loss)

    # We will get an output shape of (10, 9)
    # 3 metrics over the whole network parameters
    # 3 metrics over the feature part
    # 3 metrics over the fully connected part
    print(f"Shape of the gradient metric output: {grad_metrics.shape}")

    # Now let's say we want to have the minimum and maximum of the absolute gradient
    # values in the first convolution's kernel. We can achieve that by using a
    # `grad_transform`:
    mcollector = GradientMetricCollector(
        [
            Min(net.features[0].weight, grad_transform=lambda grad: grad.abs()),
            Max(net.features[0].weight, grad_transform=lambda grad: grad.abs()),
        ]
    )

    grad_metrics = mcollector(loss)

    print(f"Value range of absolute gradient values:\n{grad_metrics}")
