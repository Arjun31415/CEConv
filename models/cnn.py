"""Model definitions for the color MNIST experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ceconv.ceconv2d import CEConv2d
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d


_DROPOUT_FACTOR = 0.3


class CNN(nn.Module):
    """Vanilla Convolutional Neural Network with 7 layers."""

    def __init__(
        self,
        planes: int,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, planes, kernel_size=3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=4)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn5 = nn.BatchNorm2d(planes)
        self.bn6 = nn.BatchNorm2d(planes)
        self.bn7 = nn.BatchNorm2d(planes)

        self.fc = nn.Linear(planes, num_classes)

        self.mp = nn.MaxPool2d(2)
        self.do = nn.Dropout2d(_DROPOUT_FACTOR)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.do(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = self.do(F.relu(self.bn3(self.conv3(x))))
        x = self.do(F.relu(self.bn4(self.conv4(x))))
        x = self.do(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(x.size(0), -1)
        return self.fc(x)

class LECNN(CNN):
    """Luminance Equivariant Convolutional Neural Network with 7 layers."""

    def __init__(self, planes: int, le_layers: int = 7, num_classes: int = 10) -> None:
        super().__init__(planes, num_classes)
        self.eps = 1e-6
        assert 1 <= le_layers <= 7, "LE stages must be between 1 and 7"
        self.le_layers = le_layers
        self.luminance_scale = lambda x: (x.mean(dim=[1, 2, 3], keepdim=True) / (x.mean() + self.eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.luminance_scale(x)
        x = L * self.do(F.relu(self.bn1(self.conv1(x))))

        if self.le_layers > 1:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = F.relu(self.bn2(self.conv2(x)))
        x = L * self.mp(x)

        if self.le_layers > 2:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = L * self.do(F.relu(self.bn3(self.conv3(x))))

        if self.le_layers > 3:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = L * self.do(F.relu(self.bn4(self.conv4(x))))

        if self.le_layers > 4:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = L * self.do(F.relu(self.bn5(self.conv5(x))))

        if self.le_layers > 5:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = L * F.relu(self.bn6(self.conv6(x)))

        if self.le_layers > 6:
            L = self.luminance_scale(x)
        else:
            L = 1
        x = L * F.relu(self.bn7(self.conv7(x)))

        x = x.view(x.size(0), -1)
        return self.fc(x)


class CECNN(nn.Module):
    """Color Equivariant Convolutional Neural Network (CECNN) with 7 layers."""

    def __init__(
        self,
        planes: int,
        rotations: int,
        ce_layers: int = 7,
        groupcosetmaxpool: bool = False,
        num_classes: int = 10,
        separable: bool = True,
    ) -> None:
        super().__init__()

        assert rotations >= 2, "Rotations must be >= 2."
        assert ce_layers >= 1, "CE stages must be >= 1."
        assert ce_layers <= 7, "CE stages must be <= 7."

        kernels = [3, 3, 3, 3, 3, 3, 4]
        do = [True, False, True, True, True, False, False]
        mp = GroupMaxPool2d(2) if ce_layers >= 2 else nn.MaxPool2d(2)
        planes_ce = rotations * planes if not groupcosetmaxpool else planes

        self.ceconv_list = nn.ModuleList(
            [
                nn.Sequential(
                    CEConv2d(
                        1 if i == 0 else rotations,
                        rotations,
                        3 if i == 0 else planes,
                        planes,
                        kernel_size=kernels[i],
                        separable=separable,
                    ),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(_DROPOUT_FACTOR) if do[i] is True else nn.Identity(),
                    mp if i == 1 else nn.Identity(),
                )
                for i in range(ce_layers)
            ]
        )
        self.conv2d_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        planes_ce,
                        planes_ce,
                        kernel_size=kernels[j],
                    ),
                    nn.BatchNorm2d(planes_ce),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(_DROPOUT_FACTOR) if do[j] is True else nn.Identity(),
                    mp if j == 1 else nn.Identity(),
                )
                for j in range(ce_layers, 7)
            ]
        )

        if groupcosetmaxpool is True:
            self.gmp = GroupCosetMaxPool()
        else:
            self.gmp = None
        self.fc = nn.Linear(planes_ce, num_classes)

    def forward(self, x) -> torch.Tensor:
        for layer in self.ceconv_list:
            x = layer(x)

        if self.gmp is not None:
            x = self.gmp(x)
        else:
            xs = x.shape
            x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])

        for layer in self.conv2d_list:
            x = layer(x)

        # Flatten and apply fully-connected layer.
        x = x.view(x.size()[0], -1)
        return self.fc(x)


if __name__ == "__main__":
    from torchinfo import summary

    planes = 20
    batch_size = 8
    input_shape = (batch_size, 3, 28, 28)

    print("\n=== CNN ===")
    summary(CNN(planes=planes), input_shape, device="cpu")

    print("\n=== LECNN ===")
    summary(LECNN(planes=planes, le_layers=7), input_shape, device="cpu")

    print("\n=== CECNN ===")
    rotations = 3
    summary(
        CECNN(
            planes=19,
            rotations=rotations,
            separable=True,
            groupcosetmaxpool=True,
            ce_layers=2,
        ),
        input_shape,
        device="cpu",
    )
