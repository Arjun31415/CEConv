"""Luminance Equivariant Convolutional Layer."""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

def _get_luminance_scaling_matrix(scalings: int) -> torch.Tensor:
    """Returns a 1x1 luminance scaling matrix.

    Scales brightness levels by different factors.

    Args:
      scalings: int, number of brightness levels
    """
    assert scalings > 0, "Number of scalings must be positive."

    # Define different brightness factors
    scale_values = torch.linspace(0.5, 1.5, scalings)  # Low to High brightness

    # Convert to tensor
    return torch.tensor(scale_values, dtype=torch.float32).view(-1, 1, 1, 1)

def _trans_input_filter(weights, scalings, scaling_matrix) -> torch.Tensor:
    """Apply brightness scaling to filter.

    Args:
      weights: float32, input filter of size [c_out, c_in, 1, k, k]
      scalings: int, number of brightness scalings applied to filter
      scaling_matrix: float32, scaling factors of size [scalings, 1, 1, 1]
    """

    weights_shape = weights.shape
    weights_flat = weights.reshape((1, -1))  # Flatten the weights

    # Apply scaling to weights
    transformed_weights = scaling_matrix * weights_flat  # Element-wise multiplication
    transformed_weights = transformed_weights.view((scalings,) + weights_shape)

    return transformed_weights

def _trans_hidden_filter(weights: torch.Tensor, scalings: int) -> torch.Tensor:
    """Apply brightness permutations to hidden layer filters."""

    w_shape = weights.shape
    transformed_weights = torch.zeros(
        ((w_shape[0],) + (scalings,) + w_shape[1:]), device=weights.device
    )

    # Apply brightness level adjustments
    for i in range(scalings):
        transformed_weights[:, i, :, :, :, :] = weights * (0.5 + (i / (scalings - 1)))

    return transformed_weights

class LEConv2d(nn.Conv2d):
    """
    Applies a Luminance Equivariant convolution over an input.

    Args:
        in_scalings (int): Number of input brightness scalings (1 for input layer, >1 for hidden layers).
        out_scalings (int): Number of output brightness scalings.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        learnable (bool): If True, the transformation matrix is learnable.
        separable (bool): If True, the convolution is separable.
        stride (int or tuple, optional): Stride of the convolution.
        padding (int, tuple or str, optional): Padding applied to all four sides.
    """

    def __init__(
        self,
        in_scalings: int,
        out_scalings: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        learnable: bool = False,
        separable: bool = True,
        stride: int = 1,
        **kwargs
    ) -> None:
        self.in_scalings = in_scalings
        self.out_scalings = out_scalings
        self.separable = separable

        stride = stride if isinstance(stride, (int, tuple)) else int(stride[0])  # Ensure stride is a tuple
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, **kwargs)


        # Initialize transformation matrix and weights
        self.transformation_matrix = Parameter(
            _get_luminance_scaling_matrix(out_scalings), requires_grad=learnable
        )
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, 1, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        n = self.in_channels * self.in_scalings * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with luminance scaling."""

        if self.in_scalings == 1:
            tw = _trans_input_filter(self.weight, self.out_scalings, self.transformation_matrix)
        else:
            tw = _trans_hidden_filter(self.weight, self.out_scalings)

        expected_elements = torch.numel(tw)  # Get total elements in tw
        
        # Compute correct shape ensuring all elements match
        kernel_size_tuple = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)

        correct_shape = (
            self.out_channels,
            self.out_scalings,
            self.in_channels,
            self.in_scalings,
            *kernel_size_tuple,
        )

        if math.prod(correct_shape) != expected_elements:
            raise ValueError(f"Mismatch in expected elements. Expected: {expected_elements}, Got: {math.prod(correct_shape)}")
        
        tw = tw.view(correct_shape)


        expected_channels = self.in_channels * self.in_scalings
        if input.size(1) != expected_channels:
            input = input.repeat(1, self.in_scalings, 1, 1)  # Expand channels by repeating across scalings


        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride, padding=self.padding)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_scalings, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y
