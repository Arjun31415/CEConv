"""Color Equivariant Convolutional Layer."""

import math
import typing
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


# class LearnableHueTransformation(nn.Module):
#     def __init__(self, rotations: int, init, lambda_orth: float = 10):
#         super().__init__()
#         self.rotations = rotations
#         # self.transformation_matrix = nn.Parameter(
#         #     torch.eye(3) + 0.01 * torch.randn(3, 3)
#         # )  # Start close to identity
#
#         self.transformation_matrix = nn.Parameter(
#             init + 0.1 * torch.randn(3, 3)
#         )  # Start close to identity
#         self.lambda_orth = lambda_orth  # Regularization strength
#
#     def forward(self, x):
#         """Apply the learned hue transformation"""
#         return torch.matmul(self.transformation_matrix, x)
#
#     def orthogonality_loss(self):
#         """Encourage learned matrix to be near orthogonal"""
#         I = torch.eye(
#             self.transformation_matrix.shape[0],
#             device=self.transformation_matrix.device,
#         )
#         return self.lambda_orth * torch.norm(
#             self.transformation_matrix @ self.transformation_matrix.T - I, p="fro"
#         )
#


class LearnableHueTransformation(nn.Module):
    def __init__(
        self, rotations: int, axis: torch.types.Tensor, lambda_orth: float = 0
    ):
        super().__init__()
        self.rotations: torch.types.Tensor = torch.nn.Parameter(
            torch.rand(1) * torch.Tensor(rotations), requires_grad=True
        )
        self.axis = torch.nn.Parameter(data=axis, requires_grad=True)
        # self.transformation_matrix = nn.Parameter(
        #     torch.eye(3) + 0.01 * torch.randn(3, 3)
        # )  # Start close to identity

        self.lambda_orth = lambda_orth  # Regularization strength
        # self.transformation_matrix = nn.Parameter(self.generate_rotation_matrix(
        #     self.axis, (2 * torch.pi) / self.rotations
        # ))

    def skew_symmetric(self, u):
        """
        Given a 3D vector u, return the skew-symmetric matrix [u]_x
        """
        ux, uy, uz = u
        return torch.tensor(
            [[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], device=u.device, dtype=u.dtype
        )

    def generate_rotation_matrix(self, u, theta):
        """
        Compute the 3x3 rotation matrix using the concise Rodrigues' formula.
        Args:
            u: Tensor of shape (3,) - axis of rotation (not necessarily normalized)
            theta: Scalar tensor - rotation angle in radians
        Returns:
            R: Tensor of shape (3, 3)
        """
        u = F.normalize(u, dim=0)  # Ensure u is unit vector
        I = torch.eye(3, device=u.device)
        K = self.skew_symmetric(u)  # Cross product matrix [u]_x
        outer = torch.outer(u, u)  # Outer product u ⊗ u

        R = torch.cos(theta) * I + torch.sin(theta) * K + (1 - torch.cos(theta)) * outer
        return R

    def forward(self, x):
        """Apply the learned hue transformation"""
        trans_matrix = self.generate_rotation_matrix(
            self.axis, (2 * torch.pi) / self.rotations
        )
        return torch.matmul(trans_matrix, x).to(self.device)

    def orthogonality_loss(self):
        """Encourage learned matrix to be near orthogonal"""
        # I = torch.eye(
        #     self.transformation_matrix.shape[0],
        #     device=self.transformation_matrix.device,
        # )
        # return self.lambda_orth * torch.norm(
        #     self.transformation_matrix @ self.transformation_matrix.T - I, p="fro"
        # )
        return torch.Tensor(0).to(self.device)


def _get_hue_rotation_matrix(rotations: int) -> torch.Tensor:
    """Returns a 3x3 hue rotation matrix.

    Rotates a 3D point by 360/rotations degrees along the diagonal.

    Args:
      rotations: int, number of rotations
    """

    assert rotations > 0, "Number of rotations must be positive."

    # Constants in rotation matrix
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return torch.tensor(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ],
        dtype=torch.float32,
    )


def _trans_input_filter(weights, rotations, rotation_matrix) -> torch.Tensor:
    """Apply linear transformation to filter.

    Args:
      weights: float32, input filter of size [c_out, 3 (c_in), 1, k, k]
      rotations: int, number of rotations applied to filter
      rotation_matrix: float32, rotation matrix of size [3, 3]
    """

    # Flatten weights tensor.
    weights_flat = weights.permute(2, 1, 0, 3, 4)  # [1, 3, c_out, k, k]
    weights_shape = weights_flat.shape
    weights_flat = weights_flat.reshape((1, 3, -1))  # [1, 3, c_out*k*k]

    # Construct full transformation matrix.

    # rotation_matrix = torch.stack(
    #     [torch.matrix_power(rotation_matrix, i) for i in range(rotations)], dim=0
    # )
    rotation_matrix = torch.stack(
        [
            torch.matrix_power(rotation_matrix, i)
            for i in range(rotations)
        ],
        dim=0,
    ).to(weights_flat.device)

    # Apply transformation to weights.
    # [rotations, 3, 3] * [1, 3, c_out*k*k] --> [rotations, 3, c_out*k*k]
    transformed_weights = torch.matmul(rotation_matrix, weights_flat)
    # [rotations, 1, c_in (3), c_out, k, k]
    transformed_weights = transformed_weights.view((rotations,) + weights_shape)
    # [c_out, rotations, c_in (3), 1, k, k]
    tw = transformed_weights.permute(3, 0, 2, 1, 4, 5)

    return tw.contiguous()


def _trans_hidden_filter(weights: torch.Tensor, rotations: int) -> torch.Tensor:
    """Perform cyclic permutation on hidden layer filter parameters."""

    # Create placeholder for output tensor
    w_shape = weights.shape
    transformed_weights = torch.zeros(
        ((w_shape[0],) + (rotations,) + w_shape[1:]), device=weights.device
    )

    # Apply cyclic permutation on output tensor
    for i in range(rotations):
        transformed_weights[:, i, :, :, :, :] = torch.roll(weights, shifts=i, dims=2)

    return transformed_weights


class CEConv2d(nn.Conv2d):
    """
    Applies a Color Equivariant convolution over an input signal composed of several
    input planes.


    Args:
        in_rotations (int): Number of input rotations: 1 for input layer, >1 for
            hidden layers.
        out_rotations (int): Number of output rotations.
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        learnable (bool): If True, the transformation matrix is learnable.
        separable (bool): If True, the convolution is separable.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_rotations: int,
        out_rotations: int,
        in_channels: int,
        out_channels: int,
        kernel_size: typing.Union[int, typing.Tuple[int, int]],
        learnable: bool = False,
        separable: bool = True,
        **kwargs,
    ) -> None:
        self.in_rotations = in_rotations
        self.out_rotations = out_rotations
        self.separable = separable

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.rotations: torch.types.Tensor = torch.nn.Parameter(
             torch.Tensor(in_rotations), requires_grad=True
        )
        self.axis: torch.types.Tensor = torch.nn.Parameter(
            data=torch.Tensor([1, 2, 1]), requires_grad=True
        )

        # Initialize transformation matrix and weights.
        if in_rotations == 1:
            init = (
                torch.rand((3, 3)) * 2.0 / 3 - (1.0 / 3)
                if learnable
                else _get_hue_rotation_matrix(out_rotations)
            )
            # self.transformation_matrix = Parameter(init, requires_grad=learnable)
            # self.transformation_matrix = LearnableHueTransformation(
            #     axis=torch.rand(3), rotations=self.in_rotations
            # )
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
            )
        else:
            if separable:
                if in_rotations > 1:
                    self.weight = Parameter(
                        # torch.Tensor(out_channels, 1, 1, *self.kernel_size)
                        torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                    )
                    self.pointwise_weight = Parameter(
                        torch.Tensor(out_channels, in_channels, self.in_rotations, 1, 1)
                    )
            else:
                self.weight = Parameter(
                    torch.Tensor(
                        out_channels, in_channels, self.in_rotations, *self.kernel_size
                    )
                )

        self.reset_parameters()

    def skew_symmetric(self, u:torch.types.Tensor):
        """
        Given a 3D vector u, return the skew-symmetric matrix [u]_x
        """
        ux, uy, uz = u
        return torch.tensor(
            [[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], device=u.device, dtype=u.dtype
        )

    def generate_rotation_matrix(
        self, u: torch.types.Tensor, theta: torch.types.Tensor
    ) -> torch.types.Tensor:
        """
        Compute the 3x3 rotation matrix using the concise Rodrigues' formula.
        Args:
            u: Tensor of shape (3,) - axis of rotation (not necessarily normalized)
            theta: Scalar tensor - rotation angle in radians
        Returns:
            R: Tensor of shape (3, 3)
        """
        u = F.normalize(u, dim=0)  # Ensure u is unit vector
        I = torch.eye(3, device=u.device)
        K = self.skew_symmetric(u)  # Cross product matrix [u]_x
        outer = torch.outer(u, u)  # Outer product u ⊗ u

        R = torch.cos(theta) * I + torch.sin(theta) * K + (1 - torch.cos(theta)) * outer
        return R

    def reset_parameters(self) -> None:
        """Initialize parameters."""

        # Compute standard deviation for weight initialization.
        n = self.in_channels * self.in_rotations * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(n)

        # Initialize weights.
        self.weight.data.uniform_(-stdv, stdv)
        if hasattr(self, "pointwise_weight"):
            self.pointwise_weight.data.uniform_(-stdv, stdv)

        # Initialize bias.
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        # Compute full filter weights.
        if self.in_rotations == 1:
            # Apply rotation to input layer filter.
            transformation_matrix = self.generate_rotation_matrix(
                self.axis, torch.tensor((2 * torch.pi) / self.out_rotations)
            )
            tw = _trans_input_filter(
                self.weight, self.out_rotations, transformation_matrix
            )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations)

        tw_shape = (
            self.out_channels * self.out_rotations,
            self.in_channels * self.in_rotations,
            *self.kernel_size,
        )
        tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        # if self.in_rotations == 1:
        #     return y, self.transformation_matrix.orthogonality_loss()
        # else:
        #     return y
        return y
