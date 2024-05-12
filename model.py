import gzip
import math
import pickle

import numpy as np
import torch
import e3nn
import torch.nn as nn
from e3nn import o3
from e3nn.nn import SO3Activation


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> torch.Tensor:
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_near_identity_grid(
    max_beta: float = math.pi / 8, max_gamma: float = 2 * math.pi, n_alpha: int = 8, n_beta: int = 3, n_gamma=None
) -> torch.Tensor:
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)

class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)

class SO3ConvolutionToS2(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        # self.register_parameter(
        #     "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        # )  # [f_in, f_out, n_so3_pts]
        # self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), s2_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=True)

    def forward(self, x):
        # psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x)

class S2ConvNet_Autoencoder(torch.nn.Module):
    def __init__(self, lmax: int, l_list: int, channels: int) -> None:
        """
        Create SO3 Equivariant Autoencoder

        Args:
            lmax (int):  Max degree of input tensor
            l_list (int):  degrees of data through network.  Must begin with lmax and end with latent l
            channels (int): Channels at each layer of the network
        """
        #         self.l_list = [lmax, 3, 3, 3, 2, 2, 1]
        # f_list = [1] + [4,   8, 8, 8, 16, 16, 32]

        super().__init__()

        assert(all(l <= lmax for l in l_list))

        self.l_list = l_list
        f_list = [1] + channels

        assert len(f_list) == len(self.l_list) + 1

        self.latent_repr = f_list[-1] * so3_irreps(self.l_list[-1])
        self.model_sphten_repr = e3nn.io.SphericalTensor(lmax=lmax, p_val=1, p_arg=1)

        # We start out on S2 and go to SO3
        encoder_list = []
        encoder_list.append(e3nn.o3.Linear(s2_irreps(self.l_list[0]), so3_irreps(self.l_list[0]), f_in=f_list[0], f_out=f_list[1], internal_weights=True))
        encoder_list.append(SO3Activation(self.l_list[0], self.l_list[0], torch.relu, 11))
        encoder_list.append(e3nn.nn.BatchNorm(so3_irreps(self.l_list[0])))
        for i in range(len(self.l_list) - 1):
            encoder_list.append(e3nn.o3.Linear(so3_irreps(self.l_list[i]), so3_irreps(self.l_list[i]), f_in=f_list[i+1], f_out=f_list[i+2], internal_weights=True))
            encoder_list.append(e3nn.nn.BatchNorm(so3_irreps(self.l_list[i])))
            encoder_list.append(SO3Activation(self.l_list[i], self.l_list[i+1], torch.relu, 11))
        self.encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        for i in range(len(self.l_list) - 1, 0, -1):
            decoder_list.append(e3nn.o3.Linear(so3_irreps(self.l_list[i]), so3_irreps(self.l_list[i]), f_in=f_list[i+1], f_out=f_list[i-1+1], internal_weights=True))
            # decoder_list.append(e3nn.nn.S2Activation(s2_irreps(self.l_list[i]), torch.relu, 10, lmax_out=self.l_list[i-1]))
            decoder_list.append(e3nn.nn.BatchNorm(so3_irreps(self.l_list[i])))
            decoder_list.append(SO3Activation(self.l_list[i], self.l_list[i-1], torch.relu, 11))

        decoder_list.append(e3nn.o3.Linear(so3_irreps(self.l_list[0]), s2_irreps(self.l_list[0]), f_in=f_list[1], f_out=f_list[0], internal_weights=True))

        self.decoder = nn.Sequential(*decoder_list)


    def forward(self, x):
        lat = self.encoder(x)
        out = self.decoder(lat)
        return x, lat, out