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


    # def __init__(self, lmax):
class S2ConvNet_Autoencoder(torch.nn.Module):
    def __init__(self, lmax) -> None:
        super().__init__()

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.l_list = [lmax, 3, 2, 1]
        f_list = [1, 4, 8, 16, 32]

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












        # S2Convolution(1, f_list[0], self.l_list[0], grid_s2)

        # conv_out = SO3ConvolutionToS2(f_list[0], 1, self.l_list[0], grid_so3)

        # encoder_list = []
        # for i in range(len(self.l_list) - 1):
        #     # encoder_list.append(e3nn.o3.Linear(s2_irreps(self.l_list[i]), s2_irreps(self.l_list[i]), f_list[i], f_list[i+1], internal_weights=True))
        #     # encoder_list.append(S2Activation(s2_irreps(self.l_list[i]), s2_irreps(self.l_list[i+1]), torch.relu, 1)

        #     # #     s2_irreps(self.l_list[i]), f_list[i]))
        #     encoder_list.append(SO3Convolution(f_list[i], f_list[i+1], self.l_list[i], grid_so3))
        #     encoder_list.append(SO3Activation(self.l_list[i], self.l_list[i+1], torch.relu, 1))
        # self.encoder = nn.Sequential(conv_in, *encoder_list)


        # decoder_list = []
        # for i in range(len(self.l_list) - 1, 0, -1):
        #     decoder_list.append(SO3Convolution(f_list[i], f_list[i-1], self.l_list[i], grid_so3))
        #     decoder_list.append(SO3Activation(self.l_list[i], self.l_list[i-1], torch.relu, 1))

        # # self.decoder = nn.Sequential(*decoder_list)
        # self.decoder = nn.Sequential(*decoder_list, conv_out)


        # self.lin1 = e3nn.o3.Linear(s2_irreps(self.l_list[0]), s2_irreps(self.l_list[0]), f_in=f_list[0],f_out=f_list[1], internal_weights=True)

        # # # TODO Add conv out

        # # f1 = 20
        # # f2 = 40
        # # f_output = 10

        # # b_in = 60
        # # lmax1 = 4

        # b_l1 = 10
        # lmax2 = 5

        # b_l2 = 6

        # self.conv1 = S2Convolution(1, f1, lmax1, kernel_grid=grid_s2)
        # # self.conv1 = SO3Convolution(1, f1, lmax1, kernel_grid=grid_so3)

        # self.act1 = SO3Activation(lmax1, lmax1, torch.relu, b_l1)

        # self.conv2 = SO3Convolution(f1, f2, lmax1, kernel_grid=grid_so3)

        # # self.act2 = SO3Activation(lmax2, 0, torch.relu, b_l2)

        # # self.w_out = torch.nn.Parameter(torch.randn(f2, f_output))

    def forward(self, x):
        lat = self.encoder(x)
        out = self.decoder(lat)
        return x, lat, out
        return self.lin1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x

        # x = self.conv1(x)
        # print("x after conv1: ", x.shape)
        # x = self.act1(x)
        # print("x after act: ", x.shape)
        # x = self.conv2(x)
        # print("x after conv2: ", x.shape)

        lat = self.encoder(x)
        out = self.decoder(lat)

        # x = x.transpose(-1, -2)  # [batch, features, alpha, beta] -> [batch, features, beta, alpha]
        # x = self.from_s2(x)  # [batch, features, beta, alpha] -> [batch, features, irreps]
        # x = self.conv1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        # x = self.act1(x)  # [batch, features, irreps] -> [batch, features, irreps]
        # x = self.conv2(x)  # [batch, features, irreps] -> [batch, features, irreps]
        # x = self.act2(x)  # [batch, features, scalar]
        # x = x.flatten(1) @ self.w_out / self.w_out.shape[0]

        return x, lat, out
