import math
import e3nn
import torch.nn as nn
import torch


class WeightedLoss(nn.Module):
    def __init__(self, lmax):
        super(WeightedLoss, self).__init__()
        self.lmax = lmax

    def forward(self, input, target):
        loss = 0
        start_idx = 0
        for l in range(self.lmax):
            end_idx = start_idx + 2 * l + 1
            weight = 1 / torch.mean(target[..., start_idx:end_idx])
            l_loss = torch.nn.functional.mse_loss(input[..., start_idx:end_idx], target[..., start_idx:end_idx]) * weight
            loss += l_loss
            # torch.exp(torch.tensor(-l))
            start_idx = end_idx
        return loss

class LogLoss(nn.Module):
    def __init__(self, lmax):
        super(LogLoss, self).__init__()
        self.lmax = lmax

    def forward(self, input, target):
        input_log = torch.log(input / 10)
        target_log = torch.log(target / 10)
        return torch.nn.functional.mse_loss(input, target)
        # loss = 0
        # start_idx = 0
        # for l in range(self.lmax):
        #     end_idx = start_idx + 2 * l + 1
        #     loss += torch.nn.functional.mse_loss(input_log[..., start_idx:end_idx], target_log[..., start_idx:end_idx])
        #     start_idx = end_idx
        # return loss

class GridLoss(nn.Module):
    def __init__(self, lmax, res=100):
        super(GridLoss, self).__init__()

        res_beta = 100
        res_alpha = 51
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

    def forward(self, input, target):
        input_s2grid = self.m(input)
        target_s2grid = self.m(target)
        # return torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="sum")


        # Sum over all the grid points, mean over the batch
        # loss = torch.nn.functional.l1_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        loss = torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        # loss = loss / target_s2grid  # Normalize by distance
        loss = torch.sum(loss, dim=(1, 2, 3))  # B
        loss = torch.mean(loss)
        return loss

class WeightedGridLoss(nn.Module):
    def __init__(self, lmax, res=100):
        super(WeightedGridLoss, self).__init__()

        res_beta = 100
        res_alpha = 51
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

    def forward(self, input, target):
        input_s2grid = self.m(input)
        target_s2grid = self.m(target)
        # return torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="sum")


        # Sum over all the grid points, mean over the batch
        # loss = torch.nn.functional.l1_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha

        # s2grid is the distance of each point from the center of the sphere.
        # We want the density of rays to be constant for a given shell of the sphere
        # The area of the sphere is 4 * pi * r^2
        # So we multiply each loss value by r^2

        loss = torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        loss = loss / target_s2grid  # Normalize by distance
        # weight = target_s2grid**2
        # loss *= weight
        # loss = torch.sum(loss, dim=(1, 2, 3))  # B
        loss = torch.mean(loss)
        return loss


class WeightedGridLossWithRotation(nn.Module):
    def __init__(self, lmax, res=100):
        super(WeightedGridLossWithRotation, self).__init__()

        res_beta = 20
        res_alpha = 41
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)

    def forward(self, input, target):

        random_axis = torch.randn((3,))
        random_axis /= torch.linalg.norm(random_axis)
        random_angle = torch.rand((1,)) * 2 * math.pi



        rotation_matrix = self.irreps.D_from_axis_angle(random_axis, random_angle[0]).to(input.device)

        input_s2grid = self.m(torch.einsum("ij, ...j->...i", rotation_matrix, input))
        target_s2grid = self.m(torch.einsum("ij, ...j->...i", rotation_matrix, target))
        # return torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="sum")


        # Sum over all the grid points, mean over the batch
        # loss = torch.nn.functional.l1_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha

        # s2grid is the distance of each point from the center of the sphere.
        # We want the density of rays to be constant for a given shell of the sphere
        # The area of the sphere is 4 * pi * r^2
        # So we multiply each loss value by r^2

        # loss = torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        loss = torch.nn.functional.l1_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        weight = target_s2grid**2
        loss *= weight
        loss = torch.sum(loss, dim=(1, 2, 3))  # B
        loss = torch.mean(loss)
        return loss


class WeightedPointLoss(nn.Module):
    def __init__(self, lmax, n_points=1, topk=10):
        super(WeightedPointLoss, self).__init__()

        normalization = "integral"
        # self.
        # self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)
        self.sphTen = e3nn.io.SphericalTensor(lmax, 1, 1)
        self.n_points = n_points
        self.topk = topk
        self.scale = 10

    def forward(self, input, target):

        random_vector = torch.randn((self.n_points, 3), device=input.device)
        random_vector /= torch.linalg.norm(random_vector, dim=1, keepdim=True)
        # print(random_vector.shape)

        input_points = self.sphTen.signal_xyz(input, random_vector)
        target_points = self.sphTen.signal_xyz(target, random_vector)

        # difference = input_points - target_points
        # loss = torch.where(difference > 0, difference, -self.scale * difference)
        loss = torch.nn.functional.huber_loss(input_points, target_points, reduction="none")
        loss /= target_points


        # loss = torch.nn.functional.l1_loss(input_points, target_points, reduction="none")  # B x 1 x n_points
        # # print("Loss shape: ", loss.shape)

        # weight = target_points**2

        # loss = loss / target_points  # B x n_points  Percent error for each point

        # loss *= weight

        # loss = loss.reshape(-1, self.n_points)
        # return torch.mean(loss)


        # loss = torch.sum(loss, dim=(1,)) / self.n_points # B

        loss = loss.reshape(-1, self.n_points)
        loss = torch.topk(loss, k=self.topk, dim=1).values / self.topk
        # loss = torch.max(loss, dim=1).values
        # print(loss.shape)
        loss = torch.mean(loss)
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, lmax, n_points=1):
        super(WeightedPointLoss, self).__init__()

        normalization = "integral"

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)
        self.sphTen = e3nn.io.SphericalTensor(lmax, 1, 1)
        self.n_points = n_points
        self.topk = topk
        self.scale = 10

    def forward(self, input, target):

        random_vector = torch.randn((self.n_points, 3), device=input.device)
        random_vector /= torch.linalg.norm(random_vector, dim=1, keepdim=True)
        # print(random_vector.shape)

        input_points = self.sphTen.signal_xyz(input, random_vector)
        target_points = self.sphTen.signal_xyz(target, random_vector)

        # difference = input_points - target_points
        # loss = torch.where(difference > 0, difference, -self.scale * difference)
        
        b = input.shape[0]
        assert input_points.shape == (b,)

        input_points = input_points.abs()
        target_points = target_points.abs()
        mx = torch.max(input_points, target_points)
        mn = torch.min(input_points, target_points)
        # are all the evaluated points positive?
        return mx.sum() - mn.sum()