import math
import e3nn
import torch.nn as nn
import torch


class WeightedLoss(nn.Module):
    """Loss on the spherical harmonic coefficents.  Does not work well"""

    def __init__(self, lmax):
        super(WeightedLoss, self).__init__()
        self.lmax = lmax

    def forward(self, input, target):
        loss = 0
        start_idx = 0
        for l in range(self.lmax):
            end_idx = start_idx + 2 * l + 1
            weight = 1 / torch.mean(target[..., start_idx:end_idx])
            l_loss = (
                torch.nn.functional.mse_loss(
                    input[..., start_idx:end_idx], target[..., start_idx:end_idx]
                )
                * weight
            )
            loss += l_loss
            # torch.exp(torch.tensor(-l))
            start_idx = end_idx
        return loss


class GridLoss(nn.Module):
    """Project spherical harmonics to S2 Grid and compute loss on the grid."""

    def __init__(self, lmax, res=100):
        super(GridLoss, self).__init__()

        res_beta = 100
        res_alpha = 51
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

    def forward(self, input, target):
        input_s2grid = self.m(input)
        target_s2grid = self.m(target)

        # Sum over all the grid points, mean over the batch
        loss = torch.nn.functional.mse_loss(
            input_s2grid, target_s2grid, reduction="none"
        )  # B x 1 x beta x alpha

        loss = torch.sum(loss, dim=(1, 2, 3))  # B
        loss = torch.mean(loss)
        return loss


class WeightedGridLoss(nn.Module):
    """
    Project spherical harmonics to S2 Grid and compute loss on the grid.
    Weighted by distance from center of sphere.
    """

    def __init__(self, lmax, res=100):
        super(WeightedGridLoss, self).__init__()

        res_beta = 100
        res_alpha = 51
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

    def forward(self, input, target):
        input_s2grid = self.m(input)
        target_s2grid = self.m(target)

        # loss = torch.nn.functional.mse_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        loss = torch.nn.functional.huber_loss(
            input_s2grid, target_s2grid, reduction="none"
        )  # B x 1 x beta x alpha
        loss = loss / target_s2grid  # Normalize by distance
        loss = torch.mean(loss)
        return loss


class WeightedGridLossWithRotation(nn.Module):
    """
    Rotate grid to even out the density of rays.
    Weighted by distance from center of sphere.

    Has to be debugged
    """

    def __init__(self, lmax, res=100):
        super(WeightedGridLossWithRotation, self).__init__()

        res_beta = 20
        res_alpha = 41
        normalization = "integral"
        self.m = e3nn.o3.ToS2Grid(lmax, (res_beta, res_alpha), normalization)

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)

    def forward(self, input, target):
        # Compute random rotation by sampling a 3d gaussian
        random_axis = torch.randn((3,))
        random_axis /= torch.linalg.norm(random_axis)
        random_angle = torch.rand((1,)) * 2 * math.pi
        rotation_matrix = self.irreps.D_from_axis_angle(
            random_axis, random_angle[0]
        ).to(input.device)

        input_s2grid = self.m(torch.einsum("ij, ...j->...i", rotation_matrix, input))
        target_s2grid = self.m(torch.einsum("ij, ...j->...i", rotation_matrix, target))

        loss = torch.nn.functional.mse_loss(
            input_s2grid, target_s2grid, reduction="none"
        )  # B x 1 x beta x alpha
        # loss = torch.nn.functional.l1_loss(input_s2grid, target_s2grid, reduction="none")  # B x 1 x beta x alpha
        loss = loss / target_s2grid  # Normalize by distance
        # weight = target_s2grid**2
        # loss *= weight

        # Sum over all the grid points, mean over the batch
        loss = torch.sum(loss, dim=(1, 2, 3))  # B
        loss = torch.mean(loss)
        return loss


class WeightedPointLoss(nn.Module):
    """Compute loss on random points on the sphere.  Weighted by distance from center of sphere."""

    def __init__(self, lmax, n_points=1, topk=10):
        super(WeightedPointLoss, self).__init__()

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)
        self.sphTen = e3nn.io.SphericalTensor(lmax, 1, 1)
        self.n_points = n_points
        self.topk = topk
        self.scale = 10

    def forward(self, input, target):
        random_vector = torch.randn((self.n_points, 3), device=input.device)
        random_vector /= torch.linalg.norm(random_vector, dim=1, keepdim=True)

        input_points = self.sphTen.signal_xyz(input, random_vector)
        target_points = self.sphTen.signal_xyz(target, random_vector)

        loss = torch.nn.functional.huber_loss(
            input_points, target_points, reduction="none"
        )
        loss /= target_points

        # Only keep top k
        loss = loss.reshape(-1, self.n_points)
        loss = torch.topk(loss, k=self.topk, dim=1).values / self.topk
        loss = torch.mean(loss)

        return loss


class IOULoss(nn.Module):
    """Has to be debugged"""

    def __init__(self, lmax, n_points=100):
        super(IOULoss, self).__init__()

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)
        self.sphTen = e3nn.io.SphericalTensor(lmax, 1, 1)
        self.n_points = n_points

    def compute_iou(self, input, target):
        random_vector = torch.randn((self.n_points, 3), device=input.device)
        random_vector /= torch.linalg.norm(random_vector, dim=1, keepdim=True)

        input_points = self.sphTen.signal_xyz(input[:1, :, :], random_vector)
        target_points = self.sphTen.signal_xyz(target[:1, :, :], random_vector)  # b x 1 x n_points

        # # One by one:
        # input_points_idv = []
        # target_points_idv = []
        # for _ in range(len(input)):
        #     input_points_idv.append(self.sphTen.signal_xyz(input[_:_+1, :, :], random_vector))
        #     target_points_idv.append(self.sphTen.signal_xyz(target[_:_+1, :, :], random_vector))

        # input_points_idv = torch.concat(input_points_idv)
        # target_points_idv = torch.concat(target_points_idv)

        # print(torch.max(input_points_idv - input_points))
        # print(torch.max(target_points_idv - target_points))


        # input_points = input_points_idv
        # target_points = target_points_idv

        # Fine below here...

        input_points = input_points[:1, :, :]
        target_points = target_points[:1, :, :]

        input_points = input_points.abs()
        target_points = target_points.abs()

        # mx_idv = []
        # mn_idv = []
        # for i in range(len(input_points)):
        #     mx_idv.append(torch.max(input_points[i], target_points[i]))
        #     mn_idv.append(torch.min(input_points[i], target_points[i]))
        # mx_idv = torch.stack(mx_idv)
        # mn_idv = torch.stack(mn_idv)

        mx = torch.max(input_points, target_points)  # b x 1 x n_points
        mn = torch.min(input_points, target_points)  # b x 1 x n_points

        # Assume equal weight on each point and compute riemann integral for
        # min and max.  Then compute the intersection over union.
        iou = mn.sum(dim=(1, 2)) / mx.sum(dim=(1, 2))

        return iou  # (b, )

    def forward(self, input, target):
        iou = self.compute_iou(input, target)  # between 0 and 1
        # print(iou)
        # https://arxiv.org/pdf/1608.01471
        return (-torch.log(iou)).mean()

class ReconstructionLoss(nn.Module):
    """Has to be debugged"""

    def __init__(self, lmax, n_points=1):
        super(WeightedPointLoss, self).__init__()

        self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)
        self.sphTen = e3nn.io.SphericalTensor(lmax, 1, 1)
        self.n_points = n_points
        self.scale = 10

    def forward(self, input, target):
        random_vector = torch.randn((self.n_points, 3), device=input.device)
        random_vector /= torch.linalg.norm(random_vector, dim=1, keepdim=True)

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
