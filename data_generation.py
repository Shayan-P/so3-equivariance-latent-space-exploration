import os

import e3nn
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from utils import get_data_path, load_data, save_data

# from plotly.subplots import make_subplots
# from functools import partial, reduce
# import plotly
# import plotly.graph_objects as go
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from e3nn import o3, io, nn


def mesh_to_sphTen_by_grid(
    res_beta: int, res_alpha: int, lmax: int, mesh: trimesh.Trimesh
) -> torch.tensor:
    """
    Convert a mesh to the coefficients of a spherical tensor using an sgrid

    Args:
        res_beta (int): Resolution of beta in grid
        res_alpha (int): Resolution of alpha in grid
        lmax (int): max irrep dimension
        mesh (trimesh.Trimesh): input mesh (smaller than radius 10 sphere centered at origin)

    Returns:
        torch.tensor: Spherical tensor coefficients
    """
    # Create spherical tensor creator
    fromS2Grid = e3nn.o3.FromS2Grid(
        (res_beta, res_alpha), lmax, normalization="integral"
    )

    grid_coords = fromS2Grid.grid
    grid_coords_flat = grid_coords.reshape(-1, 3)

    # Calculate using grid
    ray = trimesh.ray.ray_pyembree
    intersector = ray.RayMeshIntersector(mesh)
    index_tri, index_ray, loc = intersector.intersects_id(
        ray_origins=10 * grid_coords_flat,
        ray_directions=-grid_coords_flat,
        multiple_hits=False,
        return_locations=True,
    )

    # loc is the location of intersection point in world coordinate frame
    loc = torch.tensor(loc, dtype=torch.float32)
    distances = torch.norm(loc, dim=1)

    signal = distances.reshape(1, res_beta, res_alpha)
    sph_coeff = fromS2Grid(signal)

    return sph_coeff


def mesh_to_sphten_by_sample(
    lmax: int, mesh: trimesh.Trimesh, n_samples: int
) -> torch.tensor:
    """
    Convert mesh to spherical tensor coefficients using random samples

    Note: Not currently userd due to normalization issues.

    Args:
        lmax (int): max irrep dimension
        mesh (trimesh.Trimesh): input mesh (smaller than radius 10 sphere centered at origin)
        n_samples (int): number of samples

    Returns:
        torch.tensor: Spherical tensor coefficients
    """
    # -- Sample uniformly on the sphere -- #
    vectors = torch.normal(0, 1.0, (n_samples, 3))
    vectors = vectors / torch.linalg.norm(vectors, dim=1, keepdim=True)

    ray = trimesh.ray.ray_pyembree
    intersector = ray.RayMeshIntersector(mesh)
    index_tri, index_ray, loc = intersector.intersects_id(
        ray_origins=10 * vectors,
        ray_directions=-vectors,
        multiple_hits=False,
        return_locations=True,
    )

    x = torch.tensor(loc, dtype=torch.float32).reshape(-1, 3)

    sphten = e3nn.io.SphericalTensor(lmax, 1, 1)
    sph_coeff = e3nn.o3.spherical_harmonics(sphten, x, normalize=False)

    return sph_coeff


class SimpleShapeGridDataset(Dataset):
    def __init__(self, res_beta=100, res_alpha=51, lmax=11):
        """
        Initialize a dataset with a single 1x1x1 cube

        Args:
            res_beta (int, optional): resolution of beta on grid. Defaults to 100.
            res_alpha (int, optional): resolution of alpha on grid. Defaults to 51.
            lmax (int, optional): max irrep dimension. Defaults to 11.
        """
        self.shapes = []

        # just add a cube
        box_mesh = trimesh.creation.box([1, 1, 1])
        scoeff = mesh_to_sphTen_by_grid(res_beta, res_alpha, lmax, box_mesh)
        self.shapes.append(scoeff)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = self.shapes[0]
        return data


class BoxesDataset(Dataset):
    def __init__(self, res_beta=100, res_alpha=51, lmax=11, n_samples=2):
        """
        Initialize a dataset with n_samples of random boxes with side lengths in
        [0.5, 2.0]

        Args:
            res_beta (int, optional): resolution of beta on grid. Defaults to 100.
            res_alpha (int, optional): resolution of alpha on grid. Defaults to 51.
            lmax (int, optional): max irrep dimension. Defaults to 11.
            n_samples (int, optional): Number of boxes in dataset. Defaults to 2.
        """

        data_fname = f"boxes_n{n_samples}_beta{res_beta}_alpha{res_alpha}_l{lmax}"
        if os.path.exists(get_data_path(data_fname)):
            self.data = load_data(data_fname)
        else:
            print("Generating Data")
            self.data = BoxesDataset.generate_data(n_samples, res_beta, res_alpha, lmax)
            save_data(self.data, data_fname)
            # torch.save(self.data, data_fname)
        self.shapes = []

        # just add a cube
        box_mesh = trimesh.creation.box([1, 1, 1])
        scoeff = mesh_to_sphTen_by_grid(res_beta, res_alpha, lmax, box_mesh)
        self.shapes.append(scoeff)

    @staticmethod
    def generate_data(n_samples: int, res_beta: int = 100, res_alpha: int = 51, lmax: int = 11
    ) -> torch.tensor:
        """
        Sample boxes with side lengths between 0.5 and 2

        Args:
            n_samples (int): Number of boxes
            res_beta (int, optional): resolution of beta on grid. Defaults to 100.
            res_alpha (int, optional): resolution of alpha on grid. Defaults to 51.
            lmax (int, optional): max irrep dimension. Defaults to 11.

        Returns:
            _type_: _description_
        """
        min_size = 0.5
        max_size = 2
        samples = []
        pbar = tqdm(range(n_samples))
        for i in pbar:
            side_lengths = np.random.rand(3) * (max_size - min_size) + min_size

            box_mesh = trimesh.creation.box(side_lengths)
            scoeff = mesh_to_sphTen_by_grid(res_beta, res_alpha, lmax, box_mesh)
            samples.append(scoeff)

        samples = torch.stack(samples)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
