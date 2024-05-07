import os
import matplotlib.pyplot as plt
from e3nn import o3, io, nn
import numpy as np
import torch
import e3nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import plotly
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots
from functools import partial, reduce

from tqdm.notebook import tqdm



class SimpleShapeDataset(Dataset):
    def __init__(self, sample_points=512):
        self.shapes = []

        # just add a cube
        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        data = np.asarray(pcd.points)
        data -= data.mean(axis=0, keepdims=True)
        self.shapes.append(data)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.shapes[0]


class SimpleShapeUniformRayDataset(Dataset):
    def __init__(self, sample_points=512):
        rays = np.random.randn(sample_points, 3)
        self.start = np.zeros_like(rays)
        self.directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        self.ray_tensors = o3d.core.Tensor(np.concatenate([self.start, self.directions], axis=-1), dtype=o3d.core.Dtype.Float32)

        self.shapes = []

        # just add a cube
        mesh = o3d.geometry.TriangleMesh.create_box()
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        data = np.asarray(pcd.points)
        center = data.mean(axis=0)
        mesh = mesh.translate(-center)

        self.shapes.append(self.ray_casting(mesh))

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.shapes[0]

    def ray_casting(self, mesh):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        intersections = scene.cast_rays(self.ray_tensors)
        ts = intersections['t_hit'].numpy()
        return self.start + self.directions * ts[:, None]



def mesh_to_sgrid(res_beta: int, res_alpha: int, lmax: int, mesh: trimesh.Trimesh):
    # Create spherical tensor creator
    fromS2Grid = e3nn.o3.FromS2Grid((res_beta, res_alpha), lmax, normalization="integral")

    grid_coords = fromS2Grid.grid
    grid_coords_flat = grid_coords.reshape(-1, 3)

    # -- Calculate using grid -- #
    ray = trimesh.ray.ray_pyembree
    intersector = ray.RayMeshIntersector(mesh)
    index_tri, index_ray, loc = intersector.intersects_id(ray_origins=10 * grid_coords_flat, ray_directions=-grid_coords_flat,
                            multiple_hits=False, return_locations=True)

    # loc is the location of intersection point in world coordinate frame
    loc = torch.tensor(loc, dtype=torch.float32)
    distances = torch.norm(loc, dim=1)

    signal = distances.reshape(1, res_beta, res_alpha)
    sph_coeff = fromS2Grid(signal)

    return sph_coeff

def mesh_to_sphDeltas(lmax: int, mesh: trimesh.Trimesh, n_samples: int) -> torch.tensor:
    # -- Sample uniformly on the sphere -- #

    vectors = torch.normal(0, 1.0, (n_samples, 3))
    vectors = vectors / torch.linalg.norm(vectors, dim=1, keepdim=True)


    ray = trimesh.ray.ray_pyembree
    intersector = ray.RayMeshIntersector(mesh)
    index_tri, index_ray, loc = intersector.intersects_id(ray_origins=10 * vectors, ray_directions=-vectors,
                            multiple_hits=False, return_locations=True)

    x = torch.tensor(loc, dtype=torch.float32).reshape(-1, 3)

    sphten = e3nn.io.SphericalTensor(lmax, 1, 1)
    sph_coeff = e3nn.o3.spherical_harmonics(sphten, x, normalize=False)

    return sph_coeff


class SimpleShapeGridDataset(Dataset):
    def __init__(self, res_beta=100, res_alpha=51, lmax=11):
        self.shapes = []

        # just add a cube
        box_mesh = trimesh.creation.box([1, 1, 1])
        scoeff= mesh_to_sgrid(res_beta, res_alpha, lmax, box_mesh)
        self.shapes.append(scoeff)

        # self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = self.shapes[0]

        # random_axis = torch.randn((3,))
        # random_axis /= torch.linalg.norm(random_axis)
        # random_angle = torch.rand((1,)) * 2 * math.pi

        # rotation_matrix = self.irreps.D_from_axis_angle(random_axis, random_angle[0]).to(data.device)
        # data = torch.einsum("ij, ...j->...i", rotation_matrix, data)

        return data

def generate_data(n_samples, res_beta=100, res_alpha=51, lmax=11):
    min_size = 0.5
    max_size = 2
    samples = []
    pbar = tqdm(range(n_samples))
    for i in pbar:
        side_lengths = np.random.rand(3) * (max_size - min_size) + min_size

        box_mesh = trimesh.creation.box(side_lengths)
        scoeff = mesh_to_sgrid(res_beta, res_alpha, lmax, box_mesh)
        samples.append(scoeff)

    samples = torch.stack(samples)
    return samples


class BoxesDataset(Dataset):
    def __init__(self, res_beta=100, res_alpha=51, lmax=11, n_samples=2):

        data_fname = f'boxes_n{n_samples}_beta{res_beta}_alpha{res_alpha}_l{lmax}.pt'
        if os.path.exists(data_fname):
            self.data = torch.load(data_fname)
        else:
            print("Generating Data")
            self.data = generate_data(n_samples, res_beta, res_alpha, lmax)
            torch.save(self.data, data_fname)
        self.shapes = []

        # just add a cube
        box_mesh = trimesh.creation.box([1, 1, 1])
        scoeff= mesh_to_sgrid(res_beta, res_alpha, lmax, box_mesh)
        self.shapes.append(scoeff)

        # self.irreps = e3nn.o3.Irreps.spherical_harmonics(lmax, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        # random_axis = torch.randn((3,))
        # random_axis /= torch.linalg.norm(random_axis)
        # random_angle = torch.rand((1,)) * 2 * math.pi

        # rotation_matrix = self.irreps.D_from_axis_angle(random_axis, random_angle[0]).to(data.device)
        # data = torch.einsum("ij, ...j->...i", rotation_matrix, data)

        return data
