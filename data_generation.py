import matplotlib.pyplot as plt
from e3nn import o3, io, nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import partial, reduce


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
