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
