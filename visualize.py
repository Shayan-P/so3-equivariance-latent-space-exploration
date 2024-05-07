import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from e3nn import io, o3


def visualize_points(sphten: io.SphericalTensor, data):
    signal = o3.spherical_harmonics(sphten, x=data, normalize=False).mean(dim=[0, 1])

    fig = make_subplots(rows=1, cols=2, specs=[[{'is_3d': True} for j in range(2)] for i in range(1)])
    # fig.add_trace(go.Scatter3d(x=0.12 * data[0, :, 0], y=0.12 * data[0, :, 1], z=0.12 * data[0, :, 2]), row=1, col=2)
    fig.add_trace(go.Surface(sphten.plotly_surface(signal, radius=True)[0]), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=data[0, :, 0], y=data[0, :, 1], z=data[0, :, 2]), row=1, col=1)
    return fig


def visualize_signal(sphten: io.SphericalTensor, signal):
    return go.Figure(go.Surface(sphten.plotly_surface(signal, radius=True)[0]))
