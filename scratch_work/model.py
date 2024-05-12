import torch
import torch.nn as nn
import e3nn
from e3nn import o3, io


class SelfTP(nn.Module):
    def __init__(self, irreps_in, irreps_out, copies_in, copies_out, non_linearity=torch.tanh, resolution=100):
        super(SelfTP, self).__init__()

        self.tp = o3.FullyConnectedTensorProduct(irreps_in1=copies_in * irreps_in, irreps_in2=copies_in * irreps_in, irreps_out=copies_out * irreps_out)
        self.irrep_in = irreps_in
        self.irrep_out = irreps_out
        self.act = e3nn.nn.S2Activation(self.irrep_out, non_linearity, resolution)
        self.size_in = len(irreps_in.randn(-1))
        self.size_out = len(irreps_in.randn(-1))
        self.copies_in = copies_in
        self.copies_out = copies_out

    def forward(self, x):
        x = self.tp(x, x)
        x = x.reshape(-1, self.copies_out, self.size_out)
        x = self.act(x)
        x = x.reshape(-1, self.copies_out * self.size_out)
        return x


class NoiseLayer(nn.Module):
    def __init__(self, irreps, std_dev):
        super(NoiseLayer, self).__init__()

        self.noise = lambda x: (1 + torch.normal(0, std_dev, (1,), device=x.device)) * x

        #     1 + torch.randn(1, device=x.device)

        # (torch.rand(1, device=x.device) - 0.5) * max_noise) * x
        self.norm_act = e3nn.nn.NormActivation(irreps, self.noise)
        # self.norm_act = e3nn.nn.NormActivation(irreps, torch.sigmoid)

    def forward(self, x):
        return self.norm_act(x)


class EncoderDecoder(nn.Module):
    def __init__(self, lmax, n_repeats=4, tw_max=4):
        super(EncoderDecoder, self).__init__()

        self.irrep_layer = []
        self.single_irrep_layer = []
        self.counts = []
        tw = 1
        # lmax = 5
        # self.irrep_layer.append(io.SphericalTensor(lmax=lmax, p_arg=1, p_val=1))  # Add a copy of the last layer
        for l in range(lmax, 0, -1):  # last one sould be lmax=1
            # for l in range(lmax, -1, -1): # last one sould be lmax=0
            # for l in range(lmax, 1, -1): # last one sould be lmax=2

            for _ in range(n_repeats):
                rep = o3.Irreps('+'.join([f'1x{ir}e' for ir in range(l+1)]))
                self.single_irrep_layer.append(rep)
                self.irrep_layer.append(tw * rep)
                self.counts.append(tw)
            tw = min(tw * 2, tw_max)

        self.model_sphten_repr = io.SphericalTensor(lmax=lmax, p_arg=1, p_val=1)
        self.latent_repr = self.irrep_layer[-1]

        # todo should the last layer have batch norm?
        # todo use fully connected tensor product?
        self.encoder = nn.Sequential(*[
            SelfTP(irreps_in=self.single_irrep_layer[i], irreps_out=self.single_irrep_layer[i + 1],
                   copies_in=self.counts[i], copies_out=self.counts[i + 1], non_linearity=torch.tanh,
                   resolution=100)
            for i in range(len(self.irrep_layer) - 1)
        ])

        # New
        self.noise_layer = NoiseLayer(self.irrep_layer[-1], 1.0)
        # self.linear = e3nn.o3.Linear(irreps_in=self.irrep_layer[-1], irreps_out=self.irrep_layer[-1])

        # todo is the information bottleneck limited enough?
        self.decoder = nn.Sequential(*[
            SelfTP(irreps_in=self.single_irrep_layer[i+1], irreps_out=self.single_irrep_layer[i],
                   copies_in=self.counts[i+1], copies_out=self.counts[i], non_linearity=torch.tanh,
                   resolution=100)
            for i in range(len(self.irrep_layer) - 2, -1, -1)
        ])

    def forward(self, x):
        # inp = o3.spherical_harmonics(l=self.irrep_layer[0], x=points, normalize=False).mean(dim=1) # todo fix the normalization later
        latent = self.encoder(x)
        # out = self.decoder(latent)
        out = self.decoder(self.noise_layer(latent))
        # return out
        return x, latent, out
