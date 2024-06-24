# so3-equivariance-latent-space-exploration


This is our final project for Prof. Tess Smidt's course on Symmetry and its Application to Machine Learning.

Leveraging the spherical harmonic datatype, we present a novel, SO(3) equivariant, auto-encoding architecture to generate 3D assets parameterized by the first l spherical harmonics. Specifically, by building an SO(3) equivariant autoencoder, we separate the rotation components of the latent space from the object-defining information, allowing for more robust object generation. For shape generation, we perform traversals of the latent space using representations of elements of the SO(3) rotation group and the scale of the norm of the irreps. We demonstrate reconstruction and generation on a dataset consisting of rotated boxes.

overview: [https://www.youtube.com/watch?v=BuKRtS3pf70](https://www.youtube.com/watch?v=BuKRtS3pf70)

Run `so3_equivariant_autoencoder.ipynb` notebook to reproduce the results of the [paper](./final_report.pdf).

![poster](./poster.pdf)
