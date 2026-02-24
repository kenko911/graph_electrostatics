# `graph_longrange`

`graph_longrange` provides blah in pytorch, for use in X. The repo works mostly with Gaussian type orbitals, and provides the following fundamental operations. All the code is pytorch and for systems up to a few thousand atoms on a single GPU, the code is reasonably fast compared to many modern MLIPs.

- [x] **Coulomb Energy** from a set of atoms with **atomic multipole moments**
- [x] **Atom Centered Electrostatic Features**, similar to the LODE descriptor 
- [x] Damped Coulomb Interactions via **Gaussian Type Orbitals**, instead of point charges
- [x] Consistent **Realspace and Periodic** Implementations
- [x] Correction terms to energy and features for handling **slab geometries**, as well as molecules in boxes
- [x] Efficient **batching** for pytorch training
- [x] Considerable **precopmutation** for electrostatic features to speed up self-consistent-field loops
- [x] Interpolation functions for probing the potential and density around atomistic systems

Additional functionality will also be coming in the near future, including:

- [ ] Confined Jellium slabs of charge, useful when simulating electrochemical interfaces

This repo currently supports several long-range MLIP architectures including the MACE-POLAR-1 foundation model \cite{macepolar1}.

## Atomic Multipoles and Gaussian Type Orbitals

Take a collection of $N$ atoms with positions $`\{\mathbf{x}_i\}_i^N`$, which have associtated electric multipole moments in spherical notation $`\{p_{ilm}\}_{ilm}`$. This repo assumes that these correspond to a smooth charge density

$$\rho(\mathbf{x}) = \sum_{i} \sum_{lm} p_{ilm} \phi_{lm}(\mathbf{x}-\mathbf{x}_i)$$

$`\phi_{lm}`$ is a basis function which in this repo is always a Gaussian Type Orbital. We provide efficient batched functions for the electrostatic energy in realspace and periodic boundary conditions:

$$E = \frac{1}{2} \iint \frac{\rho(\mathbf{x})\rho(\mathbf{x}')}{4\pi\epsilon_0|\mathbf{x}-\mathbf{x}'|}d\mathbf{x}d\mathbf{x}'$$

The repo also provides functions for electrostatic features, such as:

$$v_{inlm} = \int \phi_{nlm}(\mathbf{x}-\mathbf{x}') v(\mathbf{x}) d\mathbf{x}$$

Where $`v(\mathbf{x})`$ is the potential coming from a $`\rho`$ like that above.

## Important Conventions

The choice Normalization conditions for the Fourier series coefficients are all stated in `docs/`. Importantly, when using top level functions, the reciprocal cell (often `rcell`) is defined as `rcell = 2 * pi * torch.linalg.inv(cell.T)`, and **not** just `torch.linalg.inv(cell)`.


