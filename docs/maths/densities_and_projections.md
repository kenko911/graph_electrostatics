# Densities and Projections

This file derives the expressions used in the code for building the k-space density
and projecting electrostatic features. Conventions for Fourier series are in
`docs/maths/fourier.md`, and the GTO basis is defined in `docs/maths/gto.md`.

## Density from multipoles

Given atomic multipoles $p_{ilm}$ at positions $\mathbf{r}_i$, the periodic density is

$$
\rho(\mathbf{r}) = \sum_{\mathbf{a}\in\Lambda} \sum_{ilm}
 p_{ilm}\,\phi_{nlm}(\mathbf{r}-\mathbf{r}_i-\mathbf{a}).
$$

The lattice sum over $\mathbf{a}$ builds the periodic extension of each local basis
function.

## Density Fourier series coefficients

Using the lattice transform result (see `docs/maths/fourier.md`), the Fourier-series
coefficients are

$$
\tilde{\rho}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega}
\sum_{ilm} p_{ilm}\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot\mathbf{r}_i}.
$$

In code we store only one half-space (typically $\mathbf{k}_x>0$) and work with
real/imaginary parts. Define

$$
C^R_{lm}(\mathbf{k}) = \sum_i p_{ilm}\cos(\mathbf{k}\cdot\mathbf{r}_i),\quad
C^I_{lm}(\mathbf{k}) = \sum_i p_{ilm}\sin(\mathbf{k}\cdot\mathbf{r}_i),
$$

then

$$
\Re\{\tilde{\rho}\} = \frac{(2\pi)^3}{\Omega} \sum_{lm}\left(\Re\{\tilde{\phi}_{nlm}\}\,C^R_{lm} + \Im\{\tilde{\phi}_{nlm}\}\,C^I_{lm}\right),
$$

$$
\Im\{\tilde{\rho}\} = \frac{(2\pi)^3}{\Omega} \sum_{lm}\left(\Im\{\tilde{\phi}_{nlm}\}\,C^R_{lm} - \Re\{\tilde{\phi}_{nlm}\}\,C^I_{lm}\right).
$$

These are the expressions implemented by `assemble_fourier_series_batch`.

## Potential and projection

The electrostatic potential is

$$
v(\mathbf{r}) = \int \frac{\rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}\,d\mathbf{r}',
$$

and in k-space

$$
\tilde{v}(\mathbf{k}) = \frac{4\pi}{\epsilon_0}\frac{\tilde{\rho}(\mathbf{k})}{k^2},
\quad \mathbf{k}\neq\mathbf{0}.
$$

For feature projections we need the periodic extension of the basis function:

$$
\phi_{nlm}^{\mathrm{repeated}}(\mathbf{r})
= \sum_{\mathbf{a}\in\Lambda} \phi_{nlm}(\mathbf{r}-\mathbf{a}).
$$

This “repeated” basis is just the periodic extension of the local orbital. It is used
so the projection integral can be taken over the unit cell while remaining equivalent
to an integral over all space.

The projection is

$$
v^i_{nlm} = \int_\Omega v(\mathbf{r})\,\phi_{nlm}^{\mathrm{repeated}}(\mathbf{r}-\mathbf{r}_i)\,d\mathbf{r}.
$$

Using Parseval and the lattice transform for $\phi_{nlm}^{\mathrm{repeated}}$, we obtain

$$
v^i_{nlm} = \frac{1}{(2\pi)^3}\sum_{\mathbf{k}\in\Lambda^\star} \tilde{v}(\mathbf{k})\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot\mathbf{r}_i}.
$$

For a real potential, the half-space form is

$$
v^i_{nlm} = \frac{2}{(2\pi)^3}\sum_{\mathbf{k}_x>0}\left[A_{nlm}(\mathbf{k})\cos(\mathbf{k}\cdot\mathbf{r}_i) + B_{nlm}(\mathbf{k})\sin(\mathbf{k}\cdot\mathbf{r}_i)\right],
$$

where

$$
A_{nlm} = \Re\{\tilde{v}\}\Re\{\tilde{\phi}_{nlm}\} + \Im\{\tilde{v}\}\Im\{\tilde{\phi}_{nlm}\},
$$

$$
B_{nlm} = \Re\{\tilde{v}\}\Im\{\tilde{\phi}_{nlm}\} - \Im\{\tilde{v}\}\Re\{\tilde{\phi}_{nlm}\}.
$$

These contractions are implemented by `project_to_features_batch`.

## Code mapping summary

- Density coefficients: `assemble_fourier_series_batch`
- Coulomb kernel: `apply_coulomb_kernel_batch`
- Feature projection: `project_to_features_batch`
