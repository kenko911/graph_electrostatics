# Definitions and Results for GTO basis sets

The local basis looks like: 
$$\phi_{nlm}(x) = C_{l} e^{-\frac{x^2}{2\sigma_n^2}} x^{l} Y_{lm}(x)$$
NOTE: I will always use the $Y_{lm}$ sign convention which is set by e3nn's spherical harmoincs implementation (normalize=True, normalization='integral'), which means that $\int d\hat{r} Y_{lm} Y_{LM} = \delta_{Ll}\delta_{Mn}$. 

NOTE: This means that the *value* of $Y_{00}(x) = 1/\sqrt{4\pi}$.

The value at the origin is $C_l/\sqrt{4 \pi}$

The code currently has $C_l=1$, but a sensible normislation would be $C_l = \frac{1}{\sigma^{2l}}$


## Intergrals and Normalisation
The key expression is 
$$\int_0^\infty e^{-\frac{r^2}{2\sigma^2}} r^{n} dr = 2^\frac{n-1}{2} \Gamma\left(\frac{n+1}{2}\right) \sigma^{n+1} $$
or $$\int_0^\infty e^{-\frac{r^2}{\sigma^2}} r^{n} dr = \frac{1}{2} \Gamma\left(\frac{n+1}{2}\right) \sigma^{n+1} $$
Integrals
$$\int_{R^3} \phi_{nlm}(x) dx = C_{l} \int e^{-\frac{x^2}{2\sigma_n^2}} r^{l+2} dr \int Y_{lm}(\hat{r}) d\hat{r} = C_{l} \ 2^\frac{2l+1}{2} \Gamma\left(\frac{l+3}{2}\right) \sigma^{2l+3} \ \int Y_{lm}(\hat{r}) d\hat{r} $$
$$= C_{l}2^\frac{2l+1}{2} \cdot \Gamma\left(\frac{l+3}{2}\right) \sigma^{2l+3} \cdot \delta_{l0} \sqrt{4\pi} $$

square integrals:
$$\int_{R^3} \phi_{nlm}(x)^2 dx = C_{l}^2 \int e^{-\frac{x^2}{\sigma_n^2}} r^{2l+2} dr \int Y_{lm}(\hat{r})^2 d\hat{r} = C_{l}^2  \int e^{-\frac{x^2}{\sigma_n^2}} r^{2l+2} dr $$
$$= C_{l}^2 \frac{1}{2} \Gamma\left(\frac{2l+3}{2}\right) \sigma^{2l+3} $$
#### Summary
radial integral:
$$C_{l} 2^\frac{2l+1}{2} \Gamma\left(\frac{l+3}{2}\right) \sigma^{2l+3}$$
square 
$$C_{l}^2 \frac{1}{2} \Gamma\left(\frac{2l+3}{2}\right) \sigma^{2l+3}$$
moments 
$$C_l \sqrt{\frac{4\pi}{2l+1}}2^\frac{2l+1}{2} \Gamma\left(\frac{2l+3}{2}\right) \sigma^{2l+3}$$

## Moments of local basis functions
The spherical multipoles are defined (see https://en.wikipedia.org/wiki/Spherical_multipole_moments) by
$$Q_{lm} = \int \rho(x) R_{lm}(x) dx$$
This is consistent with the above definitions. The $LM$ moments of basis function $nlm$ are zero unless $l=L,m=M$. We therefore define the following constants as the basis function multipole moments:
$$\bar{Q}^n_{lm} = \int_{R^3} \phi_{nlm}(x) R_{lm}(x) dx = C_{l} \sqrt{\frac{4\pi}{2l+1}} \int e^{-\frac{x^2}{2\sigma_n^2}} r^{2l+2} dr \int Y_{lm}(\hat{r})^2 d\hat{r}$$
$$= C_{l} \sqrt{\frac{4\pi}{2l+1}} 2^\frac{2l+1}{2} \Gamma\left(\frac{2l+3}{2}\right) \sigma^{2l+3}$$

NOTE: to get the total multipoles of a molecular system, its easiest to sum them directly using these definitions, and not use the fourier representation.
for $l=0$, 
$$= C_{0}2^\frac{1}{2} \sqrt{4\pi} \cdot \Gamma\left(\frac{3}{2}\right) \sigma^{3} = \pi \sqrt{2} \sigma^3$$


## Fourier Transform of a GTO
This can be found in [this paper]

$$\tilde{\phi}_{nlm} = 4\pi C_{l} (-i)^l Y_{lm}(\hat{k}) \int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}} dr$$
$$\tilde{\phi}_{nlm}(k) := C_l (-i)^l Y_{lm}(\hat{k}) f_{l,\sigma}(k)$$
with $f_{l,\sigma}(k)= 4\pi \int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}} dr$ . The real and imaginary parts are:
for even $l$: 
$$Re = C_l (-1)^{l/2} Y_{lm}(\hat{k}) f_{l,\sigma}(k)$$
$$Im = 0$$
for odd $l$: 
$$Re = 0$$
$$Im = - C_l (-1)^{(l-1)/2} Y_{lm}(\hat{k}) f_{l,\sigma}(k)$$

#### The Radial Integral
we need $f_{l,\sigma}(k)= 4\pi \int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}} dr$

therefore,
$$f_{l,\sigma} =  4\pi \sqrt{\frac{\pi}{2}} \ k^l \sigma^{(3+2l)}F_1\left(\frac{3}{2}+l, \frac{3}{2}+l, -\frac{(k\sigma)^2}{2}\right)$$





## Charge Density Fourier Series
When calculating objects like the charge density in fourier space, we will use the true fourier series coeficients. From the section on fourier transform of a lattice of functions, they are

$$\tilde{\rho}(k) = \frac{(2\pi)^3}{\Omega} \sum_{inlm} \rho^i_{nlm} \tilde{\phi}_{nlm}(k) e^{- ik\cdot r_i}$$

Here we need only $k_x > 0$, and we need to compute the real and complex parts seperately

$$Re\{\tilde{\rho}(k)\} = \frac{(2\pi)^3}{\Omega} \sum_{inlm} \rho^i_{nlm} \left( Re\{\tilde{\phi}_{nlm}(k)\}  \cos(k \cdot x_i) + Im\{\tilde{\phi}_{nlm}(k)\}  \sin(k \cdot x_i)\right)$$

$$Im\{\tilde{\rho}(k)\} = \frac{(2\pi)^3}{\Omega} \sum_{inlm} \rho^i_{nlm} \left( Im\{\tilde{\phi}_{nlm}(k)\}  \cos(k \cdot x_i) - Re\{\tilde{\phi}_{nlm}(k)\}  \sin(k \cdot x_i)\right)$$




## Local Orbital Projection
In this section I use $\phi^{repeated}$ to mean a lattice of $\phi_{nlm}$, to avoid broken definitions. $\phi^{repeated}(r) = \sum_{r'\in\Lambda} \phi(r-r')$.
$$v^j_{nlm} = \int_\Omega \rho(r) \phi_{nlm}^{repeated}(r-r_i)dr$$
We can use 
$$\int_\Omega f(r)^\star g(r)dr = \frac{\Omega}{(2\pi)^6} \sum_{k \in \Lambda^\star} \tilde{f}^\star \tilde{g} $$
therefore 
$$\int_\Omega f(r)^\star g(r-r_i)dr = \frac{\Omega}{(2\pi)^6} \sum_{k \in \Lambda^\star} \tilde{f}^\star \tilde{g} \ e^{-ik\cdot r_i}$$
substituing: $f=\rho$ and $g=\phi_{nlm}^{repeated}(r-r_i)$ 
therefore:
$$\tilde{f} = \tilde{\rho}$$ 
and 

$$\tilde{g} = \tilde{\phi}_{nlm} \frac{(2\pi)^3}{\Omega}$$

so 

$$v^j_{nlm} = \frac{\Omega}{(2\pi)^6}\sum_k \tilde{\rho}(k)^\star \frac{(2\pi)^3}{\Omega}\tilde{\phi}_{nlm}(k) e^{-ik\cdot r_j}$$

$$v^j_{nlm} = \frac{1}{(2\pi)^3}\sum_k \tilde{\rho}(k)^\star \tilde{\phi}_{nlm}(k) e^{-ik\cdot r_j}$$

This is essentially the inverse of the density block. For a real density, we get

$$v^j_{nlm} = \frac{2}{(2\pi)^3} \sum_{k_x>0} A_{nlm}(k) \cos(k \cdot r_j) + B_{nlm}(k) \sin(k \cdot r_j)$$

with 

$$A_{nlm}(k) = Re(\tilde{\rho})Re(\tilde{\phi}_{nlm}) + Im(\tilde{\rho})Im(\tilde{\phi}_{nlm})$$

$$B_{nlm}(k) = Re(\tilde{\rho})Im(\tilde{\phi}_{nlm}) - Im(\tilde{\rho})Re(\tilde{\phi}_{nlm})$$
