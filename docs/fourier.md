# Definitions used in Fourier Transforms and Series

## Fourier Transforms and Series
The fourier transform and inverse will be defined as
$$\tilde{\phi} = \int_{\mathbb{R}^3} e^{-ik\cdot x} \phi(x)dx$$
$$\phi = \frac{1}{(2\pi)^3}\int_{\mathbb{R}^3} e^{-ik\cdot x} \tilde{\phi}(k)dk$$

For a periodic signal, we instead get fourier series: 
$$\tilde{f} = \int_{\mathbb{R}^3} e^{-ik\cdot x} f(x)dx$$
$$= \frac{(2\pi)^3}{\Omega} \sum_{p \in \Lambda^\star}\delta({k-p}) \int_{\Omega} e^{-ik\cdot x} f(x)dx$$
(see, e.g. David Tong's AQM notes). We define the fourier series reconstruction as 
$$f(x) = \frac{1}{(2\pi)^3}\sum_{k \in \Lambda^\star} e^{ik\cdot x} \tilde{f}(k)$$ with the fourier series coeficients $$\tilde{f}(k) := \frac{(2\pi)^3}{\Omega} \int_{\Omega} e^{-ik\cdot x} f(x)dx$$
This equation is the definition of the fourier series terms which describe densities in the code. 


## Real Signal Fourier Reconstuction
for real signals $$\tilde{f}(k) = \tilde{f}(-k)^\star$$
Therefore, the reconstruction theorem can be simplified 
$$f(x) = \frac{1}{(2\pi)^3}\sum_{k \in \Lambda^\star} e^{ik\cdot x} \tilde{f}(k)$$
$$\rho(r) = \frac{2}{(2\pi)^3}\sum_{k^+}\left[ Re\{\tilde{\rho}\} \cos(k\cdot x) - Im\{\tilde{\rho}\} \sin(k\cdot x) \right] + \frac{1}{(2\pi)^3}\sum_{k=\mathbf{0}}\left[ Re\{\tilde{\rho}\} \cos(k\cdot x) - Im\{\tilde{\rho}\} \sin(k\cdot x) \right]$$
where $k^+$ means sum over all 'positive' k-vectors, in the sense that every k-vector $k$ has an opposite one ($-k$), so just take one of these two. In practice we take the positive $x$ hemisphere. In these docs, I will not write the $k=0$ term (which has a different prefactor) and will just write $\sum_{k_x>0}$ to mean the equation above. 

Hence:
$$\rho(r) = \frac{1}{(2\pi)^3}\sum_{k_x > 0} 2Re\{\tilde{\rho}\} \cos(k\cdot x) - 2 Im\{\tilde{\rho}\} \sin(k\cdot x)$$


## Fourier Series Parsevals Theorem 
For these definitions, it is:
$$\int_\Omega f(x)^\star g(x)dx = \frac{\Omega}{(2\pi)^6} \sum_{k \in \Lambda^\star} \tilde{f}^\star \tilde{g} $$
for real signals we can simplify
$$=\frac{2\Omega}{(2\pi)^6} \sum_{k_x > 0} \left(Re(\tilde{f}^\star)Re(\tilde{g}) - Im(\tilde{f}^\star)Im(\tilde{g}) \right)$$
$$=\frac{2\Omega}{(2\pi)^6} \sum_{k_x > 0} \left(Re(\tilde{f})Re(\tilde{g}) + Im(\tilde{f})Im(\tilde{g}) \right)$$


## Fourier Transform a Lattice of Functions
let $f(x) = \sum_{r \in \Lambda} g(x-r) = g(x) \star \sum_{r \in \Lambda} \delta(x-r)$. Then $\mathcal{F}(f) = \tilde{g} \times \mathcal{F}(\sum_{r \in \Lambda} \delta(x-r)) = \tilde{g} \times \sum_{r \in \Lambda} e^{-ik\cdot r}) = \tilde{g} \frac{(2\pi)^3}{\Omega} \sum_{p \in \Lambda^\star}\delta({k-p})$. 
In short: 
$$\mathcal{F}(\sum_{r \in \Lambda} g(x-r)) = \tilde{g} \ \frac{(2\pi)^3}{\Omega} \sum_{p \in \Lambda^\star}\delta({k-p})$$
In other words, the fourier series coeficients are 
$$\tilde{f}(k)=\tilde{g} \ \frac{(2\pi)^3}{\Omega}$$
the $2\pi$'s cancel in the reconstruction theorem: 
$$f(x) = \frac{1}{\Omega} \sum_{k \in \Lambda^\star} \tilde{g}(k) e^{ik\cdot x}$$
For real signals, we find that the reconstuction theorem above can be applied again, except that we have a new factor of $(2\pi)^3/\Omega$.



## Structure Factor
For a lot of more general long range 'stuff', the key object is this:

$$S_{\eta}(\mathbf{k}) = \sum_{i} e^{-i\mathbf{k}\cdot \mathbf{r}_i} a_{i\eta}$$

$$Re(S_{\eta}) = \sum_i \cos(k\cdot x_i) a_{i\eta}$$

$$Im(S_{\eta}) = - \sum_i \sin(k\cdot x_i) a_{i\eta}$$


