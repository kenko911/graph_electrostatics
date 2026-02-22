import torch
import pytest
from scipy.constants import pi
import scipy.special
from e3nn.o3 import spherical_harmonics
from graph_longrange.utils import permute_to_e3nn_convention, to_dense_batch

torch.set_default_dtype(torch.float64)

from graph_longrange.kspace import (
    compute_k_vectors,
    FourierReconstructionBlock,
    ParsevalsBlock
)
from graph_longrange.gto_electrostatics import (
    GTOChargeDensityFourierSeriesBlock,
    GTOLocalOrbitalProjectionBlock,
    GTOFourierSeriesCoeficientsBlock,
    gto_basis_kspace_cutoff
)
from graph_longrange.realspace import (
    tensor_realspace_GTO_evaluation,
    tensor_get_grid_cell,
    GTO_no_shift_two_site_overlap,
    GTO_simple_shift_two_site_overlap
)



def get_trial_arguments(include_cell=False):
    cells = [
        1.*torch.eye(3),
        2.5*torch.eye(3),
        1.*torch.eye(3),
        2.5*torch.eye(3),
    ]
    cells[2][0,1] = 0.25
    cells[3][0,1] = 0.5

    max_l = 2

    sigmas = [0.1, 0.25]

    args = []
    for l in range(max_l+1):
        for sigma in sigmas:
            for cell in cells:
                if include_cell:
                    # don't test big sigma small cell
                    if not(sigma > 0.15 and cell[0,0] < 2.0):
                        args.append((l, sigma, 0.05, cell))
                else:
                    args.append((l, sigma))
    return args


trial_args_cell = get_trial_arguments(include_cell=True)
trial_args_nocell = get_trial_arguments()



def get_fourier_reconstruction_of_single_GTO(l, sigma, position, cell, kspace_cutoff, sample_points, normalize='none'):
    """ computes fourier reconstruction of a single GTO """

    assert position.dim() == 1 and position.size()[0] == 3
    assert cell.dim() == 2
    assert sample_points.dim() == 2 and sample_points.size()[1] == 3

    # stuff
    num_angular_channels = (l+1)**2
    
    # blocks
    basis_coefs_block = GTOFourierSeriesCoeficientsBlock([sigma], l, kspace_cutoff=kspace_cutoff, normalize=normalize)
    density_block = GTOChargeDensityFourierSeriesBlock()
    evaluation_block = FourierReconstructionBlock()

    # graph
    batch = torch.tensor([0])
    cell_batch = cell.unsqueeze(0)
    r_cell_batch = 2 * pi * torch.linalg.inv(cell_batch.mT)
    volumes_batch = torch.linalg.det(cell_batch)
    positions_batch = position.unsqueeze(0)

    # samples
    sample_batch = torch.zeros(sample_points.size()[0], dtype=torch.int64)
    batch_sample_points, sample_points_mask = to_dense_batch(sample_points, sample_batch)
    sample_values = torch.zeros((sample_points.size(0), 2*l+1))

    # k grid
    k_vectors, k_vectors_normed_squared, k_mask = compute_k_vectors(kspace_cutoff, cell_batch, r_cell_batch)
    
    # basis coefs
    basis_fs = basis_coefs_block(k_vectors, k_vectors_normed_squared, k_mask)

    for m_index in range(2*l+1):
        charge_coefficients = torch.zeros(1,1,num_angular_channels)
        charge_coefficients[0,0,-(2*l+1)+m_index] = 1.

        density = density_block(
            charge_coefficients,
            positions_batch,
            k_vectors,
            basis_fs,
            volumes_batch,
            batch
        )

        values = evaluation_block(
            k_vectors, 
            density, 
            batch_sample_points, 
            sample_points_mask
        )

        # now return only the requied angular indices
        assert values.dim() == 2 #[n_graph, n_sample]
        assert values.size()[0] == 1

        sample_values[:,m_index] = values[0,:]

    return sample_values


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_fourier_reconstruction_of_GTO(l, sigma, Delta, cell):
    sample_points = torch.rand(100, 3)
    sample_points = torch.matmul(sample_points, cell)

    center = 0.5*torch.sum(cell, axis=0)
    sample_points = sample_points*0.1
    sample_points += center.unsqueeze(0)
    real_space_values = tensor_realspace_GTO_evaluation(
        sample_points, 
        l, 
        sigma, 
        center
    )

    kspace_cutoff = 2.0*gto_basis_kspace_cutoff([sigma], l)
    fourier_recon_values = get_fourier_reconstruction_of_single_GTO(
        l, sigma, center, cell, kspace_cutoff, sample_points
    )

    assert real_space_values.size() == fourier_recon_values.size()
    max_val = torch.max(real_space_values)
    print(max_val)
    print(torch.abs(torch.divide(real_space_values - fourier_recon_values, max_val)))
    assert torch.all(torch.abs(torch.divide(real_space_values - fourier_recon_values, max_val)) < 1e-3)


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_fourier_reconstruction_of_GTO_normalized(l, sigma, Delta, cell):
    sample_points = torch.rand(100, 3)
    sample_points = torch.matmul(sample_points, cell)

    center = 0.5*torch.sum(cell, axis=0)
    sample_points = sample_points*0.1
    sample_points += center.unsqueeze(0)
    real_space_values = tensor_realspace_GTO_evaluation(
        sample_points, 
        l, 
        sigma, 
        center,
        normalize='multipoles'
    )

    kspace_cutoff = 2.0*gto_basis_kspace_cutoff([sigma], l)
    fourier_recon_values = get_fourier_reconstruction_of_single_GTO(
        l, sigma, center, cell, kspace_cutoff, sample_points, normalize='multipoles'
    )

    assert real_space_values.size() == fourier_recon_values.size()
    max_val = torch.max(real_space_values)
    print(max_val)
    print(torch.abs(torch.divide(real_space_values - fourier_recon_values, max_val)))
    assert torch.all(torch.abs(torch.divide(real_space_values - fourier_recon_values, max_val)) < 1e-3)


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_kspace_GTO_square_integral(l, sigma, Delta, cell):
    kspace_cutoff = 1.5*gto_basis_kspace_cutoff([sigma], l)

    # blocks
    density_block = GTOChargeDensityFourierSeriesBlock()
    basis_coefs_block = GTOFourierSeriesCoeficientsBlock([sigma], l, kspace_cutoff=kspace_cutoff)
    product_block = ParsevalsBlock()

    # graph
    position = 0.5*torch.sum(cell, axis=0)
    batch = torch.tensor([0])
    cell_batch = cell.unsqueeze(0)
    r_cell_batch = 2 * pi * torch.linalg.inv(cell_batch.mT)
    volumes_batch = torch.linalg.det(cell_batch)
    positions_batch = position.unsqueeze(0)

    # k grid
    k_vectors, k_vectors_normed_squared, k_mask = compute_k_vectors(kspace_cutoff, cell_batch, r_cell_batch)

    # basis coefs
    basis_fs = basis_coefs_block(k_vectors, k_vectors_normed_squared, k_mask)

    # get integrals
    num_angular_channels = 2*l+1
    integrals = torch.zeros((num_angular_channels, num_angular_channels))
    for m1 in range(2*l+1):
        for m2 in range(2*l+1):
            charge_coefficients = torch.zeros(1,1,(l+1)**2)
            charge_coefficients[0,0,-(2*l+1)+m1] = 1.

            density1 = density_block(
                charge_coefficients,
                positions_batch,
                k_vectors,
                basis_fs,
                volumes_batch,
                batch
            )

            charge_coefficients = torch.zeros(1,1,(l+1)**2)
            charge_coefficients[0,0,-(2*l+1)+m2] = 1.

            density2 = density_block(
                charge_coefficients,
                positions_batch,
                k_vectors,
                basis_fs,
                volumes_batch,
                batch
            )

            integrals[m1, m2] = product_block(
                density1,
                density2,
                volumes_batch
            )

    analytic_square_integral = 0.5 * scipy.special.gamma((2*l + 3) / 2) * sigma**(2*l+3)
    analytic_square_integral_matrix = torch.diag(torch.ones(2*l+1) * analytic_square_integral)

    print(f'normal, {l}, {sigma}, {cell}')
    print(integrals)
    print(analytic_square_integral_matrix)

    assert torch.all(torch.abs(analytic_square_integral_matrix - integrals) < 1e-5)


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_kspace_square_integral_by_projection(l, sigma, Delta, cell):
    kspace_cutoff = 1.5*gto_basis_kspace_cutoff([sigma], l)

    # blocks
    density_block = GTOChargeDensityFourierSeriesBlock()
    basis_coefs_block = GTOFourierSeriesCoeficientsBlock([sigma], l, kspace_cutoff=kspace_cutoff)
    projection_block = GTOLocalOrbitalProjectionBlock()

    # graph
    position = 0.5*torch.sum(cell, axis=0)
    batch = torch.tensor([0])
    cell_batch = cell.unsqueeze(0)
    r_cell_batch = 2 * pi * torch.linalg.inv(cell_batch.mT)
    volumes_batch = torch.linalg.det(cell_batch)
    positions_batch = position.unsqueeze(0)

    # k grid
    k_vectors, k_vectors_normed_squared, k_mask = compute_k_vectors(kspace_cutoff, cell_batch, r_cell_batch)

    # basis coefs
    basis_fs = basis_coefs_block(k_vectors, k_vectors_normed_squared, k_mask)

    # get integrals
    num_angular_channels = 2*l+1
    integrals = torch.zeros((num_angular_channels, num_angular_channels))
    for m1 in range(2*l+1):
        charge_coefficients = torch.zeros(1,1,(l+1)**2)
        charge_coefficients[0,0,-(2*l+1)+m1] = 1.

        density1 = density_block(
            charge_coefficients,
            positions_batch,
            k_vectors,
            basis_fs,
            volumes_batch,
            batch
        )

        projections = projection_block(
            k_vectors,  # [n_graph, max_n_k, 3]
            positions_batch, # [n_nodes, 3]
            density1, # [n_graph, max_n_k, 2]
            batch,
            k_mask,
            basis_fs, # [n_graph, max_n_k, n_sigma, (max_l+1)**2, 2]
        )

        integrals[m1, :] = projections[0,0,-(2*l+1):]

    analytic_square_integral = 0.5 * scipy.special.gamma((2*l + 3) / 2) * sigma**(2*l+3)
    analytic_square_integral_matrix = torch.diag(torch.ones(2*l+1) * analytic_square_integral)

    print(f'projection, {l}, {sigma}, {cell}')
    print(integrals)
    print(analytic_square_integral_matrix)

    assert torch.all(torch.abs(analytic_square_integral_matrix - integrals) < 1e-5)


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_overlap_integrals(l, sigma, Delta, cell):
    if (cell[0,0] > 2.0 and sigma < 0.15):
        # the numerical integrals are too expensive
        return

    """ tests the oerlap between two basis functions, real space grid vs kspace projection """
    kspace_cutoff = 1.5*gto_basis_kspace_cutoff([sigma], l)
    center1 = 0.5*torch.sum(cell, axis=0)
    center2 = center1 + torch.tensor([0.5,0.5,0.5])

    # real space version
    coordinates = tensor_get_grid_cell(Delta, cell)
    overlaps_realspace = GTO_simple_shift_two_site_overlap(coordinates, cell, center1, center2, sigma, sigma, l, l)

    # k-space version
    # blocks
    density_block = GTOChargeDensityFourierSeriesBlock()
    basis_coefs_block = GTOFourierSeriesCoeficientsBlock([sigma], l, kspace_cutoff=kspace_cutoff)
    projection_block = GTOLocalOrbitalProjectionBlock()

    # graph
    batch = torch.tensor([0,0])
    cell_batch = cell.unsqueeze(0)
    r_cell_batch = 2 * pi * torch.linalg.inv(cell_batch.mT)
    volumes_batch = torch.linalg.det(cell_batch)
    positions_batch = torch.vstack((center1, center2))

    # k grid
    k_vectors, k_vectors_normed_squared, k_mask = compute_k_vectors(kspace_cutoff, cell_batch, r_cell_batch)
    print(k_vectors.shape)

    # basis coefs
    basis_fs = basis_coefs_block(k_vectors, k_vectors_normed_squared, k_mask)

    # get integrals
    num_angular_channels = 2*l+1
    kspace_overlaps = torch.zeros((num_angular_channels, num_angular_channels))
    for m1 in range(2*l+1):
        charge_coefficients = torch.zeros(2,1,(l+1)**2)
        charge_coefficients[0,0,-(2*l+1)+m1] = 1.

        density1 = density_block(
            charge_coefficients,
            positions_batch,
            k_vectors,
            basis_fs,
            volumes_batch,
            batch
        )

        projections = projection_block(
            k_vectors,  # [n_graph, max_n_k, 3]
            positions_batch, # [n_nodes, 3]
            density1, # [n_graph, max_n_k, 2]
            batch,
            k_mask,
            basis_fs, # [n_nodes, n_sigma, (max_l+1)**2]
        )

        kspace_overlaps[m1, :] = projections[1,0,-(2*l+1):]

    print(overlaps_realspace*1000)
    print(kspace_overlaps*1000)

    assert torch.all(torch.abs(overlaps_realspace - kspace_overlaps) < 1e-5)


@pytest.mark.parametrize("l, sigma", trial_args_nocell)
def test_GTO_fourier_transform(l, sigma):
    # basically a second implementation
    """ the result is:
    tilde{\phi}_{nlm} = 4\pi C_{l} (-i)^l Y_{lm}(\hat{k}) \int_0^\infty r^{l+2} j_l(kr) e^{-frac{r^2}{2\sigma^2}} dr
    
    where we define:
    f_{l,\sigma}(k)= 4\pi \int_0^\infty r^{l+2} j_l(kr) e^{-\frac{r^2}{2\sigma^2}} dr

    and use
    f_{l,\sigma} =  4\pi \sqrt{\frac{\pi}{2}} \ k^l \sigma^{(3+2l)}F_1\left(\frac{3}{2}+l, \frac{3}{2}+l, -\frac{(k\sigma)^2}{2}\right)
    """
    kspace_cutoff = 0.5/sigma

    # set up the basis
    basis_coefs_block = GTOFourierSeriesCoeficientsBlock([sigma], l, kspace_cutoff=kspace_cutoff)

    # generate some k-vectors
    k_vectors = torch.randn(100,3)
    k_vectors_normed_squared = torch.einsum('ki,ki->k', k_vectors, k_vectors)
    k_vectors_normed = torch.sqrt(k_vectors_normed_squared)
    k_mask = (k_vectors_normed_squared < kspace_cutoff**2)

    # compute all
    basis_fs = basis_coefs_block(k_vectors.unsqueeze(0), k_vectors_normed_squared.unsqueeze(0), k_mask.unsqueeze(0))  # [n_graph, max_n_k, n_sigma, (max_l+1)**2, 2]
    assert basis_fs.size() == torch.Size([1,100,1,(l+1)**2,2])

    # remove unwanted bits
    kcoeffs_graph = basis_fs[0,:,0,-(2*l+1):,:]

    # compute directly. 
    # get f_l,sigma
    prefac = 4*pi*(pi/2)**0.5 * sigma**(3+2*l)
    k_l = torch.pow(k_vectors_normed, l) * k_mask.to(float)
    special_func = scipy.special.hyp1f1(l+1.5, l+1.5, - 0.5 * k_vectors_normed_squared * sigma**2)
    f_l_sigma = prefac * torch.multiply(k_l, special_func)
    assert f_l_sigma.size() == torch.Size([100])

    # spherical harmonics
    Ylmk = spherical_harmonics(l, permute_to_e3nn_convention(k_vectors), normalize=True)
    assert Ylmk.size() == torch.Size([100, 2*l+1])
    
    if l%2 == 0:
        real_part = (-1)**(l//2) * torch.einsum('km,k->km', Ylmk, f_l_sigma)
        imag_part = torch.zeros_like(real_part)
    else:
        imag_part = - (-1)**((l-1)//2) * torch.einsum('km,k->km', Ylmk, f_l_sigma)
        real_part = torch.zeros_like(imag_part)

    assert real_part.size() == torch.Size([100, 2*l+1])
    assert torch.all(torch.abs(real_part - kcoeffs_graph[...,0]) < 1e-8)
    assert torch.all(torch.abs(imag_part - kcoeffs_graph[...,1]) < 1e-8)
