import torch
from graph_longrange.realspace import (
    tensor_get_grid_cell, 
    tensor_realspace_GTO_evaluation, 
    GTO_no_shift_two_site_overlap, 
    GTO_simple_shift_two_site_overlap, 
    realspace_cluster_multipoles
)
from scipy.special import gamma
import numpy as np
from scipy.constants import pi
import pytest


torch.set_default_dtype(torch.float64)


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


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_realspace_GTO_square_integrals(l, sigma, Delta, cell):
    print(f"realspace_GTO_square_integrals: (l, sigma, Delta, cell) = ({l}, {sigma}, {Delta}, {cell})")
    coordinates = tensor_get_grid_cell(Delta, cell)
    phi = tensor_realspace_GTO_evaluation(coordinates, l, sigma, 0.5*torch.sum(cell, axis=0)) # [..., 2l+1]
    square_integrals = torch.zeros((2*l+1,2*l+1))

    for l1 in range(2*l+1):
        for l2 in range(2*l+1):
            values = torch.multiply(phi[...,l1], phi[...,l2])
            integ1 = torch.trapz(values, x=coordinates[...,0], axis=0)
            integ2 = torch.trapz(integ1, x=coordinates[0,:,:,1], axis=0)
            integ3 = torch.trapz(integ2, x=coordinates[0,0,:,2], axis=0)

            square_integrals[l1,l2] = integ3

    assert square_integrals.size() == torch.Size([2*l+1,2*l+1])

    # analytic
    analytic_square_integral = 0.5 * gamma((2*l + 3) / 2) * sigma**(2*l+3)
    analytic_square_integral_matrix = torch.diag(torch.ones(2*l+1) * analytic_square_integral)
    assert square_integrals.size() == torch.Size([2*l+1,2*l+1])
    
    assert torch.all(torch.abs(analytic_square_integral_matrix - square_integrals)/sigma**(2*l+3) < 1e-6)


@pytest.mark.parametrize("l, sigma", trial_args_nocell)
def test_realspace_GTO_origin_values(l, sigma):
    print(f"realspace_GTO_origin_values: (l, sigma) = ({l}, {sigma})")
    eval_points = torch.Tensor([[0.,0.,0.]])
    GTO_center = torch.Tensor([0.,0.,0.])
    origin_value = tensor_realspace_GTO_evaluation(eval_points, l, sigma, GTO_center)
    assert origin_value.size() == torch.Size([1, 2*l+1])
    origin_value = origin_value[0,:]

    analytic_value = torch.zeros_like(origin_value)
    if l==0:
        analytic_value[0] = (4*pi)**(-0.5)
    
    assert torch.all(torch.abs(origin_value - analytic_value) < 1e-10)


@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def test_realspace_GTO_square_integrals_a_different_way(l, sigma, Delta, cell):
    coordinates = tensor_get_grid_cell(Delta, cell)
    center = 0.5*torch.sum(cell, axis=0)
    square_integrals = GTO_no_shift_two_site_overlap(coordinates, center, center, sigma, sigma, l, l)

    assert square_integrals.size() == torch.Size([2*l+1,2*l+1])

    # analytic
    analytic_square_integral = 0.5 * gamma((2*l + 3) / 2) * sigma**(2*l+3)
    analytic_square_integral_matrix = torch.diag(torch.ones(2*l+1) * analytic_square_integral)
    assert square_integrals.size() == torch.Size([2*l+1,2*l+1])
    
    assert torch.all(torch.abs(analytic_square_integral_matrix - square_integrals)/sigma**(2*l+3) < 1e-6)


#@pytest.mark.parametrize("l, sigma, Delta, cell", trial_args_cell)
def t_realspace_GTO_square_integrals_with_shifts(l, sigma, Delta, cell):
    coordinates = tensor_get_grid_cell(Delta, cell)
    #center = 0.5*torch.sum(cell, axis=0)

    for center in [0.0*torch.sum(cell, axis=0), 0.5*torch.sum(cell, axis=0), 1.*torch.sum(cell, axis=0)]:
        square_integrals = GTO_simple_shift_two_site_overlap(coordinates, cell, center, center, sigma, sigma, l, l)
        assert square_integrals.size() == torch.Size([2*l+1,2*l+1])

        # analytic
        analytic_square_integral = 0.5 * gamma((2*l + 3) / 2) * sigma**(2*l+3)
        analytic_square_integral_matrix = torch.diag(torch.ones(2*l+1) * analytic_square_integral)
        assert square_integrals.size() == torch.Size([2*l+1,2*l+1])
        
        assert torch.all(torch.abs(analytic_square_integral_matrix - square_integrals)/sigma**(2*l+3) < 1e-6)


def get_trial_arguments_mpoles():
    cells = [
        1.*torch.eye(3),
        20.*torch.eye(3),
    ]
    max_l = 2
    sigmas = [0.1, 2.0]
    args = [
        (max_l, sigmas[0], 0.01, cells[0]),
        (max_l, sigmas[1], 0.2, cells[1]),
    ]
    return args

trial_args_multipoles = get_trial_arguments_mpoles()


@pytest.mark.parametrize("max_test_l, sigma, Delta, cell", trial_args_multipoles)
def test_orbital_multipoles_normalized(max_test_l, sigma, Delta, cell):
    coordinates = tensor_get_grid_cell(Delta, cell)
    center = 0.5*torch.sum(cell, axis=0)

    multipoles = torch.zeros(((max_test_l+1)**2,(max_test_l+1)**2))

    for l_basis in range(max_test_l + 1):        
        orbital = tensor_realspace_GTO_evaluation(coordinates, l_basis, sigma, center, normalize='multipoles')

        for m in range(2*l_basis+1):
            multipoles[l_basis**2+m, :] = realspace_cluster_multipoles(coordinates, orbital[...,m], center, max_test_l)  # [ (max_test_l+1)**2 ]

    multipoles_analytic = torch.zeros((max_test_l+1)**2)
    for l in range(max_test_l+1):
        multipoles_analytic[l**2:(l+1)**2] = 1.0
    
    multipoles_analytic = torch.diag(multipoles_analytic)

    print(multipoles_analytic)
    print(multipoles)

    assert torch.all(torch.abs(multipoles - multipoles_analytic) < 1e-3)
