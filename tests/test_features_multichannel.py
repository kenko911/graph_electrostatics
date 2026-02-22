import torch

from graph_longrange.features import (
    GTOElectrostaticFeatures,
    GTOElectrostaticFeaturesMultiChannel,
)
from graph_longrange.kspace import compute_k_vectors_flat


def test_multichannel_matches_looped_single_channel():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    density_max_l = 1
    feature_max_l = 1
    density_smearing_width = 1.2
    feature_smearing_widths = [0.8, 1.6]
    kspace_cutoff = 4.0

    n_nodes = 5
    batch = torch.zeros(n_nodes, dtype=torch.long)
    node_positions = torch.randn(n_nodes, 3)
    source_feats_base = torch.randn(n_nodes, (density_max_l + 1) ** 2, 2)
    source_feats = source_feats_base.transpose(1, 2)
    assert not source_feats.is_contiguous()

    cell = torch.eye(3).unsqueeze(0) * 8.0
    rcell = torch.inverse(cell)
    volumes = torch.det(cell)
    pbc = torch.tensor([[True, True, True]], dtype=torch.bool)

    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        kspace_cutoff, cell, rcell
    )

    multi = GTOElectrostaticFeaturesMultiChannel(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        include_self_interaction=True,
        kspace_cutoff=kspace_cutoff,
    )
    single = GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        include_self_interaction=True,
        kspace_cutoff=kspace_cutoff,
    )

    multi_out = multi(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=node_positions,
        batch=batch,
        volume=volumes,
        pbc=pbc,
    )

    looped = []
    for channel in range(source_feats.size(1)):
        looped.append(
            single(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                source_feats=source_feats[:, channel : channel + 1],
                node_positions=node_positions,
                batch=batch,
                volume=volumes,
                pbc=pbc,
            )
        )
    looped_out = torch.stack(looped, dim=1)

    torch.testing.assert_close(multi_out, looped_out, rtol=1e-7, atol=1e-9)
