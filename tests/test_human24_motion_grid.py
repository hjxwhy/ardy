from dataclasses import asdict

import torch

from ardy.human24.model import Human24FSQVAE, TokenizerConfig


def test_motion_grid_keeps_coarse_codes_and_bounds_joint_displacement(tmp_path):
    torch.manual_seed(927)
    model = Human24FSQVAE(TokenizerConfig(width=16, heads=2, layers=1, code_dim=8,
                                          motion_grid=True, grid_position_weight=2.5))
    grid = model.motion_grid
    x = torch.randn(16, 13, 218)
    valid = torch.arange(13)[None, :] < torch.arange(6, 22)[:, None]
    grid.fit(model, [(x, valid, None)])
    x[~valid] = 1e6
    center = grid.center.clone()
    grid.fit(model, [(x, valid, None)])
    torch.testing.assert_close(grid.center, center)
    x = torch.randn(8, 13, 218)
    codes = model.encode(x)
    before = ((codes + 1) * 32).long() // 4
    with torch.no_grad():
        for layer in (model.encoder[-1], model.decoder[-1]):
            layer.weight.normal_(std=20)
            layer.bias.normal_(std=20)
    perturbed = model.encode(x)
    indices = (perturbed + 1) * 32
    torch.testing.assert_close(indices, indices.round(), atol=0, rtol=0)
    assert indices.min() >= 0 and indices.max() <= 63
    torch.testing.assert_close(indices.long() // 4, before)
    y = model.decode(perturbed, length=13)
    coarse = grid.coordinates(perturbed)[0][:, :13]
    offset = y[..., :69] - (coarse[..., :69] * model.std[:69] + model.mean[:69])
    assert offset.unflatten(-1, (23, 3)).norm(dim=-1).max() <= grid.max_pose_residual_m + 1e-6
    torch.testing.assert_close(y, model(x))
    path = tmp_path / "stable.pt"
    torch.save({"config": asdict(model.config), "model": model.state_dict()}, path)
    torch.testing.assert_close(Human24FSQVAE.from_checkpoint(path)(x), y)
