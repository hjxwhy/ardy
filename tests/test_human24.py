import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from ardy.human24.data import make_chunk, prepare_clip
from ardy.human24.model import Human24FSQVAE, TokenizerConfig


@pytest.mark.parametrize("frames", [1, 3, 8, 10])
def test_contacts_follow_sampled_world_positions_in_training_windows(tmp_path, frames):
    from concurrent.futures import ThreadPoolExecutor
    from ardy.human24.paper_data import Corpus

    # Stays planted on source frames 0 and 1, then moves before frame 4.
    root = np.zeros((10, 3), np.float32)
    root[2:, 0] = 0.02
    joints = np.zeros((10, 24, 3), np.float32)
    joints[:, 10, 0] = -root[:, 0]  # This foot stays planted in world coordinates.
    joints[2:, 11, 0] = -0.017  # Moves 0.003 m: 0.09 m/s at 30 Hz, below threshold.
    joints[:, 8, 1] = 0.2
    root[9, 2] = 1.0  # Past the last sampled frame, so its speed is irrelevant.
    quat = np.zeros((frames, 24, 4), np.float32)
    quat[..., 3] = 1
    name = "contact.npz"
    np.savez(tmp_path/name, joints=joints[:frames], root_positions=root[:frames], local_rot_quat=quat, fps=120)
    parents = [-1] + [0] * 23
    n = (frames + 3) // 4
    with ThreadPoolExecutor(1) as pool:
        corpus = Corpus(tmp_path, [(name, frames)], parents, "cpu", pool)
        x, valid, _ = corpus.batch((n+1, [(0, n)]))
        expected = np.tile([1, 1, 0, 1], (n, 1))
        if n >= 2:
            expected[0] = [0, 1, 0, 1]
        if n == 2:
            expected[-1] = expected[0]  # Last label uses the last sampled interval.
        np.testing.assert_array_equal(x[valid][:, 213:217], expected)
        legacy = Corpus(tmp_path, [(name, frames)], parents, "cpu", pool, 120)
        old, _, _ = legacy.batch((n, [(0, n)]))
        np.testing.assert_array_equal(old[0, 0, 213:217], [1, 1, 0, 1])


def test_overlapping_windows(tmp_path):
    # Distinct 3-D rotations catch swapped composition order and row-major 6D.
    rng = np.random.default_rng(4)
    rotations = Rotation.random(8 * 24, random_state=rng)
    local = rotations.as_matrix().reshape(8, 24, 3, 3)
    joints = rng.normal(size=(8, 24, 3)).astype("f")
    joints[:, 0] = 0
    root = np.zeros((8, 3), dtype="f")
    root[:, 1] = 1
    root[5:, 0] = 2
    feet = [7, 10, 8, 11]
    joints[:, feet] = 0
    joints[:, feet, 1] = [-1, -0.95, -0.8, -0.99]
    joints[7, 11, 2] = 1
    path = tmp_path / "clip.npz"
    np.savez(path, joints=joints, root_positions=root, local_rot_quat=rotations.as_quat().reshape(8, 24, 4), fps=120)
    parents = [-1, *range(23)]
    clip = prepare_clip(path, parents)
    identity = np.tile([1, 0, 0, 0, 1, 0], 24)
    windows = [make_chunk(clip, s, 5) for s in (0, 3)]
    for s, x in zip((0, 3), windows):
        assert x.shape == (5, 218)
        np.testing.assert_array_equal(x[:, :69], joints[s:s + 5, 1:].reshape(5, 69))
        np.testing.assert_allclose(x[0, 69:213], identity, atol=1e-6)
        np.testing.assert_array_equal(x[:, 217], root[s:s + 5, 1])
        for k in range(5):
            g0, gt = np.eye(3), np.eye(3)
            for j in range(24):
                g0 = g0 @ local[s, j]
                gt = gt @ local[s + k, j]
                expected = (g0.T @ gt).T[:2].reshape(6)
                np.testing.assert_allclose(x[k, 69 + 6*j:75 + 6*j], expected, atol=1e-6)
    np.testing.assert_array_equal(windows[0][3:, 213:217], windows[1][:2, 213:217])
    np.testing.assert_array_equal(windows[0][0, 213:217], [1, 1, 0, 1])
    np.testing.assert_array_equal(windows[0][-1, 213:217], [0, 0, 0, 0])
    np.testing.assert_array_equal(windows[1][-1, 213:217], [1, 1, 0, 0])
    with pytest.raises(ValueError):
        make_chunk(clip, 4, 5)


def test_quantized_roundtrip_and_gradient():
    model = Human24FSQVAE(TokenizerConfig(width=32, layers=1, code_dim=8))
    x = torch.randn(2, 7, 218)
    z = model.encode(x)
    assert z.shape == (2, 2, 8)
    torch.testing.assert_close(z * 32, (z * 32).round())
    assert z.min() >= -1 and z.max() <= 31 / 32
    y = model.decode(z, length=7)
    assert y.shape == x.shape
    y.mean().backward()
    assert model.patch_in.weight.grad.norm() > 0


def test_full_frame_windows_and_ddp_epoch_cover_tail():
    import math
    from ardy.human24.data import covering_starts
    from ardy.human24.train import epoch_batches

    for frames in (1, 63, 64, 65, 128, 129, 599):
        hits = np.zeros(frames)
        for start in covering_starts(frames, 64):
            hits[start:start + 64] += 1
        assert hits.min() == 1 and hits.max() <= 2

    count, batch, world = 103, 4, 4
    steps = math.ceil(count / (batch * world))
    for sharded in (True, False):
        for seed in (123, 124):
            seen = []
            for rank in range(world):
                first = rank * count // world if sharded else 0
                end = (rank + 1) * count // world if sharded else count
                ids = epoch_batches(end - first, batch, world, rank, steps, seed, sharded)
                assert ids.shape == (steps, batch)
                seen.extend((ids.flatten() + first).tolist())
            assert set(seen) == set(range(count))
