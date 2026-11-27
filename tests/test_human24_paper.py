from math import sqrt

import numpy as np
import pytest
import torch

from ardy.human24.paper_data import FPS, STRIDE

from ardy.human24.paper_train import AdamAtan2, BATCH, TOTAL_STEPS, learning_rate, plan_epoch, skating_terms


def test_schedule_and_bias_corrected_adam_atan2():
    assert learning_rate(1) == 2e-9
    assert learning_rate(10_000) == 2e-5
    assert learning_rate(TOTAL_STEPS) == 0
    assert abs(learning_rate((TOTAL_STEPS + 10_000) // 2) - 1e-5) < 1e-16
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = AdamAtan2([p])
    expected = np.zeros(3)
    m, v = np.zeros(3), np.zeros(3)
    for t, grad in enumerate(([1., -1., 0.], [-2., 3., 0.], [0., -7., 0.]), 1):
        p.grad = torch.tensor(grad, dtype=torch.float64)
        opt.step()
        g = np.asarray(grad)
        m, v = 0.9*m + 0.1*g, 0.99*v + 0.01*g**2
        expected -= 2e-5 * 1.27 * np.arctan2(m / (1-0.9**t), np.sqrt(v) / sqrt(1-0.99**t))
        np.testing.assert_allclose(p.detach().numpy(), expected, atol=1e-14)


def test_paper_position_encoding_matches_released_model_with_bf16_projection():
    import runpy
    from pathlib import Path
    from ardy.human24.model import Human24FSQVAE, TokenizerConfig

    # Load the published implementation without importing the text-encoder dependencies.
    source = Path(__file__).resolve().parents[1] / "ardy/model/autoencoder/transformer.py"
    official = runpy.run_path(str(source))["PositionalEncoding"](32, dropout=0, max_len=75)
    h = torch.randn(2, 75, 32).bfloat16()
    cfg = dict(width=32, layers=1, paper_backbone=True)
    current = Human24FSQVAE(TokenizerConfig(**cfg, fp32_position_encoding=True))
    torch.testing.assert_close(current.add_time(h), official(h), rtol=0, atol=0)
    legacy = Human24FSQVAE(TokenizerConfig(**cfg))
    assert legacy.add_time(h).dtype == torch.bfloat16


def test_paper_epoch_coverage_and_padding():
    assert (FPS, STRIDE) == (30, 4)
    offsets = np.array([0, 17, 220, 800, 2000])
    seen = np.zeros(offsets[-1], bool)
    for length, entries in plan_epoch(offsets, 2026, True):
        assert 30 <= length <= 300 and length % 30 == 0 and len(entries) == BATCH
        for start, size in entries:
            assert size <= length and start + size <= offsets[np.searchsorted(offsets, start, side="right")]
            seen[start:start+size] = True
    assert seen.all()


def test_skating_is_world_space_and_ignores_right_padding():
    pred = torch.zeros(1, 4, 218)
    pred[..., 213:217] = 1
    # Pelvis moves by +0.2 m in X and +0.1 m in Y while all feet remain fixed in world space.
    for t in range(3):
        pred[0,t,:69] = torch.tensor([-0.2*t, -0.1*t, 0.]).repeat(23)
        pred[0,t,217] = 0.1*t
    pred[0,3,:69] = 100
    delta = torch.tensor([[[0.2, 0.]]]).expand(1,4,2)
    mask = torch.tensor([[True, True, True, False]])
    value, weight = skating_terms(pred, mask, delta)
    assert weight == 8 and value < 1e-6
    value, _ = skating_terms(pred, mask, delta*0)
    assert abs((value/weight).item() - 6.0) < 1e-5


def test_skating_speed_and_contact_gradients_match_explicit_world_trajectory():
    from ardy.human24.paper_train import global_loss

    rng = torch.Generator().manual_seed(11)
    x = torch.randn(2, 5, 218, generator=rng, dtype=torch.double)
    x[..., 213:217] = x[..., 213:217].sigmoid()
    x.requires_grad_()
    delta = torch.randn(2, 5, 2, dtype=torch.double, generator=rng)
    valid = torch.arange(5)[None] < torch.tensor([[4], [2]])
    skate, count = skating_terms(x, valid, delta)
    rec = x.new_tensor(0.25)
    actual = global_loss(rec, valid, skate, count, 0.001)
    # Reconstruct absolute positions first, independently of the implementation.
    xz = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, :-1].cumsum(1)), dim=1)
    root = torch.stack((xz[..., 0], x[..., 217], xz[..., 1]), -1)
    terms, weights = [], []
    for b, length in enumerate([4, 2]):
        for j, channel in zip((7, 10, 8, 11), range(213, 217)):
            world = root[b, :length] + x[b, :length, 3*(j-1):3*j]
            w = x[b, :length-1, channel]
            terms.append((w * world.diff(dim=0).norm(dim=-1) * 30).sum())
            weights.append(w.sum())
    expected = rec + 0.001 * sum(terms) / sum(weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])


def test_validation_detects_input_independent_codes_and_poses(monkeypatch):
    from types import SimpleNamespace
    from ardy.human24 import paper_train as paper

    x = torch.zeros(2, 8, 218)
    x[1, :, :69:3] = 0.25
    valid = torch.arange(8)[None] < torch.tensor([[8], [5]])
    x[1, 5:, :69] = 1000
    model = paper.Human24FSQVAE(paper.TokenizerConfig(width=16, heads=2, layers=1, code_dim=8))
    corpus = SimpleNamespace(device=x.device, batches=lambda _: [(x, valid, torch.zeros(2, 8, 2))])
    # Vary codes and poses with time only, even though the two inputs differ.
    codes = torch.arange(2).float()[None, :, None].expand(2, 2, 8) / 32
    pose = torch.arange(8).float()[None, :, None].expand_as(x)
    monkeypatch.setattr(model, "encode", lambda _: codes)
    monkeypatch.setattr(model, "decode_raw", lambda _: pose)
    result = paper.validate(model, corpus, [])
    assert result["paired_target_cm"] == 25.0
    assert result["paired_reconstruction_cm"] == result["paired_code_change_pct"] == 0.0


@pytest.mark.parametrize("skating_weight", [0, 0.001, 0.01])
def test_continue_from_weights_keeps_schedule_and_resumes_new_optimizer(monkeypatch, tmp_path, skating_weight):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    from ardy.human24 import paper_train as paper

    cfg = paper.TokenizerConfig(width=16, layers=1, code_dim=8)
    model = paper.Human24FSQVAE(cfg)
    model.mean.fill_(0.2)
    source = tmp_path / "source.pt"
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "step": 30_000}, source)
    out = tmp_path / "output"
    out.mkdir()
    device = torch.device("cpu")
    x = torch.zeros(2, 8, 218)
    batch = (x, torch.ones(2, 8, dtype=torch.bool), torch.zeros(2, 8, 2))
    corpus = SimpleNamespace(device=device, offsets=[0, 8], batches=lambda plans: (batch for _ in plans))
    monkeypatch.setattr(paper, "CONFIG", cfg)
    monkeypatch.setattr(paper, "Corpus", lambda *args: corpus)
    monkeypatch.setattr(paper, "index_clips", lambda *args: ([], {"train": [], "val": []}))
    monkeypatch.setattr(paper, "describe_index", lambda *args: {"data": str(tmp_path)})
    monkeypatch.setattr(paper, "plan_epoch", lambda *args: [None, None])
    if skating_weight == 0:
        def unexpected_skating_gradient(*args):
            raise AssertionError("Disabled skating loss entered the training path")
        monkeypatch.setattr(paper, "skating_terms", unexpected_skating_gradient)
    paper.train(tmp_path, out, False, 47, device, init_from=source, stop_after=30_001, skating_weight=skating_weight)
    state = torch.load(out / "last.pt", weights_only=True)
    assert state["setup"]["skating_weight"] == skating_weight
    metrics = state["metrics"]
    assert metrics["loss"] == pytest.approx(metrics["reconstruction"] + skating_weight * metrics["skate_mps"])
    assert state["optimizer"]["param_groups"][0]["steps"] == 1
    assert state["optimizer"]["param_groups"][0]["lr"] == paper.learning_rate(30_001)
    torch.testing.assert_close(state["model"]["mean"], model.mean)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["resume_step"] == rows[1]["step"] == 30_000
    with pytest.raises(ValueError, match="configuration differs"):
        paper.train(tmp_path, out, True, 47, device, stop_after=30_002, skating_weight=0.11)
    paper.train(tmp_path, out, True, 47, device, stop_after=30_002, skating_weight=skating_weight)
    state = torch.load(out / "last.pt", weights_only=True)
    assert state["step"] == 30_002 and state["optimizer"]["param_groups"][0]["steps"] == 2
