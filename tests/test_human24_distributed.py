"""Exercise the production DDP loop on a synthetic CPU corpus, including a topology change on resume."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ardy.human24 import paper_train as paper
from ardy.human24.data import split_name


def configure():
    torch.set_num_threads(1)
    paper.CONFIG = paper.TokenizerConfig(width=16, heads=2, layers=1, code_dim=8)
    paper.BATCH = 4
    paper.TOTAL_STEPS = 102
    paper.WARMUP = 2


def distributed_worker(rank, directory, motion_grid):
    configure()
    path = Path(directory)
    dist.init_process_group("gloo", init_method=(path/"rendezvous").as_uri(), world_size=2, rank=rank)
    try:
        if rank:
            def unexpected_write(*args):
                raise AssertionError("A nonzero rank attempted to write a checkpoint")
            paper.save_atomic = unexpected_write
        paper.train(path/"data", path/"distributed", True, 47, torch.device("cpu"), motion_grid=motion_grid)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("motion_grid", [False, True])
def test_distributed_checkpoint_resume(tmp_path, monkeypatch, motion_grid):
    # Preserve module-level production settings in the pytest process.
    for key in ("CONFIG", "BATCH", "TOTAL_STEPS", "WARMUP"):
        monkeypatch.setattr(paper, key, getattr(paper, key))
    configure()
    cache = tmp_path/"data"
    cache.mkdir()
    parents = [-1] + [0] * 23
    (cache/"skeleton.json").write_text(json.dumps({"parents": parents}))
    (cache/"takes").mkdir()
    rng = np.random.default_rng(47)
    counts = {"train": 0, "val": 0}
    i = 0
    while min(counts.values()) < 4:
        path = Path(f"takes/motion_{i}.npz")
        i += 1
        split = split_name(path)
        if counts[split] == 4:
            continue
        n = (17, 21, 45, 60)[counts[split]] * 4
        np.savez(cache/path, fps=120, joints=rng.normal(size=(n,24,3)).astype("f"),
                 root_positions=rng.normal(size=(n,3)).astype("f"), local_rot_quat=rng.normal(size=(n,24,4)))
        counts[split] += 1
    single = tmp_path/"single"
    multi = tmp_path/"distributed"
    single.mkdir()
    multi.mkdir()
    save = paper.save_atomic
    def capture(path, payload):
        save(path, payload)
        if path.name == "last.pt" and payload["step"] == 100:
            legacy = dict(payload["setup"])
            legacy.pop("per_device_batch")
            if not motion_grid:
                legacy.pop("contact_velocity_fps")
                legacy["config"] = {k: v for k, v in legacy["config"].items()
                                    if k not in ("motion_grid", "grid_position_weight")}
            save(multi/"last.pt", {**payload, "setup": legacy})
            (multi/"run.json").write_text(json.dumps(legacy))
    monkeypatch.setattr(paper, "save_atomic", capture)
    paper.train(cache, single, False, 47, torch.device("cpu"), motion_grid=motion_grid)
    # Independently compute the next two updates on one CPU using the combined
    # batch from both ranks, weighting by valid frames and contact probabilities.
    initial = torch.load(multi/"last.pt", weights_only=True)
    ref = paper.Human24FSQVAE(paper.TokenizerConfig(**initial["config"]))
    ref.load_state_dict(initial["model"])
    opt = paper.AdamAtan2(ref.parameters())
    opt.load_state_dict(initial["optimizer"])
    epoch, cursor = initial["epoch"], initial["cursor"]
    pool = ThreadPoolExecutor(2)
    parents, splits = paper.index_clips(cache, pool)
    expected_contact_fps = 30 if motion_grid else 120
    source = paper.Corpus(cache, splits["train"], parents, torch.device("cpu"), pool, expected_contact_fps)
    for step in (101, 102):
        plan = paper.plan_epoch(source.offsets, 47 + epoch, True)
        if cursor == len(plan):
            cursor = 0
            epoch += 1
            plan = paper.plan_epoch(source.offsets, 47 + epoch, True)
        pair = [plan[(cursor + rank) % len(plan)] for rank in (0, 1)]
        entries = np.concatenate([batch[1] for batch in pair])
        assert len(entries) == 2 * paper.BATCH
        x, valid, delta = source.batch((max(batch[0] for batch in pair), entries))
        raw, pred = paper.predict(ref, x)
        rec = paper.reconstruction_loss(raw, x, valid, ref)
        skate, weight = paper.skating_terms(pred, valid, delta)
        opt.zero_grad(set_to_none=True)
        (rec + 0.01 * skate / weight).backward()
        opt.param_groups[0]["lr"] = paper.learning_rate(step)
        opt.step()
        cursor = min(cursor + 2, len(plan))

    pool.shutdown()
    mp.spawn(distributed_worker, args=(str(tmp_path), motion_grid), nprocs=2)
    records = [json.loads(s) for s in (multi/"metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["resume_step"] == 100 and records[0]["per_rank_batch_max"] == paper.BATCH
    assert records[0]["global_batch"] == 2 * paper.BATCH
    assert records[1]["step"] == 102 and "val" in records[1]
    last = torch.load(multi/"last.pt", weights_only=True)
    assert last["setup"]["per_device_batch"] == paper.BATCH
    assert last["setup"]["global_batch"] == 2 * paper.BATCH
    assert last["setup"]["contact_velocity_fps"] == expected_contact_fps
    assert last["step"] == last["optimizer"]["param_groups"][0]["steps"] == 102
    assert last["cursor"] == cursor and last["epoch"] == epoch
    for key, value in ref.state_dict().items():
        torch.testing.assert_close(value, last["model"][key], atol=1e-6, rtol=1e-5)
