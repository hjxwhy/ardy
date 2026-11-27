"""Train a Human24 tokenizer with python -m or torchrun -m ardy.human24.train."""

import argparse
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .model import Human24FSQVAE, TokenizerConfig


def load_split(cache, name, device, rows=slice(None)):
    # Transfer in blocks to avoid a second full-size copy in host memory.
    array = np.load(cache / f"{name}.npy", mmap_mode="c")[rows]
    data = torch.empty(array.shape, dtype=torch.float32, device=device)
    for i in range(0, len(array), 2048):
        data[i:i + 2048].copy_(torch.from_numpy(array[i:i + 2048]))
    lengths = np.load(cache / f"{name}_spans.npy", mmap_mode="c")[rows, 1]
    return data, torch.as_tensor(lengths, device=device)


def valid_frames(target, lengths):
    return torch.arange(target.shape[1], device=target.device)[None, :] < lengths[:, None]


@torch.no_grad()
def set_normalization(model, data, lengths, seed, distributed=False):
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data), generator=rng)[:8192]
    sums = torch.zeros(218, device=data.device, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    frames = 0
    for ids in indices.split(128):
        x = data[ids]
        values = x[valid_frames(x, lengths[ids])].double()
        sums += values.sum((0,))
        squares += values.square().sum((0,))
        frames += len(values)
    if distributed:
        count = torch.tensor(frames, dtype=sums.dtype, device=data.device)
        for value in (sums, squares, count):
            dist.all_reduce(value)
        frames = count
    mean = sums / frames
    model.mean.copy_(mean)
    model.std.copy_((squares / frames - mean.square()).clamp_min(1e-4).sqrt())


def reconstruction_loss(raw, target, valid, model):
    normalized = (target - model.mean) / model.std
    errors = F.smooth_l1_loss(raw, normalized, reduction="none")
    contacts = F.binary_cross_entropy_with_logits(raw[..., 213:217], target[..., 213:217], reduction="none")
    per_frame = (errors[..., :69].mean(-1) + errors[..., 69:213].mean(-1)
                 + errors[..., 217] + contacts.mean(-1))
    return (per_frame * valid).sum() / valid.sum()


class TrainingPass(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x, valid):
        raw = self.tokenizer.decode_raw(self.tokenizer.encode(x))[:, :x.shape[1]]
        return reconstruction_loss(raw, x, valid, self.tokenizer)


def rotation_matrix(x):
    columns = x.reshape(*x.shape[:-1], 24, 2, 3)
    a, b = columns.unbind(-2)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack((a, b, torch.linalg.cross(a, b)), dim=-1)


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(model, data, lengths, batch):
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=data.device)
    for x, lens in zip(data.split(batch), lengths.split(batch)):
        valid = valid_frames(x, lens)
        with precision(data.device):
            raw = model.decode_raw(model.encode(x))[:, :x.shape[1]]
        pred = model.unpack(raw)
        loss = reconstruction_loss(raw, x, valid, model)
        p, t = pred[valid], x[valid]
        pos = (p[:, :69] - t[:, :69]).reshape(-1, 23, 3).norm(dim=-1)
        a, b = (rotation_matrix(y[:, 69:213]) for y in (p, t))
        trace = (a * b).sum((-2, -1))
        angles = torch.rad2deg(((trace - 1) / 2).clamp(-1, 1).acos())
        pred_contact = p[:, 213:217] >= 0.5
        contact = t[:, 213:217] == 1
        n = valid.sum()
        totals += torch.stack((loss * n, pos.sum() / 23, angles.sum() / 24,
                               (p[:, 217] - t[:, 217]).abs().sum(), (pred_contact & contact).sum(),
                               pred_contact.sum(), contact.sum(), n))
    loss, pos, angle, height, tp, predicted, actual, n = totals.tolist()
    return {"loss": loss / n, "mpjpe_cm": 100 * pos / n, "rotation_deg": angle / n,
            "root_y_mae_cm": 100 * height / n, "contact_f1": 2 * tp / max(1, predicted + actual),
            "valid_frames": int(n)}


def save_atomic(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        try:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)



def epoch_batches(count, batch, world, rank, steps, seed, sharded):
    # For a resident shard the indices are local and every row is used before padding.
    gen = torch.Generator().manual_seed(seed + (rank if sharded else 0))
    permutation = torch.randperm(count, generator=gen)
    groups = 1 if sharded else world
    total = steps * groups * batch
    permutation = permutation.repeat(math.ceil(total / count))[:total]
    return permutation.reshape(steps, groups, batch)[:, 0 if sharded else rank]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128, help="Windows per device per update")
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-from", type=Path, help="Start a new schedule from saved tokenizer weights and stats")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    last_path = args.output / "last.pt"
    log_path = args.output / "metrics.jsonl"
    if not args.resume and (last_path.exists() or log_path.exists()):
        raise FileExistsError("Output already contains a run. Use --resume or a new output directory.")
    train_spec = manifest["splits"]["train"]
    global_count = train_spec["windows"]
    sharded = train_spec.get("sampling") == "all_frames" and world > 1
    rows = slice(rank * global_count // world, (rank + 1) * global_count // world) if sharded else slice(None)
    train, lengths = load_split(args.cache, "train", device, rows)
    if rank == 0:
        val, val_lengths = load_split(args.cache, "val", device)
    init = None
    if args.init_from:
        model = Human24FSQVAE.from_checkpoint(args.init_from, device)
        cfg = model.config
        init = {"path": str(args.init_from.resolve()), "sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest()}
    else:
        cfg = TokenizerConfig()
        model = Human24FSQVAE(cfg).to(device)
        if not args.resume:
            set_normalization(model, train, lengths, args.seed, distributed=sharded)
    wrapper = TrainingPass(model)
    net = DistributedDataParallel(wrapper, device_ids=[local_rank] if device.type == "cuda" else None,
                                  broadcast_buffers=False) if world > 1 else wrapper
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device.type == "cuda")
    steps = math.ceil(global_count / (args.batch * world))
    setup = {"config": asdict(cfg), "data": manifest, "epochs": args.epochs, "seed": args.seed,
             "batch_per_device": args.batch, "world_size": world, "peak_lr": args.lr,
             "steps_per_epoch": steps, "parameters": sum(p.numel() for p in model.parameters())}
    if init is not None:
        setup["initial_checkpoint"] = init
    best_loss = float("inf")
    start_epoch = 0
    if args.resume:
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        previous_setup = dict(previous["setup"])
        previous_setup["config"] = asdict(TokenizerConfig(**previous_setup["config"]))
        if previous_setup != setup:
            raise ValueError("Resume requires the same data and training configuration")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start_epoch, best_loss = previous["epoch"], previous["best_loss"]
        del previous

    def log(record):
        print(json.dumps(record), flush=True)
        with log_path.open("a") as f:
            print(json.dumps(record), file=f)

    if rank == 0 and start_epoch == 0:
        (args.output / "run.json").write_text(json.dumps(setup, indent=2))
        log({"epoch": 0, "step": 0, "val": evaluate(model, val, val_lengths, args.batch)})
    total_steps = args.epochs * steps
    warmup = min(200, max(1, total_steps // 10))
    for epoch in range(start_epoch, args.epochs):
        if world > 1:
            dist.barrier()
        net.train()
        batches = epoch_batches(len(train), args.batch, world, rank, steps, args.seed + epoch, sharded).to(device)
        running = torch.zeros((), device=device)
        for offset, ids in enumerate(batches):
            step = epoch * steps + offset + 1
            progress = (step - warmup) / max(1, total_steps - warmup)
            scale = step / warmup if step <= warmup else 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            optimizer.param_groups[0]["lr"] = args.lr * scale
            x = train[ids]
            optimizer.zero_grad(set_to_none=True)
            with precision(device):
                loss = net(x, valid_frames(x, lengths[ids]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += loss.detach()
            if rank == 0 and step % 200 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}", flush=True)
        if world > 1:
            dist.reduce(running, dst=0)
        if rank == 0:
            metrics = evaluate(model, val, val_lengths, args.batch)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("Non-finite validation metrics")
            record = {"epoch": epoch + 1, "step": step, "train_loss": running.item() / (steps * world), "val": metrics}
            payload = {"config": asdict(cfg), "model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_atomic(args.output / "best.pt", payload)
            save_atomic(last_path, {**payload, "optimizer": optimizer.state_dict(), "setup": setup, "best_loss": best_loss})
            log(record)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
