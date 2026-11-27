"""Train the 218D Human24 adaptation with the published ARDY tokenizer schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, replace
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.nn.parallel import DistributedDataParallel

from .data import FOOT_JOINTS
from .model import Human24FSQVAE, TokenizerConfig
from .paper_data import Corpus, FPS, STRIDE, describe_index, index_clips
from .train import precision, reconstruction_loss, rotation_matrix, save_atomic


TOTAL_STEPS = 4_000_000
WARMUP = 10_000
PEAK_LR = 2e-5
BATCH = 128
DEFAULT_SKATING_WEIGHT = 0.01
CONFIG = TokenizerConfig(width=512, heads=4, layers=8, code_dim=128, levels=64,
                         paper_backbone=True, fp32_position_encoding=True)


def learning_rate(step):
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    return PEAK_LR * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP))) / 2


class AdamAtan2(torch.optim.Optimizer):
    """Bias-corrected AdamAtan2 with a=1.27, b=1, no weight decay or epsilon."""
    def __init__(self, params):
        super().__init__(params, dict(lr=PEAK_LR, betas=(0.9, 0.99), steps=0))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            group["steps"] += 1
            t = group["steps"]
            params = [p for p in group["params"] if p.grad is not None]
            grads = [p.grad for p in params]
            for p in params:
                if not self.state[p]:
                    self.state[p]["m"] = torch.zeros_like(p)
                    self.state[p]["v"] = torch.zeros_like(p)
            m = [self.state[p]["m"] for p in params]
            v = [self.state[p]["v"] for p in params]
            torch._foreach_lerp_(m, grads, 1 - beta1)
            torch._foreach_mul_(v, beta2)
            torch._foreach_addcmul_(v, grads, grads, value=1-beta2)
            denom = torch._foreach_sqrt(v)
            # atan2(c*m, sqrt(v)) equals atan2(m_hat, sqrt(v_hat)).
            numer = torch._foreach_mul(m, math.sqrt(1-beta2**t) / (1-beta1**t))
            updates = [a.atan2_(b) for a, b in zip(numer, denom)]
            torch._foreach_add_(params, updates, alpha=-1.27 * group["lr"])


def plan_epoch(offsets, seed, training):
    """Tile every clip, shuffle same-length buckets, then shuffle their batches."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(10)]
    for base, end in zip(offsets[:-1], offsets[1:]):
        start = base
        while start < end:
            frames = min(int(end - start), int(rng.integers(1, 11)) * FPS)
            # Shift short tails back into the same clip to form a one-second window.
            if frames < FPS:
                start = max(base, end - FPS)
                frames = int(end - start)
            bucket = (frames - 1) // FPS
            buckets[bucket].append((start, frames))
            start += frames
    batches = []
    for bucket, entries in enumerate(buckets):
        if not entries:
            continue
        entries = np.asarray(entries, dtype=np.int64)
        if training:
            rng.shuffle(entries)
            pad = -len(entries) % BATCH
            if pad:
                entries = np.concatenate((entries, entries[rng.integers(len(entries), size=pad)]))
        for i in range(0, len(entries), BATCH):
            batches.append(((bucket + 1) * FPS, entries[i:i+BATCH]))
    if training:
        rng.shuffle(batches)
    return batches


def foot_speed(pred, planar_delta):
    # Source root XZ is auxiliary supervision. World Y uses the predicted root height.
    feet = pred[..., :69].reshape(*pred.shape[:2], 23, 3)[:, :, [j-1 for j in FOOT_JOINTS]]
    velocity = feet.diff(dim=1)
    height_delta = pred[..., 217].diff(dim=1)
    root_delta = torch.stack((planar_delta[:, :-1, 0], height_delta, planar_delta[:, :-1, 1]), dim=-1)
    return ((velocity + root_delta[:, :, None, :]) * FPS).norm(dim=-1)


def contact_weighted_speed(speed, valid, contacts):
    weights = contacts[:, :-1] * (valid[:, :-1] & valid[:, 1:]).unsqueeze(-1)
    return (weights * speed).sum(), weights.sum()


def skating_terms(pred, valid, planar_delta):
    return contact_weighted_speed(foot_speed(pred, planar_delta), valid, pred[..., 213:217])


class RawPrediction(torch.nn.Module):
    """DDP must see the encoder and decoder together in one forward pass."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.decode_raw(self.model.encode(x))[:, :x.shape[1]]


def global_loss(rec, valid, skate, weight, skating_weight=DEFAULT_SKATING_WEIGHT):
    if not dist.is_initialized():
        return rec + skating_weight * skate / weight.clamp_min(1e-8)
    # Differentiable SUM also propagates gradients through the predicted contact denominator.
    count = valid.sum()
    totals = all_reduce(torch.stack((rec * count, count, skate, weight)))
    return totals[0] / totals[1] + skating_weight * totals[2] / totals[3].clamp_min(1e-8)


def predict(model, x, *, return_codes=False):
    with precision(x.device):
        codes = model.encode(x)
        raw = model.decode_raw(codes)[:, :x.shape[1]]
    pred = model.unpack(raw)
    return (raw, pred, codes) if return_codes else (raw, pred)


@torch.no_grad()
def fit_stats(model, corpus, batches):
    sums = torch.zeros(218, dtype=torch.float64, device=corpus.device)
    squares = torch.zeros_like(sums)
    count = 0
    for x, mask, _ in corpus.batches(batches[:64]):
        values = x[mask].double()
        sums += values.sum(0)
        squares += values.square().sum(0)
        count += len(values)
    mean = sums / count
    model.mean.copy_(mean)
    model.std.copy_((squares / count - mean.square()).clamp_min(1e-4).sqrt())


@torch.inference_mode()
def validate(model, corpus, batches, skating_weight=DEFAULT_SKATING_WEIGHT):
    model.eval()
    total = torch.zeros(12, dtype=torch.float64, device=corpus.device)
    pair_totals = torch.zeros(5, dtype=torch.float64, device=corpus.device)
    for x, valid, delta in corpus.batches(batches):
        raw, pred, codes = predict(model, x, return_codes=True)
        if len(x) > 1:
            # Compare different windows at the same time index. This excludes
            # variation due solely to the shared positional encoding.
            paired = valid & valid.roll(1, 0)
            token_pairs = paired[:, ::model.config.patch]
            pose_distances = []
            for values in (x, pred):
                pose = values[..., :69].reshape(*valid.shape, 23, 3)
                distance = (pose - pose.roll(1, 0)).norm(dim=-1).mean(dim=-1)
                pose_distances.append(distance[paired].sum(dtype=torch.float64))
            bins = (codes * (model.config.levels // 2)).round()
            changed = (bins != bins.roll(1, 0)).sum(dim=-1)
            pair_totals += torch.stack((*pose_distances, paired.sum(),
                                        changed[token_pairs].sum(), token_pairs.sum() * codes.shape[-1]))
        speed = foot_speed(pred, delta)
        skate, contacts = contact_weighted_speed(speed, valid, pred[..., 213:217])
        gt_skate, gt_contacts = contact_weighted_speed(speed, valid, x[..., 213:217])
        p, t = pred[valid], x[valid]
        pos = (p[:, :69] - t[:, :69]).reshape(-1, 23, 3).norm(dim=-1)
        a, b = (rotation_matrix(z[:, 69:213]) for z in (p, t))
        ang = torch.rad2deg((((a*b).sum((-2,-1))-1)/2).clamp(-1,1).acos())
        c, truth = p[:, 213:217] >= 0.5, t[:, 213:217] == 1
        n = valid.sum()
        rec = reconstruction_loss(raw, x, valid, model)
        total += torch.stack((rec*n, skate, contacts, pos.sum()/23, ang.sum()/24,
                              (p[:, 217]-t[:, 217]).abs().sum(), (c & truth).sum(), c.sum(), truth.sum(), n, gt_skate, gt_contacts))
    rec, skate, contacts, pos, ang, height, tp, positives, truth, n, gt_skate, gt_contacts = total.tolist()
    speed = skate / max(contacts, 1e-8)
    metrics = {"loss": rec/n + skating_weight * speed, "reconstruction": rec/n, "skate_mps": speed,
            "skate_gt_contact_mps": gt_skate / max(gt_contacts, 1e-8),
            "mpjpe_cm": 100*pos/n, "rotation_deg": ang/n, "root_y_mae_cm": 100*height/n,
            "contact_f1": 2*tp/max(1,positives+truth), "valid_frame_observations": int(n)}
    target_gap, pred_gap, frames, changed, scalars = pair_totals.tolist()
    if frames:
        metrics.update(paired_target_cm=100 * target_gap / frames,
                       paired_reconstruction_cm=100 * pred_gap / frames,
                       paired_code_change_pct=100 * changed / scalars)
    return metrics


def train(data, output, resume, seed, device, workers=4, *, motion_grid=False, stop_after=None, init_from=None,
          skating_weight=DEFAULT_SKATING_WEIGHT):
    if not math.isfinite(skating_weight) or skating_weight < 0:
        raise ValueError("skating_weight must be finite and nonnegative")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        _train(data, output, resume, seed, device, pool, motion_grid, stop_after, init_from, skating_weight)


def _train(data, output, resume, seed, device, pool, motion_grid, stop_after, init_from, skating_weight):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    leader = rank == 0
    recorded = json.loads((output/"run.json").read_text()) if resume else None
    if recorded is not None and recorded["config"].get("motion_grid", False) != motion_grid:
        raise ValueError(f"The FSQ mode differs from the checkpoint in {output}. Select the matching OUTPUT_DIR.")
    global_batch = world * BATCH
    index = [index_clips(data, pool) if leader else None]
    if world > 1:
        dist.broadcast_object_list(index, src=0, device=device)
    parents, splits = index[0]
    manifest = describe_index(data, parents, splits)
    if leader and not resume and (output/"run.json").exists():
        raise FileExistsError("Existing run; use --resume with its last.pt")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    # Runs written before the rate was recorded used source-frame contact labels.
    contact_fps = recorded.get("contact_velocity_fps", FPS * STRIDE) if resume else FPS
    train_data = Corpus(data, splits["train"], parents, device, pool, contact_fps)
    if leader:
        val_data = Corpus(data, splits["val"], parents, device, pool, contact_fps)
        val_batches = plan_epoch(val_data.offsets, seed, False)
    fp32_positions = (recorded["config"].get("fp32_position_encoding", False)
                      if resume else CONFIG.fp32_position_encoding and not motion_grid)
    config = replace(CONFIG, motion_grid=motion_grid, grid_position_weight=2.5 if motion_grid else 1.0,
                     fp32_position_encoding=fp32_positions)
    if resume and motion_grid:
        config = replace(config, grid_position_weight=recorded["config"].get("grid_position_weight", 1.0))
    model = Human24FSQVAE(config).to(device)
    opt = AdamAtan2(model.parameters())
    setup = {"config": asdict(config), "data": manifest, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
             "total_steps": TOTAL_STEPS, "warmup": WARMUP, "peak_lr": PEAK_LR, "min_lr": 0.0,
             "schedule": "linear warmup then cosine", "global_batch": global_batch, "per_device_batch": BATCH,
             "optimizer": "AdamAtan2", "contact_velocity_fps": contact_fps,
             "betas": [0.9, 0.99], "a": 1.27, "b": 1.0, "weight_decay": 0.0, "skating_weight": skating_weight,
             "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "initialization": "random", "fps": FPS,
             "clip_seconds": [1, 10], "validation_every": 5000, "checkpoint_every": 1000}
    step = epoch = cursor = 0
    best = math.inf
    loaded_source = False
    if resume:
        setup["initialization"] = recorded["initialization"]
        if init_from and str(init_from.resolve()) != setup["initialization"]:
            raise ValueError("Requested source differs from the existing run")
        # The cursor counts 128-window batches in both the original global-batch
        # run and the new per-device run, so it is portable across world sizes.
        def resume_settings(run):
            settings = dict(run)
            settings.setdefault("contact_velocity_fps", FPS * STRIDE)
            settings["config"] = asdict(TokenizerConfig(**settings["config"]))
            old_global = settings.pop("global_batch")
            settings["per_device_batch"] = settings.get("per_device_batch", old_global)
            return {**settings, "device": None, "data": {**settings["data"], "data": None}}
        if resume_settings(recorded) != resume_settings(setup):
            raise ValueError("Run configuration differs from requested training")
    if resume and (output/"last.pt").exists():
        last = torch.load(output/"last.pt", map_location="cpu", weights_only=True)
        if resume_settings(last["setup"]) != resume_settings(setup):
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["optimizer"])
        epoch, cursor, step, best = (last[k] for k in ("epoch", "cursor", "step", "best_loss"))
        if leader and step == TOTAL_STEPS and not (output/"final.pt").exists():
            save_atomic(output/"final.pt", {k: last[k] for k in ("config", "model", "step", "metrics")})
        del last
    elif init_from or setup["initialization"] != "random":
        source = Path(init_from or setup["initialization"]).resolve()
        weights = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        source_config = TokenizerConfig(**weights["config"])
        if replace(source_config, fp32_position_encoding=config.fp32_position_encoding) != config:
            raise ValueError("Source tokenizer configuration differs from the requested model")
        model.load_state_dict(weights["model"])
        step = weights["step"]
        if "optimizer" in weights:
            opt.load_state_dict(weights["optimizer"])
            epoch, cursor = weights["epoch"], weights["cursor"]
        if stop_after is not None and stop_after <= step:
            raise ValueError("--stop-after must exceed the source checkpoint step")
        setup["initialization"] = str(source)
        loaded_source = True
        del weights
    elif leader:
        calibration = plan_epoch(train_data.offsets, seed, True)
        fit_stats(model, train_data, calibration)
        if model.motion_grid is not None:
            model.motion_grid.fit(model, train_data.batches(calibration[:64]))
    if leader:
        tmp = output/"run.json.tmp"
        tmp.write_text(json.dumps(setup, indent=2))
        tmp.replace(output/"run.json")

    def record(row):
        print(json.dumps(row), flush=True)
        with (output/"metrics.jsonl").open("a") as f:
            print(json.dumps(row), file=f)

    net = RawPrediction(model)
    if world > 1:
        # DDP synchronizes parameters and buffers (including fitted feature statistics) at initialization.
        net = DistributedDataParallel(net, device_ids=[device.index] if device.type == "cuda" else None)
    if leader:
        record({"event": "start", "resume_step": step, "world_size": world, "global_batch": global_batch,
                "per_rank_batch_min": BATCH, "per_rank_batch_max": BATCH})
    if loaded_source:
        if leader:
            metrics = validate(model, val_data, val_batches, skating_weight)
            best = metrics["loss"]
            weights = {"config": asdict(config), "model": model.state_dict(), "step": step, "metrics": metrics}
            save_atomic(output/"best.pt", weights)
            save_atomic(output/"last.pt", {**weights, "setup": setup, "optimizer": opt.state_dict(),
                                           "epoch": epoch, "cursor": cursor, "best_loss": best})
            record({"step": step, "val": metrics, "event": "initial_checkpoint"})
        if world > 1:
            dist.barrier()
    t0, start_step = time.monotonic(), step
    end_step = TOTAL_STEPS if stop_after is None else min(stop_after, TOTAL_STEPS)
    while step < end_step:
        batches = plan_epoch(train_data.offsets, seed + epoch, True)
        indices = range(cursor, len(batches), world)
        local_plans = (batches[(idx + rank) % len(batches)] for idx in indices)
        for idx, (x, valid, delta) in zip(indices, train_data.batches(local_plans)):
            step += 1
            model.train()
            with precision(device):
                raw = net(x)
            rec = reconstruction_loss(raw, x, valid, model)
            if skating_weight:
                skate, weight = skating_terms(model.unpack(raw), valid, delta)
                loss = global_loss(rec, valid, skate, weight, skating_weight)
            elif world > 1:
                count = valid.sum()
                totals = all_reduce(torch.stack((rec * count, count)))
                loss = totals[0] / totals[1]
            else:
                loss = rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr = learning_rate(step)
            opt.param_groups[0]["lr"] = lr
            opt.step()
            if leader and (step == 1 or step % 100 == 0 or (loaded_source and step == start_step + 1)):
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite loss at {step}")
                record({"step": step, "lr": lr, "train_loss": loss.item(),
                        "steps_per_sec": (step-start_step)/(time.monotonic()-t0)})
            validate_now = step in (100, end_step) or step % 5000 == 0
            if validate_now or step % 1000 == 0:
                if leader:
                    weights = {"config": asdict(config), "model": model.state_dict(), "step": step}
                    if validate_now:
                        metrics = validate(model, val_data, val_batches, skating_weight)
                        if not all(math.isfinite(v) for v in metrics.values()):
                            raise FloatingPointError(f"Non-finite validation at step {step}")
                        weights["metrics"] = metrics
                        record({"step": step, "val": metrics})
                        if metrics["loss"] < best:
                            best = metrics["loss"]
                            save_atomic(output/"best.pt", weights)
                    save_atomic(output/"last.pt", {**weights, "setup": setup, "optimizer": opt.state_dict(),
                                                    "epoch": epoch, "cursor": min(idx + world, len(batches)), "best_loss": best})
                    if step == TOTAL_STEPS:
                        save_atomic(output/"final.pt", weights)
                if world > 1:
                    dist.barrier()
                if step == end_step:
                    return
        epoch += 1
        cursor = 0


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--skating-weight", type=float, default=DEFAULT_SKATING_WEIGHT,
                     help="Foot-skating loss coefficient (0 disables its training gradient)")
    cli.add_argument("--workers", type=int, default=4, help="NPZ reader threads per GPU")
    cli.add_argument("--motion-grid", action="store_true", help="Use protected coarse FSQ levels fitted on training motions")
    cli.add_argument("--stop-after", type=int, help="Validate and save at this absolute step, keeping the original LR schedule")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seed", type=int, default=2026)
    cli.add_argument("--init-from", type=Path, help="Start a new run at the source step with saved weights and statistics")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--auto-resume", action="store_true", help="Resume the existing run if present")
    args = cli.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        cli.error("--stop-after must be a positive step")
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and "LOCAL_RANK" not in os.environ:
        cli.error("Multi-node jobs should use scripts/train_human24_paper.sh or torchrun")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    with ExitStack() as stack:
        if world > 1:
            dist.init_process_group("nccl")
            stack.callback(dist.destroy_process_group)
        leader = world == 1 or dist.get_rank() == 0
        decisions = [None, None]
        if leader:
            args.output.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((args.output/"trainer.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.seek(0)
            lock.truncate()
            print(os.getpid(), file=lock, flush=True)
            decisions = [args.resume or (args.auto_resume and (args.output/"run.json").is_file()),
                         (args.output/"final.pt").is_file()]
        if world > 1:
            dist.broadcast_object_list(decisions, src=0, device=device)
        resume, finished = decisions
        if not finished:
            train(args.data, args.output, resume, args.seed, device, args.workers,
                  motion_grid=args.motion_grid, stop_after=args.stop_after, init_from=args.init_from,
                  skating_weight=args.skating_weight)


if __name__ == "__main__":
    main()
