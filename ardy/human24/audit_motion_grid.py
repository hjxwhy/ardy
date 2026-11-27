"""Audit the two-GPU FSQ repair, including replacement by the failed Transformer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch

from .data import clip_frame_count, split_name
from .model import Human24FSQVAE, TokenizerConfig
from .paper_data import Corpus, describe_index
from .paper_train import plan_epoch, predict
from .train import reconstruction_loss


def effective_levels(hist):
    p = hist.double() / hist.sum(1, keepdim=True)
    return (-(p * p.clamp_min(1e-30).log()).sum(1)).exp()


class Measurements:
    def __init__(self, device, dimensions):
        self.hist = torch.zeros(dimensions, 64, dtype=torch.long, device=device)
        self.sums = torch.zeros(11, dtype=torch.float64, device=device)
        self.max_residual = torch.zeros((), device=device)
        self.rec_sum = torch.zeros((), device=device, dtype=torch.float64)

    def observe(self, model, x, valid):
        raw, pred, codes = predict(model, x, return_codes=True)
        self.rec_sum += reconstruction_loss(raw, x, valid, model).double() * valid.sum()
        indices = ((codes + 1) * 32).round().long()
        mask = valid[:, ::model.config.patch]
        ids = indices[mask] + torch.arange(indices.shape[-1], device=x.device) * 64
        self.hist += torch.bincount(ids.flatten(), minlength=self.hist.numel()).reshape_as(self.hist)
        coarse_bins = indices // model.motion_grid.fine_levels
        coarse_pose = model.motion_grid.coordinates(codes)[0][:, :x.shape[1], :69]
        coarse_pose = coarse_pose * model.std[:69] + model.mean[:69]
        target, rec, coarse = (v[..., :69].unflatten(-1, (23, 3)) for v in (x, pred, coarse_pose))
        def distance(a, b):
            return (a - b).norm(dim=-1).mean(dim=-1)
        residual = (rec - coarse).norm(dim=-1)
        self.max_residual = torch.maximum(self.max_residual, residual[valid].max())
        self.sums[:3] += torch.stack((valid.sum(), distance(target, rec)[valid].sum(dtype=torch.float64),
                                      distance(target, coarse)[valid].sum(dtype=torch.float64)))
        if len(x) > 1:
            pair = valid & valid.roll(1, 0)
            tokens = pair[:, ::model.config.patch]
            self.sums[3:] += torch.stack((
                pair.sum(), *(distance(t, t.roll(1, 0))[pair].sum(dtype=torch.float64)
                              for t in (target, rec, coarse)),
                distance(target, rec.roll(1, 0))[pair].sum(dtype=torch.float64),
                tokens.sum() * indices.shape[-1], (indices != indices.roll(1, 0))[tokens].sum(),
                (coarse_bins != coarse_bins.roll(1, 0))[tokens].sum(),
            ))
        return indices[mask]

    def report(self, radius):
        n, err, coarse_err, pairs, real, rec, coarse, swapped, dims, changed, coarse_changed = self.sums.tolist()
        effective = effective_levels(self.hist)
        coarse_eff = effective_levels(self.hist.reshape(len(self.hist), 16, 4).sum(2))
        return dict(valid_frames=int(n), tokens=int(self.hist[0].sum()), mpjpe_cm=100 * err / n,
                    reconstruction_loss=self.rec_sum.item() / n,
                    coarse_mpjpe_cm=100 * coarse_err / n, paired_target_cm=100 * real / pairs,
                    paired_reconstruction_cm=100 * rec / pairs, swapped_mpjpe_cm=100 * swapped / pairs,
                    maximum_joint_correction_cm=100 * self.max_residual.item(),
                    paired_code_change_pct=100 * changed / dims,
                    effective_levels_median=effective.median().item(), effective_levels_min=effective.min().item(),
                    effective_fine_levels_given_coarse_median=(effective / coarse_eff).median().item(),
                    dominant_over_99pct_channels=int((self.hist.max(1).values / self.hist.sum(1) >= .99).sum()),
                    fixed_grid_bounds=dict(paired_code_change_pct_min=100 * coarse_changed / dims,
                        effective_levels_min_per_channel=coarse_eff.min().item(),
                        effective_levels_median_min=coarse_eff.median().item(),
                        mpjpe_cm_max=100 * (coarse_err / n + radius),
                        paired_reconstruction_cm_min=100 * max(0, coarse / pairs - 2 * radius)))


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", required=True, type=Path)
    cli.add_argument("--failed-checkpoint", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    if torch.cuda.device_count() != 2:
        cli.error("Expose the two verification GPUs using CUDA_VISIBLE_DEVICES")
    run = json.loads((args.checkpoint.parent / "run.json").read_text())
    root = Path(run["data"]["data"])
    ckpt = torch.load(args.checkpoint, mmap=True, weights_only=True, map_location="cpu")
    failed = torch.load(args.failed_checkpoint, mmap=True, weights_only=True, map_location="cpu")
    models = []
    for i in range(2):
        model = Human24FSQVAE(TokenizerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        if i:
            for name, parameter in model.named_parameters():
                parameter.copy_(failed["model"][name])
        models.append(model.eval().to(torch.device("cuda", i)))
    original = Human24FSQVAE(TokenizerConfig(**failed["config"]))
    original.load_state_dict(failed["model"])
    original.eval().to(models[1].mean.device)
    original_totals = torch.zeros(5, dtype=torch.float64, device=original.mean.device)
    original_hist = torch.zeros(original.config.code_dim, 64, dtype=torch.long, device=original.mean.device)
    ablation = torch.zeros(4, dtype=torch.float64, device=models[0].mean.device)
    counts = [Measurements(m.mean.device, m.config.code_dim) for m in models]
    mismatch = 0
    paths = sorted(p.relative_to(root) for p in root.glob("*/*.npz") if split_name(p.relative_to(root)) == "val")
    with ThreadPoolExecutor(6) as pool:
        entries = list(zip(paths, pool.map(clip_frame_count, (root / p for p in paths))))
        index = describe_index(root, run["data"]["parents"], {"val": entries})["splits"]["val"]
        if index != run["data"]["splits"]["val"]:
            raise ValueError("Validation paths or frame counts differ from the training run")
        contact_fps = run.get("contact_velocity_fps", run["data"]["source_fps"])
        corpus = Corpus(root, entries, run["data"]["parents"], torch.device("cpu"), pool, contact_fps)
        for x, valid, _ in corpus.batches(plan_epoch(corpus.offsets, run["seed"], False)):
            bins = [stats.observe(m, x.to(m.mean.device), valid.to(m.mean.device)).cpu()
                    for stats, m in zip(counts, models)]
            mismatch += int((bins[0] // 4 != bins[1] // 4).sum())
            # Same fitted grid with both learned corrections set to zero.
            active = models[0]
            inputs, frame_mask = x.to(active.mean.device), valid.to(active.mean.device)
            normalized = (inputs - active.mean) / active.std
            length = normalized.shape[1]
            pad = -length % active.config.patch
            patches = torch.cat((normalized, normalized[:, -1:].expand(-1, pad, -1)), dim=1)
            patches = patches.reshape(len(x), -1, 218 * active.config.patch)
            blank = patches.new_zeros((*patches.shape[:2], active.config.code_dim))
            fixed_codes = active.motion_grid.encode(patches, blank)
            fixed_raw = active.motion_grid.decode(fixed_codes, torch.zeros_like(patches).reshape(len(x), -1, 218),
                                                  active.mean, active.std)[:, :length]
            fixed = active.unpack(fixed_raw)
            pos_error = (fixed[..., :69] - inputs[..., :69]).unflatten(-1, (23, 3)).norm(dim=-1).mean(-1)
            tokens = frame_mask[:, ::active.config.patch]
            ablation += torch.stack((pos_error[frame_mask].sum(dtype=torch.float64),
                       (bins[0].to(inputs.device) != ((fixed_codes[tokens] + 1) * 32).round()).sum(),
                       tokens.sum() * fixed_codes.shape[-1],
                       reconstruction_loss(fixed_raw, inputs, frame_mask, active).double() * frame_mask.sum()))
            old_x, mask = x.to(original.mean.device), valid.to(original.mean.device)
            _, old_pred, old_codes = predict(original, old_x, return_codes=True)
            old_bins = (old_codes * 32).round().long() + 32
            old_ids = old_bins[mask[:, ::original.config.patch]] + torch.arange(len(original_hist), device=mask.device) * 64
            original_hist += torch.bincount(old_ids.flatten(), minlength=original_hist.numel()).reshape_as(original_hist)
            old_joints = old_pred[..., :69].unflatten(-1, (23, 3))
            errors = (old_joints - old_x[..., :69].unflatten(-1, (23, 3))).norm(dim=-1).mean(-1)
            original_totals[0] += errors[mask].sum(dtype=torch.float64)
            if len(x) > 1:
                pair = mask & mask.roll(1, 0)
                gap = (old_joints - old_joints.roll(1, 0)).norm(dim=-1).mean(-1)
                token_pair = pair[:, ::original.config.patch]
                original_totals[1:] += torch.stack((gap[pair].sum(dtype=torch.float64), pair.sum(),
                       (old_bins != old_bins.roll(1, 0))[token_pair].sum(), token_pair.sum() * old_bins.shape[-1]))
    radius = models[0].motion_grid.max_pose_residual_m
    report = dict(step=ckpt["step"], failed_step=failed["step"], validation_clips=len(paths),
                  validation_index_sha256=index["index_sha256"],
                  matching_input_normalization=all(torch.equal(ckpt["model"][k], failed["model"][k])
                                                   for k in ("mean", "std")),
                  gpus=[str(torch.cuda.get_device_properties(i).uuid) for i in range(2)],
                  coarse_bin_mismatches=mismatch, trained=counts[0].report(radius),
                  failed_weights_in_repaired_model=counts[1].report(radius))
    err, gap, pairs, changed, dims = original_totals.tolist()
    report["original_failed_model"] = dict(mpjpe_cm=100 * err / report["trained"]["valid_frames"],
                                          paired_reconstruction_cm=100 * gap / pairs,
                                          paired_code_change_pct=100 * changed / dims,
                                          effective_levels_median=effective_levels(original_hist).median().item(),
                                          dominant_over_99pct_channels=int((original_hist.max(1).values /
                                                                           original_hist.sum(1) >= .99).sum()))
    position_sum, changed, dims, rec_sum = ablation.tolist()
    n = report["trained"]["valid_frames"]
    report["zero_learned_corrections"] = dict(mpjpe_cm=100 * position_sum / n,
                                              learned_encoder_changed_scalar_pct=100 * changed / dims,
                                              reconstruction_loss=rec_sum / n)
    if mismatch or any(s.max_residual.item() > radius + 1e-5 for s in counts):
        raise RuntimeError("Coarse codes or bounded reconstruction invariant violated")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
