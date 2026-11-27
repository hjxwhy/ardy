"""Coarse motion coordinates and learned fine bins on the same scalar FSQ grid."""

import math

import torch
from torch import nn


class MotionGrid(nn.Module):
    fine_levels = 4
    max_pose_residual_m = 0.025
    companding_scale = math.asinh(6.)

    def __init__(self, patch, dimensions, levels, position_weight):
        super().__init__()
        if levels != 64 or dimensions > patch * 218:
            raise ValueError("The motion grid requires 64 levels and at most 218 scalars per frame")
        self.patch = patch
        self.levels = levels
        if not math.isfinite(position_weight) or position_weight <= 0:
            raise ValueError("Expected a finite positive position weight")
        weight = torch.ones(patch, 218)
        weight[:, :69] = position_weight
        weight[:, 217] = position_weight
        self.register_buffer("feature_weight", weight.flatten(), persistent=False)
        self.register_buffer("center", torch.zeros(patch * 218))
        self.register_buffer("basis", torch.zeros(dimensions, patch * 218))
        self.register_buffer("scale", torch.ones(dimensions))

    @torch.no_grad()
    def fit(self, model, batches):
        """Fit only on complete training patches, after feature normalization."""
        width = self.center.numel()
        sums = self.center.double().new_zeros(width)
        second = sums.new_zeros((width, width))
        count = 0
        for x, valid, _ in batches:
            end = x.shape[1] // self.patch * self.patch
            full = valid[:, :end].reshape(len(x), -1, self.patch).all(-1)
            patches = ((x[:, :end] - model.mean) / model.std).reshape(len(x), -1, width)
            values = patches[full].double()
            sums += values.sum(0)
            second.addmm_(values.T, values)
            count += len(values)
        if count <= len(self.scale):
            raise ValueError("Insufficient complete training patches to fit the motion grid")
        mean = sums / count
        covariance = second / count - mean.outer(mean)
        covariance *= self.feature_weight[:, None] * self.feature_weight[None, :]
        variance, vectors = torch.linalg.eigh(covariance.cpu())
        top = slice(-len(self.scale), None)
        self.center.copy_(mean)
        self.basis.copy_(vectors[:, top].flip(1).T)
        self.scale.copy_(variance[top].flip(0).clamp_min(1e-8).sqrt())

    def encode(self, patches, correction):
        with torch.autocast(patches.device.type, enabled=False):
            score = (((patches.float() - self.center) * self.feature_weight) @ self.basis.T) / self.scale
            coordinate = (score.asinh() / self.companding_scale + 1) * (self.levels / 2)
            coarse = (coordinate / self.fine_levels).floor().clamp(0, self.levels // self.fine_levels - 1)
            base = coarse * self.fine_levels
            fine = coordinate - base - 0.5 + 2 * correction.float().tanh()
            fine = fine.clamp(0, self.fine_levels - 1)
            quantized = fine + (fine.round() - fine).detach()
            return (base + quantized) / (self.levels / 2) - 1

    def coordinates(self, codes):
        """Both reconstructions come exclusively from the discrete token."""
        with torch.autocast(codes.device.type, enabled=False):
            index = ((codes.float() + 1) * (self.levels / 2)).clamp(0, self.levels - 1)
            group = index.detach().round().div(self.fine_levels, rounding_mode="floor")
            center = (group + 0.5) * self.fine_levels
            midpoint = torch.cat((center, index + 0.5), dim=0)
            score = ((midpoint / (self.levels / 2) - 1) * self.companding_scale).sinh()
            features = ((score * self.scale) @ self.basis) / self.feature_weight + self.center
            features = features.reshape(len(midpoint), -1, 218)
            return features.chunk(2, dim=0)

    def decode(self, codes, correction, mean, std):
        coarse, fine = self.coordinates(codes)
        refined = fine + 0.5 * correction.float().tanh()
        # A fixed physical bound prevents even an arbitrary learned branch
        # from mapping distinct coarse poses to a single reconstructed pose.
        offset = ((refined[..., :69] - coarse[..., :69]) * std[:69]).unflatten(-1, (23, 3))
        norm = offset.norm(dim=-1, keepdim=True)
        offset = offset * (self.max_pose_residual_m / norm.clamp_min(self.max_pose_residual_m))
        positions = coarse[..., :69] + offset.flatten(-2) / std[:69]
        contacts = (fine[..., 213:217] * std[213:217] + mean[213:217]).clamp(0.005, 0.995)
        return torch.cat((positions, refined[..., 69:213],
                          contacts.logit() + correction[..., 213:217], refined[..., 217:]), dim=-1)
