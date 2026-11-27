"""Read 30 Hz training windows directly from the original Human24 NPZ files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
import torch

from .data import clip_frame_count, make_chunk, prepare_clip, split_name

FPS = 30
STRIDE = 4


def index_clips(data, pool):
    """Only paths and header lengths are kept in RAM. Rank zero scans the shared mount."""
    parents = json.loads((data / "skeleton.json").read_text())["parents"]
    paths = sorted(data.glob("*/*.npz"))
    splits = {"train": [], "val": []}
    for i in range(0, len(paths), 512):
        group = paths[i:i + 512]
        for path, frames in zip(group, pool.map(clip_frame_count, group)):
            relative = path.relative_to(data)
            splits[split_name(relative)].append((relative.as_posix(), frames))
    if not all(splits.values()):
        raise ValueError(f"Expected Human24 motion files in both splits at {data}")
    return parents, splits


def describe_index(data, parents, splits):
    manifest = {"data": str(data.resolve()), "source_fps": FPS * STRIDE, "fps": FPS,
                "parents": parents, "splits": {}}
    for split, entries in splits.items():
        frames = np.fromiter((n for _, n in entries), dtype=np.int64)
        sizes = (frames + STRIDE - 1) // STRIDE
        digest = hashlib.sha256()
        for path, n in entries:
            digest.update(f"{path}\t{n}\n".encode())
        manifest["splits"][split] = {"clips": len(entries), "source_frames": int(frames.sum()),
                                     "sampled_frames": int(sizes.sum()), "shorter_than_1s": int((sizes < FPS).sum()),
                                     "index_sha256": digest.hexdigest()}
    return manifest


class Corpus:
    def __init__(self, data, entries, parents, device, pool: ThreadPoolExecutor, contact_velocity_fps=FPS):
        if contact_velocity_fps not in (FPS, FPS * STRIDE):
            raise ValueError(f"Unsupported foot contact velocity FPS: {contact_velocity_fps}")
        self.source_rate_contacts = contact_velocity_fps != FPS
        self.paths = [data / p for p, _ in entries]
        sizes = [(n + STRIDE - 1) // STRIDE for _, n in entries]
        self.offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
        self.parents = parents
        self.device = device
        self.pool = pool

    def _window(self, start, size):
        idx = int(np.searchsorted(self.offsets, start, side="right")) - 1
        local_start = int(start - self.offsets[idx])
        clip, xz = prepare_clip(self.paths[idx], self.parents, stride=STRIDE,
                                expected_fps=FPS * STRIDE, return_planar=True,
                                source_rate_contacts=self.source_rate_contacts)
        if len(xz) != self.offsets[idx + 1] - self.offsets[idx]:
            raise ValueError(f"Length changed since indexing {self.paths[idx]}")
        x = make_chunk(clip, local_start, size)
        delta = np.zeros((size, 2), dtype=np.float32)
        delta[:-1] = np.diff(xz[local_start:local_start + size], axis=0)
        return x, delta

    def _request(self, planned):
        length, entries = planned
        return length, [self.pool.submit(self._window, int(start), int(size)) for start, size in entries]

    def _collect(self, request):
        length, futures = request
        count = len(futures)
        x = np.empty((count, length, 218), dtype=np.float32)
        delta = np.zeros((count, length, 2), dtype=np.float32)
        sizes = np.empty(count, dtype=np.int64)
        for i, future in enumerate(futures):
            values, planar = future.result()
            n = len(values)
            x[i, :n] = values
            x[i, n:] = values[-1]
            delta[i, :n] = planar
            sizes[i] = n
        valid = np.arange(length) < sizes[:, None]
        return tuple(torch.from_numpy(a).to(self.device) for a in (x, valid, delta))

    def batch(self, planned):
        return self._collect(self._request(planned))

    def batches(self, plans):
        """Overlap reading the next local batch with computation of the current batch."""
        plans = iter(plans)
        first = next(plans, None)
        if first is None:
            return
        pending = self._request(first)
        for plan in plans:
            current, pending = pending, self._request(plan)
            yield self._collect(current)
        yield self._collect(pending)
