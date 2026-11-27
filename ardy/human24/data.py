"""Build Human24 windows at the source frame rate and prepare a reusable training cache."""

import argparse
import hashlib
import json
import os
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DATA = Path("/mnt/data/datas/bones-seed/human24")
FEATURE_DIM = 218
FOOT_JOINTS = [7, 10, 8, 11]


def prepare_clip(path, parents, *, stride=1, expected_fps=None, return_planar=False, source_rate_contacts=False):
    with np.load(path, allow_pickle=False) as motion:
        joints = motion["joints"]
        root = motion["root_positions"]
        quaternions = motion["local_rot_quat"]
        fps = float(motion["fps"])
    if expected_fps is not None and fps != expected_fps:
        raise ValueError(f"Expected {expected_fps} Hz source in {path}, found {fps}")
    frames = len(root)
    if frames < 1 or joints.shape != (frames, 24, 3) or quaternions.shape != (frames, 24, 4):
        raise ValueError(f"Invalid Human24 motion in {path}")
    rotations = Rotation.from_quat(quaternions[::stride].reshape(-1, 4)).as_matrix().reshape(-1, 24, 3, 3)
    for j in range(1, 24):
        rotations[:, j] = rotations[:, parents[j]] @ rotations[:, j]

    # Label t describes the same forward interval [t, t+1] as the skating loss.
    # Only historical runs sampled precomputed source-rate labels.
    contact_stride = 1 if source_rate_contacts else stride
    feet = joints[::contact_stride, FOOT_JOINTS] + root[::contact_stride, None, :]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        speed[:-1] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * (fps / contact_stride)
        speed[-1] = speed[-2]
    contacts = (feet[..., 1] < 0.10) & (speed < 0.15)
    if source_rate_contacts:
        contacts = contacts[::stride]
    clip = (joints[::stride, 1:].reshape(-1, 69), rotations, contacts, root[::stride, 1:2])
    if return_planar:
        return clip, root[::stride][:, [0, 2]]
    return clip


def make_chunk(clip, start, length):
    positions, rotations, contacts, height = clip
    if start < 0 or length < 1 or start + length > len(rotations):
        raise ValueError("Window outside clip")
    window = slice(start, start + length)
    relative = rotations[start].swapaxes(-1, -2) @ rotations[window]
    six = np.concatenate((relative[..., 0], relative[..., 1]), axis=-1)
    return np.concatenate((positions[window], six.reshape(length, 144), contacts[window], height[window]),
                          axis=-1).astype(np.float32)


def split_name(relative_path):
    group = relative_path.with_name(relative_path.stem.removesuffix("_M"))
    digest = hashlib.blake2b(group.as_posix().encode(), digest_size=8).digest()
    return "val" if int.from_bytes(digest, "little") % 100 < 2 else "train"


def clip_frame_count(path):
    # Read only the NPY header, without decompressing the motion arrays.
    with ZipFile(path) as archive, archive.open("joints.npy") as file:
        version = np.lib.format.read_magic(file)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(file)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(file)
        else:
            raise ValueError(f"Unsupported NPY header {version} in {path}")
    return shape[0]


def covering_starts(frames, length):
    # Align the last full window to the end of the clip. Only the tail can overlap.
    return [min(i, max(0, frames - length)) for i in range(0, frames, length)]


def prepare_shard(task):
    data, cache, paths, split, first, length, windows, parents, seed, first_row, frame_counts = task
    x = np.load(cache / f"{split}.npy", mmap_mode="r+")
    spans = np.load(cache / f"{split}_spans.npy", mmap_mode="r+")
    row = first_row
    for i, path in enumerate(paths, first):
        clip = prepare_clip(data / path, parents)
        frames = len(clip[0])
        valid = min(length, frames)
        count = frames - valid + 1
        rng = np.random.default_rng([seed, i])
        if frame_counts is not None:
            if frames != frame_counts[i - first]:
                raise ValueError(f"Unexpected clip length in {path}")
            starts = covering_starts(frames, length)
        else:
            starts = []
            for k in range(windows):
                low = k * count // windows
                high = max(low + 1, (k + 1) * count // windows)
                starts.append(int(rng.integers(low, high)))
        for start in starts:
            chunk = make_chunk(clip, start, valid)
            if not np.isfinite(chunk).all():
                raise ValueError(f"Non-finite features in {path}")
            x[row, :valid] = chunk
            x[row, valid:] = chunk[-1]
            spans[row] = (start, valid)
            row += 1
    return len(paths)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=DATA)
    cli.add_argument("--cache", type=Path, required=True)
    cli.add_argument("--length", type=int, default=64)
    cli.add_argument("--full-train", action="store_true", help="Tile every training frame, including the tail")
    cli.add_argument("--train-windows", type=int, default=4)
    cli.add_argument("--workers", type=int, default=24)
    cli.add_argument("--seed", type=int, default=2026)
    args = cli.parse_args()
    if args.length < 1 or args.train_windows < 1:
        cli.error("length and train-windows must be positive")
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=False)
    parents = json.loads((args.data / "skeleton.json").read_text())["parents"]
    files = sorted(p.relative_to(args.data) for p in args.data.glob("*/*.npz"))
    splits = {"train": [], "val": []}
    for path in files:
        splits[split_name(path)].append(path)
    manifest = {"data": str(args.data.resolve()), "length": args.length, "feature_dim": FEATURE_DIM,
                "parents": parents, "seed": args.seed, "splits": {}}
    tasks = []
    for split, paths in splits.items():
        if not paths:
            raise ValueError(f"Empty {split} split")
        windows = args.train_windows if split == "train" else 1
        frames = None
        if split == "train" and args.full_train:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                frames = np.fromiter(pool.map(clip_frame_count, (args.data / p for p in paths), chunksize=128),
                                     dtype=np.int64, count=len(paths))
            counts = (frames + args.length - 1) // args.length
            offsets = np.concatenate(([0], np.cumsum(counts)))
            n = int(offsets[-1])
            np.save(cache / "train_clip_offsets.npy", offsets)
            np.save(cache / "train_clip_frames.npy", frames)
            manifest["splits"][split] = {"clips": len(paths), "windows": n, "sampling": "all_frames",
                                         "source_frames": int(frames.sum())}
        else:
            n = len(paths) * windows
            offsets = np.arange(len(paths) + 1) * windows
            manifest["splits"][split] = {"clips": len(paths), "windows_per_clip": windows, "windows": n}
        (cache / f"{split}_clips.txt").write_text("".join(f"{p}\n" for p in paths))
        for name, shape, dtype in [(split, (n, args.length, FEATURE_DIM), "float32"),
                                   (f"{split}_spans", (n, 2), "int32")]:
            np.lib.format.open_memmap(cache / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for i in range(0, len(paths), 128):
            tasks.append((args.data, cache, paths[i:i + 128], split, i, args.length, windows, parents, args.seed,
                          int(offsets[i]), None if frames is None else frames[i:i + 128]))
    print(manifest["splits"], flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, count in enumerate(pool.map(prepare_shard, tasks)):
            done += count
            if i % 64 == 0:
                print(f"Prepared {done}/{len(files)} clips", flush=True)
    tmp = cache / "manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, cache / "manifest.json")
    print(f"Prepared all {done} clips", flush=True)


if __name__ == "__main__":
    main()
