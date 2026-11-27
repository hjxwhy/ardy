"""Export a standalone, synchronized SOMA BVH / Human24 motion viewer."""
import argparse
import base64
import json
from pathlib import Path
import tarfile

import numpy as np
import rapidgzip
from scipy.spatial.transform import Rotation

from retarget_bones_seed import MOTION, hierarchy

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = [
    "230315/egypt_dance_R_003__A275",
    "230315/reach_jump_R_002__A274_M",
]


def clip_id(value):
    path = Path(value)
    if len(path.parts) != 2 or not path.parts[0].isdigit() or path.name in {".", ".."}:
        raise argparse.ArgumentTypeError("Use date/clip_name, for example " + DEFAULT_CLIPS[0])
    return path.with_suffix("").as_posix() if path.suffix in {".npz", ".bvh"} else path.as_posix()


def source_files(dataset, clips):
    """Prefer extracted files and scan the archive at most once for the remainder."""
    missing = {}
    for clip in clips:
        member = f"soma_uniform/bvh/{clip}.bvh"
        extracted = dataset / member
        if extracted.is_file():
            yield clip, extracted.read_bytes()
        else:
            missing[member] = clip
    if missing:
        with rapidgzip.open(str(dataset / "soma_uniform.tar.gz"), parallelization=8) as stream:
            with tarfile.open(fileobj=stream, mode="r|", bufsize=1024 * 1024) as archive:
                for member in archive:
                    if member.isfile() and member.name in missing:
                        yield missing.pop(member.name), archive.extractfile(member).read()
                        if not missing:
                            break
    if missing:
        raise FileNotFoundError("Missing original BVH: " + ", ".join(missing))


def soma_world_positions(raw):
    """Evaluate the original BVH channels and offsets in their own skeleton."""
    header = MOTION.search(raw, 0, 90000)
    if header is None:
        raise ValueError("Invalid BVH motion header")
    names, parents, offsets, channels = hierarchy(raw[:header.start()])
    frames = int(header[1])
    fps = round(1 / float(header[2]))
    values = np.fromstring(raw[header.end():], dtype=np.float64, sep=" ").reshape(frames, sum(channels))
    if names[:2] != ["Root", "Hips"] or list(channels[:2]) != [6, 6]:
        raise ValueError("Expected the SOMA Uniform Root / Hips position channels")
    positions = np.empty((frames, len(names), 3), np.float32)
    for start in range(0, frames, 512):
        block = values[start:start + 512]
        rotations = np.empty((len(block), len(names), 3, 3))
        world = np.empty((len(block), len(names), 3))
        cursor = 0
        for j, (p, n) in enumerate(zip(parents, channels)):
            joint = block[:, cursor:cursor + n]
            cursor += n
            local_rotation = Rotation.from_euler("ZYX", joint[:, -3:], degrees=True).as_matrix()
            translation = joint[:, :3] if n == 6 else offsets[j]
            if p < 0:
                rotations[:, j] = local_rotation
                world[:, j] = translation
            else:
                rotations[:, j] = rotations[:, p] @ local_rotation
                world[:, j] = world[:, p] + (rotations[:, p] @ translation[..., None])[..., 0]
        positions[start:start + len(block)] = world * 0.01
    # Omit the virtual scene Root so the rendered tree starts at Hips.
    return positions[:, 1:], names[1:], (parents[1:] - 1).tolist(), fps


def encode_positions(positions):
    return base64.b64encode(np.asarray(positions, dtype="<f4").tobytes()).decode("ascii")


def make_clip(dataset, spec, clip, raw):
    source, names, parents, fps = soma_world_positions(raw)
    with np.load(dataset / "human24" / (clip + ".npz"), allow_pickle=False) as data:
        relative, root = data["joints"], data["root_positions"]
        if fps != int(data["fps"]) or relative.shape != (len(source), len(spec["parents"]), 3):
            raise ValueError(f"Source and target timeline mismatch for {clip}")
        if not np.allclose(source[:, 0] * spec["root_motion_scale_after_cm_to_m"], root, atol=2e-6):
            raise ValueError(f"Source and target pelvis trajectories do not match for {clip}")
        target = relative + root[:, None, :]
    return dict(
        id=clip, frames=len(source), fps=fps,
        source=dict(names=names, parents=parents, xyz=encode_positions(source)),
        target=dict(names=spec["joint_names"], parents=spec["parents"], xyz=encode_positions(target)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/data/datas/bones-seed"))
    parser.add_argument("--clip", action="append", type=clip_id, help="date/clip_name (repeat for multiple clips)")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "human24_comparison.html")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    clips = list(dict.fromkeys(args.clip or DEFAULT_CLIPS))
    spec = json.loads((dataset / "human24" / "skeleton.json").read_text())
    for clip in clips:
        if not (dataset / "human24" / (clip + ".npz")).is_file():
            parser.error(f"Missing Human24 clip: {clip}")
    samples = {clip: make_clip(dataset, spec, clip, raw) for clip, raw in source_files(dataset, clips)}
    template = Path(__file__).with_name("bones_motion_viewer.html").read_text()
    payload = json.dumps([samples[key] for key in clips], separators=(",", ":")).replace("<", "\\u003c")
    output = args.output.resolve()
    if output.suffix.lower() != ".html" or output == Path(__file__).with_name("bones_motion_viewer.html"):
        parser.error("--output must name an HTML export distinct from the viewer template")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("<!--MOTION_DATA-->", payload))
    print(f"{output} ({len(clips)} clips, {sum(s['frames'] for s in samples.values())} frames)")


if __name__ == "__main__":
    main()
