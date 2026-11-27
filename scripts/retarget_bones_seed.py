"""Retarget the Bones SEED SOMA Uniform archive onto the supplied receiver skeleton."""
import argparse
import csv
import ctypes
import functools
import io
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import re
import sys
import subprocess
import tarfile
import time
import zipfile
from tempfile import NamedTemporaryFile

import numpy as np
import rapidgzip
from scipy.spatial.transform import Rotation

REST = np.array([
    [0,0,0],[.074741,-.093560,-.003486],[-.074741,-.093560,-.003486],[0,.112003,-.031303],
    [.109339,-.477648,-.014084],[-.109339,-.477648,-.014084],[0,.245255,-.032571],
    [.094756,-.885206,-.060656],[-.094756,-.885206,-.060656],[0,.299528,-.001681],
    [.122099,-.938555,.063346],[-.122099,-.938555,.063346],[0,.523536,-.056669],
    [.087723,.427340,-.043115],[-.087723,.427340,-.043115],[0,.584206,-.005528],
    [.187642,.462535,-.054506],[-.187642,.462535,-.054506],[.453092,.449081,-.081596],
    [-.453092,.449081,-.081596],[.711685,.458878,-.082685],[-.711685,.458878,-.082685],
    [.798903,.451279,-.095703],[-.798903,.451279,-.095703]], dtype=np.float64)
PARENT = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],dtype=np.int32)
NAMES = 'pelvis left_hip right_hip spine1 left_knee right_knee spine2 left_ankle right_ankle spine3 left_foot right_foot neck left_collar right_collar head left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_hand right_hand'.split()
SOURCE = 'Hips LeftLeg RightLeg Spine1 LeftShin RightShin Spine2 LeftFoot RightFoot Chest LeftToeBase RightToeBase Neck2 LeftShoulder RightShoulder Head LeftArm RightArm LeftForeArm RightForeArm LeftHand RightHand LeftHandMiddle2 RightHandMiddle2'.split()
MOTION = re.compile(rb'(?m)^MOTION\s*\nFrames:\s*(\d+)\s*\nFrame Time:[ \t]*([\d.eE+-]+)[ \t\r]*\n')
REST_FILE = 'soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh'
P = ctypes.c_void_p
ROTATION_FIELD = "local_rot_quat"
ROTATION_DESCRIPTION = ("float32 [T,24,4], unit quaternions in xyzw order; active parent-local "
                        "rotations in target rest axes; joint 0 is world-relative; sign continuous per clip")
REST_OFFSETS = REST.copy()
REST_OFFSETS[1:] -= REST[PARENT[1:]]


def hierarchy(header):
    names, parents, offsets, channels, stack = [], [], [], [], []
    for line in header.decode('ascii').splitlines():
        line = line.strip()
        if line.startswith(('JOINT ', 'ROOT ')):
            name = line.split()[1]
            node = len(names)
            names.append(name)
            parents.append(stack[-1] if stack else -1)
        elif line == 'End Site':
            node = None
        elif line == '{':
            stack.append(node)
        elif line == '}':
            stack.pop()
        elif line.startswith('OFFSET') and stack[-1] is not None:
            offsets.append([float(v) for v in line.split()[1:]])
        elif line.startswith('CHANNELS'):
            fields = line.split()
            expected = ('Xposition Yposition Zposition ' if fields[1] == '6' else '')+'Zrotation Yrotation Xrotation'
            if fields[2:] != expected.split():
                raise ValueError(f'Unexpected channels: {line}')
            channels.append(int(fields[1]))
    assert not stack and len(names)==len(set(names))==len(offsets)==len(channels)
    return names, np.array(parents,np.int32), np.array(offsets), np.array(channels,np.int32)


class Converter:
    def __init__(self, base):
        raw = base.read_bytes()
        h = MOTION.search(raw)
        self.names,self.parents,self.offsets,self.base_channels = hierarchy(raw[:h.start()])
        angles = np.fromstring(raw[h.end():].splitlines()[0],sep=' ')
        assert self.names[:2] == ['Root','Hips'] and len(angles)==sum(self.base_channels)
        self.ref = {}
        ref_pos = {}
        cursor = 0
        for i,name in enumerate(self.names):
            count = self.base_channels[i]
            values = angles[cursor:cursor+count]
            cursor += count
            parent = self.names[self.parents[i]] if i else None
            parent_r = self.ref.get(parent, np.eye(3))
            off = values[:3] if count==6 else self.offsets[i]
            ref_pos[name] = ref_pos.get(parent,np.zeros(3)) + parent_r @ off
            self.ref[name] = parent_r @ Rotation.from_euler('ZYX',values[-3:],degrees=True).as_matrix()
        drop_cm = ref_pos['Hips'][1] - np.mean([ref_pos[n][1] for n in ('LeftToeBase','RightToeBase')])
        self.root_scale = -REST[[10,11],1].mean() / drop_cm
        self.mapped = np.array([self.names.index(name) for name in SOURCE], dtype=np.int32)
        self.driver = self.mapped[PARENT]
        self.inverse_reference = np.ascontiguousarray(
            Rotation.from_matrix(np.stack([self.ref[name] for name in SOURCE])).inv().as_quat()
        )
        self.local = np.zeros_like(REST)
        self.active = np.zeros(len(self.names),np.int32)
        for j in range(1,24):
            i = self.driver[j]
            self.local[j] = self.ref[self.names[i]].T @ (REST[j]-REST[PARENT[j]])
        for i in self.mapped:
            while i>=0:
                self.active[i] = 1
                i = self.parents[i]
        kernel_dir = Path(sys.prefix) / 'retarget'
        kernel_dir.mkdir(exist_ok=True)
        source = Path(__file__).with_suffix('.cpp')
        binary = kernel_dir / 'convert.so'
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(['g++', '-std=c++17', '-O3', '-fPIC', '-shared',
                            str(source), '-o', str(binary)], check=True)
        library = ctypes.CDLL(str(binary))
        self.kernel = library.convert
        self.kernel.argtypes = [P,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,P,P,P,P,P,P,ctypes.c_double,P,P,P,P,P]
        self.kernel.restype = ctypes.c_int
        self.check_fk = library.check_rotation_fk
        self.check_fk.argtypes = [ctypes.c_int, P, P, P, P]
        self.check_fk.restype = ctypes.c_double

    @functools.lru_cache(maxsize=8)
    def channels(self, header):
        names,parents,offsets,channels = hierarchy(header)
        if (names!=self.names or not np.array_equal(parents,self.parents)
                or not np.isfinite(offsets).all()
                or list(channels[:2])!=[6,6] or not np.all(channels[2:]==3)):
            raise ValueError('Unexpected SOMA Uniform motion skeleton')
        return channels

    def convert(self, raw, expected_frames):
        h = MOTION.search(raw,0,90000)
        if not h: raise ValueError('Missing BVH motion header')
        channels = self.channels(raw[:h.start()])
        frames = int(h[1])
        if frames!=expected_frames or frames==0: raise ValueError(f'Frame count {frames} vs metadata {expected_frames}')
        frame_time = float(h[2])
        fps = round(1/frame_time)
        if fps!=120 or abs(fps*frame_time-1)>.0001: raise ValueError(f'Unexpected frame time {frame_time}')
        pos = np.empty((frames,24,3),np.float32)
        root = np.empty((frames,3),np.float32)
        rotations = np.empty((frames,24,4),np.float32)
        ptr = ctypes.cast(ctypes.c_char_p(raw),P).value+h.end()
        code = self.kernel(ptr,len(raw)-h.end(),frames,len(channels),channels.ctypes.data,
                           self.parents.ctypes.data,self.active.ctypes.data,self.driver.ctypes.data,
                           PARENT.ctypes.data,self.local.ctypes.data,self.root_scale,pos.ctypes.data,root.ctypes.data,
                           self.mapped.ctypes.data,self.inverse_reference.ctypes.data,rotations.ctypes.data)
        if code: raise ValueError(f'Invalid numeric data at frame {code}')
        if not (np.isfinite(pos).all() and np.isfinite(root).all()): raise ValueError('Nonfinite output')
        return pos,root,fps,rotations

    def rotation_error(self, rotations, positions):
        error = self.check_fk(len(positions), PARENT.ctypes.data, REST_OFFSETS.ctypes.data,
                              rotations.ctypes.data, positions.ctypes.data)
        if not 0 <= error < 2e-6:
            raise ValueError(f"Invalid quaternions or rotation FK mismatch: {error} m")
        return error


def save_clip(converter, raw, frames, path, *, private_directory=False, add_only=False):
    if add_only:
        return frames, 0.0, add_rotations(converter, raw, frames, path)
    pos,root,fps,rotations = converter.convert(raw,frames)
    rotation_error = converter.rotation_error(rotations, pos)
    # Verify rigid bone lengths after float32 storage on every frame.
    delta = pos[:,1:] - pos[:,PARENT[1:]]
    lengths = np.linalg.norm(delta,axis=2)
    maximum = float(np.max(np.abs(lengths-np.linalg.norm(REST[1:]-REST[PARENT[1:]],axis=1))))
    if maximum>2e-6: raise ValueError(f'Bone length error {maximum}')
    stream = io.BytesIO()
    np.savez(stream,joints=pos,root_positions=root,fps=np.int32(fps),local_rot_quat=rotations)
    if private_directory:
        # The caller publishes the enclosing directory atomically after its final clip.
        with path.open('xb') as f:
            f.write(stream.getbuffer())
        return frames,maximum,rotation_error
    replace_npz(path, stream, 0o644)
    return frames, maximum, rotation_error


def replace_npz(path, stream, mode):
    with NamedTemporaryFile(dir=path.parent, prefix='.writing-',delete=False) as f:
        temp = Path(f.name)
        try:
            os.fchmod(f.fileno(), mode)
            f.write(stream.getbuffer())
            f.close()
            os.replace(temp,path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise


def add_rotations(converter, raw, frames, path):
    # Append a member to a copy of the original ZIP. Existing NPY members and any
    # additional user fields are carried through byte-for-byte.
    pos, root, fps, rotations = converter.convert(raw, frames)
    with io.BytesIO(path.read_bytes()) as stream:
        with np.load(stream, allow_pickle=False) as old:
            original = old["joints"]
            if (original.dtype != np.float32 or original.shape != pos.shape
                    or not np.array_equal(original, pos) or not np.array_equal(old["root_positions"], root)
                    or int(old["fps"]) != fps):
                raise ValueError(f"Original pose or root differs from BVH: {path}")
            error = converter.rotation_error(rotations, pos)
            if ROTATION_FIELD in old.files:
                previous = old[ROTATION_FIELD]
                if previous.dtype != rotations.dtype or not np.array_equal(previous, rotations):
                    raise ValueError(f"Conflicting existing rotation field: {path}")
                return error
        with zipfile.ZipFile(stream, "a", compression=zipfile.ZIP_STORED) as archive:
            with archive.open(ROTATION_FIELD + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, rotations, allow_pickle=False)
        replace_npz(path, stream, path.stat().st_mode & 0o777)
        return error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--readers',type=int,default=24)
    parser.add_argument('--add-rotations', action='store_true',
                        help='Add local_rot_quat to existing Human24 files after matching them to the source BVH')
    args=parser.parse_args()
    base=args.dataset.resolve()
    out=base/'human24'
    if args.add_rotations:
        existing_spec = json.loads((out/'skeleton.json').read_text())
        if (existing_spec['joint_names'] != NAMES or existing_spec['parents'] != PARENT.tolist()
                or not np.array_equal(existing_spec['rest_positions_m'], REST)):
            raise ValueError('Existing Human24 skeleton does not match the target template')
    else:
        out.mkdir(exist_ok=False)
    converter=Converter(base/REST_FILE)
    metadata=base/'metadata/seed_metadata_v004.csv'
    with metadata.open() as f:
        expected={row['move_soma_uniform_path']: int(row['move_duration_frames']) for row in csv.DictReader(f)}
    manifest={'joint_names':NAMES,'parents':PARENT.tolist(),'rest_positions_m':REST.tolist(),
              'source_joints':SOURCE,'units':'metres','up_axis':'Y','positive_x_in_rest':'anatomical left',
              'root_motion_scale_after_cm_to_m':converter.root_scale*100,
              'retarget':'Apply each mapped source parent global rotation relative to the shared SOMA bind pose to the receiver rest bone offset, then FK. Scale the complete world pelvis path by the ratio of rest pelvis-to-toe-base vertical distances. No contact IK.',
              'arrays':{'joints':'float32 [T,24,3], pelvis-relative, world axes (heading retained)',
                        'root_positions':'float32 [T,3], world pelvis trajectory in metres',
                        'fps':'int32 scalar, native frame rate',
                        ROTATION_FIELD: ROTATION_DESCRIPTION},
              'world_positions':'joints + root_positions[:,None,:]',
              'path_mapping':'soma_uniform/bvh/<date>/<clip>.bvh -> human24/<date>/<clip>.npz'}
    if not args.add_rotations:
        (out/'skeleton.json').write_text(json.dumps(manifest,indent=2)+'\n')
    parents=set()
    pending={}
    done=total=0
    maximum=rotation_maximum=0.
    start=time.monotonic()
    def collect():
        nonlocal done,total,maximum,rotation_maximum
        for future in wait(pending,return_when=FIRST_COMPLETED).done:
            name=pending.pop(future)
            try: frames,err,rotation_error=future.result()
            except Exception as e: raise ValueError(name) from e
            done+=1
            total+=frames
            maximum=max(err,maximum)
            rotation_maximum=max(rotation_error,rotation_maximum)
            if done%2000==0:
                print(f'{done:,} clips, {total:,} frames, {time.monotonic()-start:.0f}s',flush=True)
    with ThreadPoolExecutor(args.workers) as pool, rapidgzip.open(str(base/'soma_uniform.tar.gz'),parallelization=args.readers) as file:
        with tarfile.open(fileobj=file,mode='r|',bufsize=512*1024) as tar:
            for member in tar:
                if not member.isfile(): continue
                if member.name not in expected: raise ValueError(f'Unknown or duplicate archive path: {member.name}')
                frames=expected.pop(member.name)
                rel=Path(member.name).relative_to('soma_uniform/bvh')
                if len(rel.parts)!=2 or '..' in rel.parts or rel.suffix!='.bvh': raise ValueError(rel)
                path=(out/rel).with_suffix('.npz')
                if not args.add_rotations and path.parent not in parents:
                    path.parent.mkdir()
                    parents.add(path.parent)
                while len(pending)>=args.workers*2: collect()
                raw=tar.extractfile(member).read()
                pending[pool.submit(save_clip,converter,raw,frames,path,add_only=args.add_rotations)] = member.name
        while pending: collect()
    if expected: raise ValueError(f'Missing {len(expected)} archive entries, e.g. {next(iter(expected))}')
    summary={'clips':done,'frames':total,'max_bone_length_error_m':maximum,'max_rotation_fk_joint_error_m':rotation_maximum}
    if args.add_rotations:
        summary = json.loads((out/'summary.json').read_text())
        summary.update(rotation_clips=done, rotation_frames=total, max_rotation_fk_joint_error_m=rotation_maximum)
        existing_spec['arrays'][ROTATION_FIELD] = ROTATION_DESCRIPTION
        (out/'skeleton.json').write_text(json.dumps(existing_spec,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':
    main()
