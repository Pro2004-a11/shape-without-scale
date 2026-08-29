#!/usr/bin/env python3
"""What monocular SfM actually recovers: shape, not size.

Compares a COLMAP reconstruction against ground-truth poses two ways.

    SE(3)  rotation + translation only. This is the honest question "where was the
           camera, in metres?" Monocular SfM cannot answer it - the reconstruction is
           correct up to an unknown scale factor, so this number is meaningless-large.

    Sim(3) rotation + translation + a single global scale, solved by Umeyama. This asks
           "was the SHAPE of the trajectory right?" - and it usually is, to millimetres.

Every impressive monocular accuracy figure you have ever read is the second number. That
is not cheating, but the scale had to come from somewhere, and it is worth saying so out
loud. Run this on your own reconstruction and see the gap.

Usage:
    python ate_scale.py --est sparse/0 --gt gt_sparse/0

Both arguments are COLMAP text models (cameras.txt / images.txt / points3D.txt); the GT
model just needs images.txt with the reference poses under matching image filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def camera_centres(model_dir: Path) -> dict[str, np.ndarray]:
    """COLMAP images.txt -> {image_name: camera centre in world coords}.

    COLMAP stores world-to-camera as (quaternion, translation) with the quaternion in
    QW QX QY QZ order - w FIRST, unlike most libraries. The camera centre is -R^T t.
    A pose line has exactly 10 tokens and ends with the filename; the POINTS2D line that
    follows every pose does not, which is how they are told apart.
    """
    txt = model_dir / "images.txt"
    if not txt.exists():
        extra = ""
        if (model_dir / "images.bin").exists():
            extra = (f"\n  Found images.bin - this is a binary model. Convert it first:\n"
                     f"    colmap model_converter --input_path {model_dir} "
                     f"--output_path {model_dir} --output_type TXT")
        raise SystemExit(f"no images.txt in {model_dir}{extra}")

    out: dict[str, np.ndarray] = {}
    for line in txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 10 or not f[-1].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        qw, qx, qy, qz = (float(v) for v in f[1:5])
        t = np.array([float(v) for v in f[5:8]])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        out[f[-1]] = -R.T @ t
    return out


def align(A: np.ndarray, B: np.ndarray, with_scale: bool) -> tuple[np.ndarray, float]:
    """Umeyama: best rigid (or similarity) map taking A onto B. Returns per-point error, scale."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    # reflection guard - without this a mirrored solution can win on noisy data
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    s = float((S * [1, 1, d]).sum() / (X ** 2).sum()) if with_scale else 1.0
    return np.linalg.norm(Y - s * (X @ R), axis=1), s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", type=Path, required=True, help="COLMAP model to evaluate")
    ap.add_argument("--gt", type=Path, required=True, help="COLMAP model holding reference poses")
    a = ap.parse_args()

    est, gt = camera_centres(a.est), camera_centres(a.gt)
    shared = sorted(set(est) & set(gt))
    if not shared:
        raise SystemExit("no image names in common - the two models name their frames differently")
    A = np.array([est[k] for k in shared])
    B = np.array([gt[k] for k in shared])
    span = float(np.linalg.norm(B.max(0) - B.min(0)))

    print(f"{len(est)} estimated poses, {len(gt)} reference, {len(shared)} matched")
    print(f"reference trajectory extent: {span:.2f} m\n")

    for label, with_scale in (("SE(3)   scale NOT solved", False),
                              ("Sim(3)  scale solved out", True)):
        e, s = align(A, B, with_scale)
        print(f"  {label}   scale {s:8.4f}   ATE median {np.median(e) * 1000:9.1f} mm"
              f"   RMSE {np.sqrt((e ** 2).mean()) * 1000:9.1f} mm"
              f"   ({100 * np.median(e) / span:6.2f}% of extent)")

    print("\n  The gap between those two lines is the part monocular vision does not give you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
