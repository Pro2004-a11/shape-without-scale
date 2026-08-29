# Shape without scale

![Estimated vs ground-truth trajectory](trajectory.png)

**There are two trajectories in that plot.** One is motion capture. One is a single RGB
camera. You cannot see the difference, because it is 8.2 mm over a 7.10 m path.

You also cannot see the thing that makes it work — a scale factor of 0.566 that came from
the ground truth, not from the images. This repo is about that second sentence.

Reproduces the numbers: **622/622 frames registered, 8.2 mm from motion capture, 29.3 dB
Gaussian splat** — from a single RGB stream with no depth, no IMU and no LiDAR.

Everything here runs on **TUM RGB-D `fr3/long_office_household`**, which is public and has
motion-capture ground truth, so you can check the result rather than take my word for it.
The Kinect's depth channel is present in that dataset and is **never used** below.

    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

Total runtime on an RTX 4070 Ti: ~6 min of SfM, ~25 min of splat training.

---

## 1. Structure from motion — RGB only

```bash
IM=path/to/fr3_long_office/rgb          # 622 PNGs, 640x480, real handheld Kinect
W=work/sfm

colmap feature_extractor \
  --database_path $W/db.db --image_path "$IM" \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.max_num_features 16384

colmap sequential_matcher \
  --database_path $W/db.db --SequentialMatching.overlap 15

colmap mapper \
  --database_path $W/db.db --image_path "$IM" --output_path $W/sparse
```

Sequential matching, not exhaustive — this is a continuous handheld video, so neighbours in
time are neighbours in space and you get the same result for a fraction of the pairs.

Expected: `622 / 622 registered, 49,044 points, 0.739 px mean reprojection, 6.1 min`.

> **Gotcha.** `--SiftExtraction.max_image_size` does not exist in current COLMAP; it is
> `--FeatureExtraction.max_image_size`. An unrecognised option makes `feature_extractor`
> exit immediately, and the rest of the chain then runs happily on an **empty database**
> and reports `No images with matches` — which points nowhere near the real cause. Check
> that feature extraction actually logged features before moving on.

## 2. Undistort → the canonical PINHOLE dataset

```bash
colmap image_undistorter \
  --image_path "$IM" --input_path $W/sparse/0 \
  --output_path work/undistorted --output_type COLMAP

mkdir -p work/undistorted/sparse/0
mv work/undistorted/sparse/*.bin work/undistorted/sparse/0/
```

> **Gotcha.** Do this even if your trainer claims to undistort on the fly. LichtFeld Studio
> undistorts every image at load time for a `SIMPLE_RADIAL` model and blows its pinned-memory
> allocator — 3,396 failures, dead at iteration 0, with 47 GB of host RAM free, so it is not
> what it looks like. Feeding it an already-PINHOLE dataset makes the problem vanish.

## 3. Train the splat — seeded from SfM points only

```bash
LichtFeld-Studio --headless \
  -d work/undistorted -o work/undistorted/run \
  -i 30000 --eval --test-every 8 --max-cap 2000000
```

No depth supervision, no priors. The only geometry entering training is COLMAP's own
49k triangulated points.

Expected: `29.27 dB / 0.913 SSIM at 30k, 2M gaussians`.

## 4. The measurement that matters

```bash
python ate_scale.py --est $W/sparse/0 --gt path/to/groundtruth_model/0
```

```
622 estimated poses, 622 reference, 622 matched
reference trajectory extent: 7.10 m

  SE(3)   scale NOT solved   scale   1.0000   ATE median  1570.2 mm   ( 22.12% of extent)
  Sim(3)  scale solved out   scale   0.5661   ATE median     8.2 mm   (  0.11% of extent)

  The gap between those two lines is the part monocular vision does not give you.
```

**That is the whole story of monocular 3D in two lines.** The shape of the trajectory is
recovered to 8 millimetres over seven metres. The *size* of it is not recovered at all —
the scale factor of 0.566 came from fitting against ground truth, not from the images. Run
dense RGB-D SLAM on the same footage and it returns real metres and **124 mm** of drift:
fifteen times worse, and correct in a way the 8.2 mm never is.

Every impressive monocular accuracy number you have seen is the second line. That is not
cheating — it is the standard and correct way to evaluate a monocular system — but the
scale had to come from somewhere, and it is worth saying which line you are quoting.

---

## Files

| file | what it does |
|---|---|
| `ate_scale.py` | COLMAP model vs reference poses, Umeyama alignment with and without scale |

`ate_scale.py` has no dependencies beyond numpy and works on any COLMAP text model, so it
is reusable on your own reconstructions.
