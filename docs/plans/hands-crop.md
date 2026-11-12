# Plan: depth-localized hand crops for the hologram and rps demos

Status: READY (2026-11-12). Supersedes the full-frame hand detector of `docs/plans/hands-demos.md`
("Model (hand detector + 21 keypoints)"); everything else in that plan (modes, protocol F-E, web, RPS game and
classifier) stays.

## Why

On the board (2026-11-12) the full-frame YOLOv8n-pose hand21 model (notebook 08) detected hands on the user's
bust, arm and background, and showed keypoints with no hand in view. Host FP32 on 270 frames of the final
setup reproduced it: 57 boxes 0.5 m wide at 0.77 m, scores up to 0.58. When the hand was found, keypoints were
stable (keypoint conf 0.97). The failure is localization, not keypoints.

## Decision

MediaPipe-style two stages, the first without a network:

```
F200 INZI (IR + depth, pixel-aligned)
 -> 1. depth localizer (CPU): table background, foreground band, blobs, hand end of the blob, palm centre,
       crop centre/side/angle per hand (max 2)
 -> 2. upright square crop of the preprocessed IR (cv2.warpAffine), 192x192
 -> 3. YOLOv8n-pose hand21 on the crop (DPU): keep the best box; its score is the hand-present score
 -> 4. keypoints back to the full frame (inverse affine) -> depth lift -> HandFrame (protocol unchanged)
```

## Setup facts (2026-11-12 captures)

- `build/ir_capture_20261112_empty/` 30 frames of the empty table; `build/ir_capture_20261112_hand/` 300 frames,
  one hand moving (open, fist, scissors, rotations, entering/leaving); `build/ir_capture_20261112/` 270 frames,
  one hand plus the user's bust in view at the start. Each has `ir_XXXXX.png` (raw 8-bit IR 640x480),
  `depth_XXXXX.npy` (raw Z16), `meta.json`. They may contain faces: local only, never committed or uploaded.
- Depth: `depth_m = raw * 3.125e-5`. Empty table: 67% valid pixels; RANSAC plane fits 91% of them; camera
  ~0.42 m above the table, tilted ~29 deg (image top = far side, 0.55-0.85 m; bottom = near, ~0.39 m).
- Intrinsics: nominal `fx = fy = 430, cx = 320, cy = 240` (config `hands.intrinsics`).

## Frozen interfaces

### F-G — crop convention (training and runtime MUST match)

- Model input `N = 192` (square), 3 channels.
- Preprocessing happens on the **full frame first**, then the crop:
  runtime `cpsa.adapters.camera.irdepth.ir_to_input(ir)` (min-max + CLAHE clip 3.0, tiles 8x8, 3ch);
  training: IR-look (as `cpsa_ml.datasets.handframe.ir_look`) then the same min-max + CLAHE on the full image.
- A crop is `(cx, cy, side, angle)`: centre in full-frame pixels, side in full-frame pixels, `angle` in radians.
  Let `u` be the hand's "up" direction in the image (wrist -> fingers). `angle` is the rotation that maps `u` onto
  image up `(0, -1)`.
- Forward map of a full-frame point `p` to crop pixels:
  `q = R(angle) @ (p - c) * (N / side) + (N/2, N/2)`, with `R(a) = [[cos a, -sin a], [sin a, cos a]]`
  (image coordinates, y down). Crop with `cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_LINEAR,
  borderMode=cv2.BORDER_CONSTANT, borderValue=0)` where `M` is the 2x3 matrix of that map. The inverse map takes
  crop keypoints back to the full frame.
- Training crops from GT 21 keypoints (MediaPipe order): palm centre `c0 = mean(kpts[[0, 5, 9, 13, 17]])`;
  `u = kpts[9] - kpts[0]`; hand size `h = max side of the bbox of the labelled keypoints`.
  Jitter per sample: fill `f ~ U(0.40, 0.80)`, `side = h / f`; centre shift `U(-0.15, 0.15) * side` on each
  axis; angle `angle_gt + U(-45, +45) deg`. Keypoints outside the crop get `v = 0`. The box label is the bbox of
  the in-crop labelled keypoints, padded by 10% of its size, clipped to the crop. Several hands in one image:
  each hand gives its own crop; other hands' keypoints in that crop are not labelled (so prefer images whose
  other hands do not overlap the crop; drop the crop if another hand's box overlaps it by > 20% IoU).
- Negatives: about 10% of the crops contain no hand (random crops away from all hand boxes, random side in the
  same pixel range, random angle); empty label file.
- Runtime crops: `c` = palm centre from the localizer, `u` = blob axis (arm entry -> hand tip),
  `side = hand_extent / 0.6`.

### F-H — depth localizer (runtime)

`src/cpsa/adapters/pose/hand_locator.py` (adapters layer: OpenCV allowed), pure functions + small dataclasses:

```python
@dataclass(frozen=True)
class TableModel:            # from the empty-table calibration
    background_m: NDArray[np.float32]   # (H, W) per-pixel median depth, NaN where never valid
    plane: tuple[float, float, float, float]  # n_x, n_y, n_z, d in camera metres (n . X + d = 0), RANSAC
    zone: NDArray[np.bool_]  # (H, W) table region where hands are searched

@dataclass(frozen=True)
class HandRoi:
    cx: float; cy: float; side: float; angle: float   # F-G crop, full-frame pixels / radians
    palm_depth_m: float; area_m2: float; tip: tuple[float, float]; entry: tuple[float, float]

def calibrate_table(depth_frames: Sequence[NDArray[np.uint16]], *, scale_m: float, intrinsics) -> TableModel
def locate_hands(depth: NDArray[np.uint16], table: TableModel, *, scale_m: float, intrinsics,
                 config: LocatorConfig) -> list[HandRoi]   # at most config.max_hands, largest first
```

Foreground: valid depth and height above the background (fallback: above the plane where the background is NaN)
within `[config.min_height_m, config.max_height_m]` (default 0.02, 0.35), inside `zone`. Morphological open/close,
connected components, drop components with real area `< config.min_area_m2` (default 0.002 = 20 cm^2), using
`area_px * (z / fx)^2`. Entry point = where the component touches the zone border (centroid of that contact);
no contact -> the component's point nearest to the near image edge. Tip = component point farthest from the
entry. Palm centre = maximum of the distance transform restricted to the part of the component within
`config.hand_length_m` (default 0.20) of the tip. `u` = entry -> tip. Hand extent = the larger side of the
`u`-aligned bbox of that hand part. Compute on a 2x-downsampled mask (320x240); return full-res coordinates.

## Tasks

### C1 — depth localizer + offline evaluation (worker)

Implement F-H with unit tests on synthetic depth (plane + box "arm" + blob "hand": entry, tip, palm, angle,
size, the empty-table case returns []). Add `scripts/hand_locator_preview.py` (host): calibrate on
`build/ir_capture_20261112_empty`, run on `build/ir_capture_20261112_hand` and `build/ir_capture_20261112`,
write (a) a contact sheet of full frames with mask outline, entry, tip, palm, crop square; (b) a sheet of the
192x192 crops (via `ir_to_input` then the F-G warp); (c) `stats.json` (frames with 0/1/2 hands, timing per frame).
Also a pure-numpy F-G helper `src/cpsa/vision/hand_crop.py` (`crop_matrix`, `to_crop`, `from_crop`) with tests
(round trip, angle convention). Acceptance: no ROI on the empty frames; 1 ROI on frames with a hand clearly in
view; crops show the hand upright (coordinator + user review the sheet); locate_hands < 10 ms on the host.

### C2 — crop dataset builder + notebook 09 (worker)

`ml/src/cpsa_ml/datasets/handcrop.py` (+ CLI), reusing `handframe` sources, IR-look and CLAHE; F-G crops,
2 crops per hand instance (different jitter), ~10% negatives; YOLO-pose dataset 192x192, `kpt_shape [21, 3]`,
`flip_idx` identity, `names {0: hand}`, `stats.json`, contact sheet; tests (F-G maths, label validity,
negatives, determinism with a seed). Notebook `ml/notebooks/09_hand_crop.ipynb` (+ kernel dir, sync script,
notebook tests), mirroring notebook 08: build the dataset on Kaggle, train YOLOv8n-pose hand21 LeakyReLU at
`imgsz 192`, `mosaic 0`, `degrees 0` (rotation is in the crops), `fliplr 0.5`, `hsv_h 0`, `hsv_s 0`,
export ONNX at (192, 192), zip the outputs. Kernel id `fabiocabeccia/cpsa-hand-crop`, GPU T4, internet on.

### C3 — quantize, compile, evaluate (after Kaggle)

T6 tools at 192x192; hand21 PCK@0.2 FP32 vs INT8 on val crops; overlay on the C1 crops from the captures.

### C4 — runtime integration, C5 — board test

`HandPerception`: locator -> crop -> DPU per ROI -> inverse map -> lift; table calibration at mode start
(hands out of view, ~1 s), config keys under `hands.locator` and `hands.model` (192x192); then the board test
(AC-13, AC-14 of `hands-demos.md`).
