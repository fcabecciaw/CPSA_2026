# AI showcase mode ("Cosa vede l'IA") — Implementation Plan

**Status:** READY (2026-11-12)
**Goal:** a new supervisor mode `showcase` runs a pre-trained Vitis AI model-zoo detector (YOLOX-nano, COCO 80
classes) on the F200 RGB camera on the KV260 DPU, draws boxes and Italian labels on the live `/video.mjpeg` stream,
and publishes a `showcase.state` summary that the dashboard shows next to the video.
**Spec:** user requests 2026-11-12: "focus on models I dont have to train, that can be re-used with minimal effort,
and that use the camera. As simple as possible." / "yes start with yolox-nano" / later "MoveNet and SESR-S via a
model selector".
**Scope:** YOLOX-nano end to end (model package, runtime, mode, protocol, config, deploy, web view, board test).
The config and protocol shapes allow more models later.
**Non-goals:** training or re-quantizing anything; MoveNet / SESR-S (later plans reuse this mode); a model-selector
command (later); changes to medical, hologram or rps behaviour.
**Constraints:**
- Medical mode is frozen: additive changes only (mode enums, new message type, new mode button). Existing tests pass
  unmodified except additive mode-enum assertions.
- One mode active at a time (supervisor rule); the RGB camera and the DPU are released on `stop()`.
- Board: `ssh kria@10.42.0.83`; deploys keep `--model build/vitis/hagrid_lrelu512/compiled/pose_halfbust62.xmodel`
  and `--hand-model build/vitis/hand_topdown/compiled/hand21_topdown.xmodel`.

## Facts established (2026-11-12)

- Zoo entry: Vitis-AI v3.0 `model_zoo/model-list/pt_yolox-nano_coco_416_416_1G_3.0/model.yaml`.
  The prebuilt KV260 xmodel (`yolox_nano_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz`) targets DPU fingerprint
  `0x101000056010407`; **our board's DPU is `0x101000016010407`**, so the prebuilt file does not load.
- The float+quantized package (`pt_yolox-nano_coco_416_416_1G_3.0.zip`, 23.7 MB,
  `https://www.xilinx.com/bin/public/openDownload?filename=pt_yolox-nano_coco_416_416_1G_3.0.zip`) contains the
  QAT-converted `qat/convert_qat_results/YOLOX_0_int.xmodel`. Compiled with the repo toolchain
  (`tools/vitis/run_vitis.sh bash tools/vitis/compile.sh <int.xmodel> build/vitis/yolox_nano/compiled yolox_nano`)
  it gives fingerprint `0x101000016010407` (match). No retraining, no requantization.
- Compiled graph (inspected with `tools/vitis/inspect_xmodel.py`): 1 DPU subgraph; input `[1, 416, 416, 3]` xint8
  NHWC, fix_point **-2**; three outputs NHWC xint8, fix_point 3: stride 8 `[1,52,52,85]`, stride 16 `[1,26,26,85]`,
  stride 32 `[1,13,13,85]`.
- Zoo prototxt (`yolox_nano_pt.prototxt`): `mean 0, scale 1` (raw 0–255 pixels), `type: YOLOX_NANO`,
  `conf_threshold 0.3`, `nms_threshold 0.65`, 80 classes, anchor-free, strides 32/16/8.
- Upstream YOLOX preprocessing is BGR, letterbox with pad value 114 (top-left placed); the per-cell 85 channels are
  `[reg x, y, w, h, objectness, 80 class scores]` and decode as `xy = (t_xy + grid) * stride`,
  `wh = exp(t_wh) * stride`, `score = obj * cls`. Whether the zoo graph's DPU outputs are pre- or post-sigmoid
  is **to verify** in Task S1 against the package code (`quantized/YOLOX.py`) and a reference run; fix_point 3
  (range ±16) suggests raw logits.
- Reusable code: `cpsa.adapters.pose.dpu` (`quantize_fix_point`, `quantize_rgb_lut`, `dequantize_fix_point`,
  `order_outputs_by_stride`, runner setup pattern in `DpuPoseEstimator.__init__`), `cpsa.vision.pose_decode`
  (`letterbox(rgb, size, pad_value, resize=...)`, `unletterbox`, class-aware `_nms`), the RGB camera
  `cpsa.adapters.camera.uvc.UvcVideoSource` (constructed in `cpsa.board.__main__` from the `vision.camera` config),
  the video sink pattern `_HandMode._offer_video` in `cpsa.board.hand_modes` (`ctx.publisher.offer_video_frame`).
- `input_quantization_lut(fix_point)` assumes `value / 255`; YOLOX needs `value * 1.0` → a LUT with a value scale.

## Approach

```
UvcVideoSource (RGB 640x480@30, vision.camera settings)
 -> DpuYoloxDetector.detect(rgb): RGB->BGR, letterbox 416 (pad 114), LUT quantize (scale 1.0, fix -2), DPU,
    dequantize, decode_yolox, NMS, unletterbox -> list[Detection] (source pixels)
 -> ShowcaseMode: draw boxes + labels on a copy of the frame -> offer_video_frame (MJPEG, left pane)
                  publish showcase.state (<= 10 Hz) -> dashboard right pane (object list + counts)
```

- Overlay is drawn on the board (always aligned with the video; one web view for all future models).
- Labels Italian by default, ASCII only (OpenCV Hershey fonts have no accents); English list kept for `labels: en`.
- Medical's `DpuPoseEstimator` is not refactored (frozen); the detector reuses its public helpers.

## Frozen interfaces

### F-S1 — detector runtime (`src/cpsa/vision/detection.py`, numpy only)

```python
@dataclass(frozen=True)
class Detection:
    box: tuple[
        float, float, float, float
    ]  # x1, y1, x2, y2 in source-image pixels, clipped to the image
    score: float  # 0..1
    class_id: int  # COCO-80 index (0 = person)


class ObjectDetector(Protocol):
    @property
    def input_size(self) -> tuple[int, int]: ...  # (height, width) of the model input
    @property
    def stage_ms(self) -> dict[str, float]: ...  # keys: preprocess, dpu, decode (last call)
    def detect(
        self, rgb: NDArray[np.uint8]
    ) -> list[Detection]: ...  # (H, W, 3) RGB uint8; score-desc order
```

`src/cpsa/vision/coco.py`: `COCO80_EN: tuple[str, ...]` (80, standard order) and `COCO80_IT: tuple[str, ...]`
(80, ASCII Italian, e.g. `persona`, `bicicletta`, `auto`, `tazza`, `cellulare`, `bottiglia`, `forbici`,
`orsacchiotto`, `spazzolino`). Test pins length 80, ASCII, index 0 = `person` / `persona`.

`src/cpsa/vision/yolox_decode.py`:
`decode_yolox(heads: Sequence[NDArray[np.floating]], strides: Sequence[int], *, conf_thr: float, iou_thr: float,
logits: bool) -> tuple[NDArray (N,4) xyxy input-space, NDArray (N,) scores, NDArray (N,) int class_ids]`
(`heads[i]` is `(H_i, W_i, 85)`; `logits=True` applies sigmoid to objectness and classes). Max 100 boxes after NMS.

### F-S2 — protocol v1 `showcase.state` (new schema + example in `docs/protocol/v1/`)

```jsonc
{ "model": "yolox_nano",
  "width": 640, "height": 480,
  "fps": 29.7 | null, "dpu_ms": 4.1 | null, "total_ms": 21.0 | null,
  "objects": [ { "label": "persona", "class_id": 0, "score": 0.91, "box": [x1, y1, x2, y2] } ],  // <= 20, score desc
  "counts":  [ { "label": "persona", "count": 2 } ] }                                            // count desc, then label
```

Mode enums gain `"showcase"`: `cpsa.supervisor.contract.ModeName`, `cpsa.application.config.SUPERVISOR_MODES`,
`cpsa.adapters.web.server.MODES` (+ `CAPABILITIES` `"showcase.state"`, `_applies_to`: `showcase.*` only in
`showcase`), and the 4 schemas with a mode enum (`hello`, `status`, `mode.progress`, `mode.set`). Builder
`cpsa.protocol.showcase.showcase_state(...)` (stdlib only, layer rule).

### F-S3 — config `showcase:` (optional section; the mode is registered only when present)

```yaml
showcase:
  default_model: yolox_nano
  labels: it                      # it | en
  max_fps: 30                     # processing cap
  models:
    - name: yolox_nano
      kind: yolox                 # decoder family; only "yolox" is accepted now
      path: models/yolox_nano.xmodel
      conf_thr: 0.35
      iou_thr: 0.45
```

The RGB camera settings come from the existing `vision.camera` section (device/match/size/fps/exposure/gain/
power_line_hz), as medical uses them. `ShowcaseConfig` follows the `HandsConfig` pattern in
`cpsa.application.config`.

## Verification contract

- **AC-S1** — compiled model: `build/vitis/yolox_nano/compiled/yolox_nano.xmodel` has fingerprint
  `0x101000016010407`, 1 DPU subgraph, input 1×416×416×3. Evidence: compile log + `inspect_xmodel.py` JSON.
- **AC-S2** — decode correctness: on ≥ 20 real images, `DpuYoloxDetector`'s pre/post-processing applied to the
  reference outputs of the package model matches the package's own postprocess (same class, IoU ≥ 0.9 for every
  box with score ≥ 0.35; no extra/missing boxes). Unit tests pin decode maths (grid, stride, exp, sigmoid switch,
  NMS, unletterbox) on synthetic heads. Evidence: reference script report + tests.
- **AC-S3** — mode contract: `showcase` satisfies `Mode` (start/stop/health keys); missing model file →
  `ModeStartError("model_missing")`; no RGB node → `camera_missing`; `stop()` releases camera and runner; overlay
  frames reach `offer_video_frame`; `showcase.state` published ≤ 10 Hz. Evidence: `tests/board/test_showcase_mode.py`
  with fake camera/detector/publisher.
- **AC-S4** — protocol/config: new schema validates its example and builder output; enums include `showcase`
  (Python + JSON); config parses the section, rejects unknown `kind`/`labels`, and omitting `showcase:` changes
  nothing. Evidence: `tests/unit/test_protocol_showcase.py`, contract tests, `tests/application/test_config.py`.
- **AC-S5** — web: mode button "Cosa vede l'IA"; right pane lists the counts (large) and the top objects; top bar
  shows fps/latency; mock stream renders it. Evidence: `npm run typecheck && npm run build` + screenshots.
- **AC-S6** — board: in `showcase`, ≥ 25 fps processed, DPU ≤ 10 ms/frame, boxes plausible on real objects
  (person, cup, bottle, phone, book) seen by the user; switching to medical / hologram / rps and back works.
  Evidence: `/health` + `showcase.state` samples + user observation.
- **AC-S7** — regression: full gate green except the pre-existing `test_hygiene` kalman2d failure.

## Tasks

### S1 — YOLOX-nano runtime and reference check
**Depends on:** none.
**Files:** new `src/cpsa/vision/detection.py`, `src/cpsa/vision/coco.py`, `src/cpsa/vision/yolox_decode.py`,
new `src/cpsa/adapters/detect/__init__.py` + `src/cpsa/adapters/detect/yolox_dpu.py` (`DpuYoloxDetector(xmodel_path,
*, conf_thr=0.35, iou_thr=0.45)` implementing F-S1 with VART, reusing `cpsa.adapters.pose.dpu` helpers; add a LUT
helper with a value scale without changing the existing function's behaviour), new `tools/vitis/zoo_yolox_nano.sh`
(download the package, extract the QAT int xmodel, compile, inspect), new `tools/vitis/yolox_reference.py`
(AC-S2 report), tests `tests/unit/test_yolox_decode.py`, `tests/unit/test_coco.py`,
`tests/adapters/test_yolox_dpu.py` (fake runner), `tools/vitis/README.md` section.
**Requirements:** AC-S1, AC-S2. "No training, no quantization."
**Actions:** confirm the output channel order and sigmoid state from `quantized/YOLOX.py`; build the reference path
(package code in the Vitis docker with `pytorch_nndct`, or the float `float/yolox_nano.pth` with the package code)
on ≥ 20 images (e.g. `build/models/hand_topdown/val_subset/images` people images plus any COCO-like images on disk);
implement decode + detector; compare.
**Verification:** `uv run --no-sync pytest -q tests/unit/test_yolox_decode.py tests/unit/test_coco.py tests/adapters/test_yolox_dpu.py tests/architecture/test_layers.py`;
reference report; ruff; mypy.

### S2 — mode, protocol, config, supervisor wiring, deploy
**Depends on:** F-S1 (use a fake detector until S1 lands).
**Files:** new `src/cpsa/board/showcase_mode.py`, new `src/cpsa/protocol/showcase.py`, new
`docs/protocol/v1/showcase.state.schema.json` + example; additive edits to `cpsa.supervisor.contract`,
`cpsa.application.config` (`ShowcaseConfig`, `SUPERVISOR_MODES`), `cpsa.adapters.web.server` (MODES, CAPABILITIES,
`_applies_to`), the 4 mode-enum schemas, `cpsa.board.supervisor_main` (register when `showcase:` exists),
`config/board.demo.yaml` (`showcase:` section), `deploy/deploy.sh` (repeatable `--showcase-model PATH` → copied to
`~/cpsa/models/<basename>`), `deploy/README.md`; tests `tests/board/test_showcase_mode.py`,
`tests/unit/test_protocol_showcase.py`, additive cases in `tests/application/test_config.py`,
`tests/adapters/test_web_server.py`, contract tests.
**Requirements:** AC-S3, AC-S4.
**Actions:** mode loop: read RGB frame → detect → draw (stable per-class colour, `"<label> <score%>"`, 2 px boxes) →
`offer_video_frame` (check the colour order the server encoder expects) → throttled `showcase.state`; health keys
`camera`, `model`, `fps`, `dpu_ms`, `errors`; release camera + runner on stop and on failed start.
**Verification:** `uv run --no-sync pytest -q tests/board/test_showcase_mode.py tests/unit/test_protocol_showcase.py tests/application/test_config.py tests/adapters/test_web_server.py tests/supervisor tests/contract`;
`bash -n deploy/deploy.sh`; ruff; mypy.

### S3 — dashboard view
**Depends on:** F-S2 (mock until S2 lands).
**Files:** `web/src/protocol.ts` (mode + message type), new `web/src/showcase.ts`, `web/src/main.ts`,
`web/src/views.ts`, `web/src/labels.ts`, `web/src/mock.ts`, `web/src/styles.css`, `web/index.html` as needed.
**Requirements:** AC-S5. Layout from D1 (slim top bar, video left, panel right).
**Actions:** mode button "Cosa vede l'IA"; right panel: counts as large rows ("2 × persona"), then the top objects with
score bars; empty state "Mostra un oggetto alla telecamera"; fps/latency in the top bar from `showcase.state`.
**Verification:** `cd web && npm run typecheck && npm run build`; screenshots (mock) at 1280×720 and 1920×1080.

### S4 — board test (coordinator + user)
**Depends on:** S1–S3 integrated, full gate run, user OK to deploy.
**Actions:** `tools/vitis/zoo_yolox_nano.sh` (or reuse the compiled file); `deploy/deploy.sh --model
build/vitis/hagrid_lrelu512/compiled/pose_halfbust62.xmodel --hand-model
build/vitis/hand_topdown/compiled/hand21_topdown.xmodel --showcase-model build/vitis/yolox_nano/compiled/yolox_nano.xmodel`;
switch to "Cosa vede l'IA"; sample `showcase.state` for 30 s; switch through the other modes and back.
**Acceptance:** AC-S6.

## Integration verification

Full gate once on the integrated commit:
`uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src`;
`cd ml && .venv/bin/python -m pytest -q`; `cd web && npm run typecheck && npm run build`.
Expected: all green except the pre-existing `tests/architecture/test_hygiene.py::test_no_forbidden_artifacts_are_tracked`.

## Open decisions / risks

- **Sigmoid state of the DPU outputs** — resolved in S1 by the reference check (decoder flag `logits`).
- **Board CPU budget** — drawing + JPEG encode of 640×480 on the A53 alongside decode; if < 25 fps, cap the state
  rate, draw on a half-size copy, or lower the MJPEG quality.
- **Label language** — Italian ASCII by default (`labels: it`); switchable to English in config.
- **Licence** — Vitis AI model zoo terms (Apache-2.0 for this entry per `model.yaml` link; YOLOX upstream Apache-2.0).
