# Native 2D keypoint Kalman kernel (C) — Implementation Plan

**Status:** READY

**Goal:** the per-frame Kalman predict/update of `KeypointTracker2D` runs in a C kernel loaded through `ctypes`, so tracking on the KV260 drops from 52 ms/frame (62 keypoints, measured 2026-11-11) to well under 5 ms/frame, with results equal to the NumPy path.

**Requested by:** the user, 2026-11-11 ("launch another [worker] that implement the kalman filter in c"), in parallel with the NumPy vectorization of `src/cpsa/vision/tracker2d.py` (task TV, worker `ctx_6db8cc1e91a1`).

**Scope:**
- the C kernel;
- its build (on the host for the tests, on the board in `deploy/deploy.sh`);
- the `ctypes` adapter and the NumPy reference kernel;
- the integration behind a configuration switch, with a NumPy fallback.

**Non-goals:**
- RTS smoothing (`KeypointTracker2D.smoothed`, run once per episode);
- subject selection, skeleton, face width and framing;
- GTSAM or factor graphs;
- Rust.

**Constraints:**
- Layer rules (`tests/architecture/test_layers.py`):
  - `cpsa.vision` may import only `domain` and `numpy`;
  - `ctypes` and native libraries belong only in `cpsa.adapters`, `cpsa.board` or `cpsa.recorder`;
  - `cpsa.application` must not import `cpsa.adapters`.
- Board: Ubuntu 22.04 aarch64, `gcc 11.4` present, Python 3.10, Cortex-A53.
- Host: x86_64, `gcc` present, Python 3.14.
- No new Python dependency. No committed binaries for this kernel; the board build happens at deploy.
- The demo is Friday 2026-11-13: if the library is missing or broken, the tracker falls back to NumPy and logs one WARNING. It never stops the demo.

## Approach

The C kernel does only the batched filter math. Everything else (masks, re-initialization, dropout counters, status, history) stays in Python on the stacked arrays that task TV already builds. A frame then makes one `ctypes` call instead of ~600 NumPy calls.

### F15 — frozen C ABI (`native/kalman2d/kalman2d.c`, header `kalman2d.h`)

```c
#include <stdint.h>
int32_t cpsa_kalman2d_abi_version(void);          /* returns 1 */
int32_t cpsa_kalman2d_step(
    int32_t k,                 /* keypoints, >= 0 */
    const double *F,           /* 6x6 transition, row-major */
    const double *Q,           /* 6x6 process noise */
    const double *x,           /* k x 6 prior state */
    const double *P,           /* k x 6 x 6 prior covariance */
    const uint8_t *predict,    /* k flags: 1 = apply F, Q */
    const uint8_t *update,     /* k flags: 1 = run the measurement update */
    const double *z,           /* k x 2 measured position (px) */
    const double *r,           /* k isotropic measurement variance, R_i = r_i * I2 */
    double gate_chi2,
    double *x_pred, double *p_pred,   /* outputs, k x 6 and k x 6 x 6 */
    double *x_filt, double *p_filt,   /* outputs */
    uint8_t *accepted);        /* output, k flags */
/* returns 0, or -1 when k < 0 or a required pointer is NULL (k == 0 needs no arrays) */
```

All arrays are C-contiguous `float64` (flags `uint8`). The kernel:
- has no globals and does not allocate, so it is thread-safe;
- compiles as C99 with `-O2` (no `-ffast-math`: equality with NumPy matters more than the last nanosecond);
- writes every output element for every keypoint.

**Semantics, per keypoint `i`** (this is the contract the NumPy reference and the C code must both meet):
1. **Predict.**
   - If `predict[i]`: `x_pred = F x`, `P_pred = F P Fᵀ + Q`.
   - Else: `x_pred = x`, `P_pred = P` (copies).
2. **Update.** If `update[i]`:
   - `y = z − x_pred[0:2]`;
   - `S = P_pred[0:2, 0:2] + r·I₂`;
   - `det = S₀₀S₁₁ − S₀₁S₁₀`;
   - reject if `det` is not finite or `det ≤ 0`;
   - `S⁻¹ = [[S₁₁, −S₀₁], [−S₁₀, S₀₀]] / det`;
   - `d² = yᵀ S⁻¹ y`; reject if `d²` is not finite or `d² > gate_chi2`;
   - `K = P_pred[:, 0:2] · S⁻¹` (6×2);
   - `x_filt = x_pred + K y`;
   - `P_filt = P_pred − K · P_pred[0:2, :]`, which equals `(I − KH)P_pred` with `H` selecting the position;
   - `P_filt = ½(P_filt + P_filtᵀ)`;
   - `accepted[i] = 1`.
3. **Otherwise** (no update, or rejected): `x_filt = x_pred`, `P_filt = P_pred`, `accepted[i] = 0`.

This is the math of `KeypointTracker2D._measurement_update` at commit `535ceef`. There `solve` and `pinv` equal the explicit inverse for an invertible symmetric 2×2 `S`. A `LinAlgError` there is a rejection here.

### F16 — frozen Python interface

`src/cpsa/vision/kalman_kernel.py` (new, pure `numpy`):

```python
@dataclass(frozen=True)
class KalmanStep:
    x_pred: NDArray[np.float64]    # (K, 6)
    p_pred: NDArray[np.float64]    # (K, 6, 6)
    x_filt: NDArray[np.float64]    # (K, 6)
    p_filt: NDArray[np.float64]    # (K, 6, 6)
    accepted: NDArray[np.bool_]    # (K,)

class FilterKernel(Protocol):
    def step(self, F, Q, x, P, predict, update, z, r, gate_chi2: float) -> KalmanStep: ...

class NumpyKalmanKernel:           # reference + default; batched NumPy, F15 semantics
    def step(...) -> KalmanStep: ...
```

Inputs: `F`, `Q` (6, 6); `x` (K, 6); `P` (K, 6, 6); `predict`, `update` (K,) bool; `z` (K, 2); `r` (K,). Inputs are never modified.

`src/cpsa/adapters/native/kalman.py` (new):
- `NativeKalmanKernel(lib_path)` implements `FilterKernel` through `ctypes`. It converts inputs with `np.ascontiguousarray`, allocates outputs, calls `cpsa_kalman2d_step` once, and raises `RuntimeError` on a non-zero return.
- `load_native_kalman(path: Path | None = None) -> FilterKernel | None`:
  - tries `path`, then the environment variable `CPSA_KALMAN2D_LIB`, then `LIB_DIR / "libcpsa_kalman2d.so"` (same `LIB_DIR` as `classifier.py`);
  - checks `cpsa_kalman2d_abi_version() == 1`;
  - on any failure logs one WARNING with the reason and returns `None`;
  - never raises.

**Library location:** `src/cpsa/adapters/native/lib/libcpsa_kalman2d.so`, git-ignored.
- Host tests build it into `tmp_path` with `cc -O2 -std=c99 -fPIC -shared -o … native/kalman2d/kalman2d.c`, and skip the native tests when `cc` is absent.
- `deploy/deploy.sh` builds it on the board after the source sync.

### Integration (Task K2, after TV and K1)
- `KeypointTracker2D(cfg, layout, *, kernel: FilterKernel | None = None)`: the batched predict/update from TV calls `kernel.step(...)`. When `None`, it uses `NumpyKalmanKernel` or TV's equivalent inline code, whichever is simpler; results must be identical.
- Pass-through:
  - `EpisodeAnalyzer(..., kernel=None)` passes it to the tracker it builds;
  - `VisionVerifier(..., tracker_kernel=None)` passes it to every `EpisodeAnalyzer` it builds (the episode analyzer and the watch analyzer).
- Config `vision.tracker_backend: numpy | native`. The default is `numpy`; `config/board.demo.yaml` sets `native`.
- `cpsa.board.__main__` (or the medical-mode wiring, wherever `VisionVerifier` is built) calls `load_native_kalman()` when the backend is `native`. If that returns `None`, it uses NumPy, and the WARNING says the native kernel was requested but unavailable.

## Verification contract

- **AC-K1:** `NumpyKalmanKernel.step` equals a per-keypoint reference within `rtol = atol = 1e-12`.
  - The reference is a verbatim copy, in the test, of the predict/update math of `tracker2d.py` at commit `535ceef` (loop, `solve`, `pinv`).
  - Inputs: random states and covariances (SPD); K ∈ {0, 1, 62}; mixed masks; gated outliers; non-finite `z`; singular `S` (a rejection in both).
  - Evidence: `pytest tests/unit/test_kalman_kernel.py`.
- **AC-K2:** `NativeKalmanKernel.step` equals `NumpyKalmanKernel.step` on the same cases plus 10 000 random keypoint-steps.
  - Tolerances: `rtol = 1e-12`, `atol = 1e-12`; the `accepted` flags are identical; the inputs are unmodified.
  - The ABI version is checked.
  - Evidence: `pytest tests/adapters/test_native_kalman.py` (the library is built in a fixture).
- **AC-K3:** `load_native_kalman` returns `None` with one WARNING for each of these:
  - a missing file;
  - a non-ELF or wrong-arch file;
  - a missing symbol;
  - an ABI mismatch.

  It never raises. The layer test passes (`ctypes` only in adapters).
- **AC-K4:** `deploy/deploy.sh` builds `libcpsa_kalman2d.so` on the board with `cc -O2 -std=c99 -fPIC -shared … -lm` after the sync.
  - It skips the build when the `.so` is newer than `kalman2d.c`.
  - When `cc` is missing or the build fails, it warns and continues (NumPy fallback).
  - It prints the library path and size.
  - Evidence: `bash -n deploy/deploy.sh`; `deploy/deploy.sh --model /dev/null --dry-run` shows the step.
- **AC-K5:** performance.
  - `NativeKalmanKernel.step` with K = 62 has a median ≤ 0.05 ms on the host and ≤ 0.3 ms on the board.
  - Evidence: `tools/bench_kalman.py`, which reports NumPy and native medians, with no CI timing assertions.
- **AC-K6 (K2):**
  - `KeypointTracker2D` with the native kernel equals the NumPy tracker within 1e-9 on all of TV's equivalence scenes.
  - `vision.tracker_backend` parses, defaults to `numpy` and rejects other values.
  - The fallback path is tested.
  - On the board: `perf_snapshot` `track_ms` ≤ 5 ms with a person in view, and the analysis-queue `dropped` count does not grow at `analysis_fps: 10`.

## Tasks

### Task K1 — C kernel, NumPy reference, ctypes adapter, build (parallel with TV; start now)

**Depends on:** none. It must not edit `src/cpsa/vision/tracker2d.py`, `src/cpsa/vision/skeleton.py` or `tests/unit/test_tracker2d.py`, which task TV owns.

**Files (new unless noted):**
- `native/kalman2d/kalman2d.c`, `native/kalman2d/kalman2d.h`;
- `src/cpsa/vision/kalman_kernel.py`;
- `src/cpsa/adapters/native/kalman.py`;
- `tests/unit/test_kalman_kernel.py`, `tests/adapters/test_native_kalman.py`;
- `tools/bench_kalman.py`;
- `.gitignore` (add `src/cpsa/adapters/native/lib/libcpsa_kalman2d.so`);
- `deploy/deploy.sh` and `deploy/README.md` (a build step plus one paragraph);
- `src/cpsa/adapters/native/__init__.py` (docstring only).

**Interfaces:** F15 and F16, frozen.

**Actions:**
1. Write the kernel to F15 with fixed-size loops (6×6 and 6×2), no allocation and no globals.
2. Write `NumpyKalmanKernel` (batched, F15 semantics) and the AC-K1 test against the verbatim reference copied from `git show 535ceef:src/cpsa/vision/tracker2d.py`.
3. Write the adapter and the loader (AC-K3), and the native tests with a session fixture that compiles into `tmp_path` (AC-K2).
4. Add the deploy build step (AC-K4) and the benchmark script (AC-K5).

**Verification:**
- `uv run --no-sync pytest -q tests/unit/test_kalman_kernel.py tests/adapters/test_native_kalman.py tests/architecture`;
- `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src`;
- `bash -n deploy/deploy.sh`;
- `uv run --no-sync python tools/bench_kalman.py`.

### Task K2 — integration behind `vision.tracker_backend` (after TV and K1)

**Depends on:** TV, K1.

**Files:**
- `src/cpsa/vision/tracker2d.py`, `src/cpsa/vision/episode.py`;
- `src/cpsa/application/vision_verifier.py`, `src/cpsa/application/config.py`;
- the board wiring that builds `VisionVerifier` (`src/cpsa/board/__main__.py`, `src/cpsa/board/medical_mode.py`);
- `config/board.yaml`, `config/board.demo.yaml`;
- tests: `tests/unit/test_tracker2d.py`, `tests/application/test_config.py`, `tests/board`.

**Actions:** follow the "Integration" section above. Add a tracker-level equivalence test (native against NumPy) that reuses TV's scenes, and skips without `cc`.

**Verification:**
- `uv run --no-sync pytest -q tests/unit tests/application tests/board tests/architecture`;
- after the coordinator's deploy, `track_ms` and `dropped` from `status.perception.timing` on the board.

## Integration verification

Run once by the coordinator after K2:

```
uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src && bash -n deploy/deploy.sh
```

Then `npm --prefix web run build`, the deploy, and a 60 s capture of `status.perception.timing` with a person moving in view.

## Open decisions / risks

- **ctypes overhead:** one call per frame costs ~10–30 µs on the A53. That is negligible next to the 52 ms target.
- **Floating point:** the operation order in C differs from NumPy's batched `matmul`, so results can differ around 1e-15 relative. The 1e-12 tolerance covers that. `-ffast-math` is forbidden for this reason.
- **Python overhead after TV:** the ~0.25 ms/frame of host-side Python that remains is mostly skeleton and subject bookkeeping, which TV is reducing in `skeleton.py`. At the board's ~27× host-to-board ratio that is about 6–7 ms. If the board misses the 5 ms target, the next step is the skeleton and bone loop, not the kernel.
