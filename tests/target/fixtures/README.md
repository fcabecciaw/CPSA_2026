# Classifier golden fixture

**Status: NOT CAPTURED.** `classifier_golden.npz` does not exist yet. It must be
captured on a KV260 (aarch64) at the baseline commit `fcf3ec9` (PLAN.md Task 0).
Until then `tests/target/test_classifier_golden.py` skips and classifier
equivalence with the baseline (FR-004, AC-003) is **not verified**.

When the fixture is captured, replace this status block with:

- baseline commit: `fcf3ec9`
- device: (board, OS image, Python version)
- date: (YYYY-MM-DD)
- command(s) used, exactly as run
- input sets included and their window counts

## Schema

`classifier_golden.npz` holds one or more input sets; for a set named `<set>`:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `<set>_rows` | `(N, 20)` float32 | raw synchronized rows: R acc xyz (mg), R gyr xyz (dps), R quat xyzw (×10000), L acc, L gyr, L quat |
| `<set>_ts` | `(N,)` float64 | row timestamps (s) |
| `<set>_features` | `(K, 18)` float32 | baseline features of every classified window |
| `<set>_tags` | `(K,)` int64 | baseline tags (1–3) of the same windows |

`K = floor((N - 150) / 75) + 1 - 2` (window #1 is discarded and window #2 only
calibrates the extractor). Each set must have `K >= 10`.

Expected sets: `synthetic` (seeded random rows, always) and `recorded` (a real
session, when available; preferred for tag diversity).

## Capture

The native feature extractor keeps process-global state, so every set is captured
in a fresh Python process; the target test replays each set in a fresh process
too. On the KV260:

```bash
git checkout fcf3ec9
# save the script below as capture_golden.py in the repository root, then:
python3 capture_golden.py classifier_golden.npz                     # synthetic only
python3 capture_golden.py classifier_golden.npz --recorded rec.npz  # plus a real session
```

`rec.npz` (optional) holds `rows (N, 20)` and `ts (N,)` from a real session, in the
raw units above.

```python
"""Capture the classifier golden fixture at baseline fcf3ec9 (run from the repo root)."""

import argparse
import multiprocessing

import numpy as np

SYNTHETIC_SEED = 20260101
SYNTHETIC_ROWS = 150 + 75 * 12  # 11 classified windows


def synthetic_rows():
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n = SYNTHETIC_ROWS
    acc = rng.uniform(-2000.0, 2000.0, (n, 2, 3))
    gyr = rng.uniform(-500.0, 500.0, (n, 2, 3))
    quat = rng.normal(size=(n, 2, 4))
    quat = np.round(quat / np.linalg.norm(quat, axis=2, keepdims=True) * 10000.0)
    rows = np.concatenate(
        [acc[:, 0], gyr[:, 0], quat[:, 0], acc[:, 1], gyr[:, 1], quat[:, 1]], axis=1
    ).astype(np.float32)
    ts = np.arange(n, dtype=np.float64) * 0.01
    return rows, ts


def capture(rows, ts):
    # Imported here so each set gets a fresh library state in its own process.
    from IMU_pipeline.data_stream.data_buffer import DataBuffer
    from IMU_pipeline.classifiers.stereotipy_classifier.predict_models_wrapper_quat import (
        initialize,
        predict_pericolosa_wrists_quat,
    )

    features, tags = [], []

    def sink(feats, _ts):
        features.append(np.asarray(feats, dtype=np.float32).copy())
        tags.append(predict_pericolosa_wrists_quat(feats))

    buffer = DataBuffer()  # calls ProcessDataWristsQuat_initialize
    initialize()  # c_Predict_Pericolosa_Wrists_Qua, as StereotipyClassifier does
    buffer.set_features_sink(sink)
    for row, row_ts in zip(rows, ts):
        r = [float(v) for v in row]
        buffer.add_buffer_row(r[0:3], r[3:6], r[6:10], r[10:13], r[13:16], r[16:20], float(row_ts))
    return np.stack(features), np.asarray(tags, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--recorded")
    args = parser.parse_args()

    sets = {"synthetic": synthetic_rows()}
    if args.recorded:
        with np.load(args.recorded) as data:
            sets["recorded"] = (data["rows"].astype(np.float32), data["ts"].astype(np.float64))

    arrays = {}
    context = multiprocessing.get_context("spawn")
    for name, (rows, ts) in sets.items():
        with context.Pool(1) as pool:
            features, tags = pool.apply(capture, (rows, ts))
        expected = (len(rows) - 150) // 75 + 1 - 2
        assert len(tags) == expected >= 10, (name, len(tags), expected)
        arrays.update(
            {
                f"{name}_rows": rows,
                f"{name}_ts": ts,
                f"{name}_features": features,
                f"{name}_tags": tags,
            }
        )
        print(f"{name}: {len(rows)} rows, {len(tags)} windows, tags={np.bincount(tags).tolist()}")
    np.savez(args.output, **arrays)


if __name__ == "__main__":
    main()
```

Then copy `classifier_golden.npz` into `tests/target/fixtures/` of the refactored
package and run, on the KV260:

```bash
uv sync --extra dev --extra board
uv run pytest -m target
```
