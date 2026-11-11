# Single continuous vision analyzer (camera-only mode) — Implementation Plan

**Status:** READY

**Goal:** in camera-only mode (`medical.trigger: vision`, so `VisionVerifier(continuous=True)`), every analyzed frame goes through the pose estimator and the tracker **exactly once**. An episode is a time window over that one tracked stream. Episode start no longer restarts the tracker or replays the pre-roll, so the drop bursts and the `Vision verification of … timed out` warnings at episode start disappear.

**Requested by:** the user, 2026-11-11: "We need a single one that measures if something's dangerous … Simply track the points, then if dangerous report it." Scope decision: "let's simply use camera only. The imu pipeline is kept but never used."

**Scope:**
- the `continuous=True` path of `VisionVerifier`;
- the episode-window support it needs in `EpisodeAnalyzer`;
- tests.

**Non-goals:**
- the IMU-triggered path (`continuous=False`: `_analyze_frame` with pre-roll replay), which stays byte-for-byte unchanged and keeps its tests;
- protocol, dispatcher, decisions, dashboard;
- the C Kalman kernel (`docs/plans/native-kalman2d.md` K1/K2; K2 runs after this plan).

**Constraints:**
- Keep the camera-only behaviour of AC-019…AC-026 (`docs/specs/vision-2d-rasd.md`, "Revision 2026-11-11"), especially:
  - "AC-023 … one `medical.decision`, vision tag, `CONFIRMED`, event-log line, actuation; the end yields a NON_STEREOTIPY tag change";
  - the oscillation start (AC-021);
  - the refractory (AC-022);
  - the camera-failure handling (AC-025).
- Unchanged:
  - the outputs: `medical.episode` / `episode_report` JSON, the `EpisodeTrace` file format, evidence clips (raw pre-roll ring), `perf_snapshot`, `VisionObserver` / `VisionFrameObserver` calls;
  - the frozen interfaces F7 to F14 (`docs/plans/camera-only-trigger.md`, `scratchpad/specs/dv_interface.md`).
- Layer rules: `cpsa.vision` uses only domain and numpy; `cpsa.application` does not import adapters.
- Memory stays bounded over hours of runtime.
- Demo Friday 2026-11-13.

## Current behaviour (evidence, `src/cpsa/application/vision_verifier.py`)

- `_run` (≈ line 815): when `continuous` and no episode is active, it calls `_watch_frame` (watch `EpisodeAnalyzer`, restarted every `_WATCH_MAX_S = 300` s). Otherwise it calls `_analyze_frame` (the episode analyzer `self._analyzer`).
- `_start_or_extend` (≈ 856): calls `self._analyzer.start(...)`, which resets the tracker. It then synchronously re-runs `_analyze_frame` on every pre-roll frame with `t ≥ window_end_ts − lookback_s`. That re-runs the DPU and the tracker in a burst, which blocks the queue, causes the drops and makes `verify` time out at `fast_budget_s = 0.8`.
- `verify` / `_serve_requests`: assess `[window_end − lookback_s, window_end + fast_after_s]` on the episode analyzer once frames up to `t_to` are analyzed.
- `_maybe_end_episode` / `_finish_episode` (≈ 1066 / 1078): `analyzer.active(now)` and `analyzer.report(now)`, then `analyzer.end()`.
- Trigger feeding: `_feed_trigger` / `_feed_episode_trigger` / `_trigger_t`. The `_trigger_t` guard stops replayed frames from feeding the trigger twice.
- `EpisodeAnalyzer` (`src/cpsa/vision/episode.py`):
  - `start` resets the tracker and all per-episode state;
  - `push` tracks and records events;
  - `assess` and `report` operate on `self._frames`;
  - `assessment.build_report` already keeps only frames with `t ≥ t_start`.

## Approach

**One long-lived `EpisodeAnalyzer`** (the "stream analyzer") per verifier start in continuous mode. `_WATCH_MAX_S` and the separate `self._analyzer` stop being used in this mode.

### F17 — frozen `EpisodeAnalyzer` additions (`src/cpsa/vision/episode.py`)

```python
def open_window(self, episode_id: str, t_start: float) -> None
def close_window(self) -> None
def prune(self, before_t: float) -> None
@property
def window_open(self) -> bool
```

- **`open_window`**
  - Precondition: started and not ended; no window already open (otherwise `RuntimeError`).
  - It sets the id and start time that `report`, `assess`, `active`, `episode_id` and `t_start` use.
  - It does **not** reset the tracker, the recorded frames, the contact runs or the events.
  - Contact events and grades with `t < t_start` must not count for this window. Specifically:
    - `_recent_grade` and `contact_events` used by the report only consider events with `t ≥ t_start`;
    - `latest_grade` from before `t_start` does not confirm.
- **`close_window`**
  - Returns the analyzer to its idle stream id (the id given to `start`).
  - It does not finalize in-progress contact runs (the stream continues), and it keeps the tracker.
- **`prune`**
  - Drops recorded frames, contact events and grades with `t < before_t`.
  - When a window is open, the effective cut is `min(before_t, window t_start)`.
  - Finished contact runs older than the cut are dropped; in-progress runs are kept.
  - It never touches the tracker state.
- **Unchanged:** `start`, `push`, `end`, `assess`, `report` and `active` keep their current behaviour when no window is ever opened. The IMU path and existing tests rely on that.

### Verifier, continuous mode only (`src/cpsa/application/vision_verifier.py`)

**Frame path.** In continuous mode, `_run` sends *every* frame item to one stream-frame path (today's `_watch_frame`, generalized). On the `analysis_fps` grid it does, in order:
1. `decode`, `estimate` (once), `stream.push`;
2. the perf record;
3. `_notify_frame_image` and `_notify_tracked`;
4. `_remember_oscillation_frame`, `_feed_trigger` (contacts and grades), `trigger.oscillation`, `trigger.poll(t, episode_active=self._episode_active)`, `_emit_fire`;
5. when an episode is active: the trace frame, the peak-window update and `_serve_requests`;
6. `stream.prune(t − retain_s)`, with `retain_s = max(preroll_s, lookback_s + fast_after_s)`.

**Episode start** (`_start_or_extend`, continuous): **no** `start()` and **no** replay. It:
- calls `stream.open_window(episode_id, t_start = window_end_ts − lookback_s)`;
- initializes the episode bookkeeping as today;
- seeds `_trace_frames` and the peak window from the stream frames already recorded with `t ≥ t_start`;
- serves pending requests right away, because frames up to `window_end_ts` are already tracked.

**Episode end** (`_maybe_end_episode` / `_finish_episode`, continuous): uses `stream.active(now)` and `stream.report(now)`, then `stream.close_window()` instead of `end()`. Outputs, the `try/finally` around `_finish_trigger`, `_last_report` and `_last_trace` stay as today.

**Removed in continuous mode:**
- the 5-min watch restart (memory is bounded by `prune`);
- `_feed_episode_trigger` and the `_trigger_t` replay guard (there are no replayed frames).

Keep them only if the IMU path needs them; it does not.

**Startup:** start the stream analyzer at `start()`. After a camera restart, it is restarted with the existing source-health mechanism.

**`analyzed_until_t`** and the request serving use the stream analyzer's last frame time.

**`EpisodeTrace`** in continuous mode: the same format. The frames are the stream frames in the window, and `skeleton` is the stream's skeleton state at `open_window`. Replay equivalence (`tests/application/test_vision_replay_equivalence.py`) is defined only for the IMU path. Document in the verifier docstring that continuous-mode traces start with a warm tracker.

## Verification contract

- **AC-S1** — Each analyzed frame is estimated exactly once in continuous mode.
  - Expected: with a counting fake estimator over idle, then episode, then idle (a contact pair starts the episode; the idle timeout ends it), the number of `estimate` calls equals the number of frames pushed to the stream analyzer. No frame is estimated twice, including at episode start.
  - Evidence: a new test in `tests/application/test_vision_verifier.py`.
- **AC-S2** — No tracker reset at episode start.
  - Expected: `tracker.reset` (a spy through the `tracker=` injection, or the `KeypointTracker2D` state) is called 0 times between `start()` and `stop()` while an episode opens and closes. `TrackedFrame` keypoints just after `open_window` continue from the previous frame instead of re-initializing.
  - Evidence: a new test.
- **AC-S3** — Verification is immediate.
  - Expected: `verify()` for the vision-originated start detection returns a non-timeout `VisionAssessment` (`features` without `timeout`) as soon as frames up to `t_to` are tracked, with no `Vision verification … timed out` log in the real-class tests.
  - Evidence: `test_runtime.py` real-class tests plus a `caplog` assertion.
- **AC-S4** — Unchanged camera-only behaviour. All existing continuous-mode and real-class tests pass unchanged:
  - AC-023 (`-k real_contacts`, the NON_STEREOTIPY end), the oscillation start, the refractory, the grade-upgrade test, camera failure, the import guard;
  - the episode report holds only contacts with `t_start ≥` the episode start;
  - `episode_report` JSON and the trace format validate as before.
  - Evidence: `pytest tests/application tests/board tests/adapters tests/contract`.
- **AC-S5** — Bounded memory.
  - Expected: over 10 simulated minutes at 10 fps with periodic episodes, the stream analyzer's recorded frames never exceed `retain_s·analysis_fps + episode length·analysis_fps + 5`, and the contact events stay bounded the same way.
  - Evidence: a new test.
- **AC-S6** — The IMU path is unchanged.
  - Expected: with `continuous=False`, every existing test passes unmodified, including `test_vision_replay_equivalence.py`, `test_runner.py` and the IMU-mode tests in `test_vision_verifier.py`. The `git diff` touches no IMU-only branch logic.
  - Evidence: the full suite, plus diff review.
- **AC-S7** — Board (coordinator, after deploy, provisional model).
  - Expected: during 5 minutes with repeated hand-to-face gestures:
    - the `dropped` count does not jump at episode starts;
    - no `verification … timed out` lines appear;
    - `analyzed_fps` stays within ±10% of `analysis_fps`.
  - Evidence: `status.perception.timing` capture plus `journalctl`.

## Tasks

### Task S — single stream analyzer (one worker)

**Depends on:** none. It runs in parallel with K1 (`native/`, `src/cpsa/vision/kalman_kernel.py`, `src/cpsa/adapters/native/kalman.py`, `tools/bench_kalman.py`, `deploy/`), which shares no files. K2 waits for this task.

**Files:**
- `src/cpsa/vision/episode.py` (F17);
- `src/cpsa/application/vision_verifier.py` (the continuous path only);
- `tests/unit/test_episode.py` (or the existing EpisodeAnalyzer test module) for F17;
- `tests/application/test_vision_verifier.py`, `tests/application/test_runtime.py` (AC-S1 to AC-S5);
- docstrings.

**Requirements (verbatim, `docs/specs/vision-2d-rasd.md` Revision 2026-11-11):** FR-026 to FR-032, AC-019 to AC-026 as cited above. User intent: "Simply track the points, then if dangerous report it."

**Interfaces:** F17 (frozen). F7 `VisionTrigger` (unchanged; `poll(t, episode_active)` follow-ups now come from the same stream frames). F12 to F14 (unchanged).

**Actions:**
1. Add F17 with unit tests:
   - a window over a running stream;
   - events before `t_start` are excluded from `assess`, `report` and `active`;
   - `close_window` keeps the tracker;
   - `prune` with and without a window;
   - no-window behaviour unchanged.
2. Generalize `_watch_frame` into the single stream path. Route `_run` to it for all frames in continuous mode.
3. Change `_start_or_extend`, `_maybe_end_episode`, `_finish_episode`, `analyzed_until_t` and request serving for continuous mode as in Approach. Delete the continuous-mode replay and restart and the now-dead `_trigger_t` / `_feed_episode_trigger` logic.
4. Write the AC-S1 to AC-S5 tests. Keep every existing test unmodified unless it asserts on removed internals (`_watch` restart, replay of the pre-roll in continuous mode). In that case update the assertion to the new contract and list it in the handoff.

**Verification:**
- `uv run --no-sync pytest -q tests/unit -k "episode or trigger or oscillation" tests/application tests/board`;
- `uv run --no-sync pytest -q -k real_contacts`, run 3 times;
- `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src`.

## Integration verification

The coordinator runs it once:

```
uv run --no-sync pytest -q && uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync mypy src
```

Then `npm --prefix web run build`, the deploy and AC-S7 on the board.

## Open decisions / risks

- **Warm tracker at episode start:** this differs from the old cold start. Contact grading may now see the approach before `t_start`, which is the intended improvement. Severity counts only events with `t ≥ t_start`.
- **Continuous-mode trace replay:** traces are no longer exactly reproducible from the trace frames alone, because the tracker was warm at `t_start`. This is accepted: the replay tool is an IMU-path debugging aid. It is documented in the verifier docstring.
- **Pruning cost:** `prune` runs on every frame and must be O(dropped). Use `collections.deque` or an index offset, not a list rebuild.
