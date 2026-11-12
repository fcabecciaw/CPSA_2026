# Protocol v1: the showcase dashboard

The dashboard talks to `cpsa-showcase` over one WebSocket at `/ws` (F-K1).
There is exactly one server-to-client channel and four client commands (`ping`;
since R2-G, `model.set`; for the debug panel's live calibration,
`calibration.set` and `calibration.reset`).

## Envelope

Every server message is a JSON object with `v`, `type`, `seq`, `t`, `wall` and
`data` (`envelope.schema.json`):

| Field | Meaning |
|---|---|
| `v` | Protocol version, always `1`. |
| `type` | Message type; the payload is validated by `<type>.schema.json`. |
| `seq` | Server-wide counter, `1`, `2`, `3`, …: a client detects a missed or reordered message. |
| `t` | Monotonic seconds, for latency measurement. |
| `wall` | UTC wall clock, ISO-8601 with milliseconds. |
| `data` | The payload, an object. |

A client command is `{"v": 1, "type": <command>, "data": {...}}` with an
optional `seq` (`client-envelope.schema.json`): the server echoes it as
`error.ref_seq` when the command is rejected. The commands are `ping`,
`model.set`, `calibration.set` and `calibration.reset`.

## Messages

| Type | Direction | Payload |
|---|---|---|
| `hello` | server → client | `server_version`, `protocol_version` (`1`). Sent first, on every connection. |
| `status` | server | `state` (`starting`, `running`, `error`), `camera` (`ok`, `missing`, `error`), `model`, `active_model` (`yolox`, `movenet`), `switching`, `swap_ms`, `fps`, `dpu_ms`, `errors`, `uptime_s`, `calibration` (`zoom`, `zoom_center`, `gain`, `exposure`, `saturation`, or `null`). 1 Hz and on change. |
| `showcase.state` | server | `model`, `width`, `height`, `fps`, `dpu_ms`, `total_ms`, `objects`, `counts`. At most 10 Hz, while YOLOX is loaded. |
| `pose.state` | server | `model`, `width`, `height`, `fps`, `dpu_ms`, `total_ms`, `score`, `keypoints` (17 × `[x, y, score]`). At most `movenet.publish_hz`, while MoveNet is loaded (R2-G). |
| `error` | server | `code`, `message`, optional `ref_seq`. |
| `pong` | server | `id`, the value of the `ping` it answers. |
| `ping` | client → server | `id`, echoed back unchanged. |
| `model.set` | client → server | `model` (`yolox`, `movenet`): releases the loaded DPU model and loads the other one. An unknown or disabled model is `error bad_command` and nothing is switched. |
| `calibration.set` | client → server | Any non-empty subset of `zoom` (1.0–2.0), `zoom_center` (`[x, y]`, 0–1), `gain` (64–2540), `exposure` (`100` = 1/100 s or `200` = 1/50 s) and `saturation` (0–255). Applied from the next frame and kept across restarts. |
| `calibration.reset` | client → server | Empty `data`: forget the stored calibration and go back to `config/showcase.yaml`. |

`errors` lists the conditions the process reports, once each:
`model_missing`, `dpu_error`, `camera_missing`, `camera_error`, `internal`.
`error.code` accepts the same codes plus `bad_command`.

`showcase.state.objects` holds the best 20 detections by descending `score`:
`label` (Italian, or English with `labels: en`), the COCO-80 `class_id` and
`box` as `[x1, y1, x2, y2]` in source-image pixels clipped to the frame.
`counts` holds one `{label, count}` entry per label of the frame, by descending
count and then label. The boxes are drawn on the same frame the stream carries,
so the two panes of the dashboard always agree.

`pose.state.keypoints` are COCO-17 `[x, y, score]` triples in source-image
pixels (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles);
`score` is their mean and is below `0.2` when nobody is in the frame. The
skeleton is drawn on the video while `active_model` is `movenet`, the boxes
while it is `yolox`; only the loaded model publishes its state (`showcase.state`
for YOLOX, `pose.state` for MoveNet).

`status.switching` is true during a `model.set` swap, `status.swap_ms` is how
long the last swap took, and `status.model` is the stem of the loaded
`.xmodel`.

## Live calibration

The dashboard's debug panel ("Calibrazione") tunes the YOLOX centre crop and
the camera image while the demo runs. Process B validates `calibration.set` and
queues it for process A, which changes the crop of the loaded YOLOX runner from
the next frame (no restart, no model reload) and writes `gain`,
`exposure_time_absolute` and `saturation` to the open V4L2 stream from a worker
thread, so the frame loop never waits for the camera. The accepted values are
written atomically to `~/cpsa/config/calibration.json` (`cpsa-showcase
--calibration PATH` moves it) and loaded at the next start as overrides on top
of `config/showcase.yaml`; `calibration.reset` removes the file and applies the
YAML values again. `status.calibration` always reports the values in use, and a
change publishes a new `status` at once.

A measurement that does not exist yet is `null`, never an absent key. `t` is
monotonic seconds on the board; `wall` is the UTC wall clock.

## Errors

An unknown command, a malformed envelope (`v` different from `1`, missing
`data`, a `seq` that is not a positive integer), a `ping` without `id`, or a
`model.set` whose `model` is not `yolox`/`movenet`, or a `calibration.set` that
is empty, names an unknown key or holds a value out of range is answered with
one `error` whose code is `bad_command`:

```json
{"v": 1, "type": "error", "seq": 4, "t": 1001.0,
 "wall": "2026-11-12T15:00:00.000Z",
 "data": {"code": "bad_command", "message": "unknown command 'mode.set'", "ref_seq": 2}}
```

The connection stays open: the client may send more commands. The server never
closes a WebSocket on a bad command.

## Frames

`GET /video.mjpeg` streams the overlaid frames as
`multipart/x-mixed-replace; boundary=cpsa-frame` with `Cache-Control: no-store`.
Each part is the boundary line, `Content-Type: image/jpeg`, `Content-Length`,
the frame sequence as `X-Frame-Seq` and the JPEG bytes. Frames are only encoded
while at least one client is watching, at `web.video_fps`.

`GET /` serves `web.static_dir/index.html`, `/assets/...` the built bundle, and
`GET /health` returns `{"state", "fps", "uptime_s"}` from the latest `status`.

## Files

- `*.schema.json`: one schema per payload, plus the two envelopes.
- `examples/*.json`: one valid full message per type; the frontend mock and
  `tests/contract/test_protocol_schemas.py` use them.
