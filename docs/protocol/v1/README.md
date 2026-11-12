# Protocol v1: the showcase dashboard

The dashboard talks to `cpsa-showcase` over one WebSocket at `/ws` (F-K1).
There is exactly one server-to-client channel and one client command.

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
`error.ref_seq` when the command is rejected. The only command is `ping`.

## Messages

| Type | Direction | Payload |
|---|---|---|
| `hello` | server → client | `server_version`, `protocol_version` (`1`). Sent first, on every connection. |
| `status` | server | `state` (`starting`, `running`, `error`), `camera` (`ok`, `missing`, `error`), `model`, `fps`, `dpu_ms`, `errors`, `uptime_s`. 1 Hz and on change. |
| `showcase.state` | server | `model`, `width`, `height`, `fps`, `dpu_ms`, `total_ms`, `objects`, `counts`. At most 10 Hz. |
| `error` | server | `code`, `message`, optional `ref_seq`. |
| `pong` | server | `id`, the value of the `ping` it answers. |
| `ping` | client → server | `id`, echoed back unchanged. |

`errors` lists the conditions the process reports, once each:
`model_missing`, `dpu_error`, `camera_missing`, `camera_error`, `internal`.
`error.code` accepts the same codes plus `bad_command`.

`showcase.state.objects` holds the best 20 detections by descending `score`:
`label` (Italian, or English with `labels: en`), the COCO-80 `class_id` and
`box` as `[x1, y1, x2, y2]` in source-image pixels clipped to the frame.
`counts` holds one `{label, count}` entry per label of the frame, by descending
count and then label. The boxes are drawn on the same frame the stream carries,
so the two panes of the dashboard always agree.

A measurement that does not exist yet is `null`, never an absent key. `t` is
monotonic seconds on the board; `wall` is the UTC wall clock.

## Errors

An unknown command, a malformed envelope (`v` different from `1`, missing
`data`, a `seq` that is not a positive integer) or a `ping` without `id` is
answered with one `error` whose code is `bad_command`:

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
