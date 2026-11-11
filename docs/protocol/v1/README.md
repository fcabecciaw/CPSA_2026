# Protocol v1 schemas

These schemas are the machine-readable form of `docs/specs/interface-rasd.md` (D3). The server validates its outgoing messages against them in tests (`tests/contract/protocol_schemas.py`).

- `envelope.schema.json`: every server-to-client message (IF-102). `data` is validated by `<type>.schema.json`.
- `client-envelope.schema.json`: client commands (`ping`, `mode.set`) carry `v`, `type`, `data` and an optional `seq` (a per-connection client counter starting at 1). The server echoes it as `error.ref_seq` when an error answers that command.
- `examples/<type>.json`: one valid full message per type. The frontend mock and the server's fake mode replay them.

Conventions:
- Enum values are the domain enum *names* for `Tag` (`"DANGEROUS"`) and the enum *values* for every other enum (`"CONFIRMED"`, `"left"`, `"mouth"`, `"fist"`).
- A missing or not-applicable value is `null`, never an absent required key.
- `medical.hands.keypoints` holds `[x, y, conf]` in camera-frame pixels (`w` × `h`), in `layout` order. A keypoint that was not seen has `conf` 0.
- `subject_box` is `[x1, y1, x2, y2]` in the same pixels.
- `medical.hands.seq` (protocol addition, v1 compatible) is the analyzed frame sequence number, the same value the video stream stamps as `X-Frame-Seq` (F14). Clients that do not know it ignore it.
- `status.perception.timing` (protocol addition, v1 compatible) is `null` or the rolling 5 s performance snapshot (F12/F13): exactly `analyzed_fps`, `camera_fps`, `decode_ms`, `preprocess_ms`, `dpu_ms`, `postprocess_ms`, `pose_ms`, `track_ms`, `latency_ms`, each a number or `null`, plus `dropped` (cumulative analysis-queue drops, integer ≥ 0). Numbers are rounded to one decimal. The server sends the key in medical mode; clients must also accept it missing.
- `t` and `t_start` are Kria `CLOCK_MONOTONIC` seconds.
- Clients ignore unknown types and unknown fields (IF-103). Payload objects allow extra fields.
- `log` (event, protocol addition, v1 compatible): device and connection log lines for the operator panel (BlueCoin, MetaMotion, camera, supervisor). Rate-limited by the server; clients keep only the latest lines.

Live video (protocol addition, v1 compatible; F14):
- `GET /video.mjpeg` streams the analyzed camera frames as `multipart/x-mixed-replace; boundary=cpsaframe` with `Cache-Control: no-store`. Each part is `--cpsaframe`, then `Content-Type: image/jpeg`, `Content-Length`, `X-Frame-Seq` and the JPEG bytes.
- Frames are the decoded, unmirrored camera images at camera size (the `w` × `h` of `medical.hands`), JPEG quality 70, at most 10 frames/s per client; each frame is encoded once and shared.
- With no analyzed frames (idle or another mode) the response stays open and sends nothing. When `server.video` is false the endpoint answers `404` with `{"error": "video_disabled"}`.
- Frames are only displayed live; this path never writes them to disk.

