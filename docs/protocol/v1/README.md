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
- `t` and `t_start` are Kria `CLOCK_MONOTONIC` seconds.
- Clients ignore unknown types and unknown fields (IF-103). Payload objects allow extra fields.
- `log` (event, protocol addition, v1 compatible): device and connection log lines for the operator panel (BlueCoin, MetaMotion, camera, supervisor). Rate-limited by the server; clients keep only the latest lines.
