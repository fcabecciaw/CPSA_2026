# Vendored MediaPipe assets (offline)

These files are served as-is by the dashboard so the laptop can run face cues
without any network access:

- `vendor/mediapipe/vision_bundle.mjs`, `vendor/mediapipe/wasm/*` — the
  `@mediapipe/tasks-vision` 1.2.3 web runtime, Apache-2.0
  (<https://github.com/google-ai-edge/mediapipe>). Only the non-module wasm
  build (`vision_wasm_internal.*`) is shipped; `FilesetResolver.forVisionTasks`
  uses it by default.
- `models/face_landmarker.task` — the MediaPipe Face Landmarker model bundle
  (`float16/1`), Apache-2.0
  (<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>).

The license text is in `vendor/mediapipe/LICENSE`. These assets run in the
laptop browser only, never on the KV260 board.

