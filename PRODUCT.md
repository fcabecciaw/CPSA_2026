# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + TypeScript. The dashboard is built on a PC into static files, which the
Kria KV260 serves over HTTP together with the live MJPEG stream. There is no
build step on the board and no internet at runtime. Every asset is bundled
locally.

## Users

- **Primary: high-school students at the university open day.**
  - They visit Università di Cagliari (UniCa) on Friday 2026-11-13.
  - One student at a time stands or sits in front of the camera while
    classmates watch the booth display.
  - They have no prior instructions and stay a few minutes.
  - Their job: try to fool the detector, and see that a small board recognises
    what they hold up.
- **Secondary: the booth operator** (an EOLAB researcher). They start the demo
  before the event, watch `/health` and the journal during it, and recover from
  problems mid-event.

## Product Purpose

One demo on an AMD Kria KV260: **"Cosa vede l'IA"**.

- A live camera feed with the boxes and Italian labels of a pre-trained
  YOLOX-nano detector drawn on it, and a panel that lists what the model sees
  ("persona", "tazza", "bottiglia", …) with the frame rate and the DPU time.
- Success: visitors see that the AI runs live on a low-power FPGA chip with no
  cloud, and that they can watch it decide in real time.

## Positioning

- The detector runs entirely on the board's DPU at 60 frames per second, not in
  the browser and not in a cloud GPU. The browser only renders what the Kria
  sends.
- The technology is visible: the frame rate, the DPU time and the latency are on
  screen, so the demo is about the machine as much as about the model.

## Operating Context

- **Display:** a booth display driven by a laptop browser, on a direct Ethernet
  link to the Kria. No internet.
- **Room:** bright and noisy, with a crowd behind the player and classmates
  watching from a few metres away.
- **Sessions:** short, with frequent changes of user. Attract and idle states
  matter.
- **Operator:** starts the unit and, if something fails, restarts it; there is no
  mode to choose and nothing to configure at the booth.

## Capabilities and Constraints

- **Camera:** one Intel RealSense F200 RGB node at 640x480, 60 fps, manual
  exposure and gain tuned for the room.
- **Model:** YOLOX-nano from the Vitis AI model zoo, COCO-80 classes, quantized
  and compiled for the KV260. No training, no per-event tuning.
- **Protocol:** the Kria is authoritative; the browser renders protocol v1
  messages (`docs/protocol/v1/README.md`) and resyncs from a snapshot on
  reconnect. Only `ping` travels the other way.
- **Excluded:** every other demo of the previous system (holograms, medical
  gestures, rock-paper-scissors, IMUs, actuators). This is a single-purpose
  showcase.

## Brand Commitments

- On-screen text is in **Italian**; class names are translated ("persona",
  "tazza"), technical terms (FPGA, DPU, AI, fps) stay as they are.
- **UNICA** (Università di Cagliari) and **EOLAB** branding must appear. The
  logo files and exact names are to be provided by the user.

## Evidence on Hand

- Board measurements: the detector processes 60 fps median with one video client;
  the MJPEG stream reaches the browser at about 55 fps with the two-process split
  (`scripts/profile_showcase.py` reports each stage).
- No testimonials, statistics or claims beyond the measured numbers.

## Product Principles

1. **It must feel instant and alive.** The video is the demo; any lag between a
   movement and its box is a bug.
2. **Understandable without instructions.** A visitor must see what the system
   does within seconds, from the video and the label panel alone.
3. **Show the machine.** Make the edge AI visible (fps, DPU time, "running on
   FPGA"), because the technology is the point.
4. **Never look broken.** A dropped frame, a reconnecting browser or a re-plugged
   camera must recover by itself and look intentional.
5. **The model is what it is.** Show the real detections, including the mistakes:
   no curated or simulated results.

## Accessibility & Inclusion

- Readable from about 2–3 m by onlookers: large type and high contrast in a
  bright room.
- No reliance on sound, since the room is noisy.
- The demo does not require any gesture or input from the visitor.
- No flashing patterns that could trigger photosensitivity.
