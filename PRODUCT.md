# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Vite + TypeScript + Three.js. The app is built on a PC into static files, which the Kria KV260 serves over HTTP. There is no build step on the board and no internet at runtime. Every asset, including fonts and libraries, is bundled locally.

## Users

- **Primary: high-school students at the university open day.**
  - They visit Università di Cagliari (UniCa) on Friday 2026-11-13.
  - One student at a time sits in front of the camera, at 0.45–0.7 m, while classmates watch the booth display.
  - They have no prior instructions and stay a few minutes.
  - Their job: explore something impressive with their own hands, and understand that university engineering made it.
- **Secondary: the booth operator** (an EOLAB researcher). They switch between the model lab and the medical demo, watch system health, and recover from problems mid-event.

## Product Purpose

A single system on an AMD Kria KV260 with two demos, chosen from the web UI:

1. **Holographic model lab:**
   - Students manipulate 3D models (satellite, heart, engine) with bare-hand gestures: rotate, scale, explode into parts, reassemble, select a part to read about it.
   - Free exploration only: no scores, no timer.
2. **Medical demo:**
   - Wrist IMUs and the camera detect stereotyped and self-harmful hand-to-face gestures and trigger feedback actuators.
   - A live dashboard explains what the system sees and decides.

Success: visitors are impressed, they understand that the AI runs live on a low-power FPGA chip, and they connect the playful demo to its serious medical application.

## Positioning

- The hand tracking runs entirely on an FPGA DPU at the booth (about 60 fps), not in the browser or a cloud GPU. The browser only renders what the Kria decides.
- The same perception system that lets students play with holograms also powers a real research application: detecting self-harm gestures.

## Operating Context

- **Display:** a booth display driven by a laptop browser, on a direct Ethernet link to the Kria. No internet.
- **Room:** bright and noisy, with a crowd behind the seated player and classmates watching from a few metres away.
- **Sessions:** short, with frequent changes of user. Attract and idle states matter.
- **Operator:** switches modes from the same UI.
- **Medical mode only:** uses BlueCoin wrist IMUs, MetaMotion vibration, an LED strip and a speaker.

## Capabilities and Constraints

- **Gestures:** pinch, grab, two-hand scale, rotate, explode (palms apart), reassemble (fists together), swipe to change model, fist hold to reset, with open/closed-hand fallbacks when finger tracking is weak.
- **Protocol:** the Kria is authoritative and sends state at up to 30 Hz over WebSocket (`docs/specs/interface-rasd.md`). The frontend renders only, and resyncs from a snapshot on reconnect.
- **Models:** satellite, heart and engine, as glTF (`.glb`) files with a per-part manifest (names, descriptions, explode data, license). The exact files are still to be sourced.
- **Excluded:** robotic-arm content. Games and leaderboards are out of scope.
- **Undecided:** the exact model files, the final gesture set, and whether to show a camera preview.

## Brand Commitments

- On-screen text is in **Italian**. Technical terms (FPGA, DPU, AI) stay as they are.
- **UNICA** (Università di Cagliari) and **EOLAB** branding must appear. The logo files and exact names are to be provided by the user.

## Evidence on Hand

- Board measurements (`docs/notes/vitis-compile.md`): the DPU pose model runs in 7–25 ms per frame depending on the variant, and the full pipeline at 30–95 fps.
- No trained hand model yet (training this week). No 3D model assets yet. No logo files yet.
- Testimonials, statistics and claims beyond the measured numbers must not be invented.

## Product Principles

1. **It must feel instant and alive.** Every hand movement gets immediate visible feedback, even before a gesture completes.
2. **Understandable without instructions.** A student who sits down must figure out what to do within seconds.
3. **Show the machine.** Make the edge AI visible (tracking, fps, "running on FPGA"), because the technology is the point.
4. **Never look broken.** Degraded tracking, a reconnect or a mode switch must look intentional and recover by itself.
5. **The serious purpose behind the play.** The playful lab and the medical demo belong to one story.

## Accessibility & Inclusion

- Readable from about 2–3 m by onlookers: large type and high contrast in a bright room.
- No reliance on sound, since the room is noisy.
- Gestures must work for left- and right-handed users, and with only one hand tracked.
- No flashing patterns that could trigger photosensitivity.
