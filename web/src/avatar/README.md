# The particle avatar (R2-F)

One `THREE.Points` cloud in the top-right pane: a breathing, slowly turning
wave of cyan → blue → violet → magenta points. It is the whole avatar — the
round-3 decision of the user, who found the 3D head of the first draft "a bit
creepy" and kept the sphere. Everything the face used to say with a mesh, the
sphere says with colour, motion and shape.

## Files

| File | What it owns |
|---|---|
| `index.ts` | Public surface: `AvatarView`, `VoiceEnergy`, the mood presets and animator. |
| `view.ts` | The three.js scene, camera, frame loop, `voiceAudioNode()` voice hook, presence and the face API used by `main.ts` (`tick`, `setExpression`, `setGaze`, `setSpeaking`, `beginUtterance`, `hold`, bubble). |
| `particles.ts` | The point cloud: Fibonacci sphere, palette, wave/speaking/gaze shaders, sprite size, uniforms. |
| `moods.ts` | The seven moods as surface parameters, `MoodAnimator` (smooth blending, surprised impulse, reduced-motion damping). |
| `audio.ts` | `VoiceEnergy`: analyser RMS with fast attack and slower release, plus a procedural syllable envelope for Web Speech. |

No binary asset ships with the avatar: the sphere is 9 000 points generated at
runtime, so `web/public/` holds nothing for R2-F and `NOTICE.md` has no avatar
entry.

## The moods

The dashboard expressions are moods of the sphere, not blendshape presets; the
cognitive layer keeps calling `setExpression(expression)` unchanged.

| Mood | What the user sees |
|---|---|
| `neutral` | calm breathing wave, steady cyan/violet/magenta palette |
| `happy` | brighter, faster, bouncier surface, warmer tint |
| `surprised` | a quick expansion: the radius puffs, the surface glows brighter |
| `curious` | a slight tilt and a stronger bulge/glow toward the person |
| `thinking` | slow differential swirl — the top of the sphere turns more than the waist |
| `sleepy` | dim, slow breathing, slightly flattened vertically |
| `worried` | jittery, fast small ripples, a faint red tint |

`MoodAnimator` blends every parameter with a 4.5 s⁻¹ exponential, so a mood
change reads as the sphere *feeling* different rather than switching costume.
`prefers-reduced-motion` damps the surface motion to 30 % and the speed to
55 % while keeping the colours; the spin is frozen.

## Voice

`AvatarView` polls `voiceAudioNode()` from `web/src/voice.ts` once per frame and
taps the shared `AnalyserNode` directly. `VoiceEnergy` measures the waveform
RMS with a fast attack (22 s⁻¹) and a slower release (8 s⁻¹), which is what
makes each syllable a visible swell of the surface, and raises the sprite size
while speaking. Web Speech has no audio graph: there `setSpeakingLevel()` takes
an explicit level, and failing that a small three-sine envelope keeps the
sphere alive while it talks. The analyser is a passive branch and never changes
what is heard.

## Attention and presence

- **Gaze** (`setGaze`, `GazeController` in `web/src/face.ts`): the cloud leans a
  few centimetres toward the person's side, the facing side bulges out and is
  the brightest region, so the attention is readable even in a still frame. Left
  alone, the gaze wanders.
- **Presence** (`setEngaged`): a person engaged makes the cloud slightly larger
  and brighter; after 3 s alone it calms down.

## Screenshots and performance

`web/scripts/shots.mjs` starts the Vite dev server and drives Chrome over the
DevTools protocol (no extra dependency: Node 22 `fetch`/`WebSocket`), writing
the PNGs to `build/web-shots/` (git-ignored):

```sh
cd web
node scripts/shots.mjs                       # moods, speaking, gaze, switching, room at 1280x720
node scripts/shots.mjs --only mood-happy
node scripts/shots.mjs --gpu --name bench-sphere \
  --query "mock=1&presence=1&bench=1" --settle 13000   # real-GPU frame rate
```

Dev-only URL parameters: `?mock=1`, `expression=<mood>`, `mouth=<0..1>`,
`gaze=left|right|up`, `presence=<0..1>`, `view=skeleton`, `bench=1`,
`video=0`. The frame loop caps the pixel ratio at 1.5, pauses when the tab is
hidden, and draws one point cloud in one call: the measured rate on the laptop
GPU is in the R2-F spec (`build/orca-specs/spec-R2F.md`).
