/**
 * The avatar's pure pieces (R2-F, round 3): the moods of the particle sphere,
 * their animator, the voice energy driver and the uniforms of the point cloud.
 * The three.js renderer itself needs a GPU and is exercised by the mock
 * screenshots under `build/web-shots/`.
 */

import { describe, expect, it, vi } from "vitest";

import { EXPRESSIONS } from "../cognitive/rules";
import { VoiceEnergy } from "./audio";
import { isExpression, MOODS, MoodAnimator, type MoodPose } from "./moods";
import { ParticleField, SPHERE_POINTS } from "./particles";

const CHANNELS = { speaking: 0, gaze: { x: 0, y: 0 }, presence: 0 };

/** A fake analyser that feeds a fixed RMS into `VoiceEnergy.measure`. */
function fakeAnalyser(amplitude: number) {
  const analyser = {
    fftSize: 512,
    disconnect: () => undefined,
    getByteTimeDomainData: (waveform: Uint8Array) => {
      for (let index = 0; index < waveform.length; index += 1) {
        waveform[index] = 128 + Math.round(amplitude * 127 * Math.sin(index * 0.4));
      }
    },
  };
  return analyser;
}

/** A source node that branches through an analyser of its own. */
function fakeSource(amplitude: number) {
  const analyser = fakeAnalyser(amplitude);
  const connect = vi.fn();
  const source = {
    context: { createAnalyser: () => analyser },
    connect,
  };
  return { source, analyser, connect };
}

describe("moods", () => {
  it("covers every expression of the cognitive layer, with the same names", () => {
    expect(Object.keys(MOODS).sort()).toEqual([...EXPRESSIONS].sort());
    for (const expression of EXPRESSIONS) {
      expect(isExpression(expression)).toBe(true);
    }
    expect(isExpression("bored")).toBe(false);
    expect(isExpression("")).toBe(false);
  });

  it("keeps every parameter finite and within the ranges the shader expects", () => {
    for (const expression of EXPRESSIONS) {
      const mood = MOODS[expression];
      expect(mood.wave).toBeGreaterThan(0);
      expect(mood.wave).toBeLessThan(0.2);
      expect(mood.speed).toBeGreaterThan(0);
      expect(mood.radius).toBeGreaterThan(0.3);
      expect(mood.radius).toBeLessThan(0.6);
      expect(mood.flatten).toBeGreaterThan(0.5);
      expect(mood.flatten).toBeLessThanOrEqual(1);
      expect(mood.brightness).toBeGreaterThan(0.4);
      expect(mood.attention).toBeGreaterThanOrEqual(0);
      expect(mood.attention).toBeLessThanOrEqual(1);
      expect(mood.tint).toHaveLength(3);
      for (const channel of mood.tint) {
        expect(Number.isFinite(channel)).toBe(true);
      }
    }
  });

  it("gives each mood the character the user asked for", () => {
    // Happy is brighter and bouncier; sleepy is dim and slow; worried is quick
    // and jittery; thinking swirls; curious leans toward the person.
    expect(MOODS.happy.brightness).toBeGreaterThan(MOODS.neutral.brightness);
    expect(MOODS.happy.bounce).toBeGreaterThan(MOODS.neutral.bounce);
    expect(MOODS.sleepy.brightness).toBeLessThan(MOODS.neutral.brightness);
    expect(MOODS.sleepy.speed).toBeLessThan(MOODS.neutral.speed);
    expect(MOODS.sleepy.flatten).toBeLessThan(MOODS.neutral.flatten);
    expect(MOODS.worried.speed).toBeGreaterThan(MOODS.neutral.speed);
    expect(MOODS.worried.wave).toBeLessThan(MOODS.neutral.wave);
    expect(MOODS.worried.jitter).toBeGreaterThan(0);
    expect(MOODS.thinking.swirl).toBeGreaterThan(MOODS.neutral.swirl);
    expect(MOODS.curious.attention).toBeGreaterThan(MOODS.neutral.attention);
    expect(MOODS.curious.tilt).toBeGreaterThan(0);
    expect(MOODS.surprised.radius).toBeGreaterThan(MOODS.neutral.radius);
  });
});

describe("mood animator", () => {
  it("starts neutral and reports the mood it was asked for", () => {
    const animator = new MoodAnimator();
    expect(animator.mood).toBe("neutral");
    const frame = animator.update(1 / 60, CHANNELS);
    expect(frame.pose.radius).toBeCloseTo(MOODS.neutral.radius, 2);
    expect(frame.pulse).toBe(0);
    animator.setExpression("happy");
    expect(animator.mood).toBe("happy");
  });

  it("blends toward the new mood instead of jumping", () => {
    const animator = new MoodAnimator();
    animator.setExpression("happy");
    const first = animator.update(1 / 60, CHANNELS).pose;
    expect(first.brightness).toBeGreaterThan(MOODS.neutral.brightness);
    expect(first.brightness).toBeLessThan(MOODS.happy.brightness);
    expect(first.tint[1]).toBeGreaterThan(0);
    expect(first.tint[1]).toBeLessThan(MOODS.happy.tint[1]);

    let last: MoodPose = first;
    for (let frame = 0; frame < 120; frame += 1) {
      last = animator.update(1 / 60, CHANNELS).pose;
    }
    expect(last.brightness).toBeCloseTo(MOODS.happy.brightness, 2);
    expect(last.wave).toBeCloseTo(MOODS.happy.wave, 2);
    expect(last.bounce).toBeCloseTo(MOODS.happy.bounce, 2);
    expect(last.tint[0]).toBeCloseTo(MOODS.happy.tint[0], 2);
  });

  it("snaps to a mood for a pinned screenshot frame", () => {
    const animator = new MoodAnimator();
    animator.snap("sleepy");
    const frame = animator.update(1 / 60, CHANNELS);
    expect(animator.mood).toBe("sleepy");
    expect(frame.pose.brightness).toBeCloseTo(MOODS.sleepy.brightness, 3);
    expect(frame.pose.speed).toBeCloseTo(MOODS.sleepy.speed, 3);
    expect(frame.pose.flatten).toBeCloseTo(MOODS.sleepy.flatten, 3);
    // A snap of surprised still puffs up, so its screenshot is the expansion.
    animator.snap("surprised");
    expect(animator.update(1 / 60, CHANNELS).pulse).toBeGreaterThan(0.5);
  });

  it("fires the surprised pulse once and decays it", () => {
    const animator = new MoodAnimator();
    animator.setExpression("surprised");
    expect(animator.update(1 / 60, CHANNELS).pulse).toBeGreaterThan(0.9);
    let pulse = 1;
    for (let frame = 0; frame < 40; frame += 1) {
      pulse = animator.update(1 / 60, CHANNELS).pulse;
    }
    expect(pulse).toBe(0);
    // Staying in the same mood does not repeat the pulse.
    animator.setExpression("surprised");
    animator.setExpression("surprised");
    expect(animator.mood).toBe("surprised");
  });

  it("does not repeat the pulse when re-setting the current mood", () => {
    const animator = new MoodAnimator();
    animator.setExpression("happy");
    const frame = animator.update(1 / 60, { ...CHANNELS });
    expect(frame.pulse).toBe(0);
    animator.setExpression("happy");
    expect(animator.update(1 / 60, CHANNELS).pulse).toBe(0);
  });

  it("damps the surface motion when the user prefers reduced motion", () => {
    const calm = new MoodAnimator({ reducedMotion: true });
    calm.snap("happy");
    const damped = calm.update(1 / 60, CHANNELS).pose;
    const full = new MoodAnimator();
    full.snap("happy");
    const lively = full.update(1 / 60, CHANNELS).pose;
    expect(damped.wave).toBeCloseTo(lively.wave * 0.3, 5);
    expect(damped.bounce).toBeCloseTo(lively.bounce * 0.3, 5);
    expect(damped.speed).toBeCloseTo(lively.speed * 0.55, 5);
    // Colour still changes: reduced motion is not a grey avatar.
    expect(damped.brightness).toBeCloseTo(lively.brightness, 5);
  });

  it("passes the voice, gaze and presence channels through untouched", () => {
    const animator = new MoodAnimator();
    const channels = { speaking: 0.7, gaze: { x: -1, y: 0.4 }, presence: 0.9 };
    const frame = animator.update(1 / 60, channels);
    expect(frame.channels).toBe(channels);
  });
});

describe("voice energy", () => {
  it("rises with an explicit speaking level and decays in silence", () => {
    const energy = new VoiceEnergy();
    expect(energy.attached).toBe(false);
    energy.setSpeakingLevel(0.8);
    let level = 0;
    for (let frame = 0; frame < 20; frame += 1) {
      level = energy.update(frame * 16, 1 / 60, true);
    }
    expect(level).toBeGreaterThan(0.75);
    expect(level).toBeLessThanOrEqual(1);

    for (let frame = 0; frame < 120; frame += 1) {
      level = energy.update(1_000 + frame * 16, 1 / 60, false);
    }
    expect(level).toBe(0);
  });

  it("keeps a floor while speaking, so quiet syllables still move the cloud", () => {
    const energy = new VoiceEnergy();
    let level = 0;
    for (let frame = 0; frame < 60; frame += 1) {
      level = energy.update(frame * 16, 1 / 60, true);
    }
    expect(level).toBeGreaterThan(0.09);
    expect(level).toBeLessThan(0.5);
  });

  it("measures a branched audio node, louder input giving a higher level", () => {
    const quiet = new VoiceEnergy();
    const quietInput = fakeSource(0.02);
    quiet.attachAudio(quietInput.source as unknown as AudioNode);
    expect(quiet.attached).toBe(true);
    expect(quietInput.connect).toHaveBeenCalledWith(quietInput.analyser);

    const loud = new VoiceEnergy();
    loud.attachAudio(fakeSource(0.5).source as unknown as AudioNode);

    let quietLevel = 0;
    let loudLevel = 0;
    for (let frame = 0; frame < 30; frame += 1) {
      quietLevel = quiet.update(frame * 16, 1 / 60, true);
      loudLevel = loud.update(frame * 16, 1 / 60, true);
    }
    expect(loudLevel).toBeGreaterThan(quietLevel);
    expect(loudLevel).toBeGreaterThan(0.5);

    quiet.detachAudio();
    expect(quiet.attached).toBe(false);
  });

  it("keeps the procedural syllable envelope when no audio graph exists", () => {
    const energy = new VoiceEnergy();
    const first = energy.update(0, 1 / 60, true);
    let peak = first;
    let lowest = first;
    for (let frame = 1; frame < 90; frame += 1) {
      const level = energy.update(frame * 16, 1 / 60, true);
      peak = Math.max(peak, level);
      lowest = Math.min(lowest, level);
    }
    expect(peak).toBeGreaterThan(lowest);
    expect(peak).toBeGreaterThan(0.1);
    expect(lowest).toBeLessThan(0.9);
  });
});

describe("particle sphere", () => {
  it("draws one additive point cloud with the sphere's point count", () => {
    const field = new ParticleField({ viewportHeight: 800 });
    expect(field.points.geometry.getAttribute("position").count).toBe(SPHERE_POINTS);
    expect(field.points.geometry.getAttribute("aDir").count).toBe(SPHERE_POINTS);
    expect(field.material.transparent).toBe(true);
    expect(field.material.depthWrite).toBe(false);
    field.dispose();
  });

  it("feeds the mood, voice, gaze and presence into the uniforms", () => {
    const field = new ParticleField({ viewportHeight: 800 });
    field.apply(MOODS.happy, { speaking: 0.6, gaze: { x: 1, y: 0 }, presence: 0.8 }, 0.5);
    const uniforms = field.material.uniforms;
    expect(uniforms.uWave.value).toBeCloseTo(MOODS.happy.wave, 6);
    expect(uniforms.uBrightness.value).toBeCloseTo(MOODS.happy.brightness, 6);
    expect(uniforms.uSpeak.value).toBeCloseTo(0.6, 6);
    expect(uniforms.uPresence.value).toBeCloseTo(0.8, 6);
    expect(uniforms.uGaze.value.x).toBeCloseTo(1, 6);
    expect(uniforms.uExpand.value).toBeGreaterThan(1);
    expect(uniforms.uTint.value.g).toBeCloseTo(MOODS.happy.tint[1], 6);
    // Looking right leans the cloud toward the person.
    expect(field.object.position.x).toBeGreaterThan(0);
    expect(field.object.rotation.z).toBeLessThan(0);

    field.setTime(2.5);
    expect(uniforms.uTime.value).toBeCloseTo(2.5, 6);
    field.setViewportHeight(1600);
    expect(uniforms.uScale.value).toBeCloseTo(800, 6);
    field.dispose();
  });

  it("keeps the points of a rest sphere, ordered from top to bottom", () => {
    const field = new ParticleField({ viewportHeight: 800 });
    const positions = field.points.geometry.getAttribute("position");
    const first = positions.getY(0);
    const last = positions.getY(SPHERE_POINTS - 1);
    expect(first).toBeCloseTo(1, 5);
    expect(last).toBeCloseTo(-1, 5);
    for (let index = 0; index < SPHERE_POINTS; index += 1) {
      const length = Math.hypot(
        positions.getX(index),
        positions.getY(index),
        positions.getZ(index),
      );
      expect(length).toBeCloseTo(1, 5);
    }
    field.dispose();
  });
});
