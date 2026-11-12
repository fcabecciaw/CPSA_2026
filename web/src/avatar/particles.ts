/**
 * The particle sphere of the avatar (R2-F, round 3).
 *
 * One `THREE.Points` cloud on a Fibonacci sphere: a breathing wave surface with
 * the palette the user asked for — cyan at the top, blue and violet through the
 * middle, magenta at the bottom — drawn additively as soft round sprites. All
 * the character lives in the vertex shader and in the uniforms the `AvatarView`
 * feeds every frame:
 *
 * - **mood** (`wave`, `speed`, `bounce`, `swirl`, `jitter`, `flatten`, colour,
 *   brightness): the surface moves differently in each expression;
 * - **speaking**: `uSpeak` is the voice level of `VoiceEnergy`, which swells the
 *   surface into travelling pulses on each syllable;
 * - **attention**: a bulge and a white-hot glow toward the person, from the gaze
 *   controller;
 * - **presence**: the cloud is a little larger and brighter when someone is
 *   engaged, calmer when the room is empty.
 *
 * The cost is one draw call and no per-frame geometry upload: 9 000 points at
 * 60 fps on the laptop, well inside the dashboard's budget.
 */

import * as THREE from "three";

import type { MoodPose } from "./moods";

/** Points of the cloud: dense enough to read as a surface, cheap enough to draw. */
export const SPHERE_POINTS = 9_000;
/** Rest radius in world units: about 70 % of the pane height at the camera. */
const SPHERE_RADIUS = 0.43;
const FIBONACCI_TURNS = 0.618033988749895;
/** Extra radius of the surprised puff when a mood begins. */
const PULSE_EXPAND = 0.12;
/** How much the whole cloud leans toward the person. */
const LEAN_X = 0.05;
const LEAN_Y = 0.03;

const VERTEX = /* glsl */ `
attribute vec3 aDir;
attribute float aSeed;

uniform float uTime;
uniform float uSpeed;
uniform float uWave;
uniform float uBounce;
uniform float uSwirl;
uniform float uJitter;
uniform float uRadius;
uniform float uFlatten;
uniform float uBrightness;
uniform float uAttention;
uniform float uPresence;
uniform float uSpeak;
uniform float uExpand;
uniform float uMotion;
uniform float uSize;
uniform float uScale;
uniform vec2 uGaze;
uniform vec3 uTint;

varying vec3 vColor;
varying float vGlow;

vec3 palette(float t) {
  vec3 magenta = vec3(1.000, 0.180, 0.760);
  vec3 violet  = vec3(0.560, 0.290, 1.000);
  vec3 blue    = vec3(0.170, 0.450, 1.000);
  vec3 cyan    = vec3(0.360, 0.950, 1.000);
  if (t < 0.34) return mix(magenta, violet, t / 0.34);
  if (t < 0.68) return mix(violet, blue, (t - 0.34) / 0.34);
  return mix(blue, cyan, (t - 0.68) / 0.32);
}

vec3 rotateY(vec3 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

void main() {
  float t = uTime * uSpeed;

  // Thinking is a slow differential twist: the top of the sphere turns more
  // than its waist, so the surface visibly shears instead of spinning rigidly.
  float twist = uSwirl * (0.35 + 0.65 * (aDir.y * 0.5 + 0.5)) * sin(uTime * 0.42);
  vec3 dir = rotateY(aDir, twist);

  // The breathing wave the user liked, plus the faster ripple of the moods.
  float wave = sin(dir.y * 4.2 + t * 1.15) * 0.50
             + sin(dir.x * 5.1 - t * 0.85) * 0.32
             + sin(dir.z * 6.3 + t * 0.65) * 0.22;
  float breathe = uBounce * sin(uTime * 1.05);

  // Speaking: the loudness of the voice swells the surface, and a pulse travels
  // through it so each syllable reads at dashboard size.
  float syllable = 0.5 + 0.5 * sin(dir.y * 6.5 - t * 6.2);
  float speak = uSpeak * (0.35 + 0.65 * syllable);

  // Attention: the side of the sphere that faces the person bulges out and is
  // the brightest part of the cloud, so the gaze is readable in a still frame.
  vec3 gazeDir = normalize(vec3(clamp(uGaze.x, -1.0, 1.0), clamp(uGaze.y, -1.0, 1.0), 0.9));
  float toward = pow(max(dot(dir, gazeDir), 0.0), 3.0);
  float attention = uAttention * toward * (0.05 + 0.03 * sin(t * 1.8 + aSeed * 3.1));

  float shimmer = sin(uTime * 13.0 + aSeed * 40.0) * uJitter;

  // Presence: engaged is a touch bigger and brighter, alone a touch calmer.
  // The voice swells the whole surface: the voice level peaks on the syllable, so the
  // sphere visibly breathes out with every loudness peak instead of just
  // brightening.
  float radius = uRadius * uExpand * (0.96 + 0.06 * uPresence) + speak * 0.075
               + (uWave * wave + breathe + attention + shimmer) * uMotion;
  vec3 point = dir * radius;
  point.y *= uFlatten;

  vec4 mv = modelViewMatrix * vec4(point, 1.0);
  gl_Position = projectionMatrix * mv;

  float distance = max(0.35, -mv.z);
  float size = uSize * (0.55 + 0.85 * aSeed) * (1.0 + 0.35 * uSpeak);
  gl_PointSize = max(1.0, size * uScale / distance);

  float ramp = clamp(point.y * (1.0 / (2.0 * uRadius)) + 0.5, 0.0, 1.0);
  vec3 color = palette(ramp) * uBrightness * (0.86 + 0.14 * uPresence)
             * (0.72 + 0.55 * aSeed);
  color += uTint * 0.55;
  color += vec3(0.85, 0.95, 1.0) * attention * 4.2;
  // A hint of white on the loudest part of the syllable, little enough to keep
  // the cyan-magenta palette readable.
  color += vec3(0.65, 0.89, 1.0) * uSpeak * 0.22 * (0.35 + 0.65 * syllable);
  vColor = color;
  vGlow = (0.7 + 0.55 * aSeed) * (1.0 + 0.4 * uSpeak);
}
`;

const FRAGMENT = /* glsl */ `
precision mediump float;

varying vec3 vColor;
varying float vGlow;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;
  float falloff = exp(-r2 * 3.4);
  gl_FragColor = vec4(vColor, falloff * vGlow);
}
`;

export interface ParticleFieldOptions {
  /** Viewport height in device pixels: the projection scale of the sprites. */
  viewportHeight: number;
  /** Softens every oscillation of the cloud, for `prefers-reduced-motion`. */
  reducedMotion?: boolean;
}

/** The point cloud plus the uniforms the mood, voice and gaze drive. */
export class ParticleField {
  readonly points: THREE.Points;
  readonly material: THREE.ShaderMaterial;
  readonly object = new THREE.Group();

  private readonly geometry: THREE.BufferGeometry;
  private readonly gaze: THREE.Vector2;
  private readonly tint: THREE.Color;

  constructor(options: ParticleFieldOptions) {
    const directions = fibonacciSphere(SPHERE_POINTS);
    const seeds = new Float32Array(SPHERE_POINTS);
    for (let index = 0; index < SPHERE_POINTS; index += 1) {
      seeds[index] = hash(index);
    }

    this.geometry = new THREE.BufferGeometry();
    // `position` is the rest sphere (bounding volume); the shader builds the
    // frame from `aDir` and the uniforms.
    this.geometry.setAttribute("position", new THREE.BufferAttribute(directions, 3));
    this.geometry.setAttribute("aDir", new THREE.BufferAttribute(directions, 3));
    this.geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);

    this.gaze = new THREE.Vector2(0, 0);
    this.tint = new THREE.Color(0, 0, 0);
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: {
        uTime: { value: 0 },
        uSpeed: { value: 1 },
        uWave: { value: 0.05 },
        uBounce: { value: 0.012 },
        uSwirl: { value: 0 },
        uJitter: { value: 0 },
        uRadius: { value: SPHERE_RADIUS },
        uFlatten: { value: 1 },
        uBrightness: { value: 1 },
        uAttention: { value: 0.35 },
        uPresence: { value: 0 },
        uSpeak: { value: 0 },
        uExpand: { value: 1 },
        uMotion: { value: options.reducedMotion === true ? 0.4 : 1 },
        uSize: { value: 0.062 },
        uScale: { value: Math.max(1, options.viewportHeight) * 0.5 },
        uGaze: { value: this.gaze },
        uTint: { value: this.tint },
      },
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.object.add(this.points);
  }

  /** One frame of mood, voice, gaze and presence. */
  apply(
    pose: MoodPose,
    channels: { speaking: number; gaze: { x: number; y: number }; presence: number },
    pulse: number,
  ): void {
    const uniforms = this.material.uniforms;
    uniforms.uWave.value = pose.wave;
    uniforms.uSpeed.value = pose.speed;
    uniforms.uBounce.value = pose.bounce;
    uniforms.uSwirl.value = pose.swirl;
    uniforms.uJitter.value = pose.jitter;
    uniforms.uRadius.value = pose.radius;
    uniforms.uFlatten.value = pose.flatten;
    uniforms.uBrightness.value = pose.brightness;
    uniforms.uAttention.value = pose.attention;
    uniforms.uPresence.value = channels.presence;
    uniforms.uSpeak.value = channels.speaking;
    uniforms.uExpand.value = 1 + PULSE_EXPAND * pulse;
    this.gaze.set(channels.gaze.x, channels.gaze.y);
    this.tint.setRGB(pose.tint[0], pose.tint[1], pose.tint[2]);
    this.object.position.set(LEAN_X * channels.gaze.x, LEAN_Y * channels.gaze.y, 0);
    this.object.rotation.z = -0.06 * channels.gaze.x + pose.tilt;
  }

  setTime(seconds: number): void {
    this.material.uniforms.uTime.value = seconds;
  }

  setViewportHeight(heightDevice: number): void {
    this.material.uniforms.uScale.value = Math.max(1, heightDevice) * 0.5;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/** Uniform points on a sphere, ordered bottom to top along a spiral. */
function fibonacciSphere(count: number): Float32Array {
  const directions = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const y = 1 - (index / Math.max(1, count - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = 2 * Math.PI * index * FIBONACCI_TURNS;
    directions[index * 3] = Math.cos(theta) * radius;
    directions[index * 3 + 1] = y;
    directions[index * 3 + 2] = Math.sin(theta) * radius;
  }
  return directions;
}

function hash(index: number): number {
  const value = Math.sin(index * 127.1 + 311.7) * 43758.5453;
  return value - Math.floor(value);
}
