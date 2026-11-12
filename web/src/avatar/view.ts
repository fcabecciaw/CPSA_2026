/**
 * The avatar of the dashboard (R2-F, round 3): one breathing point-cloud wave
 * sphere, cyan at the top fading through blue and violet to magenta at the
 * bottom, in the top-right pane.
 *
 * The user dropped the 3D head ("a bit creepy") and kept the sphere, so the
 * sphere now carries the character itself: mood (colour, wave, speed, swirl,
 * flatten), speaking (voice level of `voiceAudioNode()`), attention (the cloud
 * leans, bulges and brightens toward the person) and presence (a little bigger
 * and brighter when someone is engaged).
 *
 * It keeps the API the dashboard already speaks: `setExpression`, `hold`,
 * `beginUtterance`/`endUtterance`, `setSpeaking`, `setGaze`/`updateGaze`,
 * `tick`. The subtitles are not its business (`subtitles.ts`): the canvas
 * subtree is never written by anything else. Three.js lives here and in
 * `particles.ts`.
 */

import * as THREE from "three";

import type { Box } from "../cognitive/perception";
import type { Expression } from "../cognitive/rules";
import { ExpressionController, GazeController, type GazePoint } from "../face";
import { VoiceEnergy } from "./audio";
import { isExpression, MoodAnimator } from "./moods";
import { ParticleField } from "./particles";

/** How long a person may be invisible before the cloud calms down. */
const DISENGAGE_MS = 3_000;
const MAX_PIXEL_RATIO = 1.5;

export interface AvatarDebug {
  /** Pin one mood (screenshot gallery and framing experiments). */
  expression?: string;
  /** Pin the voice level 0..1 as if the sphere were speaking. */
  mouth?: number;
  /** Pin the attention point, -1..1 (left/right screenshots). */
  gaze?: { x: number; y: number };
  /** Pin presence: 0 alone, 1 engaged. */
  presence?: number;
}

export interface AvatarOptions {
  debug?: AvatarDebug;
  /**
   * The analyser of the sentence being spoken (`voiceAudioNode()` of
   * `voice.ts`), polled once per frame: the surface follows the real amplitude
   * of the local voice, and falls back to its own envelope for Web Speech.
   */
  audio?: () => AudioNode | null;
}

export class AvatarView {
  readonly gaze = new GazeController();
  readonly expressions = new ExpressionController();

  private readonly root: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer | null;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(30, 1, 0.1, 40);
  private readonly spinGroup = new THREE.Group();
  private readonly animator: MoodAnimator;
  private readonly energy = new VoiceEnergy();
  private readonly reducedMotion: boolean;
  private readonly audio: (() => AudioNode | null) | null;
  private readonly pinnedExpression: Expression | null;
  private readonly pinnedMouth: number | null;
  private readonly pinnedGaze: GazePoint | null;
  private readonly pinnedPresence: number | null;

  private field: ParticleField | null = null;
  private gazeTarget: GazePoint = { x: 0, y: 0 };
  private speaking = false;
  private presenceTarget = 0;
  private presence = 0;
  private lastPresentAt = 0;
  private attachedAudio: AudioNode | null = null;
  private frameHandle = 0;
  private lastFrameAt: number | null = null;
  private lastClock = 0;
  private pixelRatio = 1;
  private disposed = false;
  private resizeObserver: ResizeObserver | null = null;

  constructor(root: HTMLElement, options: AvatarOptions = {}) {
    this.root = root;
    this.audio = options.audio ?? null;
    this.reducedMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.animator = new MoodAnimator({ reducedMotion: this.reducedMotion });
    this.pinnedExpression =
      options.debug?.expression !== undefined && isExpression(options.debug.expression)
        ? options.debug.expression
        : null;
    this.pinnedMouth =
      options.debug?.mouth === undefined ? null : clamp01(options.debug.mouth);
    this.pinnedGaze = options.debug?.gaze ?? null;
    this.pinnedPresence = options.debug?.presence ?? null;
    if (this.pinnedExpression !== null) {
      this.animator.snap(this.pinnedExpression);
      this.expressions.beginUtterance(this.pinnedExpression, 0);
    }
    if (this.pinnedMouth !== null) {
      this.energy.setLevel(this.pinnedMouth);
      this.speaking = true;
    }
    if (this.pinnedGaze !== null) {
      this.gazeTarget = this.pinnedGaze;
    }
    if (this.pinnedPresence !== null) {
      this.presence = clamp01(this.pinnedPresence);
      this.presenceTarget = this.presence;
    }
    this.root.classList.add("avatar");

    this.camera.position.set(0, 0.02, 2.3);
    this.camera.lookAt(0, 0, 0);
    this.scene.add(this.spinGroup);

    this.renderer = createRenderer(this.root);
    if (this.renderer === null) {
      this.root.dataset.status = "nogl";
      return;
    }
    this.root.append(this.renderer.domElement);
    this.renderer.domElement.setAttribute("role", "img");
    this.renderer.domElement.setAttribute(
      "aria-label", "Sfera di particelle luminose, il volto digitale di ARI",
    );

    this.resize();
    this.field = new ParticleField({
      viewportHeight: Math.max(1, this.root.clientHeight) * this.pixelRatio,
      reducedMotion: this.reducedMotion,
    });
    this.spinGroup.add(this.field.object);
    this.root.dataset.status = "ready";

    this.resizeObserver =
      typeof ResizeObserver === "function" ? new ResizeObserver(() => this.resize()) : null;
    this.resizeObserver?.observe(this.root);
    document.addEventListener("visibilitychange", this.onVisibility);
    this.start();
  }

  // --- presence ------------------------------------------------------------

  /** A person is (or is not) engaged; the cloud calms down after 3 s alone. */
  setEngaged(present: boolean, now: number): void {
    if (this.pinnedPresence !== null) {
      return;
    }
    if (present) {
      this.lastPresentAt = now;
    }
    const engaged = present || now - this.lastPresentAt < DISENGAGE_MS;
    this.presenceTarget = engaged ? 1 : 0;
  }

  // --- the face API of the cognitive layer ---------------------------------

  setExpression(expression: Expression): void {
    if (this.pinnedExpression !== null) {
      return;
    }
    const safe = isExpression(expression) ? expression : "neutral";
    this.animator.setExpression(safe);
    this.root.dataset.expression = safe;
  }

  setSpeaking(speaking: boolean): void {
    if (this.pinnedMouth !== null) {
      return;
    }
    this.speaking = speaking;
    this.root.classList.toggle("avatar-speaking", speaking);
  }

  /** Apply a gaze point; x and y are -1..1. */
  setGaze(point: GazePoint): void {
    if (this.pinnedGaze !== null) {
      return;
    }
    this.gazeTarget = point;
  }

  updateGaze(
    boxes: readonly Box[],
    frame: { width: number; height: number },
    now: number,
    idle: boolean,
  ): GazePoint {
    if (this.pinnedGaze !== null) {
      return this.pinnedGaze;
    }
    const point = this.gaze.update(boxes, frame, now, idle);
    this.setGaze(point);
    return point;
  }

  /** One cognitive step (250 ms): expression decay and blink bookkeeping. */
  tick(now: number, idleS: number): Expression {
    if (this.pinnedExpression !== null) {
      return this.pinnedExpression;
    }
    const expression = this.expressions.tick(now, idleS);
    if (expression !== this.animator.mood) {
      this.setExpression(expression);
    }
    return expression;
  }

  beginUtterance(expression: Expression, now: number): void {
    this.expressions.beginUtterance(expression, now);
    if (this.pinnedExpression === null) {
      this.setExpression(expression);
    }
    this.setSpeaking(true);
  }

  endUtterance(now: number): void {
    this.expressions.endUtterance(now);
    this.setSpeaking(false);
  }

  hold(expression: Expression, now: number, holdMs?: number): void {
    this.expressions.hold(expression, now, holdMs);
    if (this.pinnedExpression === null) {
      this.setExpression(expression);
    }
  }

  // --- voice ---------------------------------------------------------------

  /** R2-V hands the voice player's output node here for the lip-sync. */
  attachAudio(node: AudioNode): void {
    this.syncAudio(node);
  }

  detachAudio(): void {
    this.syncAudio(null);
  }

  /**
   * Follow the shared voice analyser: called once per frame with
   * `voiceAudioNode()`, it attaches the first time the local voice exists and
   * ignores the identical node afterwards.
   */
  syncAudio(node: AudioNode | null): void {
    if (node === this.attachedAudio) {
      return;
    }
    this.attachedAudio = node;
    if (node === null) {
      this.energy.detachAudio();
    } else {
      this.energy.attachAudio(node);
    }
  }

  /** Fallback level 0..1 for engines without a WebAudio graph (Web Speech). */
  setSpeakingLevel(level: number): void {
    this.energy.setSpeakingLevel(level);
  }

  // --- frame loop ----------------------------------------------------------

  private start(): void {
    const loop = (): void => {
      this.frameHandle = window.requestAnimationFrame(loop);
      this.frame();
    };
    this.frameHandle = window.requestAnimationFrame(loop);
  }

  private frame(): void {
    if (this.disposed || document.hidden || this.renderer === null) {
      return;
    }
    const now = performance.now();
    const dt = this.lastFrameAt === null ? 1 / 60 : (now - this.lastFrameAt) / 1_000;
    this.lastFrameAt = now;
    this.lastClock += dt;

    if (this.audio !== null) {
      this.syncAudio(this.audio());
    }
    const speaking = this.pinnedMouth !== null ? true : this.speaking;
    const level = this.energy.update(now, dt, speaking);
    this.presence += (this.presenceTarget - this.presence) * (1 - Math.exp(-dt * 1.6));

    const mood = this.animator.update(dt, {
      speaking: level,
      gaze: this.gazeTarget,
      presence: this.presence,
    });
    this.field?.apply(mood.pose, mood.channels, mood.pulse);
    this.field?.setTime(this.lastClock);

    // The sphere turns slowly; reduced motion freezes it.
    if (!this.reducedMotion) {
      this.spinGroup.rotation.y += dt * 0.07;
    }
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const width = Math.max(1, this.root.clientWidth);
    const height = Math.max(1, this.root.clientHeight);
    const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    this.pixelRatio = ratio;
    if (this.renderer === null) {
      return;
    }
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    // Keep the whole sphere readable on narrow panes: widen the distance a bit.
    this.camera.fov = width / height < 1.05 ? 38 : 30;
    this.camera.updateProjectionMatrix();
    this.field?.setViewportHeight(height * ratio);
  }

  private readonly onVisibility = (): void => {
    if (document.hidden) {
      this.lastFrameAt = null;
    }
  };

  dispose(): void {
    this.disposed = true;
    window.cancelAnimationFrame(this.frameHandle);
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.resizeObserver?.disconnect();
    this.field?.dispose();
    if (this.renderer !== null) {
      this.renderer.dispose();
      this.renderer.domElement.remove();
    }
    this.energy.detachAudio();
  }
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function createRenderer(root: HTMLElement): THREE.WebGLRenderer | null {
  try {
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.className = "avatar-canvas";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    root.dataset.status = "loading";
    return renderer;
  } catch (error) {
    console.warn("[cpsa] avatar: WebGL non disponibile", error);
    return null;
  }
}
