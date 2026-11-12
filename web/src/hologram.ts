/**
 * Hologram view (T5): `hologram.hands` drawn as a glowing 3D hand on canvas 2D.
 *
 * The projection is written by hand on purpose (no WebGL, no dependency): the
 * hand arrives as 21 joints in metres in the IR camera frame (x right, y down,
 * z forward), a virtual camera tilted 30° orbits around it, and the joints are
 * divided by their depth. A joint whose depth was not lifted reuses the hand's
 * median depth, so a missing depth never punches a hole in the hologram.
 */

import type { HologramHand, HologramHandsPayload, Keypoint } from "./protocol";

type Edge = readonly [number, number];

/** hand21 bones: wrist to each finger's first joint, then along the chain. */
export const HAND21_BONES: readonly Edge[] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

/**
 * Focal length guess of the IR camera in pixels, used only to give a joint
 * without depth the direction it has in `kpts2d`; the lifted joints carry the
 * calibrated geometry.
 */
const FOCAL_PX = 500;

/** Virtual camera: 30° above the hand, orbiting once every 24 s. */
const ELEVATION = (30 * Math.PI) / 180;
const CAMERA_DISTANCE = 3.2;
const FOCAL_RATIO = 1.5;
const ORBIT_PERIOD_MS = 24_000;

const COLORS = {
  background: "#050b14",
  glow: "#38d6ff",
  bone: "rgba(126, 232, 255, 0.55)",
  jointEdge: "rgba(56, 214, 255, 0.9)",
  grid: "rgba(90, 200, 255, 0.16)",
  scanline: "rgba(2, 8, 16, 0.35)",
  scanbar: "rgba(143, 208, 255, 0.06)",
} as const;

interface Viewport {
  width: number;
  height: number;
  cx: number;
  cy: number;
  focal: number;
  unit: number;
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

interface Projected {
  x: number;
  y: number;
  depth: number;
}

/** CSS-pixel viewport of the canvas; the hand sits a little above centre. */
function viewportFor(width: number, height: number): Viewport {
  const unit = Math.min(width, height);
  return {
    width,
    height,
    cx: width / 2,
    cy: height * 0.47,
    focal: unit * FOCAL_RATIO,
    unit,
  };
}

function medianDepth(hand: HologramHand): number | null {
  const depths: number[] = [];
  for (const point of hand.kpts3d) {
    if (point !== null && Number.isFinite(point[2]) && point[2] > 0) {
      depths.push(point[2]);
    }
  }
  if (depths.length === 0) {
    return null;
  }
  depths.sort((first, second) => first - second);
  return depths[Math.floor(depths.length / 2)] ?? null;
}

/**
 * Metres per joint: the lifted point when it has a usable depth, otherwise the
 * same direction in `kpts2d` carried out to the hand's median depth.
 */
function liftedPoints(hand: HologramHand, frame: HologramHandsPayload): (Vec3 | null)[] {
  const median = medianDepth(hand);
  return hand.kpts2d.map((point, index) => {
    const lifted = hand.kpts3d[index];
    if (lifted !== undefined && lifted !== null && lifted[2] > 0) {
      return { x: lifted[0], y: lifted[1], z: lifted[2] };
    }
    if (median === null) {
      return null;
    }
    return {
      x: ((point[0] - frame.width / 2) * median) / FOCAL_PX,
      y: ((point[1] - frame.height / 2) * median) / FOCAL_PX,
      z: median,
    };
  });
}

/**
 * Re-centres the hand on its centroid and scales it to a unit radius, so the
 * framing does not depend on how far from the camera the hand is.
 */
function normalise(points: (Vec3 | null)[]): (Vec3 | null)[] {
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  for (const point of points) {
    if (point !== null) {
      count += 1;
      sumX += point.x;
      sumY += point.y;
      sumZ += point.z;
    }
  }
  if (count === 0) {
    return points;
  }
  const centreX = sumX / count;
  const centreY = sumY / count;
  const centreZ = sumZ / count;

  let radius = 0;
  for (const point of points) {
    if (point !== null) {
      radius = Math.max(radius, Math.hypot(point.x - centreX, point.y - centreY, point.z - centreZ));
    }
  }
  const scale = radius > 1e-6 ? 1 / radius : 1;
  return points.map((point) =>
    point === null
      ? null
      : {
          x: (point.x - centreX) * scale,
          y: (point.y - centreY) * scale,
          z: (point.z - centreZ) * scale,
        },
  );
}

/** Orbit around the vertical axis, then tilt the camera 30° above the hand. */
function project(point: Vec3, view: Viewport, azimuth: number): Projected {
  const cosA = Math.cos(azimuth);
  const sinA = Math.sin(azimuth);
  const x1 = point.x * cosA + point.z * sinA;
  const z1 = -point.x * sinA + point.z * cosA;
  const cosE = Math.cos(ELEVATION);
  const sinE = Math.sin(ELEVATION);
  const y2 = point.y * cosE - z1 * sinE;
  const z2 = point.y * sinE + z1 * cosE;
  const depth = Math.max(0.2, z2 + CAMERA_DISTANCE);
  const scale = view.focal / depth;
  return { x: view.cx + x1 * scale, y: view.cy + y2 * scale, depth };
}

function alphaFor(point: Keypoint): number {
  const confidence = point[2] ?? 0;
  if (confidence >= 0.5) {
    return 1;
  }
  if (confidence >= 0.2) {
    return 0.5;
  }
  if (confidence > 0) {
    return 0.2;
  }
  return 0;
}

export class HologramView {
  private readonly canvas: HTMLCanvasElement;
  private readonly empty: HTMLElement;
  private readonly meta: HTMLElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver | null;

  private frame: HologramHandsPayload | null = null;
  private active = false;
  private animation: number | null = null;
  private metaKey = "";

  constructor(canvas: HTMLCanvasElement, empty: HTMLElement, meta: HTMLElement) {
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new Error("Canvas 2D non disponibile in questo browser");
    }
    this.canvas = canvas;
    this.empty = empty;
    this.meta = meta;
    this.context = context;

    this.observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.draw());
    this.observer?.observe(canvas.parentElement ?? canvas);
    window.addEventListener("resize", this.onWindowResize);

    this.updateMeta();
    this.draw();
  }

  setFrame(frame: HologramHandsPayload | null): void {
    this.frame = frame;
    this.updateMeta();
    this.draw();
  }

  /** Runs the orbit only while the hologram view is the visible one. */
  setActive(active: boolean): void {
    if (active === this.active) {
      return;
    }
    this.active = active;
    if (active) {
      this.start();
    } else {
      this.stop();
    }
    this.draw();
  }

  destroy(): void {
    this.stop();
    this.observer?.disconnect();
    window.removeEventListener("resize", this.onWindowResize);
  }

  private readonly onWindowResize = (): void => this.draw();

  private start(): void {
    if (this.animation !== null) {
      return;
    }
    const step = (): void => {
      this.animation = requestAnimationFrame(step);
      this.draw();
    };
    this.animation = requestAnimationFrame(step);
  }

  private stop(): void {
    if (this.animation !== null) {
      cancelAnimationFrame(this.animation);
      this.animation = null;
    }
  }

  private updateMeta(): void {
    const frame = this.frame;
    if (frame === null) {
      this.empty.hidden = false;
      if (this.metaKey !== "none") {
        this.metaKey = "none";
        this.meta.textContent = "Nessun frame ricevuto.";
        this.canvas.setAttribute("aria-label", "Ologramma non disponibile: in attesa di dati.");
      }
      return;
    }

    const hands = frame.hands.length;
    const joints = frame.hands.reduce((total, hand) => total + hand.kpts2d.length, 0);
    const key = `${hands}:${joints}:${frame.width}:${frame.height}`;
    this.empty.hidden = hands > 0;
    if (this.metaKey === key) {
      return;
    }
    this.metaKey = key;
    this.meta.textContent = `${hands} ${hands === 1 ? "mano" : "mani"} · ${frame.width}×${frame.height} px`;
    this.canvas.setAttribute(
      "aria-label",
      hands === 0
        ? "Ologramma: nessuna mano visibile."
        : `Ologramma 3D: ${hands} ${hands === 1 ? "mano" : "mani"}, ${joints} articolazioni.`,
    );
  }

  private draw(): void {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) {
      return;
    }

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(rect.width * ratio);
    const pixelHeight = Math.round(rect.height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }

    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.shadowBlur = 0;
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, rect.width, rect.height);

    const frame = this.frame;
    if (frame === null) {
      return;
    }

    const view = viewportFor(rect.width, rect.height);
    const azimuth = ((performance.now() % ORBIT_PERIOD_MS) / ORBIT_PERIOD_MS) * Math.PI * 2;
    this.drawTable(view, azimuth);
    for (const hand of frame.hands) {
      this.drawHand(hand, frame, view, azimuth);
    }
    this.drawScanlines(view);
    context.shadowBlur = 0;
    context.globalAlpha = 1;
  }

  /** Faint grid on the table plane, under the hand. */
  private drawTable(view: Viewport, azimuth: number): void {
    const context = this.context;
    const layer = Math.max(view.unit * 0.002, 1);
    context.save();
    context.strokeStyle = COLORS.grid;
    context.lineWidth = layer;
    context.beginPath();
    const plane = 1.25;
    const extent = 2.2;
    const step = 0.4;
    for (let offset = -extent; offset <= extent + 1e-9; offset += step) {
      const along = project({ x: offset, y: plane, z: -extent }, view, azimuth);
      const alongEnd = project({ x: offset, y: plane, z: extent }, view, azimuth);
      const across = project({ x: -extent, y: plane, z: offset }, view, azimuth);
      const acrossEnd = project({ x: extent, y: plane, z: offset }, view, azimuth);
      context.moveTo(along.x, along.y);
      context.lineTo(alongEnd.x, alongEnd.y);
      context.moveTo(across.x, across.y);
      context.lineTo(acrossEnd.x, acrossEnd.y);
    }
    context.stroke();
    context.restore();
  }

  private drawHand(
    hand: HologramHand,
    frame: HologramHandsPayload,
    view: Viewport,
    azimuth: number,
  ): void {
    if (hand.kpts2d.length < 21) {
      return;
    }
    const points = normalise(liftedPoints(hand, frame));
    const projected = points.map((point) =>
      point === null ? null : project(point, view, azimuth),
    );
    const context = this.context;
    const blur = view.unit * 0.03;

    // Bones first, farthest from the virtual camera first so the near ones
    // stay on top.
    const bones = HAND21_BONES.map(([first, second]) => {
      const from = projected[first];
      const to = projected[second];
      if (from === undefined || from === null || to === undefined || to === null) {
        return null;
      }
      const fromPoint = hand.kpts2d[first] ?? [0, 0, 0];
      const toPoint = hand.kpts2d[second] ?? [0, 0, 0];
      const alpha = Math.min(alphaFor(fromPoint), alphaFor(toPoint));
      if (alpha === 0) {
        return null;
      }
      return { from, to, alpha, depth: (from.depth + to.depth) / 2 };
    })
      .filter((bone) => bone !== null)
      .sort((first, second) => second.depth - first.depth);

    for (const bone of bones) {
      const near = CAMERA_DISTANCE / bone.depth;
      context.globalAlpha = bone.alpha * 0.85;
      context.strokeStyle = COLORS.bone;
      context.lineWidth = Math.max(1.2, view.unit * 0.008 * near);
      context.shadowColor = COLORS.glow;
      context.shadowBlur = blur * near;
      context.beginPath();
      context.moveTo(bone.from.x, bone.from.y);
      context.lineTo(bone.to.x, bone.to.y);
      context.stroke();
    }

    const joints = projected
      .map((point, index) => {
        if (point === null) {
          return null;
        }
        const alpha = alphaFor(hand.kpts2d[index] ?? [0, 0, 0]);
        return alpha === 0 ? null : { point, alpha, index };
      })
      .filter((joint) => joint !== null)
      .sort((first, second) => second.point.depth - first.point.depth);

    for (const joint of joints) {
      const near = CAMERA_DISTANCE / joint.point.depth;
      const radius = Math.max(1.5, view.unit * 0.011 * near);
      const gradient = context.createRadialGradient(
        joint.point.x - radius * 0.3,
        joint.point.y - radius * 0.35,
        radius * 0.15,
        joint.point.x,
        joint.point.y,
        radius,
      );
      gradient.addColorStop(0, "#f2feff");
      gradient.addColorStop(1, COLORS.jointEdge);
      context.globalAlpha = joint.alpha;
      context.fillStyle = gradient;
      context.shadowColor = COLORS.glow;
      context.shadowBlur = blur * 1.4 * near;
      context.beginPath();
      context.arc(joint.point.x, joint.point.y, radius, 0, Math.PI * 2);
      context.fill();
    }
  }

  /** CRT scanlines plus a slow bright band, over the whole hologram. */
  private drawScanlines(view: Viewport): void {
    const context = this.context;
    context.save();
    context.shadowBlur = 0;
    context.globalAlpha = 1;
    context.fillStyle = COLORS.scanline;
    const pitch = Math.max(3, Math.round(view.unit * 0.012));
    for (let y = 0; y < view.height; y += pitch) {
      context.fillRect(0, y, view.width, 1);
    }

    const phase = (performance.now() % 5200) / 5200;
    const bandHeight = Math.max(28, view.unit * 0.14);
    const band = context.createLinearGradient(
      0,
      phase * view.height,
      0,
      phase * view.height + bandHeight,
    );
    band.addColorStop(0, "rgba(143, 208, 255, 0)");
    band.addColorStop(0.5, COLORS.scanbar);
    band.addColorStop(1, "rgba(143, 208, 255, 0)");
    context.fillStyle = band;
    context.fillRect(0, phase * view.height, view.width, bandHeight);
    context.restore();
  }
}
