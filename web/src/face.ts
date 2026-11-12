/**
 * Holographic robot face of the dashboard (F-E3 `face.ts`).
 *
 * The face is one inline SVG: cyan glow, scanlines, a subtle flicker, blinking
 * eyes, a mouth that moves while speaking and seven expressions. The gaze
 * follows the largest person box with 30 %/1 s hysteresis and smoothing, and
 * wanders when nobody is there (F-E4 #26). Expressions are held for the
 * utterance plus 1.5 s, then decay to neutral, sleepy after a minute alone
 * (#27). `GazeController` and `ExpressionController` are DOM-free so they can
 * be unit-tested.
 */

import { EXPRESSIONS, type Expression, type GazeTarget } from "./cognitive/rules";
import type { Box } from "./cognitive/perception";

export interface GazePoint {
  x: number;
  y: number;
}

export interface GazeOptions {
  /** A box must be this much larger to steal the gaze… */
  switchMargin?: number;
  /** …and stay so for this long. */
  holdMs?: number;
  /** Smoothing rate, per second. */
  smoothing?: number;
}

/** Largest-box gaze with hysteresis and smoothing (F-E4 #26). */
export class GazeController {
  private readonly switchMargin: number;
  private readonly holdMs: number;
  private readonly smoothing: number;
  private current: GazePoint = { x: 0, y: 0 };
  private target: GazePoint = { x: 0, y: 0 };
  private targetArea = 0;
  private candidate: { area: number; since: number } | null = null;
  private lastAt: number | null = null;

  constructor(options: GazeOptions = {}) {
    this.switchMargin = options.switchMargin ?? 1.3;
    this.holdMs = options.holdMs ?? 1_000;
    this.smoothing = options.smoothing ?? 6;
  }

  update(
    boxes: readonly Box[],
    frame: { width: number; height: number },
    now: number,
    wander = boxes.length === 0,
  ): GazePoint {
    const dt = this.lastAt === null ? 0 : Math.min(0.5, Math.max(0, (now - this.lastAt) / 1000));
    this.lastAt = now;

    if (boxes.length === 0) {
      this.candidate = null;
      this.targetArea = 0;
      this.target = {
        x: Math.sin(now / 2_400) * 0.55,
        y: Math.cos(now / 3_300) * 0.35,
      };
      if (wander) {
        this.current = this.smooth(this.current, this.target, dt);
      }
      return this.current;
    }

    const largest = boxes.reduce(
      (best, box) => (area(box) > area(best) ? box : best),
      boxes[0],
    );
    const largestArea = area(largest);
    const center = centerOf(largest, frame);

    if (this.targetArea === 0) {
      this.targetArea = largestArea;
      this.target = center;
      this.candidate = null;
    } else if (largestArea >= this.targetArea * this.switchMargin) {
      if (this.candidate === null) {
        this.candidate = { area: largestArea, since: now };
      } else if (now - this.candidate.since >= this.holdMs) {
        this.targetArea = largestArea;
        this.target = center;
        this.candidate = null;
      }
    } else {
      this.candidate = null;
    }

    this.current = this.smooth(this.current, this.target, dt);
    return this.current;
  }

  reset(): void {
    this.current = { x: 0, y: 0 };
    this.target = { x: 0, y: 0 };
    this.targetArea = 0;
    this.candidate = null;
    this.lastAt = null;
  }

  private smooth(from: GazePoint, to: GazePoint, dt: number): GazePoint {
    const factor = Math.min(1, dt * this.smoothing);
    return {
      x: from.x + (to.x - from.x) * factor,
      y: from.y + (to.y - from.y) * factor,
    };
  }
}

function area(box: Box): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function centerOf(box: Box, frame: { width: number; height: number }): GazePoint {
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  return {
    x: Math.max(-1, Math.min(1, (cx / Math.max(1, frame.width) - 0.5) * 2)),
    y: Math.max(-1, Math.min(1, (cy / Math.max(1, frame.height) - 0.5) * 2)),
  };
}

/** Expression hold machine (F-E4 #27). */
export class ExpressionController {
  private readonly holdMs: number;
  private readonly sleepyAfterMs: number;
  private expression: Expression = "neutral";
  private holdUntil = 0;
  private speaking = false;

  constructor(options: { holdMs?: number; sleepyAfterMs?: number } = {}) {
    this.holdMs = options.holdMs ?? 1_500;
    this.sleepyAfterMs = options.sleepyAfterMs ?? 60_000;
  }

  get current(): Expression {
    return this.expression;
  }

  beginUtterance(expression: Expression, _now: number): void {
    this.speaking = true;
    this.expression = expression;
    this.holdUntil = Number.POSITIVE_INFINITY;
  }

  endUtterance(now: number): void {
    this.speaking = false;
    this.holdUntil = now + this.holdMs;
  }

  /** Direct hold, for connection loss and mode changes. */
  hold(expression: Expression, now: number, holdMs = this.holdMs): void {
    if (this.speaking) {
      return;
    }
    this.expression = expression;
    this.holdUntil = now + holdMs;
  }

  tick(now: number, idleS: number): Expression {
    if (this.speaking) {
      return this.expression;
    }
    if (now < this.holdUntil) {
      return this.expression;
    }
    this.expression = idleS * 1000 >= this.sleepyAfterMs ? "sleepy" : "neutral";
    return this.expression;
  }
}

const FACE_EXPRESSIONS: Record<
  Expression,
  { eyeRy: number; eyeY: number; brow: number; mouth: string }
> = {
  neutral: { eyeRy: 8, eyeY: 0, brow: 0, mouth: "M104 116 Q130 118 156 116" },
  happy: { eyeRy: 5.5, eyeY: 1, brow: -3, mouth: "M104 112 Q130 132 156 112" },
  surprised: { eyeRy: 10.5, eyeY: 0, brow: -6, mouth: "M124 111 Q130 103 136 111 Q130 123 124 111" },
  curious: { eyeRy: 7, eyeY: 0, brow: 4, mouth: "M104 116 Q118 110 130 116 Q142 122 156 116" },
  sleepy: { eyeRy: 2.6, eyeY: 2, brow: 2, mouth: "M106 118 Q130 112 154 118" },
  thinking: { eyeRy: 7, eyeY: -1, brow: 5, mouth: "M108 118 Q120 112 132 118 Q144 124 152 116" },
  worried: { eyeRy: 7.5, eyeY: 0, brow: -5, mouth: "M104 118 Q116 110 128 118 Q140 126 156 114" },
};

const SVG_NS = "http://www.w3.org/2000/svg";

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, value);
  }
  return node;
}

/**
 * The face in the dashboard. One instance owns its SVG and its two controllers;
 * the page calls `tick` on its 250 ms loop and `react` when a rule fires.
 */
export class FaceView {
  readonly gaze = new GazeController();
  readonly expressions = new ExpressionController();

  private readonly root: HTMLElement;
  private readonly shell: SVGGElement;
  private readonly eyeGroup: SVGGElement;
  private readonly leftEye: SVGEllipseElement;
  private readonly rightEye: SVGEllipseElement;
  private readonly leftPupil: SVGCircleElement;
  private readonly rightPupil: SVGCircleElement;
  private readonly leftBrow: SVGLineElement;
  private readonly rightBrow: SVGLineElement;
  private readonly mouth: SVGPathElement;
  private readonly bubble: HTMLElement | null;
  private nextBlinkAt = 0;
  private blinkTimer: number | null = null;
  private speaking = false;

  constructor(root: HTMLElement, bubble: HTMLElement | null = null) {
    this.root = root;
    this.bubble = bubble;
    this.root.classList.add("face-holo");

    const diagram = svg("svg", {
      viewBox: "0 0 260 170",
      class: "face-svg",
      role: "img",
      "aria-label": "Volto olografico del robot",
    });
    const defs = svg("defs", {});
    const glow = svg("filter", { id: "face-glow", x: "-50%", y: "-50%", width: "200%", height: "200%" });
    glow.append(
      svg("feGaussianBlur", { stdDeviation: "3.2", result: "blur" }),
      svg("feMerge", {}),
    );
    const merge = glow.lastElementChild as SVGFEMergeElement;
    merge.append(svg("feMergeNode", { in: "blur" }), svg("feMergeNode", { in: "SourceGraphic" }));
    const halo = svg("filter", { id: "face-halo", x: "-60%", y: "-60%", width: "220%", height: "220%" });
    halo.append(svg("feGaussianBlur", { stdDeviation: "9" }));
    defs.append(glow, halo);

    const shell = svg("g", { class: "face-shell", filter: "url(#face-glow)" });
    this.shell = shell;
    const haloRing = svg("ellipse", {
      cx: "130",
      cy: "86",
      rx: "86",
      ry: "66",
      class: "face-halo",
      filter: "url(#face-halo)",
    });
    const head = svg("path", {
      d: "M70 34 Q130 12 190 34 L200 98 Q196 136 130 156 Q64 136 60 98 Z",
      class: "face-head",
    });
    const headLine = svg("path", {
      d: "M82 42 Q130 24 178 42",
      class: "face-head-line",
    });
    const antenna = svg("g", { class: "face-antenna" });
    antenna.append(
      svg("line", { x1: "130", y1: "18", x2: "130", y2: "6", class: "face-antenna-stem" }),
      svg("circle", { cx: "130", cy: "5", r: "3.2", class: "face-antenna-tip" }),
    );
    const ears = svg("g", { class: "face-ears" });
    ears.append(
      svg("rect", { x: "52", y: "70", width: "8", height: "26", rx: "4", class: "face-ear" }),
      svg("rect", { x: "200", y: "70", width: "8", height: "26", rx: "4", class: "face-ear" }),
    );

    this.eyeGroup = svg("g", { class: "face-eyes" });
    this.leftEye = svg("ellipse", { cx: "98", cy: "72", rx: "15", ry: "8", class: "face-eye" });
    this.rightEye = svg("ellipse", { cx: "162", cy: "72", rx: "15", ry: "8", class: "face-eye" });
    this.leftPupil = svg("circle", { cx: "98", cy: "73", r: "4.4", class: "face-pupil" });
    this.rightPupil = svg("circle", { cx: "162", cy: "73", r: "4.4", class: "face-pupil" });
    this.leftBrow = svg("line", { x1: "82", y1: "52", x2: "112", y2: "52", class: "face-brow" });
    this.rightBrow = svg("line", { x1: "148", y1: "52", x2: "178", y2: "52", class: "face-brow" });
    this.eyeGroup.append(
      this.leftEye,
      this.rightEye,
      this.leftPupil,
      this.rightPupil,
      this.leftBrow,
      this.rightBrow,
    );

    this.mouth = svg("path", { d: FACE_EXPRESSIONS.neutral.mouth, class: "face-mouth" });
    const chin = svg("path", { d: "M108 138 Q130 146 152 138", class: "face-chin" });

    shell.append(haloRing, head, headLine, antenna, ears, this.eyeGroup, this.mouth, chin);
    const scanlines = svg("g", { class: "face-scanlines" });
    for (let y = 14; y < 168; y += 6) {
      scanlines.append(svg("line", { x1: "40", y1: String(y), x2: "220", y2: String(y) }));
    }
    const corners = svg("g", { class: "face-corners" });
    corners.append(
      svg("path", { d: "M20 28 H44 M20 28 V52" }),
      svg("path", { d: "M240 28 H216 M240 28 V52" }),
      svg("path", { d: "M20 142 H44 M20 142 V118" }),
      svg("path", { d: "M240 142 H216 M240 142 V118" }),
    );
    diagram.append(defs, scanlines, shell, corners);
    this.root.append(diagram);
    this.setExpression("neutral");
  }

  setExpression(expression: Expression): void {
    const safe = EXPRESSIONS.includes(expression) ? expression : "neutral";
    const face = FACE_EXPRESSIONS[safe];
    this.root.dataset.expression = safe;
    for (const eye of [this.leftEye, this.rightEye]) {
      eye.setAttribute("ry", String(face.eyeRy));
      eye.setAttribute("cy", String(72 + face.eyeY));
    }
    for (const pupil of [this.leftPupil, this.rightPupil]) {
      pupil.setAttribute("cy", String(73 + face.eyeY));
    }
    this.leftBrow.setAttribute("transform", `rotate(${-face.brow} 98 52)`);
    this.rightBrow.setAttribute("transform", `rotate(${face.brow} 162 52)`);
    this.mouth.setAttribute("d", face.mouth);
  }

  setSpeaking(speaking: boolean): void {
    this.speaking = speaking;
    this.root.classList.toggle("face-speaking", speaking);
  }

  /** Apply a gaze point; x and y are -1..1. */
  setGaze(point: GazePoint): void {
    this.eyeGroup.setAttribute(
      "transform",
      `translate(${(point.x * 4.5).toFixed(2)} ${(point.y * 3.4).toFixed(2)})`,
    );
    this.shell.setAttribute(
      "transform",
      `rotate(${(point.x * 1.6).toFixed(2)} 130 96)`,
    );
  }

  /** Gaze helper for the page: largest person box, frame size, idle check. */
  updateGaze(
    boxes: readonly Box[],
    frame: { width: number; height: number },
    now: number,
    idle: boolean,
  ): GazePoint {
    const point = this.gaze.update(boxes, frame, now, idle);
    this.setGaze(point);
    return point;
  }

  /** One 250 ms step: blink, expression decay, mouth state. */
  tick(now: number, idleS: number): Expression {
    if (now >= this.nextBlinkAt) {
      this.blink(now);
    }
    const expression = this.expressions.tick(now, idleS);
    this.setExpression(expression);
    this.setSpeaking(this.speaking);
    return expression;
  }

  beginUtterance(expression: Expression, now: number): void {
    this.expressions.beginUtterance(expression, now);
    this.setExpression(expression);
    this.setSpeaking(true);
  }

  endUtterance(now: number): void {
    this.expressions.endUtterance(now);
    this.setSpeaking(false);
  }

  hold(expression: Expression, now: number, holdMs?: number): void {
    this.expressions.hold(expression, now, holdMs);
    this.setExpression(expression);
  }

  setBubble(text: string | null): void {
    if (this.bubble === null) {
      return;
    }
    this.bubble.textContent = text ?? "";
    this.bubble.classList.toggle("bubble-empty", text === null || text === "");
  }

  /** Hide the bubble after the sentence has had its moment. */
  clearBubble(): void {
    this.setBubble(null);
  }

  /** Optional hint for the UI: which side of the room the person is. */
  static gazeLabel(target: GazeTarget): string {
    switch (target) {
      case "person":
        return "guarda la persona";
      case "object":
        return "guarda l'oggetto";
      case "wander":
        return "sguardo libero";
    }
  }

  private blink(now: number): void {
    this.root.classList.add("face-blinking");
    if (this.blinkTimer !== null) {
      window.clearTimeout(this.blinkTimer);
    }
    this.blinkTimer = window.setTimeout(() => {
      this.root.classList.remove("face-blinking");
      this.blinkTimer = null;
    }, 130);
    this.nextBlinkAt = now + 2_400 + Math.random() * 3_800;
  }
}
