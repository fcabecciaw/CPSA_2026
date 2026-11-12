/**
 * Live calibration of the board from the debug panel ("Calibrazione").
 *
 * The operator tunes the YOLOX centre crop (`zoom`, `zoom_center`) and the
 * camera image (`gain`, `exposure`, `saturation`) while the demo runs. This
 * module is the DOM-free half: the ranges, the debounced sender that turns a
 * dragged slider into one `calibration.set` every ~150 ms, and the geometry of
 * the dashed rectangle that shows what the network analyses.
 */

import { DETECTORS, type CalibrationStatus, type CalibrationValues, type DetectorName } from "./protocol";

export interface SliderRange {
  min: number;
  max: number;
  step: number;
}

/** `model.zoom`: 1.0 is the whole frame, 2.0 the central quarter. */
export const ZOOM_RANGE: SliderRange = { min: 1, max: 2, step: 0.05 };
/** `model.zoom_center`, each axis as a fraction of the (mirrored) frame. */
export const CENTRE_RANGE: SliderRange = { min: 0, max: 1, step: 0.01 };
/** Manual V4L2 gain of the F200 RGB node. */
export const GAIN_RANGE: SliderRange = { min: 64, max: 2540, step: 1 };

/**
 * The gain slider is logarithmic: each step is the same brightness ratio, so
 * the useful 64..800 keeps most of the track instead of its first quarter.
 */
export const GAIN_SLIDER: SliderRange = { min: 0, max: 1000, step: 1 };

export function gainToSlider(gain: number): number {
  const ratio = Math.log(clamp(gain, GAIN_RANGE.min, GAIN_RANGE.max) / GAIN_RANGE.min);
  return Math.round((ratio / Math.log(GAIN_RANGE.max / GAIN_RANGE.min)) * GAIN_SLIDER.max);
}

export function sliderToGain(position: number): number {
  const fraction = clamp(position, GAIN_SLIDER.min, GAIN_SLIDER.max) / GAIN_SLIDER.max;
  return Math.round(GAIN_RANGE.min * (GAIN_RANGE.max / GAIN_RANGE.min) ** fraction);
}
/** V4L2 saturation of the F200 RGB node (camera default 128). */
export const SATURATION_RANGE: SliderRange = { min: 0, max: 255, step: 1 };

/** `exposure_time_absolute` choices, both flicker-safe under 50 Hz lights. */
export const EXPOSURES = [
  { value: 100, label: "1/100" },
  { value: 200, label: "1/50" },
] as const;

/** The object detectors of the "Modello oggetti" control. */
export const DETECTOR_CHOICES: ReadonlyArray<{ value: DetectorName; label: string }> = [
  { value: "yolox", label: "YOLOX-nano" },
  { value: "yolov5", label: "YOLOv5-nano" },
];

/** `presence.min_box_height`: a person counts at the stand from this box height. */
export const PERSON_HEIGHT_RANGE: SliderRange = { min: 0.1, max: 0.8, step: 0.01 };
/** `presence.min_face_width`: a face counts at the stand from this width. */
export const FACE_WIDTH_RANGE: SliderRange = { min: 0.03, max: 0.25, step: 0.005 };

/** Snap a presence threshold onto its slider (laptop-side, no server command). */
export function snapPresence(value: number, range: SliderRange): number {
  return snap(value, range, 3);
}

/** One `calibration.set` at most this often while a slider moves. */
export const DEBOUNCE_MS = 150;

/**
 * After a send, `status` messages that left the board before the command
 * arrived still carry the old values: they must not snap the sliders back.
 */
export const ECHO_HOLD_MS = 1_500;

/** A rectangle as fractions (0..1) of the frame element. */
export interface FractionRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Python's `round()`: halves go to the even neighbour, as `zoom_crop` rounds. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) {
    return floor + 1;
  }
  if (diff < 0.5) {
    return floor;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The region of a `width x height` frame YOLOX sees at `zoom` around `centre`,
 * as fractions of the frame: the same arithmetic as `cpsa.yolox.zoom_crop`, so
 * the dashed outline is the crop to the pixel.
 */
export function analysisRect(
  zoom: number,
  centre: readonly [number, number],
  width = 640,
  height = 480,
): FractionRect {
  const safeZoom = Number.isFinite(zoom) && zoom >= 1 ? zoom : 1;
  const cropWidth = Math.max(1, Math.min(width, roundHalfEven(width / safeZoom)));
  const cropHeight = Math.max(1, Math.min(height, roundHalfEven(height / safeZoom)));
  const left = roundHalfEven(clamp(centre[0], 0, 1) * width - cropWidth / 2);
  const top = roundHalfEven(clamp(centre[1], 0, 1) * height - cropHeight / 2);
  return {
    left: clamp(left, 0, width - cropWidth) / width,
    top: clamp(top, 0, height - cropHeight) / height,
    width: cropWidth / width,
    height: cropHeight / height,
  };
}

/**
 * Map a rectangle of the image into its `object-fit: contain` box: the image
 * keeps its aspect inside a frame of another aspect, centred with bars.
 */
export function containRect(
  rect: FractionRect,
  imageAspect: number,
  frameAspect: number,
): FractionRect {
  if (!(imageAspect > 0) || !(frameAspect > 0)) {
    return rect;
  }
  let scaleX = 1;
  let scaleY = 1;
  if (imageAspect > frameAspect) {
    scaleY = frameAspect / imageAspect;
  } else {
    scaleX = imageAspect / frameAspect;
  }
  return {
    left: (1 - scaleX) / 2 + rect.left * scaleX,
    top: (1 - scaleY) / 2 + rect.top * scaleY,
    width: rect.width * scaleX,
    height: rect.height * scaleY,
  };
}

/**
 * Snap the edited values onto their range and step, so the board never says
 * `bad_command`. Only the keys of `patch` are touched: a value the board
 * reported between two steps (a YAML zoom of 1.33) is sent back unchanged.
 */
export function normalizePatch(patch: Partial<CalibrationValues>): Partial<CalibrationValues> {
  const normalized: Partial<CalibrationValues> = {};
  if (patch.detector !== undefined && (DETECTORS as readonly string[]).includes(patch.detector)) {
    normalized.detector = patch.detector;
  }
  if (patch.zoom !== undefined) {
    normalized.zoom = snap(patch.zoom, ZOOM_RANGE, 2);
  }
  if (patch.zoom_center !== undefined) {
    normalized.zoom_center = [
      snap(patch.zoom_center[0], CENTRE_RANGE, 2),
      snap(patch.zoom_center[1], CENTRE_RANGE, 2),
    ];
  }
  if (patch.gain !== undefined) {
    normalized.gain = snap(patch.gain, GAIN_RANGE, 0);
  }
  if (patch.exposure !== undefined) {
    const choice = EXPOSURES.find((item) => item.value === Math.round(patch.exposure ?? 0));
    if (choice !== undefined) {
      normalized.exposure = choice.value;
    }
  }
  if (patch.saturation !== undefined) {
    normalized.saturation = snap(patch.saturation, SATURATION_RANGE, 0);
  }
  return normalized;
}

function copyValues(values: CalibrationValues): CalibrationValues {
  return {
    detector: values.detector,
    zoom: values.zoom,
    zoom_center: [values.zoom_center[0], values.zoom_center[1]],
    gain: values.gain,
    exposure: values.exposure,
    saturation: values.saturation,
  };
}

export function sameCalibration(a: CalibrationValues | null, b: CalibrationValues | null): boolean {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.detector === b.detector &&
    a.zoom === b.zoom &&
    a.zoom_center[0] === b.zoom_center[0] &&
    a.zoom_center[1] === b.zoom_center[1] &&
    a.gain === b.gain &&
    a.exposure === b.exposure &&
    a.saturation === b.saturation
  );
}

/**
 * What one `calibration.set` carries: every value, so a command the board
 * dropped is repaired by the next one. An exposure the YAML set outside the
 * two panel choices is left out until the operator picks one.
 */
export function calibrationCommand(values: CalibrationValues): Partial<CalibrationValues> {
  const command: Partial<CalibrationValues> = {
    detector: values.detector,
    zoom: values.zoom,
    zoom_center: [values.zoom_center[0], values.zoom_center[1]],
    gain: values.gain,
    saturation: values.saturation,
  };
  if (EXPOSURES.some((choice) => choice.value === values.exposure)) {
    command.exposure = values.exposure;
  }
  return command;
}

export interface CalibrationControllerOptions {
  /** Send one `calibration.set`. */
  send: (values: Partial<CalibrationValues>) => void;
  /** Send `calibration.reset`. */
  reset: () => void;
  /** The values to show changed (a local edit or a new `status`). */
  onChange: (values: CalibrationValues | null) => void;
  delayMs?: number;
  holdMs?: number;
  now?: () => number;
}

/**
 * The panel's state: the values on screen, the debounce of the sends and the
 * rule that decides when `status.calibration` may overwrite a local edit.
 */
export class CalibrationController {
  private readonly options: CalibrationControllerOptions;
  private readonly delayMs: number;
  private readonly holdMs: number;
  private readonly now: () => number;
  private current: CalibrationValues | null = null;
  private installed: DetectorName[] = ["yolox"];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private holdUntil = 0;

  constructor(options: CalibrationControllerOptions) {
    this.options = options;
    this.delayMs = options.delayMs ?? DEBOUNCE_MS;
    this.holdMs = options.holdMs ?? ECHO_HOLD_MS;
    this.now = options.now ?? (() => Date.now());
  }

  /** The values on screen; `null` until the board reported them. */
  get values(): CalibrationValues | null {
    return this.current;
  }

  /** The detectors the board has installed (`status.calibration.detectors`). */
  get detectors(): readonly DetectorName[] {
    return this.installed;
  }

  /** True while an edit waits for its debounce. */
  get pending(): boolean {
    return this.timer !== null;
  }

  /** A `status.calibration` arrived: show it unless the operator is mid-edit. */
  sync(remote: CalibrationStatus | CalibrationValues | null | undefined): void {
    if (remote == null) {
      return;
    }
    if ("detectors" in remote) {
      const installed = [...remote.detectors];
      if (installed.join() !== this.installed.join()) {
        this.installed = installed;
        this.options.onChange(this.current);
      }
    }
    if (this.timer !== null || this.now() < this.holdUntil) {
      return;
    }
    const next = copyValues(remote);
    if (sameCalibration(this.current, next)) {
      return;
    }
    this.current = next;
    this.options.onChange(this.current);
  }

  /** The operator moved a control: show it at once, send it after the debounce. */
  change(patch: Partial<CalibrationValues>): void {
    if (this.current === null) {
      return;
    }
    if (patch.detector !== undefined && !this.installed.includes(patch.detector)) {
      return;
    }
    const next = { ...copyValues(this.current), ...normalizePatch(patch) };
    if (sameCalibration(this.current, next)) {
      return;
    }
    this.current = next;
    this.options.onChange(this.current);
    if (this.timer !== null) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => this.flush(), this.delayMs);
  }

  /** Send the pending edit now (also what the debounce timer calls). */
  flush(): void {
    if (this.timer === null || this.current === null) {
      return;
    }
    clearTimeout(this.timer);
    this.timer = null;
    this.holdUntil = this.now() + this.holdMs;
    this.options.send(calibrationCommand(this.current));
  }

  /** "Ripristina": drop the pending edit and let the board's next `status` speak. */
  reset(): void {
    this.cancel();
    this.holdUntil = 0;
    this.options.reset();
  }

  /** A new connection: the board's `status` is the truth again. */
  resync(): void {
    this.cancel();
    this.holdUntil = 0;
  }

  dispose(): void {
    this.cancel();
  }

  private cancel(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function snap(value: number, range: SliderRange, digits: number): number {
  const safe = Number.isFinite(value) ? value : range.min;
  const stepped = range.min + Math.round((safe - range.min) / range.step) * range.step;
  const factor = 10 ** digits;
  return clamp(Math.round(stepped * factor) / factor, range.min, range.max);
}
