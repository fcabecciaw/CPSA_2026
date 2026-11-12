/**
 * MediaPipe Face Landmarker on the laptop, for the smile and head cues (R2-D).
 *
 * The board already runs YOLO/MoveNet; faces are *not* computed there. This
 * module samples the frames of the `<img src=/video.mjpeg>` at ~10–15 Hz,
 * draws them into a small offscreen canvas and runs the Face Landmarker (GPU
 * delegate, CPU fallback, `numFaces: 4`) offline: the wasm runtime and the
 * `.task` model are served from `web/public/`.
 *
 * When the previous sample found a face, the next one runs on a square crop
 * around it (`cognitive/facecrop.ts`, C3) so a visitor 1.5–2 m away is large
 * enough for stable blendshapes; every `FULL_FRAME_EVERY` samples, and
 * whenever the crop finds nobody, it runs on the full frame again.
 *
 * Everything here is browser-only and best-effort: if WebGL, wasm or the model
 * fail, the tracker reports `unavailable` and the dashboard keeps its 60 fps
 * object path untouched. The detectors themselves are pure and live in
 * `cognitive/facecues.ts`.
 */

import { faceCrop, landmarkBox, toFramePixels, type CropRect, type NormBox } from "./cognitive/facecrop";
import { FaceCueDetector, type FaceCue, type SmileReadout } from "./cognitive/facecues";

export type FaceTrackerStatus = "off" | "loading" | "gpu" | "cpu" | "unavailable";

export interface FacePoint {
  x: number;
  y: number;
  z?: number;
}

interface Classification {
  categoryName?: string;
  score?: number;
}

interface Classifications {
  categories?: Classification[];
}

interface LandmarkerResult {
  faceLandmarks?: FacePoint[][];
  faceBlendshapes?: Classifications[];
}

interface FaceLandmarkerLike {
  detectForVideo(frame: HTMLCanvasElement, timestamp: number): LandmarkerResult;
  close(): void;
}

interface WasmFileset {
  readonly wasmLoaderPath: string;
  readonly wasmBinaryPath: string;
}

interface VisionModule {
  FilesetResolver: {
    forVisionTasks(basePath: string, useModule?: boolean): Promise<WasmFileset>;
  };
  FaceLandmarker: {
    createFromOptions(
      fileset: WasmFileset,
      options: {
        baseOptions: { modelAssetPath: string; delegate: "GPU" | "CPU" };
        runningMode: "VIDEO";
        numFaces: number;
        outputFaceBlendshapes: boolean;
      },
    ): Promise<FaceLandmarkerLike>;
  };
}

/** One landmarker sample, for the presence fusion and the debug readout. */
export interface FaceSample {
  /** Faces found (0 when the crop or the frame had nobody). */
  count: number;
  /** Face boxes normalised to the full frame. */
  boxes: NormBox[];
  /** The crop the sample ran on, `null` for the full frame. */
  crop: CropRect | null;
}

export interface FaceTrackerOptions {
  /** Cue callback; the tracker never throws into it. */
  onCue?: (cue: FaceCue) => void;
  /** Every sample: how many faces, for the presence fusion (C3). */
  onFaces?: (sample: FaceSample) => void;
  onStatus?: (status: FaceTrackerStatus) => void;
  /** Target sampling rate: 10–15 Hz. */
  fps?: number;
  /** The offscreen frame size: small on purpose, the landmarks are relative. */
  width?: number;
  height?: number;
}

/** Re-scan the whole frame this often while following a face: new arrivals. */
const FULL_FRAME_EVERY = 15;
/** The square crop canvas: a 256 px crop is drawn 1:1. */
const CROP_CANVAS = 256;

/** Landmark indices of the Face Landmarker mesh. */
const LEFT_EYE = 33;
const RIGHT_EYE = 263;
const NOSE_TIP = 1;
const FOREHEAD = 10;
const CHIN = 152;

/**
 * Yaw and pitch from five landmarks in source pixels: the same numbers whether
 * the sample ran on the full frame or on a crop (the nod and head-shake
 * detectors must not see a jump when the input changes).
 */
function headAngles(
  landmarks: readonly FacePoint[],
  toPixels: (point: FacePoint) => { x: number; y: number },
): { yaw: number; pitch: number } {
  const at = (index: number): { x: number; y: number } | undefined => {
    const point = landmarks[index];
    return point === undefined ? undefined : toPixels(point);
  };
  const left = at(LEFT_EYE);
  const right = at(RIGHT_EYE);
  const nose = at(NOSE_TIP);
  const forehead = at(FOREHEAD);
  const chin = at(CHIN);
  if (left === undefined || right === undefined || nose === undefined) {
    return { yaw: 0, pitch: 0 };
  }
  const eyeDistance = Math.hypot(right.x - left.x, right.y - left.y) || 1e-3;
  const eyeMid = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
  const faceHeight =
    forehead === undefined || chin === undefined
      ? eyeDistance * 2
      : Math.hypot(forehead.x - chin.x, forehead.y - chin.y) || eyeDistance * 2;
  const yaw = (Math.atan2(nose.x - eyeMid.x, eyeDistance * 0.5) * 180) / Math.PI;
  const pitch = (Math.atan2(nose.y - eyeMid.y, faceHeight * 0.5) * 180) / Math.PI;
  return { yaw, pitch };
}

function blendshapeMap(entry: Classifications | undefined): Record<string, number> {
  const map: Record<string, number> = {};
  for (const category of entry?.categories ?? []) {
    if (category.categoryName !== undefined && category.score !== undefined) {
      map[category.categoryName] = category.score;
    }
  }
  return map;
}

export class FaceCueTracker {
  private readonly options: FaceTrackerOptions;
  private readonly detector = new FaceCueDetector();
  private status: FaceTrackerStatus = "off";
  private landmarker: FaceLandmarkerLike | null = null;
  private image: HTMLImageElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private context: CanvasRenderingContext2D | null = null;
  private cropCanvas: HTMLCanvasElement | null = null;
  private cropContext: CanvasRenderingContext2D | null = null;
  /** Faces of the previous sample, to crop around (full-frame coordinates). */
  private lastBoxes: NormBox[] = [];
  private samples = 0;
  private frameHandle: number | null = null;
  private intervalMs: number;
  private lastSampleAt = 0;
  private lastTimestamp = 0;
  private detectEma = 0;
  private failures = 0;

  constructor(options: FaceTrackerOptions = {}) {
    this.options = options;
    const fps = Math.min(15, Math.max(10, options.fps ?? 13));
    this.intervalMs = 1_000 / fps;
  }

  get currentStatus(): FaceTrackerStatus {
    return this.status;
  }

  /** The strongest face's smile score against its threshold (debug readout). */
  get smile(): SmileReadout | null {
    return this.detector.smile;
  }

  /** One short label for the "mente" strip: the cues run on the laptop. */
  get label(): string {
    switch (this.status) {
      case "gpu":
        return "laptop · GPU";
      case "cpu":
        return "laptop · CPU";
      case "loading":
        return "laptop · carico";
      case "unavailable":
        return "laptop · n/d";
      case "off":
        return "spento";
    }
  }

  /** Start sampling; resolves once the landmarker is ready or has failed. */
  async start(image: HTMLImageElement): Promise<void> {
    if (this.status === "loading" || this.landmarker !== null) {
      return;
    }
    this.image = image;
    this.setStatus("loading");
    try {
      const landmarker = await this.createLandmarker();
      this.landmarker = landmarker;
      this.status = landmarker.delegate;
      this.options.onStatus?.(this.status);
      this.canvas = document.createElement("canvas");
      this.canvas.width = this.options.width ?? 320;
      this.canvas.height = this.options.height ?? 240;
      this.context = this.canvas.getContext("2d", { willReadFrequently: false });
      this.cropCanvas = document.createElement("canvas");
      this.cropCanvas.width = CROP_CANVAS;
      this.cropCanvas.height = CROP_CANVAS;
      this.cropContext = this.cropCanvas.getContext("2d", { willReadFrequently: false });
      if (this.context === null || this.cropContext === null) {
        this.stop("unavailable");
        return;
      }
      this.loop();
    } catch {
      this.stop("unavailable");
    }
  }

  /** Stop sampling and release the wasm runtime. */
  stop(status: FaceTrackerStatus = "off"): void {
    if (this.frameHandle !== null) {
      window.cancelAnimationFrame(this.frameHandle);
      this.frameHandle = null;
    }
    if (this.landmarker !== null) {
      try {
        this.landmarker.close();
      } catch {
        // Closing twice is harmless; the wasm runtime is already gone.
      }
      this.landmarker = null;
    }
    this.detector.reset();
    this.failures = 0;
    this.canvas = null;
    this.context = null;
    this.cropCanvas = null;
    this.cropContext = null;
    this.lastBoxes = [];
    this.samples = 0;
    this.setStatus(status);
  }

  private async createLandmarker(): Promise<FaceLandmarkerLike & { delegate: "gpu" | "cpu" }> {
    const url = new URL("vendor/mediapipe/vision_bundle.mjs", document.baseURI).href;
    const module = (await import(/* @vite-ignore */ url)) as unknown as VisionModule;
    const wasmBase = new URL("vendor/mediapipe/wasm", document.baseURI).href.replace(/\/$/, "");
    const model = new URL("models/face_landmarker.task", document.baseURI).href;
    const fileset = await module.FilesetResolver.forVisionTasks(wasmBase);
    const build = async (delegate: "GPU" | "CPU"): Promise<FaceLandmarkerLike> =>
      module.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: model, delegate },
        runningMode: "VIDEO",
        numFaces: 4,
        outputFaceBlendshapes: true,
      });
    try {
      const landmarker = await build("GPU");
      return Object.assign(landmarker, { delegate: "gpu" as const });
    } catch {
      const landmarker = await build("CPU");
      return Object.assign(landmarker, { delegate: "cpu" as const });
    }
  }

  private loop(): void {
    this.frameHandle = window.requestAnimationFrame(this.onFrame);
  }

  private readonly onFrame = (): void => {
    this.frameHandle = null;
    const now = performance.now();
    if (now - this.lastSampleAt >= this.intervalMs) {
      this.lastSampleAt = now;
      this.sample(now);
    }
    this.loop();
  };

  private sample(now: number): void {
    const image = this.image;
    const canvas = this.canvas;
    const context = this.context;
    const cropCanvas = this.cropCanvas;
    const cropContext = this.cropContext;
    const landmarker = this.landmarker;
    if (
      image === null ||
      canvas === null ||
      context === null ||
      cropCanvas === null ||
      cropContext === null ||
      landmarker === null
    ) {
      return;
    }
    if (document.hidden || !image.complete || image.naturalWidth === 0 || image.hidden) {
      return;
    }
    const started = performance.now();
    try {
      const frameWidth = image.naturalWidth;
      const frameHeight = image.naturalHeight;
      this.samples += 1;
      const crop =
        this.samples % FULL_FRAME_EVERY === 0
          ? null
          : faceCrop(this.lastBoxes, frameWidth, frameHeight);
      let input: HTMLCanvasElement = canvas;
      if (crop === null) {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
      } else {
        cropContext.drawImage(
          image,
          crop.x,
          crop.y,
          crop.size,
          crop.size,
          0,
          0,
          cropCanvas.width,
          cropCanvas.height,
        );
        input = cropCanvas;
      }
      const timestamp = now > this.lastTimestamp ? now : this.lastTimestamp + 1;
      this.lastTimestamp = timestamp;
      const result = landmarker.detectForVideo(input, timestamp);
      const landmarks = result.faceLandmarks ?? [];
      const toPixels = (point: FacePoint): { x: number; y: number } =>
        toFramePixels(point, crop, frameWidth, frameHeight);
      const faces = landmarks.slice(0, 4).map((points, index) => {
        const angles = headAngles(points, toPixels);
        return {
          blendshapes: blendshapeMap(result.faceBlendshapes?.[index]),
          yaw: angles.yaw,
          pitch: angles.pitch,
          box: landmarkBox(points, crop, frameWidth, frameHeight),
        };
      });
      // Follow these faces next time; nobody → the next sample is full frame.
      this.lastBoxes = faces.map((face) => face.box);
      this.options.onFaces?.({ count: faces.length, boxes: this.lastBoxes, crop });
      const elapsed = performance.now() - started;
      this.detectEma = this.detectEma === 0 ? elapsed : this.detectEma * 0.8 + elapsed * 0.2;
      // Keep the dashboard at >= 55 fps: down-sample when detection is slow.
      this.intervalMs = Math.min(
        200,
        Math.max(66, this.intervalMs * (this.detectEma > 45 ? 1.15 : this.detectEma < 25 ? 0.95 : 1)),
      );
      this.failures = 0;
      for (const cue of this.detector.update({ now, faces })) {
        this.options.onCue?.(cue);
      }
    } catch {
      this.failures += 1;
      if (this.failures >= 3) {
        this.stop("unavailable");
      }
    }
  }

  private setStatus(status: FaceTrackerStatus): void {
    if (this.status === status) {
      return;
    }
    this.status = status;
    this.options.onStatus?.(status);
  }
}
