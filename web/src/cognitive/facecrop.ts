/**
 * Face crops for the laptop Face Landmarker (C3), pure and DOM-free.
 *
 * A visitor 1.5–2 m away is a ~40 px face in the 640×480 `/video.mjpeg`
 * frame: the landmarker's blendshapes are noisy at that size. When the
 * previous sample found faces, the next one runs on a square crop around them
 * (1.8× margin, at least 256 px, clamped inside the frame), so the face is
 * several times larger; with no face, or every few samples to catch new
 * arrivals, it runs on the full frame. Landmarks found in the crop are mapped
 * back to the full frame, so boxes, head angles and the smile tracks never
 * depend on which input was used.
 */

/** `[x0, y0, x1, y1]` normalised to the full frame. */
export type NormBox = readonly [number, number, number, number];

/** A square region of the source frame, in source pixels. */
export interface CropRect {
  x: number;
  y: number;
  size: number;
}

export interface CropOptions {
  margin: number;
  minSize: number;
}

export const DEFAULT_CROP: CropOptions = { margin: 1.8, minSize: 256 };

/**
 * The square crop around the given faces (their union), or `null` when there
 * is no face. The crop never leaves the frame: near an edge it slides inward,
 * and it is at most as large as the frame's shorter side.
 */
export function faceCrop(
  boxes: readonly NormBox[],
  frameWidth: number,
  frameHeight: number,
  options: CropOptions = DEFAULT_CROP,
): CropRect | null {
  if (boxes.length === 0 || frameWidth <= 0 || frameHeight <= 0) {
    return null;
  }
  let x0 = Number.POSITIVE_INFINITY;
  let y0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  let y1 = Number.NEGATIVE_INFINITY;
  for (const box of boxes) {
    x0 = Math.min(x0, box[0] * frameWidth);
    y0 = Math.min(y0, box[1] * frameHeight);
    x1 = Math.max(x1, box[2] * frameWidth);
    y1 = Math.max(y1, box[3] * frameHeight);
  }
  const limit = Math.min(frameWidth, frameHeight);
  const side = Math.min(
    limit,
    Math.max(options.minSize, Math.max(x1 - x0, y1 - y0) * options.margin),
  );
  const centreX = (x0 + x1) / 2;
  const centreY = (y0 + y1) / 2;
  const x = Math.min(frameWidth - side, Math.max(0, centreX - side / 2));
  const y = Math.min(frameHeight - side, Math.max(0, centreY - side / 2));
  return { x: Math.round(x), y: Math.round(y), size: Math.round(side) };
}

/**
 * A landmark normalised to the landmarker input (the crop, or the full frame
 * when `crop` is `null`) → source pixels of the full frame.
 */
export function toFramePixels(
  point: { x: number; y: number },
  crop: CropRect | null,
  frameWidth: number,
  frameHeight: number,
): { x: number; y: number } {
  if (crop === null) {
    return { x: point.x * frameWidth, y: point.y * frameHeight };
  }
  return { x: crop.x + point.x * crop.size, y: crop.y + point.y * crop.size };
}

/** The bounding box of a face's landmarks, normalised to the full frame. */
export function landmarkBox(
  points: readonly { x: number; y: number }[],
  crop: CropRect | null,
  frameWidth: number,
  frameHeight: number,
): NormBox {
  let x0 = Number.POSITIVE_INFINITY;
  let y0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  let y1 = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    x0 = Math.min(x0, point.x);
    y0 = Math.min(y0, point.y);
    x1 = Math.max(x1, point.x);
    y1 = Math.max(y1, point.y);
  }
  if (!Number.isFinite(x0)) {
    return [0, 0, 0, 0];
  }
  const a = toFramePixels({ x: x0, y: y0 }, crop, frameWidth, frameHeight);
  const b = toFramePixels({ x: x1, y: y1 }, crop, frameWidth, frameHeight);
  const clamp = (value: number): number => Math.min(1, Math.max(0, value));
  return [
    clamp(a.x / frameWidth),
    clamp(a.y / frameHeight),
    clamp(b.x / frameWidth),
    clamp(b.y / frameHeight),
  ];
}
