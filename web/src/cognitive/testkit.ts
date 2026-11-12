/**
 * Shared, DOM-free fixtures for the cognitive tests: a 640×480 showcase frame,
 * a COCO-17 skeleton and an in-memory storage. Test-only, never imported by the
 * dashboard.
 */

import type {
  PoseStatePayload,
  ShowcaseObject,
  ShowcaseStatePayload,
} from "../protocol";
import type { StorageLike } from "./memory";

export const FRAME = { width: 640, height: 480 } as const;

export function detection(
  label: string,
  classId: number,
  box: [number, number, number, number],
  score = 0.9,
): ShowcaseObject {
  return { label, class_id: classId, score, box };
}

export function person(
  x: number,
  y = 80,
  width = 120,
  height = 360,
  score = 0.92,
): ShowcaseObject {
  return detection("persona", 0, [x, y, x + width, y + height], score);
}

export function showcase(
  objects: readonly ShowcaseObject[],
  model = "yolox_nano",
): ShowcaseStatePayload {
  const grouped = new Map<string, number>();
  for (const object of objects) {
    grouped.set(object.label, (grouped.get(object.label) ?? 0) + 1);
  }
  return {
    model,
    width: FRAME.width,
    height: FRAME.height,
    fps: 59.6,
    dpu_ms: 4.1,
    total_ms: 15.8,
    objects: [...objects],
    counts: [...grouped.entries()].map(([label, count]) => ({ label, count })),
  };
}

export type SkeletonOverrides = Record<number, [number, number, number]>;

/** A standing person in COCO-17 order, with individual keypoints overridden. */
export function keypoints(overrides: SkeletonOverrides = {}): Array<[number, number, number]> {
  const points: Array<[number, number, number]> = [
    [320, 120, 0.9], // nose
    [312, 116, 0.8], // left eye
    [328, 116, 0.8], // right eye
    [306, 118, 0.8], // left ear
    [334, 118, 0.8], // right ear
    [280, 200, 0.9], // left shoulder
    [360, 200, 0.9], // right shoulder
    [260, 250, 0.8], // left elbow
    [380, 250, 0.8], // right elbow
    [250, 300, 0.8], // left wrist
    [390, 300, 0.8], // right wrist
    [300, 300, 0.9], // left hip
    [340, 300, 0.9], // right hip
    [300, 390, 0.8], // left knee
    [340, 390, 0.8], // right knee
    [300, 470, 0.8], // left ankle
    [340, 470, 0.8], // right ankle
  ];
  for (const [index, point] of Object.entries(overrides)) {
    points[Number(index)] = point;
  }
  return points;
}

export function pose(
  overrides: SkeletonOverrides = {},
  score = 0.85,
): PoseStatePayload {
  return {
    model: "movenet",
    width: FRAME.width,
    height: FRAME.height,
    fps: 30,
    dpu_ms: 6.2,
    total_ms: 12.1,
    score,
    keypoints: keypoints(overrides),
  };
}

export function armsUp(): SkeletonOverrides {
  return {
    9: [230, 140, 0.9],
    10: [410, 140, 0.9],
  };
}

export function memoryStorage(): StorageLike & { readonly values: Map<string, string> } {
  const values = new Map<string, string>();
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value);
    },
    removeItem: (key) => {
      values.delete(key);
    },
  };
}
