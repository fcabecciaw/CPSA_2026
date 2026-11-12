/**
 * F-E1 `pose.state`: 17 keypoints, or the message is rejected whole.
 */

import { describe, expect, it } from "vitest";

import { parseServerMessage } from "./protocol";

function envelope(data: Record<string, unknown>): string {
  return JSON.stringify({
    v: 1,
    type: "pose.state",
    seq: 7,
    t: 12.5,
    wall: "2026-11-12T10:00:00Z",
    data,
  });
}

const keypoints = Array.from({ length: 17 }, (_, index) => [index, index, 0.9]);

function poseData(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model: "movenet",
    width: 640,
    height: 480,
    fps: 29.8,
    dpu_ms: 7.4,
    total_ms: 24.6,
    score: 0.88,
    keypoints,
    ...overrides,
  };
}

describe("F-E1 pose.state", () => {
  it("parses a full skeleton", () => {
    const result = parseServerMessage(envelope(poseData()));
    expect(result.kind).toBe("message");
    if (result.kind === "message" && result.message.type === "pose.state") {
      expect(result.message.data.keypoints).toHaveLength(17);
      expect(result.message.data.score).toBe(0.88);
      expect(result.message.data.fps).toBe(29.8);
    }
  });

  it("rejects a skeleton with the wrong number of keypoints", () => {
    const result = parseServerMessage(envelope(poseData({ keypoints: keypoints.slice(0, 16) })));
    expect(result.kind).toBe("invalid");
    if (result.kind === "invalid") {
      expect(result.reason).toMatch(/17/);
    }
  });
});

describe("status.state", () => {
  function status(state: string): string {
    return JSON.stringify({
      v: 1,
      type: "status",
      seq: 3,
      t: 4.5,
      wall: "2026-11-12T10:00:00Z",
      data: {
        state,
        camera: "ok",
        model: "yolox_nano",
        fps: 59.4,
        dpu_ms: 4.7,
        errors: [],
        uptime_s: 12.5,
      },
    });
  }

  it("keeps the ~0.5 s model swap of the single-model board", () => {
    const result = parseServerMessage(status("switching"));
    expect(result.kind).toBe("message");
    if (result.kind === "message" && result.message.type === "status") {
      expect(result.message.data.state).toBe("switching");
    }
  });

  it("still rejects a state the protocol does not define", () => {
    expect(parseServerMessage(status("rebooting")).kind).toBe("invalid");
  });
});

describe("status.calibration (debug panel)", () => {
  function status(calibration: unknown): string {
    return JSON.stringify({
      v: 1,
      type: "status",
      seq: 3,
      t: 1.0,
      wall: "2026-11-12T10:00:00Z",
      data: {
        state: "running",
        camera: "ok",
        model: "yolox_nano",
        fps: 59.4,
        dpu_ms: 4.7,
        errors: [],
        uptime_s: 12.5,
        ...(calibration === undefined ? {} : { calibration }),
      },
    });
  }

  function calibrationOf(raw: string): unknown {
    const result = parseServerMessage(raw);
    expect(result.kind).toBe("message");
    return result.kind === "message" && result.message.type === "status"
      ? result.message.data.calibration
      : "not a status";
  }

  it("parses the five values", () => {
    const values = { zoom: 1.5, zoom_center: [0.5, 0.45], gain: 320, exposure: 200, saturation: 110 };
    expect(calibrationOf(status(values))).toEqual(values);
  });

  it("is null when absent, null or malformed, and the status still parses", () => {
    expect(calibrationOf(status(undefined))).toBeNull();
    expect(calibrationOf(status(null))).toBeNull();
    expect(calibrationOf(status({ zoom: "big" }))).toBeNull();
    expect(
      calibrationOf(status({ zoom: 1, zoom_center: [0.5], gain: 1, exposure: 1, saturation: 1 })),
    ).toBeNull();
  });
});
