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
