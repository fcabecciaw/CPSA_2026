/**
 * Face crops for the laptop landmarker (C3): the square crop around the last
 * faces and the mapping of crop landmarks back to the 640×480 frame.
 */

import { describe, expect, it } from "vitest";

import { faceCrop, landmarkBox, toFramePixels, type NormBox } from "./facecrop";

const W = 640;
const H = 480;

describe("faceCrop — square, 1.8× margin, at least 256 px, inside the frame", () => {
  it("has no crop without a face", () => {
    expect(faceCrop([], W, H)).toBeNull();
  });

  it("uses the 256 px minimum around a small, distant face", () => {
    // A 40×48 px face centred at (320, 200).
    const box: NormBox = [300 / W, 176 / H, 340 / W, 224 / H];
    expect(faceCrop([box], W, H)).toEqual({ x: 192, y: 72, size: 256 });
  });

  it("grows with the face: 1.8× its larger side", () => {
    // A 200×180 px face centred at (320, 240) → 360 px.
    const box: NormBox = [220 / W, 150 / H, 420 / W, 330 / H];
    expect(faceCrop([box], W, H)).toEqual({ x: 140, y: 60, size: 360 });
  });

  it("slides inward at the frame edge and never exceeds the shorter side", () => {
    const corner: NormBox = [0, 0, 40 / W, 40 / H];
    expect(faceCrop([corner], W, H)).toEqual({ x: 0, y: 0, size: 256 });
    const huge: NormBox = [100 / W, 20 / H, 500 / W, 460 / H];
    const crop = faceCrop([huge], W, H)!;
    expect(crop.size).toBe(480);
    expect(crop.y).toBe(0);
    expect(crop.x + crop.size).toBeLessThanOrEqual(W);
  });

  it("covers every face of the previous sample", () => {
    const a: NormBox = [100 / W, 200 / H, 140 / W, 250 / H];
    const b: NormBox = [300 / W, 210 / H, 340 / W, 260 / H];
    const crop = faceCrop([a, b], W, H)!;
    expect(crop.x).toBeLessThanOrEqual(100);
    expect(crop.x + crop.size).toBeGreaterThanOrEqual(340);
  });
});

describe("landmarks in the crop map back to the full frame", () => {
  const crop = { x: 192, y: 72, size: 256 };

  it("maps crop-normalised points to source pixels", () => {
    expect(toFramePixels({ x: 0, y: 0 }, crop, W, H)).toEqual({ x: 192, y: 72 });
    expect(toFramePixels({ x: 0.5, y: 0.5 }, crop, W, H)).toEqual({ x: 320, y: 200 });
    expect(toFramePixels({ x: 0.5, y: 0.5 }, null, W, H)).toEqual({ x: 320, y: 240 });
  });

  it("gives the same face box from the crop and from the full frame", () => {
    // The face of the 256 px minimum test, seen in the crop…
    const inCrop = [
      { x: (300 - 192) / 256, y: (176 - 72) / 256 },
      { x: (340 - 192) / 256, y: (224 - 72) / 256 },
      { x: 0.5, y: 0.5 },
    ];
    // …and in the full frame.
    const inFrame = [
      { x: 300 / W, y: 176 / H },
      { x: 340 / W, y: 224 / H },
    ];
    const fromCrop = landmarkBox(inCrop, crop, W, H);
    const fromFrame = landmarkBox(inFrame, null, W, H);
    fromCrop.forEach((value, index) => expect(value).toBeCloseTo(fromFrame[index]!, 6));
    // Following the face: the next crop is the same.
    expect(faceCrop([fromCrop], W, H)).toEqual(crop);
  });
});
