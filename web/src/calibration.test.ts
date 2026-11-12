import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CalibrationController,
  analysisRect,
  calibrationCommand,
  containRect,
  gainToSlider,
  normalizePatch,
  sliderToGain,
} from "./calibration";
import type { CalibrationValues } from "./protocol";

const YAML: CalibrationValues = {
  zoom: 1.3,
  zoom_center: [0.5, 0.5],
  gain: 200,
  exposure: 100,
  saturation: 110,
};

/** `cpsa.yolox.zoom_crop(640, 480, zoom, centre)` as the board computes it. */
const BOARD_CROPS: Array<[number, [number, number], [number, number, number, number]]> = [
  [1.0, [0.5, 0.5], [0, 0, 640, 480]],
  [1.3, [0.5, 0.5], [74, 56, 492, 369]],
  [1.5, [0.4, 0.6], [42, 128, 427, 320]],
  [2.0, [0.0, 1.0], [0, 240, 320, 240]],
  [1.25, [0.5, 0.5], [64, 48, 512, 384]],
  [1.6, [0.5, 0.5], [120, 90, 400, 300]],
];

describe("analysisRect", () => {
  it.each(BOARD_CROPS)("matches the board crop at zoom %s around %j", (zoom, centre, crop) => {
    const rect = analysisRect(zoom, centre, 640, 480);
    expect(rect.left * 640).toBeCloseTo(crop[0], 6);
    expect(rect.top * 480).toBeCloseTo(crop[1], 6);
    expect(rect.width * 640).toBeCloseTo(crop[2], 6);
    expect(rect.height * 480).toBeCloseTo(crop[3], 6);
  });

  it("keeps the crop inside the frame and treats a bad zoom as the whole frame", () => {
    const corner = analysisRect(2, [1, 0]);
    expect(corner.left + corner.width).toBeCloseTo(1, 9);
    expect(corner.top).toBe(0);
    expect(analysisRect(Number.NaN, [0.5, 0.5])).toEqual({ left: 0, top: 0, width: 1, height: 1 });
  });
});

describe("containRect", () => {
  it("is the identity when the frame has the image aspect", () => {
    const rect = { left: 0.1, top: 0.2, width: 0.5, height: 0.6 };
    expect(containRect(rect, 4 / 3, 4 / 3)).toEqual(rect);
  });

  it("adds the side bars of a wider frame and the top bars of a taller one", () => {
    const whole = { left: 0, top: 0, width: 1, height: 1 };
    const wide = containRect(whole, 4 / 3, 16 / 9);
    expect(wide.left).toBeCloseTo(0.125, 9);
    expect(wide.width).toBeCloseTo(0.75, 9);
    expect(wide.top).toBe(0);
    const tall = containRect(whole, 16 / 9, 4 / 3);
    expect(tall.top).toBeCloseTo(0.125, 9);
    expect(tall.height).toBeCloseTo(0.75, 9);
  });
});

describe("the logarithmic gain slider", () => {
  it("spans 64..2540 and round-trips every gain within one step", () => {
    expect(sliderToGain(0)).toBe(64);
    expect(sliderToGain(1000)).toBe(2540);
    expect(gainToSlider(64)).toBe(0);
    expect(gainToSlider(2540)).toBe(1000);
    for (const gain of [64, 100, 200, 333, 800, 1500, 2540]) {
      expect(Math.abs(sliderToGain(gainToSlider(gain)) - gain) / gain).toBeLessThan(0.005);
    }
    // The low gains a lit room needs keep most of the track.
    expect(gainToSlider(800)).toBeGreaterThan(600);
  });
});

describe("normalizePatch and calibrationCommand", () => {
  it("snaps edited values onto their ranges and steps", () => {
    expect(normalizePatch({ zoom: 1.337 })).toEqual({ zoom: 1.35 });
    expect(normalizePatch({ zoom: 3 })).toEqual({ zoom: 2 });
    expect(normalizePatch({ zoom_center: [-0.2, 0.456] })).toEqual({ zoom_center: [0, 0.46] });
    expect(normalizePatch({ gain: 10.4, saturation: 300 })).toEqual({ gain: 64, saturation: 255 });
    expect(normalizePatch({ exposure: 200 })).toEqual({ exposure: 200 });
    expect(normalizePatch({ exposure: 150 })).toEqual({});
  });

  it("sends every value, but an exposure outside the two choices stays out", () => {
    expect(calibrationCommand(YAML)).toEqual(YAML);
    const { exposure: _ignored, ...rest } = YAML;
    expect(calibrationCommand({ ...YAML, exposure: 333 })).toEqual(rest);
  });
});

describe("CalibrationController", () => {
  let sent: Array<Partial<CalibrationValues>>;
  let resets: number;
  let shown: Array<CalibrationValues | null>;
  let now: number;
  let controller: CalibrationController;

  beforeEach(() => {
    vi.useFakeTimers();
    sent = [];
    resets = 0;
    shown = [];
    now = 1_000;
    controller = new CalibrationController({
      send: (values) => sent.push(values),
      reset: () => {
        resets += 1;
      },
      onChange: (values) => shown.push(values),
      now: () => now,
    });
  });

  afterEach(() => {
    controller.dispose();
    vi.useRealTimers();
  });

  it("starts empty and ignores edits until the board reports its values", () => {
    controller.change({ zoom: 1.5 });
    expect(controller.values).toBeNull();
    controller.sync(null);
    controller.sync(YAML);
    expect(controller.values).toEqual(YAML);
    expect(shown).toHaveLength(1);
  });

  it("sends one command per burst of edits, ~150 ms after the last", () => {
    controller.sync(YAML);
    controller.change({ zoom: 1.35 });
    vi.advanceTimersByTime(100);
    controller.change({ zoom: 1.4 });
    vi.advanceTimersByTime(100);
    controller.change({ zoom: 1.45 });
    expect(sent).toEqual([]);
    expect(controller.values?.zoom).toBe(1.45);
    expect(controller.pending).toBe(true);

    vi.advanceTimersByTime(149);
    expect(sent).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(sent).toEqual([{ ...YAML, zoom: 1.45 }]);
    expect(controller.pending).toBe(false);
  });

  it("does not let a stale status snap the sliders back while an edit is in flight", () => {
    controller.sync(YAML);
    controller.change({ gain: 640 });
    controller.sync(YAML); // pending edit: ignored
    vi.advanceTimersByTime(150);
    now += 500;
    controller.sync(YAML); // left the board before the command: ignored
    expect(controller.values?.gain).toBe(640);

    now += 1_500;
    controller.sync({ ...YAML, gain: 640, saturation: 90 });
    expect(controller.values?.saturation).toBe(90);
  });

  it("an unchanged value sends nothing", () => {
    controller.sync(YAML);
    controller.change({ zoom: 1.3, exposure: 150 });
    vi.advanceTimersByTime(500);
    expect(sent).toEqual([]);
  });

  it("reset drops the pending edit and takes the next status at once", () => {
    controller.sync(YAML);
    controller.change({ saturation: 60 });
    controller.reset();
    vi.advanceTimersByTime(500);
    expect(sent).toEqual([]);
    expect(resets).toBe(1);

    controller.sync({ ...YAML, zoom: 1.0 });
    expect(controller.values?.zoom).toBe(1.0);
  });

  it("flush sends a pending edit immediately", () => {
    controller.sync(YAML);
    controller.change({ exposure: 200 });
    controller.flush();
    expect(sent).toEqual([{ ...YAML, exposure: 200 }]);
    vi.advanceTimersByTime(500);
    expect(sent).toHaveLength(1);
  });
});
