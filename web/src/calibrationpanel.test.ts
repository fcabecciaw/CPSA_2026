// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { gainToSlider, sliderToGain } from "./calibration";
import { CalibrationPanel } from "./calibrationpanel";
import type { CalibrationValues } from "./protocol";

const YAML: CalibrationValues = {
  detector: "yolox",
  zoom: 1.3,
  zoom_center: [0.5, 0.5],
  gain: 200,
  exposure: 100,
  saturation: 110,
};

function setup(presence?: { thresholds: { minBoxHeight: number; minFaceWidth: number } }) {
  document.body.innerHTML = `
    <div id="calibration"></div>
    <div id="frame"><div id="rect" hidden></div></div>`;
  const sent: Array<Partial<CalibrationValues>> = [];
  let resets = 0;
  const panel = new CalibrationPanel({
    root: document.getElementById("calibration") as HTMLElement,
    rect: document.getElementById("rect") as HTMLElement,
    frame: document.getElementById("frame") as HTMLElement,
    send: (values) => sent.push(values),
    reset: () => {
      resets += 1;
    },
    presence:
      presence === undefined
        ? undefined
        : {
            get: () => presence.thresholds,
            set: (next) => {
              presence.thresholds = { ...presence.thresholds, ...next };
            },
          },
  });
  const slider = (key: string) =>
    document.querySelector<HTMLInputElement>(`input[data-key="${key}"]`) as HTMLInputElement;
  return { panel, sent, resets: () => resets, slider };
}

function drag(input: HTMLInputElement, value: number): void {
  input.value = String(value);
  input.dispatchEvent(new Event("input"));
}

describe("CalibrationPanel", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("is closed and disabled until the board is connected and reports its values", () => {
    const { panel, slider } = setup();
    const card = document.querySelector<HTMLElement>(".calibration-card") as HTMLElement;
    expect(card.hidden).toBe(true);
    expect(slider("zoom").disabled).toBe(true);

    panel.setConnected(true);
    panel.setStatus(YAML);
    expect(slider("zoom").disabled).toBe(false);
    expect(slider("zoom").value).toBe("1.3");
    expect(card.textContent).toContain("1.30×");
  });

  it("sends the dragged values after the debounce and outlines the crop while open", () => {
    const { panel, sent, slider } = setup();
    panel.setConnected(true);
    panel.setStatus(YAML);
    const rect = document.getElementById("rect") as HTMLElement;
    expect(rect.hidden).toBe(true);

    panel.toggle();
    expect(rect.hidden).toBe(false);
    expect(rect.style.left).toBe("11.563%");
    expect(rect.style.width).toBe("76.875%");

    drag(slider("zoom"), 1.5);
    drag(slider("centre_x"), 0.4);
    drag(slider("centre_y"), 0.6);
    expect(sent).toEqual([]);
    expect(rect.style.left).toBe(`${((42 / 640) * 100).toFixed(3)}%`);
    expect(rect.style.top).toBe(`${((128 / 480) * 100).toFixed(3)}%`);

    vi.advanceTimersByTime(150);
    expect(sent).toEqual([{ ...YAML, zoom: 1.5, zoom_center: [0.4, 0.6] }]);

    panel.toggle();
    expect(rect.hidden).toBe(true);
  });

  it("picks the exposure with the segmented control and resets to the YAML", () => {
    const { panel, sent, resets } = setup();
    panel.setConnected(true);
    panel.setStatus(YAML);
    const [hundredth, fiftieth] = Array.from(
      document.querySelectorAll<HTMLButtonElement>(".calibration-segment[data-value]"),
    );
    expect(hundredth.getAttribute("aria-pressed")).toBe("true");

    fiftieth.click();
    expect(fiftieth.getAttribute("aria-pressed")).toBe("true");
    vi.advanceTimersByTime(150);
    expect(sent.at(-1)?.exposure).toBe(200);

    (document.querySelector(".calibration-reset") as HTMLButtonElement).click();
    expect(resets()).toBe(1);
  });

  it("closing the card sends a pending edit at once", () => {
    const { panel, sent, slider } = setup();
    panel.setConnected(true);
    panel.setStatus(YAML);
    panel.toggle();
    expect(slider("gain").value).toBe(String(gainToSlider(200)));
    drag(slider("gain"), gainToSlider(640));
    panel.toggle();
    expect(sent).toEqual([{ ...YAML, gain: sliderToGain(gainToSlider(640)) }]);
  });
});

describe("CalibrationPanel: YOLOv5 and Distanza persone", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const detectorButton = (name: string) =>
    document.querySelector<HTMLButtonElement>(`button[data-detector="${name}"]`) as HTMLButtonElement;

  it("disables YOLOv5 when the board does not have it", () => {
    const { panel } = setup();
    panel.setConnected(true);
    panel.setStatus({ ...YAML, detectors: ["yolox"] });

    expect(detectorButton("yolox").getAttribute("aria-pressed")).toBe("true");
    expect(detectorButton("yolov5").disabled).toBe(true);
  });

  it("switches to YOLOv5, greys the zoom out and hides the crop outline", () => {
    const { panel, sent, slider } = setup();
    panel.setConnected(true);
    panel.setStatus({ ...YAML, detectors: ["yolox", "yolov5"] });
    panel.toggle();
    const rect = document.getElementById("rect") as HTMLElement;
    const note = document.querySelector<HTMLElement>(".calibration-zoom-note") as HTMLElement;
    expect(rect.hidden).toBe(false);
    expect(note.hidden).toBe(true);

    detectorButton("yolov5").click();
    expect(detectorButton("yolov5").getAttribute("aria-pressed")).toBe("true");
    expect(slider("zoom").disabled).toBe(true);
    expect(slider("centre_x").disabled).toBe(true);
    expect(slider("gain").disabled).toBe(false);
    expect(note.hidden).toBe(false);
    expect(rect.hidden).toBe(true);
    vi.advanceTimersByTime(150);
    expect(sent).toEqual([{ ...YAML, detector: "yolov5" }]);
  });

  it("sets the presence thresholds live and shows the tracked sizes", () => {
    const presence = { thresholds: { minBoxHeight: 0.35, minFaceWidth: 0.08 } };
    const { panel, sent, slider } = setup(presence);
    panel.setStatus(YAML); // not connected: the laptop-side sliders still work
    expect(slider("person_height").value).toBe("0.35");
    expect(slider("person_height").disabled).toBe(false);

    drag(slider("person_height"), 0.5);
    drag(slider("face_width"), 0.12);
    expect(presence.thresholds).toEqual({ minBoxHeight: 0.5, minFaceWidth: 0.12 });
    expect(sent).toEqual([]);

    const hint = document.querySelector(".calibration-presence-hint") as HTMLElement;
    expect(hint.textContent).toBe("nessuno in vista");
    panel.setPresenceTracks([
      { kind: "persona", id: 1, size: 0.42, atStand: true, greeted: false },
      { kind: "volto", id: 2, size: 0.06, atStand: false, greeted: false },
    ]);
    expect(hint.textContent).toBe("ora: persona 42% ● · volto 6%");
  });

  it("hides Distanza persone without a presence gate", () => {
    setup();
    const legends = Array.from(document.querySelectorAll("legend")).map((node) => node.textContent);
    expect(legends).toContain("Distanza persone");
    const group = Array.from(document.querySelectorAll<HTMLElement>("fieldset")).find(
      (node) => node.textContent?.includes("Distanza persone"),
    ) as HTMLElement;
    expect(group.hidden).toBe(true);
  });
});
