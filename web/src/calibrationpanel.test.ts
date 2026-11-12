// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { gainToSlider, sliderToGain } from "./calibration";
import { CalibrationPanel } from "./calibrationpanel";
import type { CalibrationValues } from "./protocol";

const YAML: CalibrationValues = {
  zoom: 1.3,
  zoom_center: [0.5, 0.5],
  gain: 200,
  exposure: 100,
  saturation: 110,
};

function setup() {
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
      document.querySelectorAll<HTMLButtonElement>(".calibration-segment"),
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
