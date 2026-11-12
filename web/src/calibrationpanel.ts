/**
 * The "Calibrazione" popover of the debug mode, and the dashed outline of the
 * region YOLOX analyses.
 *
 * A round button in the top bar (or `c`) opens a small card under it: zoom and
 * centre of the YOLOX crop, then gain, exposure and saturation of the camera,
 * and "Ripristina" to go back to `config/showcase.yaml`. Every change is shown
 * at once and sent to the board after a short debounce; the board stores it,
 * so the calibration survives a restart. While the card is open the video
 * shows the crop as a thin dashed rectangle. Presentation mode hides all of it
 * (`.debug-only`).
 */

import "./calibration.css";

import {
  CENTRE_RANGE,
  CalibrationController,
  EXPOSURES,
  GAIN_SLIDER,
  SATURATION_RANGE,
  ZOOM_RANGE,
  analysisRect,
  containRect,
  gainToSlider,
  sliderToGain,
  type SliderRange,
} from "./calibration";
import { el } from "./dom";
import type { CalibrationValues } from "./protocol";
import { isBareKey } from "./uimode";

export interface CalibrationPanelOptions {
  /** Empty container in the top bar; the panel builds its button and card in it. */
  root: HTMLElement;
  /** The dashed outline, a child of the video frame. */
  rect: HTMLElement;
  /** The video frame the outline is laid over (`object-fit: contain`). */
  frame: HTMLElement;
  send: (values: Partial<CalibrationValues>) => void;
  reset: () => void;
  /** Whether the debug mode is on: `c` only opens the panel there. */
  isDebug?: () => boolean;
  delayMs?: number;
}

type SliderKey = "zoom" | "centre_x" | "centre_y" | "gain" | "saturation";

interface Slider {
  input: HTMLInputElement;
  output: HTMLOutputElement;
  format: (value: number) => string;
}

const ICON_PATHS = [
  "M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16",
  "M12 9.5v5M9.5 12h5",
];

let panelCount = 0;

export class CalibrationPanel {
  readonly controller: CalibrationController;
  private readonly root: HTMLElement;
  private readonly rect: HTMLElement;
  private readonly frame: HTMLElement;
  private readonly toggleButton: HTMLButtonElement;
  private readonly card: HTMLElement;
  private readonly note: HTMLElement;
  private readonly resetButton: HTMLButtonElement;
  private readonly sliders = new Map<SliderKey, Slider>();
  private readonly exposureButtons: HTMLButtonElement[] = [];
  private readonly isDebug: () => boolean;
  private open = false;
  private connected = false;
  private frameWidth = 640;
  private frameHeight = 480;

  constructor(options: CalibrationPanelOptions) {
    this.root = options.root;
    this.rect = options.rect;
    this.frame = options.frame;
    this.isDebug = options.isDebug ?? (() => true);
    this.controller = new CalibrationController({
      send: options.send,
      reset: options.reset,
      onChange: () => this.render(),
      delayMs: options.delayMs,
    });

    panelCount += 1;
    const cardId = `calibration-card-${panelCount}`;
    this.toggleButton = el("button", "icon-button calibration-toggle");
    this.toggleButton.type = "button";
    this.toggleButton.setAttribute("aria-expanded", "false");
    this.toggleButton.setAttribute("aria-controls", cardId);
    this.toggleButton.setAttribute("aria-label", "Calibrazione (C)");
    this.toggleButton.title = "Calibrazione: zoom di YOLOX e immagine della telecamera (C)";
    this.toggleButton.append(icon());
    this.toggleButton.addEventListener("click", () => this.toggle());

    this.card = el("section", "calibration-card");
    this.card.id = cardId;
    this.card.hidden = true;
    this.card.setAttribute("aria-label", "Calibrazione");

    const header = el("header", "calibration-header");
    header.append(el("h2", "calibration-title", "Calibrazione"));
    this.note = el("p", "calibration-note");
    header.append(this.note);

    const analysis = group("Analisi YOLOX");
    analysis.append(
      this.slider("zoom", "Zoom", ZOOM_RANGE, (value) => `${value.toFixed(2)}×`),
      this.slider("centre_x", "Centro X", CENTRE_RANGE, percent),
      this.slider("centre_y", "Centro Y", CENTRE_RANGE, percent),
    );

    const image = group("Immagine");
    image.append(
      this.slider("gain", "Guadagno", GAIN_SLIDER, (value) => String(Math.round(value))),
      this.exposureRow(),
      this.slider("saturation", "Saturazione", SATURATION_RANGE, (value) =>
        String(Math.round(value)),
      ),
    );

    const footer = el("footer", "calibration-footer");
    this.resetButton = el("button", "calibration-reset", "Ripristina");
    this.resetButton.type = "button";
    this.resetButton.title = "Torna ai valori di config/showcase.yaml";
    this.resetButton.addEventListener("click", () => this.controller.reset());
    footer.append(this.resetButton, el("span", "calibration-hint", "torna a showcase.yaml"));

    this.card.append(header, analysis, image, footer);
    this.root.replaceChildren(this.toggleButton, this.card);

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.open) {
        this.setOpen(false);
      } else if (isBareKey(event, "c") && this.isDebug()) {
        event.preventDefault();
        this.toggle();
      }
    });
    window.addEventListener("resize", () => this.renderRect());
    this.render();
  }

  get isOpen(): boolean {
    return this.open;
  }

  toggle(): void {
    this.setOpen(!this.open);
  }

  setOpen(open: boolean): void {
    if (open === this.open) {
      return;
    }
    this.open = open;
    this.card.hidden = !open;
    this.toggleButton.setAttribute("aria-expanded", String(open));
    this.root.classList.toggle("calibration-open", open);
    if (!open) {
      this.controller.flush();
    }
    this.renderRect();
  }

  /** `status.calibration`, as the board applies it. */
  setStatus(values: CalibrationValues | null | undefined): void {
    this.controller.sync(values);
  }

  /** The size of the analysed frame (`showcase.state.width/height`). */
  setFrameSize(width: number, height: number): void {
    if (width > 0 && height > 0 && (width !== this.frameWidth || height !== this.frameHeight)) {
      this.frameWidth = width;
      this.frameHeight = height;
      this.renderRect();
    }
  }

  /** A reconnect: the next `status` is the truth, whatever was being edited. */
  setConnected(connected: boolean): void {
    if (connected !== this.connected) {
      this.connected = connected;
      this.controller.resync();
      this.render();
    }
  }

  private slider(
    key: SliderKey,
    label: string,
    range: SliderRange,
    format: (value: number) => string,
  ): HTMLElement {
    const row = el("label", "calibration-row");
    const name = el("span", "calibration-label", label);
    const output = el("output", "calibration-value");
    const input = el("input", "calibration-slider");
    input.type = "range";
    input.min = String(range.min);
    input.max = String(range.max);
    input.step = String(range.step);
    input.dataset.key = key;
    input.addEventListener("input", () => this.onSlider(key, Number(input.value)));
    row.append(name, output, input);
    this.sliders.set(key, { input, output, format });
    return row;
  }

  private exposureRow(): HTMLElement {
    const row = el("div", "calibration-row");
    const name = el("span", "calibration-label", "Esposizione");
    const segmented = el("div", "calibration-segmented");
    segmented.setAttribute("role", "group");
    segmented.setAttribute("aria-label", "Esposizione");
    for (const choice of EXPOSURES) {
      const button = el("button", "calibration-segment", `${choice.label} s`);
      button.type = "button";
      button.dataset.value = String(choice.value);
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => this.controller.change({ exposure: choice.value }));
      this.exposureButtons.push(button);
      segmented.append(button);
    }
    row.append(name, segmented);
    return row;
  }

  private onSlider(key: SliderKey, value: number): void {
    const current = this.controller.values;
    if (current === null) {
      return;
    }
    if (key === "centre_x") {
      this.controller.change({ zoom_center: [value, current.zoom_center[1]] });
    } else if (key === "centre_y") {
      this.controller.change({ zoom_center: [current.zoom_center[0], value] });
    } else if (key === "gain") {
      this.controller.change({ gain: sliderToGain(value) });
    } else {
      this.controller.change({ [key]: value });
    }
  }

  private render(): void {
    const values = this.controller.values;
    const ready = values !== null && this.connected;
    this.card.dataset.ready = String(ready);
    this.note.textContent = !this.connected
      ? "scheda non connessa"
      : values === null
        ? "in attesa della scheda…"
        : "salvata sulla scheda";
    for (const slider of this.sliders.values()) {
      slider.input.disabled = !ready;
    }
    for (const button of this.exposureButtons) {
      button.disabled = !ready;
      button.setAttribute("aria-pressed", String(values?.exposure === Number(button.dataset.value)));
    }
    this.resetButton.disabled = !this.connected;
    if (values !== null) {
      this.show("zoom", values.zoom);
      this.show("centre_x", values.zoom_center[0]);
      this.show("centre_y", values.zoom_center[1]);
      this.show("gain", values.gain);
      this.show("saturation", values.saturation);
    }
    this.renderRect();
  }

  private show(key: SliderKey, value: number): void {
    const slider = this.sliders.get(key);
    if (slider === undefined) {
      return;
    }
    const text = String(key === "gain" ? gainToSlider(value) : value);
    if (slider.input.value !== text) {
      slider.input.value = text;
    }
    slider.output.textContent = slider.format(value);
  }

  private renderRect(): void {
    const values = this.controller.values;
    const visible = this.open && values !== null;
    this.rect.hidden = !visible;
    if (!visible) {
      return;
    }
    const frameWidth = this.frame.clientWidth;
    const frameHeight = this.frame.clientHeight;
    const frameAspect = frameWidth > 0 && frameHeight > 0 ? frameWidth / frameHeight : 4 / 3;
    const box = containRect(
      analysisRect(values.zoom, values.zoom_center, this.frameWidth, this.frameHeight),
      this.frameWidth / this.frameHeight,
      frameAspect,
    );
    this.rect.style.left = `${(box.left * 100).toFixed(3)}%`;
    this.rect.style.top = `${(box.top * 100).toFixed(3)}%`;
    this.rect.style.width = `${(box.width * 100).toFixed(3)}%`;
    this.rect.style.height = `${(box.height * 100).toFixed(3)}%`;
  }
}

function group(title: string): HTMLFieldSetElement {
  const fieldset = el("fieldset", "calibration-group");
  fieldset.append(el("legend", "calibration-legend", title));
  return fieldset;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function icon(): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  for (const d of ICON_PATHS) {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}
