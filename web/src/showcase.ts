/**
 * "Cosa vede l'IA" view (S3): what the detector sees on the RGB camera, as one
 * booth-facing panel.
 *
 * The boxes and their Italian labels are drawn by the board on the MJPEG
 * stream, so this pane adds what the picture cannot say at a glance. In
 * presentation mode that is one chip per kind of object ("persona", with a
 * small "2×" when there are more); in debug mode the counts and the most
 * confident detections with their score bar. `CalmShowcase` debounces the 10 Hz feed into a calm view
 * (500 ms text, 5 % scores, stable order, 1 s leave hold), so the panel does not
 * flicker (coordinator feedback, 2026-11-12).
 */

import { CalmShowcase, type CalmObject, type CalmView } from "./calm";
import { clear, el } from "./dom";
import { SHOWCASE_LABELS, formatNumber } from "./labels";
import type { ShowcaseStatePayload } from "./protocol";

/** A booth reads a short list: the rest stays on the video overlay. */
const MAX_COUNTS = 6;
const MAX_OBJECTS = 6;

export interface ShowcaseElements {
  /** Presentation chips: names only, no scores. */
  chips: HTMLElement;
  countsGroup: HTMLElement;
  counts: HTMLElement;
  objectsGroup: HTMLElement;
  objects: HTMLElement;
  meta: HTMLElement;
}

interface ChipNode {
  chip: HTMLElement;
  badge: HTMLElement;
}

interface ObjectNode {
  row: HTMLElement;
  label: HTMLElement;
  score: HTMLElement;
  fill: HTMLElement;
}

export class ShowcaseView {
  private readonly elements: ShowcaseElements;
  private readonly calm = new CalmShowcase();
  private readonly objectNodes = new Map<string, ObjectNode>();
  private readonly chipNodes = new Map<string, ChipNode>();
  private more: HTMLElement | null = null;
  private rendered: CalmView | null = null;

  constructor(elements: ShowcaseElements) {
    this.elements = elements;
    this.render();
  }

  setState(state: ShowcaseStatePayload | null): void {
    if (state === null) {
      this.calm.reset();
      this.rendered = null;
      this.dropAllObjects();
      this.render();
      return;
    }
    const view = this.calm.update(state, Date.now());
    if (view !== null) {
      this.rendered = view;
      this.render();
    }
  }

  /** Keep the 1 s leave hold and the 500 ms refresh alive between frames. */
  tick(): void {
    const view = this.calm.tick(Date.now());
    if (view !== null) {
      this.rendered = view;
      this.render();
    }
  }

  private render(): void {
    // The invitation note is main.ts's alone (mode and model swap): toggling
    // it here too made it blink on every refresh.
    const { countsGroup, counts, objectsGroup, objects, meta } = this.elements;
    const view = this.rendered;
    const objectList = view?.objects ?? [];
    const countList = view?.counts ?? [];
    const visible = objectList.length > 0 || countList.length > 0;

    countsGroup.hidden = countList.length === 0;
    objectsGroup.hidden = objectList.length === 0;
    meta.hidden = !visible;

    if (!visible) {
      clear(counts);
      clear(objects);
      clear(this.elements.chips);
      this.chipNodes.clear();
      this.objectNodes.clear();
      this.more = null;
      meta.textContent = "";
      return;
    }

    renderCounts(counts, countList);
    this.renderChips(countList);
    this.renderObjects(objectList);
    meta.textContent =
      view?.meta === null || view?.meta === undefined
        ? ""
        : `${view.meta.model} · ${view.meta.width}×${view.meta.height}`;
  }

  /**
   * One chip per kind, keyed by label so a chip that stays does not blink:
   * only new kinds fade in. The count badge appears from two up.
   */
  private renderChips(counts: CalmView["counts"]): void {
    const container = this.elements.chips;
    const shown = counts.slice(0, MAX_COUNTS);
    const labels = new Set(shown.map((entry) => entry.label));
    for (const [label, node] of this.chipNodes) {
      if (!labels.has(label)) {
        node.chip.remove();
        this.chipNodes.delete(label);
      }
    }
    for (const entry of shown) {
      let node = this.chipNodes.get(entry.label);
      if (node === undefined) {
        const chip = el("li", "chip chip-entering");
        const badge = el("span", "chip-count");
        chip.append(badge, el("span", "chip-label", entry.label));
        node = { chip, badge };
        this.chipNodes.set(entry.label, node);
        window.setTimeout(() => chip.classList.remove("chip-entering"), 30);
      }
      const many = entry.count > 1;
      node.badge.hidden = !many;
      node.badge.textContent = many ? `${formatNumber(entry.count, 0)}×` : "";
      node.chip.setAttribute(
        "aria-label",
        many ? `${formatNumber(entry.count, 0)} ${entry.label}` : entry.label,
      );
      container.append(node.chip);
    }
  }

  private renderObjects(list: readonly CalmObject[]): void {
    const container = this.elements.objects;
    const shown = list.slice(0, MAX_OBJECTS);
    const keys = new Set(shown.map((object) => object.key));

    for (const [key, node] of this.objectNodes) {
      if (!keys.has(key)) {
        node.row.remove();
        this.objectNodes.delete(key);
      }
    }

    for (const object of shown) {
      let node = this.objectNodes.get(object.key);
      if (node === undefined) {
        node = createObjectNode(object);
        this.objectNodes.set(object.key, node);
        node.row.classList.add("showcase-object-entering");
        window.setTimeout(() => node?.row.classList.remove("showcase-object-entering"), 320);
      }
      node.row.classList.toggle("showcase-object-leaving", object.leaving);
      node.row.dataset.key = object.key;
      node.label.textContent = object.label;
      node.score.textContent = `${formatNumber(object.scorePercent, 0)}%`;
      const score = Math.min(1, Math.max(0, object.score));
      node.row.style.setProperty("--score", String(score));
      container.append(node.row);
    }
    if (list.length > MAX_OBJECTS) {
      if (this.more === null) {
        this.more = el("li", "showcase-more");
      }
      this.more.textContent = `e altri ${formatNumber(list.length - MAX_OBJECTS, 0)}`;
      container.append(this.more);
    } else if (this.more !== null) {
      this.more.remove();
      this.more = null;
    }
  }

  private dropAllObjects(): void {
    clear(this.elements.chips);
    this.chipNodes.clear();
    for (const node of this.objectNodes.values()) {
      node.row.remove();
    }
    this.objectNodes.clear();
    if (this.more !== null) {
      this.more.remove();
      this.more = null;
    }
  }
}

function createObjectNode(object: CalmObject): ObjectNode {
  const row = el("li", "showcase-object");
  row.dataset.key = object.key;
  const head = el("div", "showcase-object-head");
  const label = el("span", "showcase-object-label", object.label);
  const score = el("span", "showcase-object-score", `${formatNumber(object.scorePercent, 0)}%`);
  head.append(label, score);
  const bar = el("div", "showcase-bar");
  bar.setAttribute("aria-hidden", "true");
  const fill = el("span", "showcase-bar-fill");
  bar.append(fill);
  row.append(head, bar);
  return { row, label, score, fill };
}

function renderCounts(list: HTMLElement, counts: CalmView["counts"]): void {
  // The visible heading and the list's accessible name say the same thing.
  list.setAttribute("aria-label", SHOWCASE_LABELS.countsLabel);
  clear(list);
  for (const entry of counts.slice(0, MAX_COUNTS)) {
    const row = el("li", "showcase-count");
    row.append(
      el("span", "showcase-count-n", String(entry.count)),
      el("span", "showcase-count-x", "×"),
      el("span", "showcase-count-label", entry.label),
    );
    list.append(row);
  }
  appendOverflow(list, counts.length - MAX_COUNTS);
}

/** "... e altri 3": the entries that did not fit above keep a trace. */
function appendOverflow(list: HTMLElement, hidden: number): void {
  if (hidden <= 0) {
    return;
  }
  list.append(el("li", "showcase-more", `e altri ${formatNumber(hidden, 0)}`));
}
