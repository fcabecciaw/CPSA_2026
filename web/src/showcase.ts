/**
 * "Cosa vede l'IA" view (S3): what the detector sees on the RGB camera, as one
 * booth-facing panel.
 *
 * The boxes and their Italian labels are drawn by the board on the MJPEG
 * stream, so this pane adds what the picture cannot say at a glance: how many
 * objects of each kind are in the frame, and the most confident detections with
 * their score bar.
 */

import { clear, el } from "./dom";
import { SHOWCASE_LABELS, formatNumber } from "./labels";
import type { ShowcaseCount, ShowcaseObject, ShowcaseStatePayload } from "./protocol";

/** A booth reads a short list: the rest stays on the video overlay. */
const MAX_COUNTS = 6;
const MAX_OBJECTS = 6;

export interface ShowcaseElements {
  countsGroup: HTMLElement;
  counts: HTMLElement;
  objectsGroup: HTMLElement;
  objects: HTMLElement;
  empty: HTMLElement;
  meta: HTMLElement;
}

export class ShowcaseView {
  private readonly elements: ShowcaseElements;
  private state: ShowcaseStatePayload | null = null;

  constructor(elements: ShowcaseElements) {
    this.elements = elements;
    this.render();
  }

  setState(state: ShowcaseStatePayload | null): void {
    this.state = state;
    this.render();
  }

  private render(): void {
    const { countsGroup, counts, objectsGroup, objects, empty, meta } = this.elements;
    const state = this.state;
    const detections = state?.objects ?? [];
    const counted = state === null ? [] : countsOf(state);
    const visible = detections.length > 0;

    countsGroup.hidden = !visible;
    objectsGroup.hidden = !visible;
    empty.hidden = visible;
    meta.hidden = !visible;

    if (state === null || !visible) {
      clear(counts);
      clear(objects);
      meta.textContent = "";
      return;
    }

    renderCounts(counts, counted);
    renderObjects(objects, detections);
    meta.textContent = `${state.model} · ${state.width}×${state.height}`;
  }
}

/**
 * The server groups the detections by label; a server that sent objects but no
 * counts still gets a correct panel, because the counts are cheap to derive.
 */
function countsOf(state: ShowcaseStatePayload): ShowcaseCount[] {
  if (state.counts.length > 0) {
    return state.counts;
  }
  const grouped = new Map<string, number>();
  for (const object of state.objects) {
    grouped.set(object.label, (grouped.get(object.label) ?? 0) + 1);
  }
  return [...grouped.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || compareLabels(a.label, b.label));
}

function renderCounts(list: HTMLElement, counts: ShowcaseCount[]): void {
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
  appendOverflow(list, counts.length - MAX_COUNTS, "showcase-more");
}

function renderObjects(list: HTMLElement, objects: ShowcaseObject[]): void {
  list.setAttribute("aria-label", SHOWCASE_LABELS.objectsLabel);
  clear(list);
  for (const object of objects.slice(0, MAX_OBJECTS)) {
    const score = clamp01(object.score);
    const row = el("li", "showcase-object");
    row.style.setProperty("--score", String(score));

    const head = el("div", "showcase-object-head");
    head.append(
      el("span", "showcase-object-label", object.label),
      el("span", "showcase-object-score", `${formatNumber(score * 100, 0)}%`),
    );

    const bar = el("div", "showcase-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.append(el("span", "showcase-bar-fill"));
    row.append(head, bar);
    list.append(row);
  }
  appendOverflow(list, objects.length - MAX_OBJECTS, "showcase-more");
}

/** "... e altri 3": the entries that did not fit above keep a trace. */
function appendOverflow(list: HTMLElement, hidden: number, className: string): void {
  if (hidden <= 0) {
    return;
  }
  list.append(el("li", className, `e altri ${formatNumber(hidden, 0)}`));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Deterministic label order, so equal counts do not shuffle between frames. */
function compareLabels(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
