// @vitest-environment happy-dom
/**
 * ARI's subtitles and the avatar canvas live in their own subtrees: the
 * objects panel refreshes all the time and must never rewrite, replace or move
 * them (user feedback on the redesign, 2026-11-12).
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { detection, person, showcase } from "./cognitive/testkit";
import { ShowcaseView } from "./showcase";
import { Subtitles } from "./subtitles";

const INDEX = resolve(__dirname, "..", "index.html");

/** The real markup of `index.html`, without the module script. */
function mountIndex(): void {
  const html = readFileSync(INDEX, "utf8");
  const body = /<body>([\s\S]*)<\/body>/.exec(html)?.[1] ?? "";
  document.body.innerHTML = body.replace(/<script[\s\S]*?<\/script>/g, "");
}

function byId(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (node === null) {
    throw new Error(`#${id} missing`);
  }
  return node;
}

function watch(node: Node): MutationObserver & { records: MutationRecord[] } {
  const records: MutationRecord[] = [];
  const observer = new MutationObserver((batch) => records.push(...batch)) as MutationObserver & {
    records: MutationRecord[];
  };
  observer.records = records;
  observer.observe(node, {
    subtree: true,
    childList: true,
    attributes: true,
    characterData: true,
  });
  return observer;
}

/** A frame whose objects change every time: kinds, counts and scores. */
function frame(index: number) {
  const objects = [person(40 + (index % 7) * 10, 80, 120, 360, 0.8 + (index % 5) * 0.03)];
  if (index % 2 === 0) {
    objects.push(person(400, 90));
  }
  if (index % 3 !== 0) {
    objects.push(detection("tazza", 41, [300, 300, 360, 380], 0.6 + (index % 4) * 0.05));
  }
  if (index % 5 === 0) {
    objects.push(detection("libro", 73, [200, 320, 280, 400], 0.7));
  }
  return showcase(objects);
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-11-13T09:00:00Z"));
  mountIndex();
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("subtitles and avatar are independent of the objects panel", () => {
  it("keep separate subtrees in index.html", () => {
    const panel = byId("showcase-panel");
    const subtitles = byId("subtitles");
    const avatar = byId("avatar");
    expect(byId("bubble").parentElement).toBe(subtitles);
    // Siblings, never nested: no writer of one subtree owns the other.
    expect(panel.contains(subtitles)).toBe(false);
    expect(subtitles.contains(panel)).toBe(false);
    expect(panel.contains(avatar)).toBe(false);
    expect(subtitles.parentElement).toBe(panel.parentElement);
    expect(byId("hologram").parentElement).toBe(panel.parentElement);
  });

  it("100 objects-panel updates never touch the subtitle or the avatar canvas", async () => {
    const bubble = byId("bubble");
    const subtitles = new Subtitles(bubble);
    const line = "Mostrami un oggetto e provo a indovinare cos'è.";
    subtitles.show(line);

    // AvatarView's canvas, without WebGL: identity is what matters here.
    const canvas = document.createElement("canvas");
    canvas.className = "avatar-canvas";
    byId("avatar").append(canvas);

    const view = new ShowcaseView({
      chips: byId("showcase-chips"),
      countsGroup: byId("showcase-counts-group"),
      counts: byId("showcase-counts"),
      objectsGroup: byId("showcase-objects-group"),
      objects: byId("showcase-objects"),
      meta: byId("showcase-meta"),
    });

    const subtitleWatch = watch(byId("subtitles"));
    const avatarWatch = watch(byId("hologram"));
    const panelWatch = watch(byId("showcase-panel"));

    for (let index = 0; index < 100; index += 1) {
      // The calm filter refreshes every 500 ms: every frame becomes a render.
      vi.advanceTimersByTime(600);
      view.setState(frame(index));
      view.tick();
    }
    vi.advanceTimersByTime(1_000);
    await Promise.resolve();
    const subtitleRecords = [...subtitleWatch.records, ...subtitleWatch.takeRecords()];
    const avatarRecords = [...avatarWatch.records, ...avatarWatch.takeRecords()];
    const panelRecords = [...panelWatch.records, ...panelWatch.takeRecords()];

    // The panel really did change, many times.
    expect(panelRecords.length).toBeGreaterThan(100);
    expect(byId("showcase-chips").textContent).toContain("persona");

    // The subtitle: same node, same parent, same words, no mutation at all.
    expect(subtitleRecords).toEqual([]);
    expect(document.getElementById("bubble")).toBe(bubble);
    expect(bubble.parentElement).toBe(byId("subtitles"));
    expect(bubble.textContent).toBe(line);
    expect(bubble.classList.contains("bubble-empty")).toBe(false);
    expect(subtitles.text).toBe(line);

    // The avatar canvas: same node, nothing in its subtree touched.
    expect(avatarRecords).toEqual([]);
    expect(byId("avatar").firstElementChild).toBe(canvas);
  });
});

describe("Subtitles", () => {
  it("writes the node only when the spoken line changes", () => {
    const bubble = byId("bubble");
    const subtitles = new Subtitles(bubble);
    const observer = watch(bubble);

    subtitles.show("Ciao! Che piacere vederti.");
    const first = observer.takeRecords().length;
    expect(first).toBeGreaterThan(0);
    subtitles.show("Ciao! Che piacere vederti.");
    subtitles.show("Ciao! Che piacere vederti.");
    expect(observer.takeRecords()).toEqual([]);
    expect(bubble.dataset.length).toBe("short");

    subtitles.show(
      "Negli anni Cinquanta e Sessanta Hubel e Wiesel scoprirono neuroni che reagiscono a linee con un certo orientamento.",
    );
    expect(bubble.dataset.length).toBe("long");
    expect(bubble.textContent).toContain("Hubel");
  });

  it("fades out on clear and keeps the words for the fade", () => {
    const bubble = byId("bubble");
    const subtitles = new Subtitles(bubble);
    subtitles.show("Ciao, a presto!");
    subtitles.clear();
    expect(bubble.classList.contains("bubble-empty")).toBe(true);
    expect(bubble.textContent).toBe("Ciao, a presto!");
    expect(subtitles.text).toBeNull();

    const observer = watch(bubble);
    subtitles.clear();
    expect(observer.takeRecords()).toEqual([]);

    // The same line said again after a clear shows it again.
    subtitles.show("Ciao, a presto!");
    expect(bubble.classList.contains("bubble-empty")).toBe(false);
  });

  it("treats an empty line as a clear", () => {
    const bubble = byId("bubble");
    const subtitles = new Subtitles(bubble);
    subtitles.show("Ciao!");
    subtitles.show("");
    expect(bubble.classList.contains("bubble-empty")).toBe(true);
  });
});
