// @vitest-environment happy-dom
/** The attract screen: one action starts the demo, `i` brings the loop back. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SCENES } from "./scenes";
import { AttractView } from "./view";

function key(name: string, target: EventTarget = window): void {
  target.dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true }));
}

let root: HTMLElement;
let views: AttractView[] = [];

function mount(params = ""): {
  view: AttractView;
  onActivate: ReturnType<typeof vi.fn>;
  onReturn: ReturnType<typeof vi.fn>;
} {
  const onActivate = vi.fn();
  const onReturn = vi.fn();
  const view = new AttractView({
    root,
    live: () => ({ fps: 59.6, dpuMs: 4.7, labels: ["persona", "tazza"] }),
    params: new URLSearchParams(params),
    onActivate,
    onReturn,
  });
  views.push(view);
  return { view, onActivate, onReturn };
}

beforeEach(() => {
  vi.useFakeTimers();
  root = document.createElement("section");
  root.id = "attract";
  document.body.append(root);
});

afterEach(() => {
  for (const view of views) {
    view.stop();
  }
  views = [];
  document.body.innerHTML = "";
  vi.useRealTimers();
});

describe("AttractView", () => {
  it("opens on the first scene with every scene built", () => {
    const { view } = mount();
    expect(view.phase).toBe("loop");
    expect(root.hasAttribute("data-open")).toBe(true);
    const scenes = root.querySelectorAll(".scene");
    expect(scenes).toHaveLength(SCENES.length);
    expect(root.querySelectorAll(".scene[data-active]")).toHaveLength(1);
    expect(scenes[0].hasAttribute("data-active")).toBe(true);
    expect(root.querySelectorAll(".scene svg.diagram")).toHaveLength(SCENES.length);
  });

  it("starts the demo on Space, once, and the loop comes back on i", () => {
    const { view, onActivate, onReturn } = mount();
    key(" ");
    expect(view.phase).toBe("demo");
    expect(onActivate).toHaveBeenCalledTimes(1);
    expect(root.hasAttribute("data-open")).toBe(false);
    expect(root.hasAttribute("inert")).toBe(true);
    expect(root.querySelectorAll(".scene[data-active]")).toHaveLength(0);

    // Space during the demo does nothing to the loop.
    key(" ");
    expect(onActivate).toHaveBeenCalledTimes(1);

    key("i");
    expect(view.phase).toBe("loop");
    expect(onReturn).toHaveBeenCalledTimes(1);
    expect(root.hasAttribute("data-open")).toBe(true);
    expect(root.querySelector(".scene")?.hasAttribute("data-active")).toBe(true);
  });

  it("starts the demo from the button and from Enter", () => {
    const first = mount();
    root.querySelector<HTMLButtonElement>(".attract-start")?.click();
    expect(first.view.phase).toBe("demo");
    expect(first.onActivate).toHaveBeenCalledTimes(1);

    first.view.returnToLoop();
    key("Enter");
    expect(first.view.phase).toBe("demo");
    expect(first.onActivate).toHaveBeenCalledTimes(2);
  });

  it("ignores i while typing in a field", () => {
    const { view } = mount();
    view.activate();
    const input = document.createElement("input");
    document.body.append(input);
    key("i", input);
    expect(view.phase).toBe("demo");
  });

  it("moves through the scenes on its own and with the arrows", () => {
    const { view } = mount();
    const active = (): string | undefined =>
      root.querySelector<HTMLElement>(".scene[data-active]")?.dataset.scene;
    expect(active()).toBe(SCENES[0].id);
    vi.advanceTimersByTime(SCENES[0].durationMs);
    view.tick();
    expect(active()).toBe(SCENES[1].id);
    key("ArrowRight");
    expect(active()).toBe(SCENES[2].id);
    key("ArrowLeft");
    key("ArrowLeft");
    expect(active()).toBe(SCENES[0].id);
  });

  it("shows the live numbers of the board", () => {
    mount("scene=3");
    const live = root.querySelector<HTMLElement>(".scene[data-active] .scene-live");
    expect(live?.hidden).toBe(false);
    expect(live?.textContent).toContain("4,7 ms");
    expect(live?.textContent).toContain("60 al secondo");
  });

  it("skips the loop with ?attract=0", () => {
    const { view, onActivate } = mount("attract=0");
    expect(view.phase).toBe("demo");
    expect(root.hasAttribute("data-open")).toBe(false);
    // Starting in the demo is not an activation: the audio still needs a click.
    expect(onActivate).not.toHaveBeenCalled();
  });
});
