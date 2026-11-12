/**
 * The "Come funziona" attract screen: a full-screen loop of scenes that runs
 * from page load until the operator starts the demo.
 *
 * One action starts the demo and unlocks the audio: a click on "Avvia la
 * demo", Space or Enter (`onActivate` runs inside that user gesture, which is
 * what the browser needs before it plays sound). `i`, or the return button of
 * debug mode, brings the loop back. ←/→ step through the scenes.
 *
 * Every scene stays in the DOM; only `data-active` moves, so the diagrams'
 * CSS animations restart when their scene comes back and stop while hidden.
 * The live camera is copied, small, into a canvas at about 20 fps.
 */

import "./attract.css";

import { el } from "../dom";
import { isBareKey } from "../uimode";
import { AttractMachine, type AttractPhase } from "./machine";
import { SCENES, type LiveNumbers, type SceneSpec } from "./scenes";

const LIVE_REFRESH_MS = 250;
const CAMERA_FRAME_MS = 50;

export interface AttractViewOptions {
  root: HTMLElement;
  /** The demo's camera `<img>`: the loop shows a small live copy. */
  video?: HTMLImageElement | null;
  live: () => LiveNumbers;
  params?: URLSearchParams;
  /** Runs inside the activating gesture: unlock the audio here. */
  onActivate?: () => void;
  onReturn?: () => void;
  /** The debug-mode button that brings the loop back. */
  returnButton?: HTMLButtonElement | null;
  scenes?: readonly SceneSpec[];
  now?: () => number;
}

interface SceneNodes {
  section: HTMLElement;
  live: HTMLElement | null;
}

export class AttractView {
  readonly machine: AttractMachine;
  private readonly root: HTMLElement;
  private readonly video: HTMLImageElement | null;
  private readonly live: () => LiveNumbers;
  private readonly onActivate: (() => void) | null;
  private readonly onReturn: (() => void) | null;
  private readonly scenes: readonly SceneSpec[];
  private readonly now: () => number;
  private readonly sceneNodes: SceneNodes[] = [];
  private readonly segments: HTMLElement[] = [];
  private readonly camera: HTMLCanvasElement;
  private readonly cameraNote: HTMLElement;
  private shownScene = -1;
  private timer: number | null = null;
  private frame: number | null = null;
  private lastCameraAt = 0;

  constructor(options: AttractViewOptions) {
    this.root = options.root;
    this.video = options.video ?? null;
    this.live = options.live;
    this.onActivate = options.onActivate ?? null;
    this.onReturn = options.onReturn ?? null;
    this.scenes = options.scenes ?? SCENES;
    this.now = options.now ?? (() => Date.now());
    const params = options.params ?? new URLSearchParams();
    const pinned = params.has("scene") ? Number(params.get("scene")) - 1 : null;
    this.machine = new AttractMachine({
      durations: this.scenes.map((scene) => scene.durationMs),
      now: this.now(),
      start: params.get("attract") === "0" ? "demo" : "loop",
      pinnedScene: pinned !== null && Number.isFinite(pinned) ? pinned : null,
    });

    const { camera, cameraNote } = this.build();
    this.camera = camera;
    this.cameraNote = cameraNote;

    window.addEventListener("keydown", (event) => this.onKey(event));
    options.returnButton?.addEventListener("click", () => this.returnToLoop());
    this.render();
  }

  get phase(): AttractPhase {
    return this.machine.phase;
  }

  /** Start the scene clock and the live refresh. */
  start(): void {
    if (this.timer !== null) {
      return;
    }
    this.timer = window.setInterval(() => this.tick(), LIVE_REFRESH_MS);
    this.tick();
  }

  stop(): void {
    if (this.timer !== null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
    this.stopCamera();
  }

  activate(): void {
    if (!this.machine.activate()) {
      return;
    }
    this.render();
    this.onActivate?.();
  }

  returnToLoop(): void {
    if (!this.machine.returnToLoop(this.now())) {
      return;
    }
    this.render();
    this.onReturn?.();
  }

  /** Advance the scene and refresh the live line of the scene on screen. */
  tick(): void {
    if (this.machine.tick(this.now())) {
      this.render();
    }
    this.renderLive();
  }

  private onKey(event: KeyboardEvent): void {
    if (this.machine.phase === "loop") {
      if (isBareKey(event, " ", "Enter")) {
        event.preventDefault();
        this.activate();
      } else if (isBareKey(event, "ArrowRight", "ArrowLeft")) {
        event.preventDefault();
        if (this.machine.step(event.key === "ArrowRight" ? 1 : -1, this.now())) {
          this.render();
        }
      }
      return;
    }
    if (isBareKey(event, "i")) {
      event.preventDefault();
      this.returnToLoop();
    }
  }

  private render(): void {
    const open = this.machine.phase === "loop";
    this.root.toggleAttribute("data-open", open);
    this.root.toggleAttribute("inert", !open);
    this.root.setAttribute("aria-hidden", String(!open));
    const scene = open ? this.machine.scene : -1;
    if (scene !== this.shownScene) {
      this.sceneNodes.forEach((nodes, index) => {
        nodes.section.toggleAttribute("data-active", index === scene);
      });
      this.segments.forEach((segment, index) => {
        segment.classList.toggle("seg-done", scene >= 0 && index < scene);
        segment.classList.remove("seg-active");
        if (index === scene) {
          // Restart the fill animation for the new scene.
          void segment.offsetWidth;
          segment.classList.add("seg-active");
        }
      });
      this.shownScene = scene;
      this.renderLive();
    }
    if (open) {
      this.startCamera();
    } else {
      this.stopCamera();
    }
  }

  private renderLive(): void {
    if (this.shownScene < 0) {
      return;
    }
    const spec = this.scenes[this.shownScene];
    const node = this.sceneNodes[this.shownScene]?.live ?? null;
    if (spec?.live === undefined || node === null) {
      return;
    }
    const text = spec.live(this.live()) ?? "";
    if (node.textContent !== text) {
      node.textContent = text;
    }
    node.hidden = text === "";
  }

  /* --- the live camera thumbnail ------------------------------------------ */

  private startCamera(): void {
    if (this.frame !== null || typeof requestAnimationFrame !== "function") {
      return;
    }
    const step = (time: number): void => {
      this.frame = requestAnimationFrame(step);
      if (time - this.lastCameraAt < CAMERA_FRAME_MS) {
        return;
      }
      this.lastCameraAt = time;
      this.drawCamera();
    };
    this.frame = requestAnimationFrame(step);
  }

  private stopCamera(): void {
    if (this.frame !== null && typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(this.frame);
    }
    this.frame = null;
  }

  private drawCamera(): void {
    const video = this.video;
    const ready = video !== null && !video.hidden && video.naturalWidth > 0;
    this.cameraNote.hidden = ready;
    if (!ready) {
      return;
    }
    const context = this.camera.getContext("2d");
    if (context === null) {
      return;
    }
    try {
      context.drawImage(video, 0, 0, this.camera.width, this.camera.height);
    } catch {
      // A broken frame of the MJPEG stream: keep the previous one.
    }
  }

  /* --- the DOM ------------------------------------------------------------ */

  private build(): { camera: HTMLCanvasElement; cameraNote: HTMLElement } {
    const root = this.root;
    root.replaceChildren();
    root.classList.add("attract");
    root.setAttribute("aria-label", "Come funziona la demo");

    const top = el("header", "attract-top");
    const brand = el("div", "attract-brand");
    brand.append(
      el("p", "attract-title", "Cosa vede l'IA"),
      el("p", "attract-tagline", "Come funziona, in sette passi"),
    );
    const progress = el("ol", "attract-progress");
    progress.setAttribute("aria-hidden", "true");
    for (const scene of this.scenes) {
      const segment = el("li", "seg");
      segment.style.setProperty("--seg-ms", `${scene.durationMs}ms`);
      segment.append(el("span", "seg-fill"));
      progress.append(segment);
      this.segments.push(segment);
    }
    top.append(brand, progress);

    const stage = el("div", "attract-scenes");
    for (const scene of this.scenes) {
      const section = el("section", "scene");
      section.dataset.scene = scene.id;
      const text = el("div", "scene-text");
      text.append(el("h2", "scene-headline", scene.headline));
      for (const line of scene.lines) {
        text.append(el("p", "scene-line", line));
      }
      let live: HTMLElement | null = null;
      if (scene.live !== undefined) {
        live = el("p", "scene-live");
        live.hidden = true;
        text.append(live);
      }
      const art = el("div", "scene-art");
      art.append(scene.diagram());
      section.append(text, art);
      stage.append(section);
      this.sceneNodes.push({ section, live });
    }

    const bottom = el("footer", "attract-bottom");
    const cta = el("div", "attract-cta");
    const start = el("button", "attract-start");
    start.type = "button";
    start.append(playIcon(), document.createTextNode("Avvia la demo"));
    start.addEventListener("click", () => this.activate());
    const hint = el("p", "attract-hint");
    hint.append(
      document.createTextNode("oppure premi "),
      el("kbd", "", "Spazio"),
      document.createTextNode(" o "),
      el("kbd", "", "Invio"),
    );
    cta.append(start, hint);

    const figure = el("figure", "attract-cam");
    const camera = el("canvas", "attract-cam-canvas");
    camera.width = 320;
    camera.height = 240;
    camera.setAttribute("aria-hidden", "true");
    const cameraNote = el("p", "attract-cam-note", "In attesa della telecamera…");
    const caption = el("figcaption", "attract-cam-caption");
    caption.append(el("span", "live-dot"), document.createTextNode("In diretta"));
    figure.append(camera, cameraNote, caption);
    bottom.append(cta, figure);

    root.append(top, stage, bottom);
    return { camera, cameraNote };
  }
}

function playIcon(): SVGElement {
  const NS = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(NS, "svg");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("class", "icon");
  icon.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", "M8 5.5v13l10.5-6.5z");
  icon.append(path);
  return icon;
}
