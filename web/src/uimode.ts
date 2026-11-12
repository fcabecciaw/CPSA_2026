/**
 * Presentation and debug modes of the dashboard.
 *
 * The open day sees the presentation mode: the video, ARI, its subtitles and
 * the names of the objects. The debug mode adds everything the operator needs
 * (scores, the "mente" strip, fps and latency, the voice picker, the camera
 * badge). A discreet button in the top bar and the `d` key switch between
 * them; the choice survives a reload through localStorage.
 *
 * The CSS does the hiding: `<html data-ui="presentation">` hides every
 * `.debug-only` element, `data-ui="debug"` hides every `.presentation-only`.
 */

import type { StorageLike } from "./voice";

export type UiMode = "presentation" | "debug";

export const UI_MODE_KEY = "cpsa.dashboard.ui.v1";

/** A browser without storage (private window, blocked site data) still works. */
export function defaultUiStorage(): StorageLike | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/**
 * The mode at load: `?debug=1` / `?debug=0` win for this load (screenshots,
 * a quick check), then the stored choice, then presentation.
 */
export function readUiMode(storage: StorageLike | null, params: URLSearchParams): UiMode {
  const forced = params.get("debug");
  if (forced === "1") {
    return "debug";
  }
  if (forced === "0") {
    return "presentation";
  }
  try {
    return storage?.getItem(UI_MODE_KEY) === "debug" ? "debug" : "presentation";
  } catch {
    return "presentation";
  }
}

export function writeUiMode(storage: StorageLike | null, mode: UiMode): void {
  try {
    storage?.setItem(UI_MODE_KEY, mode);
  } catch {
    // A full or blocked storage only forgets the choice at the next reload.
  }
}

/** The part of a `KeyboardEvent` the shortcut looks at. */
export interface ShortcutEvent {
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  repeat: boolean;
  target: EventTarget | null;
}

/** `d` alone, never while typing in a field or with a modifier held. */
export function isToggleShortcut(event: ShortcutEvent): boolean {
  return isBareKey(event, "d");
}

/**
 * One of `keys` pressed alone (letters match either case), not auto-repeated
 * and not while typing in a field.
 */
export function isBareKey(event: ShortcutEvent, ...keys: string[]): boolean {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) {
    return false;
  }
  const pressed = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (!keys.includes(pressed)) {
    return false;
  }
  return !isTypingTarget(event.target);
}

function isTypingTarget(target: EventTarget | null): boolean {
  const element = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (element === null || typeof element.tagName !== "string") {
    return false;
  }
  const tag = element.tagName.toUpperCase();
  return (
    tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || element.isContentEditable === true
  );
}

export interface UiModeOptions {
  /** Receives `data-ui`; usually `document.documentElement`. */
  root: HTMLElement;
  button: HTMLButtonElement;
  storage?: StorageLike | null;
  params?: URLSearchParams;
  onChange?: (mode: UiMode) => void;
}

/** Binds the toggle button and the `d` shortcut to `<html data-ui>`. */
export class UiModeController {
  private readonly root: HTMLElement;
  private readonly button: HTMLButtonElement;
  private readonly storage: StorageLike | null;
  private readonly onChange: ((mode: UiMode) => void) | null;
  private current: UiMode;

  constructor(options: UiModeOptions) {
    this.root = options.root;
    this.button = options.button;
    this.storage = options.storage === undefined ? defaultUiStorage() : options.storage;
    this.onChange = options.onChange ?? null;
    this.current = readUiMode(this.storage, options.params ?? new URLSearchParams());
    this.apply();
    this.button.addEventListener("click", () => this.toggle());
    window.addEventListener("keydown", (event) => {
      if (isToggleShortcut(event)) {
        event.preventDefault();
        this.toggle();
      }
    });
  }

  get mode(): UiMode {
    return this.current;
  }

  toggle(): void {
    this.set(this.current === "debug" ? "presentation" : "debug");
  }

  set(mode: UiMode): void {
    if (mode === this.current) {
      return;
    }
    this.current = mode;
    writeUiMode(this.storage, mode);
    this.apply();
  }

  private apply(): void {
    const debug = this.current === "debug";
    this.root.dataset.ui = this.current;
    this.button.setAttribute("aria-pressed", String(debug));
    const label = debug ? "Nascondi i dettagli tecnici (D)" : "Mostra i dettagli tecnici (D)";
    this.button.setAttribute("aria-label", label);
    this.button.title = label;
    this.onChange?.(this.current);
  }
}
