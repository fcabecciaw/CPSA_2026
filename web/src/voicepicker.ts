/**
 * Voice picker of the dashboard (R2-V follow-up: «come confronto le voci?»).
 *
 * A compact control in the top bar lists every voice the local TTS service
 * offers, grouped by engine, so the user can cycle through them by ear without
 * touching a terminal:
 *
 * - `v` (or the "Prova" button) moves to the next voice, makes it this tab's
 *   voice and speaks the fixed test sentence;
 * - an explicit choice is written to localStorage *and* to the service
 *   (`PUT /default`), so a reload and every other page keep it;
 * - "Prepara frasi" asks the service to fill its cache with the fixed dashboard
 *   sentences and shows the progress of the background job.
 *
 * The state machine is DOM-free and tested in Node; only `mountVoicePicker`
 * touches the document. The click gate, mute, watchdog and single-tab rules of
 * `VoiceEngine` are untouched: the picker only chooses a voice and, on an
 * explicit click, designates this tab the speaking one.
 */

import {
  HttpTtsClient,
  TTS_URL,
  type PregenerateStatus,
  type StorageLike,
  type TimerLike,
  type TtsCatalog,
  type TtsCatalogEngine,
  type TtsDefaults,
  type TtsVoiceClient,
  type VoiceHandlers,
  type VoicePick,
  PICK_KEY,
} from "./voice";
import { el } from "./dom";
import "./voicepicker.css";

/** The sentence every "Prova" speaks: one line, no slots, easy to compare. */
export const TEST_SENTENCE =
  "Ciao! Sono la voce della dashboard: si sente bene così, o preferisci un'altra?";

/** How often a running fill is polled, in milliseconds. */
const POLL_MS = 1_500;

/** What the picker needs from the engine; `VoiceEngine` already satisfies it. */
export interface VoicePickerVoice {
  chooseVoice(engine: string, voice: string, label?: string): void;
  serviceInfo(): { engine: string; voice: string; label: string } | null;
  activate(): void;
  cancel(): void;
  speak(text: string, handlers?: VoiceHandlers): boolean;
}

/** One selectable voice, flattened with its engine for the dropdown. */
export interface VoiceEntry {
  engine: string;
  engineLabel: string;
  voice: string;
  name: string;
  /** "Microsoft Neural (edge-tts) · Elsa" */
  label: string;
}

export interface VoicePickerState {
  /** The catalog arrived: without it the picker stays hidden. */
  ready: boolean;
  error: string | null;
  engines: TtsCatalogEngine[];
  entries: VoiceEntry[];
  /** Index into `entries`, -1 when the service offers no voice. */
  index: number;
  current: VoiceEntry | null;
  /** The server-side default, after this tab changed it. */
  defaults: TtsDefaults | null;
  /** The last fill the service reported, running or finished. */
  job: PregenerateStatus | null;
  /** A one-line note for the user ("Preparo le frasi: 12/157"). */
  message: string | null;
}

export interface VoicePickerOptions {
  voice: VoicePickerVoice;
  /** Injected in the tests; the tab uses the HTTP client. */
  client?: TtsVoiceClient;
  serviceUrl?: string;
  storage?: StorageLike | null;
  timer?: TimerLike | null;
  /** Poll period while a fill runs; 0 disables the timer (manual `poll`). */
  pollMs?: number;
  onState?: (state: VoicePickerState) => void;
}

/** Every voice of every engine, in the order the service lists them. */
export function flattenCatalog(catalog: TtsCatalog): VoiceEntry[] {
  const entries: VoiceEntry[] = [];
  for (const engine of catalog.engines) {
    for (const voice of engine.voices) {
      entries.push({
        engine: engine.id,
        engineLabel: engine.label,
        voice: voice.id,
        name: voice.name,
        label: `${engine.label} · ${voice.name}`,
      });
    }
  }
  return entries;
}

function defaultStorage(): StorageLike | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function defaultTimer(): TimerLike {
  return {
    setTimeout: (callback, ms) => globalThis.setTimeout(callback, ms) as unknown as number,
    clearTimeout: (handle) =>
      globalThis.clearTimeout(handle as unknown as ReturnType<typeof setTimeout>),
  };
}

/** The voice picker of one tab: catalog, selection, test sentence and fill. */
export class VoicePicker {
  private readonly voice: VoicePickerVoice;
  private readonly client: TtsVoiceClient;
  private readonly storage: StorageLike | null;
  private readonly timer: TimerLike;
  private readonly pollMs: number;
  private onState: ((state: VoicePickerState) => void) | null;

  private engines: TtsCatalogEngine[] = [];
  private entries: VoiceEntry[] = [];
  private index = -1;
  private defaults: TtsDefaults | null = null;
  private job: PregenerateStatus | null = null;
  private message: string | null = null;
  private error: string | null = null;
  private ready = false;
  private pollTimer: number | null = null;
  private disposed = false;

  constructor(options: VoicePickerOptions) {
    this.voice = options.voice;
    this.client = options.client ?? new HttpTtsClient(options.serviceUrl ?? TTS_URL);
    this.storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.timer = options.timer ?? defaultTimer();
    this.pollMs = options.pollMs ?? POLL_MS;
    this.onState = options.onState ?? null;
  }

  /** Subscribe to state changes; the mount renders every emission. */
  setOnState(listener: ((state: VoicePickerState) => void) | null): void {
    this.onState = listener;
    listener?.(this.state());
  }

  state(): VoicePickerState {
    return {
      ready: this.ready,
      error: this.error,
      engines: this.engines,
      entries: this.entries,
      index: this.index,
      current: this.index >= 0 ? (this.entries[this.index] ?? null) : null,
      defaults: this.defaults,
      job: this.job,
      message: this.message,
    };
  }

  get current(): VoiceEntry | null {
    return this.index >= 0 ? (this.entries[this.index] ?? null) : null;
  }

  /** Read the catalog and adopt the voice saved for this browser profile. */
  async load(): Promise<VoicePickerState> {
    const catalog = await this.client.catalog();
    if (catalog === null) {
      this.error = "Servizio voce non raggiungibile";
      this.emit();
      return this.state();
    }
    this.error = null;
    this.ready = true;
    this.engines = catalog.engines;
    this.entries = flattenCatalog(catalog);
    this.defaults = catalog.defaults ?? null;
    const wanted = this.savedPick() ?? this.voice.serviceInfo() ?? catalog.defaults;
    this.index = this.indexOf(wanted?.engine ?? "", wanted?.voice ?? "");
    if (this.index < 0) {
      this.index = this.entries.length > 0 ? 0 : -1;
    }
    const current = this.current;
    if (current !== null) {
      // Re-apply the choice so a pick made before the service answered wins.
      this.voice.chooseVoice(current.engine, current.voice, current.label);
    }
    this.emit();
    return this.state();
  }

  /** Take one entry of the dropdown and make it the default everywhere. */
  select(engine: string, voice: string): VoiceEntry | null {
    const index = this.indexOf(engine, voice);
    if (index < 0) {
      return null;
    }
    this.index = index;
    this.apply(this.entries[index]);
    void this.saveDefault(this.entries[index]);
    return this.entries[index];
  }

  /** `v`: the next voice of the list, spoken immediately for comparison. */
  cycle(): VoiceEntry | null {
    if (this.entries.length === 0) {
      return null;
    }
    this.index = (this.index + 1) % this.entries.length;
    const entry = this.entries[this.index];
    this.apply(entry);
    void this.saveDefault(entry);
    this.speakTest();
    return entry;
  }

  /** Speak the fixed comparison sentence with the selected voice. */
  speakTest(): boolean {
    if (this.current === null) {
      return false;
    }
    // The click (or the `v` key) is the gesture: it designates this tab and
    // unlocks WebAudio, exactly like "Attiva la voce".
    this.voice.cancel();
    this.voice.activate();
    return this.voice.speak(TEST_SENTENCE, {});
  }

  /** `v` outside a text field: cycle and speak. Returns true when handled. */
  handleKey(event: {
    key: string;
    ctrlKey?: boolean;
    altKey?: boolean;
    metaKey?: boolean;
    target?: EventTarget | null;
  }): boolean {
    if (event.ctrlKey === true || event.altKey === true || event.metaKey === true) {
      return false;
    }
    if (isTyping(event.target)) {
      return false;
    }
    if (event.key.toLowerCase() !== "v") {
      return false;
    }
    if (this.entries.length === 0) {
      return false;
    }
    this.cycle();
    return true;
  }

  /** Ask the service to fill its cache for the selected voice, then follow it. */
  async pregenerate(): Promise<void> {
    const entry = this.current;
    if (entry === null) {
      return;
    }
    const started = await this.client.pregenerate(entry.engine, entry.voice);
    if (started === null) {
      this.message = "Preparazione non disponibile";
      this.emit();
      return;
    }
    this.job = started;
    this.message = describeJob(started);
    this.emit();
    if (started.running) {
      this.schedulePoll();
    }
  }

  /** Read the fill progress once (the timer calls this while a job runs). */
  async poll(): Promise<void> {
    const status = await this.client.pregenerateStatus();
    if (status === null) {
      this.message = "Preparazione non disponibile";
      this.emit();
      return;
    }
    this.job = status;
    this.message = describeJob(status);
    this.emit();
    if (status.running) {
      this.schedulePoll();
    }
  }

  /** Stop polling (the page is gone or the test finished). */
  dispose(): void {
    this.disposed = true;
    if (this.pollTimer !== null) {
      this.timer.clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private apply(entry: VoiceEntry): void {
    this.voice.chooseVoice(entry.engine, entry.voice, entry.label);
    this.message = null;
    this.emit();
  }

  private async saveDefault(entry: VoiceEntry): Promise<void> {
    const defaults = await this.client.setDefault(entry.engine, entry.voice);
    if (defaults === null) {
      this.message = "Scelta salvata in questa scheda (servizio non raggiungibile)";
    } else {
      this.defaults = defaults;
      this.message = `Predefinita: ${entry.label}`;
    }
    this.emit();
  }

  private schedulePoll(): void {
    if (this.pollMs <= 0 || this.pollTimer !== null || this.disposed) {
      return;
    }
    this.pollTimer = this.timer.setTimeout(() => {
      this.pollTimer = null;
      void this.poll();
    }, this.pollMs);
  }

  private indexOf(engine: string, voice: string): number {
    return this.entries.findIndex((entry) => entry.engine === engine && entry.voice === voice);
  }

  private savedPick(): VoicePick | null {
    try {
      const raw = this.storage?.getItem(PICK_KEY);
      if (raw === null || raw === undefined) {
        return null;
      }
      const parsed = JSON.parse(raw) as Partial<VoicePick>;
      if (typeof parsed.engine !== "string" || typeof parsed.voice !== "string") {
        return null;
      }
      return {
        engine: parsed.engine,
        voice: parsed.voice,
        label: typeof parsed.label === "string" ? parsed.label : undefined,
      };
    } catch {
      return null;
    }
  }

  private emit(): void {
    this.onState?.(this.state());
  }
}

function describeJob(status: PregenerateStatus): string {
  const who = status.voice ?? "";
  if (status.running) {
    return `Preparo le frasi (${status.engine ?? ""}/${who}): ${status.done}/${status.total}`;
  }
  if (status.error !== null) {
    return `Preparazione interrotta: ${status.error}`;
  }
  return `Frasi pronte: ${status.done}/${status.total} (${status.cached} già in cache)`;
}

function isTyping(target: EventTarget | null | undefined): boolean {
  if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable === true
  );
}

/**
 * Put the picker in the top bar and keep it in sync with `picker`.
 *
 * Returns the cleanup function: it removes the listeners and stops polling.
 */
export function mountVoicePicker(root: HTMLElement, picker: VoicePicker): () => void {
  root.classList.add("voice-picker");
  root.replaceChildren();

  const toggle = el("button", "voice-picker-toggle", "Voce");
  toggle.type = "button";
  toggle.setAttribute("aria-haspopup", "true");
  toggle.setAttribute("aria-expanded", "false");
  toggle.title = "Scegli la voce della dashboard (v per provarle in fila)";

  const panel = el("div", "voice-picker-panel");
  panel.hidden = true;

  const heading = el("p", "voice-picker-heading", "Voce della dashboard");

  const select = el("select", "voice-picker-select");
  select.setAttribute("aria-label", "Voce della dashboard");

  const actions = el("div", "voice-picker-actions");
  const test = el("button", "voice-picker-button", "Prova");
  test.type = "button";
  test.title = "Ascolta la frase di prova con questa voce (tasto v)";
  const fill = el("button", "voice-picker-button", "Prepara frasi");
  fill.type = "button";
  fill.title = "Genera in anticipo le frasi fisse con questa voce";
  actions.append(test, fill);

  const status = el("p", "voice-picker-status");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");

  panel.append(heading, select, actions, status);
  root.append(toggle, panel);

  let rendered = "";

  const render = (state: VoicePickerState): void => {
    root.hidden = !state.ready;
    const signature = state.entries.map((entry) => `${entry.engine}/${entry.voice}`).join("|");
    if (signature !== rendered) {
      rendered = signature;
      select.replaceChildren();
      for (const engine of state.engines) {
        const group = document.createElement("optgroup");
        group.label = engine.available ? engine.label : `${engine.label} (non disponibile)`;
        for (const voice of engine.voices) {
          const option = el("option", "", voice.name);
          option.value = `${engine.id}/${voice.id}`;
          group.append(option);
        }
        select.append(group);
      }
    }
    const current = state.current;
    select.hidden = state.entries.length === 0;
    if (current !== null) {
      select.value = `${current.engine}/${current.voice}`;
      toggle.textContent = current.name;
      toggle.title = `${current.label} — clicca per cambiare voce (v per provarle in fila)`;
    }
    test.disabled = current === null;
    fill.disabled = current === null || state.job?.running === true;
    fill.textContent = state.job?.running === true ? "Preparazione…" : "Prepara frasi";
    if (state.message !== null) {
      status.textContent = state.message;
    } else if (state.error !== null) {
      status.textContent = state.error;
    } else {
      status.textContent = "";
    }
  };

  const open = (hidden: boolean): void => {
    panel.hidden = hidden;
    toggle.setAttribute("aria-expanded", String(!hidden));
    if (!hidden) {
      select.focus();
    }
  };

  const onToggle = (): void => open(panel.hidden === false);
  const onChange = (): void => {
    const [engine = "", voice = ""] = select.value.split("/");
    picker.select(engine, voice);
  };
  const onTest = (): void => {
    picker.speakTest();
  };
  const onFill = (): void => {
    void picker.pregenerate();
  };
  const onKey = (event: KeyboardEvent): void => {
    if (event.key === "Escape" && !panel.hidden) {
      open(true);
      toggle.focus();
      return;
    }
    if (
      picker.handleKey({
        key: event.key,
        ctrlKey: event.ctrlKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
        target: event.target as EventTarget | null,
      })
    ) {
      event.preventDefault();
    }
  };
  const onDocumentClick = (event: MouseEvent): void => {
    if (panel.hidden) {
      return;
    }
    const target = event.target as Node | null;
    if (target !== null && !root.contains(target)) {
      open(true);
    }
  };

  toggle.addEventListener("click", onToggle);
  select.addEventListener("change", onChange);
  test.addEventListener("click", onTest);
  fill.addEventListener("click", onFill);
  document.addEventListener("keydown", onKey);
  document.addEventListener("click", onDocumentClick);

  picker.setOnState(render);
  void picker.load();

  return () => {
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("click", onDocumentClick);
    toggle.removeEventListener("click", onToggle);
    select.removeEventListener("change", onChange);
    test.removeEventListener("click", onTest);
    fill.removeEventListener("click", onFill);
    picker.setOnState(null);
    picker.dispose();
    root.replaceChildren();
  };
}

/**
 * One call from `main.ts`: build the picker, put it in the top bar and start
 * reading the catalog. The dashboard passes the `VoiceEngine` of the tab.
 */
export function attachVoicePicker(
  root: HTMLElement,
  voice: VoicePickerVoice,
  options: Omit<VoicePickerOptions, "voice"> = {},
): { picker: VoicePicker; detach: () => void } {
  const picker = new VoicePicker({ ...options, voice });
  return { picker, detach: mountVoicePicker(root, picker) };
}
