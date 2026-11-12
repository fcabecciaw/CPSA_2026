/**
 * Italian voice of the hologram (F-E3 `voice.ts`), Web Speech API.
 *
 * Chrome on Ubuntu 24.04 exposes its Italian voices asynchronously: prefer
 * "Google italiano", fall back to any `it-*` voice, and if there is none show a
 * muted badge instead of crashing (F-E4 #16). Speech only starts after the
 * "Attiva la voce" click (autoplay, #15); only the tab that enabled the voice
 * speaks (#18); a mute toggle keeps the face and the bubble alive (#19); a
 * watchdog cancels stuck utterances after 8 s (#17). All the browser pieces are
 * injected, so the engine is testable without Chrome.
 */

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface VoiceInfo {
  name: string;
  lang: string;
  localService?: boolean;
}

export interface UtteranceLike {
  text: string;
  lang: string;
  voice: VoiceInfo | null;
  rate: number;
  pitch: number;
  volume: number;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  onboundary: (() => void) | null;
}

export interface SpeechSynthLike {
  getVoices(): VoiceInfo[];
  speak(utterance: UtteranceLike): void;
  cancel(): void;
  addEventListener?(type: "voiceschanged", listener: () => void): void;
  removeEventListener?(type: "voiceschanged", listener: () => void): void;
}

export interface TimerLike {
  setTimeout(callback: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

export interface VoiceHandlers {
  onStart?: () => void;
  onBoundary?: () => void;
  onEnd?: (reason: "ended" | "watchdog" | "cancelled" | "unavailable") => void;
}

export type VoiceStatus =
  | "off"
  | "locked"
  | "ready"
  | "muted"
  | "no-voice"
  | "unsupported"
  | "speaking";

export interface VoiceSnapshot {
  status: VoiceStatus;
  label: string;
  enabled: boolean;
  unlocked: boolean;
  muted: boolean;
  voiceName: string | null;
  available: boolean;
}

export interface VoiceOptions {
  synth?: SpeechSynthLike | null;
  createUtterance?: (text: string) => UtteranceLike;
  storage?: StorageLike | null;
  timer?: TimerLike | null;
  /** `?voice=1` enables this tab without a click, still needs `unlock()`. */
  enabled?: boolean;
  watchdogMs?: number;
  preferredVoice?: string;
}

export const ENABLED_KEY = "cpsa.voice.enabled.v1";
export const MUTED_KEY = "cpsa.voice.muted.v1";
const WATCHDOG_MS = 8_000;

function defaultStorage(): StorageLike | null {
  try {
    // Per-tab on purpose: another tab must stay silent (F-E4 #18).
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null;
  }
}

function browserSynth(): SpeechSynthLike | null {
  try {
    const synth = window.speechSynthesis;
    return synth === undefined ? null : (synth as unknown as SpeechSynthLike);
  } catch {
    return null;
  }
}

function browserUtterance(text: string): UtteranceLike {
  const utterance = new SpeechSynthesisUtterance(text);
  return utterance as unknown as UtteranceLike;
}

function browserTimer(): TimerLike {
  return {
    setTimeout: (callback, ms) => window.setTimeout(callback, ms),
    clearTimeout: (handle) => window.clearTimeout(handle),
  };
}

/** One utterance at a time, gated by click, tab, voice and mute. */
export class VoiceEngine {
  private readonly synth: SpeechSynthLike | null;
  private readonly createUtterance: (text: string) => UtteranceLike;
  private readonly storage: StorageLike | null;
  private readonly timer: TimerLike;
  private readonly watchdogMs: number;
  private readonly preferredVoice: string;

  private voices: VoiceInfo[] = [];
  private selected: VoiceInfo | null = null;
  private enabled: boolean;
  private unlocked = false;
  private muted: boolean;
  private speaking = false;
  private simulating = false;
  private watchdog: number | null = null;
  private simulateTimer: number | null = null;
  private handlers: VoiceHandlers | null = null;

  constructor(options: VoiceOptions = {}) {
    this.synth = options.synth === undefined ? browserSynth() : options.synth;
    this.createUtterance = options.createUtterance ?? browserUtterance;
    this.storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.timer = options.timer ?? browserTimer();
    this.watchdogMs = options.watchdogMs ?? WATCHDOG_MS;
    this.preferredVoice = (options.preferredVoice ?? "Google italiano").toLowerCase();
    this.enabled =
      options.enabled === true || this.storage?.getItem(ENABLED_KEY) === "1";
    this.muted = this.storage?.getItem(MUTED_KEY) === "1";
    this.refreshVoices();
    if (this.synth !== null) {
      this.synth.addEventListener?.("voiceschanged", () => this.refreshVoices());
    }
  }

  /** The click gate: enables this tab and unlocks audio for this page. */
  activate(): void {
    this.enabled = true;
    this.unlocked = true;
    this.storage?.setItem(ENABLED_KEY, "1");
  }

  /** Allow audio after a click, without changing the tab's enabled state. */
  unlock(): void {
    this.unlocked = true;
  }

  /** `?voice=1`: this tab intends to speak, still waiting for a gesture. */
  enable(): void {
    this.enabled = true;
    this.storage?.setItem(ENABLED_KEY, "1");
  }

  disable(): void {
    this.enabled = false;
    this.storage?.setItem(ENABLED_KEY, "0");
    this.cancel();
  }

  toggleMute(): boolean {
    this.muted = !this.muted;
    this.storage?.setItem(MUTED_KEY, this.muted ? "1" : "0");
    if (this.muted) {
      this.cancel();
    }
    return this.muted;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    this.storage?.setItem(MUTED_KEY, muted ? "1" : "0");
    if (muted) {
      this.cancel();
    }
  }

  get isSpeaking(): boolean {
    return this.speaking;
  }

  get canSpeak(): boolean {
    return (
      this.enabled &&
      this.unlocked &&
      !this.muted &&
      this.selected !== null &&
      this.synth !== null
    );
  }

  get voiceName(): string | null {
    return this.selected?.name ?? null;
  }

  get available(): boolean {
    return this.synth !== null && this.selected !== null;
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  get isMuted(): boolean {
    return this.muted;
  }

  snapshot(): VoiceSnapshot {
    let status: VoiceStatus;
    if (this.speaking) {
      status = "speaking";
    } else if (!this.enabled) {
      status = "off";
    } else if (!this.unlocked) {
      status = "locked";
    } else if (this.muted) {
      status = "muted";
    } else if (this.synth === null) {
      status = "unsupported";
    } else if (this.selected === null) {
      status = "no-voice";
    } else {
      status = "ready";
    }
    return {
      status,
      label: labelFor(status, this.selected?.name ?? null),
      enabled: this.enabled,
      unlocked: this.unlocked,
      muted: this.muted,
      voiceName: this.selected?.name ?? null,
      available: this.available,
    };
  }

  /** Re-read the asynchronous voice list and pick the Italian voice. */
  refreshVoices(): void {
    if (this.synth === null) {
      this.voices = [];
      this.selected = null;
      return;
    }
    this.voices = this.synth.getVoices();
    const italian = this.voices.filter((voice) => voice.lang.toLowerCase().startsWith("it"));
    const preferred = italian.find((voice) =>
      voice.name.toLowerCase().includes(this.preferredVoice),
    );
    this.selected = preferred ?? italian[0] ?? null;
  }

  /**
   * Speak one sentence. Returns true when real speech started; a muted, locked
   * or voiceless tab still runs the callbacks for the face and the bubble.
   */
  speak(text: string, handlers: VoiceHandlers = {}): boolean {
    if (this.speaking) {
      return false;
    }
    // A simulated line (muted, locked, no voice) yields to a real one.
    if (this.simulating) {
      this.finish("cancelled");
    }
    this.handlers = handlers;
    if (!this.canSpeak) {
      this.simulate(text, handlers);
      return false;
    }
    this.speaking = true;

    const utterance = this.createUtterance(text);
    utterance.lang = "it-IT";
    utterance.voice = this.selected;
    utterance.rate = 0.98;
    utterance.pitch = 0.88;
    utterance.volume = 0.95;
    utterance.onstart = () => {
      handlers.onStart?.();
      this.armWatchdog();
    };
    utterance.onboundary = () => handlers.onBoundary?.();
    utterance.onend = () => this.finish("ended");
    utterance.onerror = () => this.finish("cancelled");
    try {
      this.synth?.speak(utterance);
    } catch {
      this.finish("unavailable");
      return false;
    }
    return true;
  }

  /** Cancel the utterance in flight (vision lost, mode switch, mute). */
  cancel(): void {
    if (!this.speaking && !this.simulating) {
      return;
    }
    if (this.speaking) {
      try {
        this.synth?.cancel();
      } catch {
        // The speech stack can be gone (tab closed): nothing to cancel.
      }
    }
    this.finish("cancelled");
  }

  private simulate(text: string, handlers: VoiceHandlers): void {
    const words = Math.max(1, text.trim().split(/\s+/).length);
    const duration = Math.min(this.watchdogMs, 600 + words * 320);
    this.simulating = true;
    handlers.onStart?.();
    this.simulateTimer = this.timer.setTimeout(() => {
      this.simulateTimer = null;
      this.finish("unavailable");
    }, duration);
  }

  private armWatchdog(): void {
    this.disarmWatchdog();
    this.watchdog = this.timer.setTimeout(() => {
      this.watchdog = null;
      try {
        this.synth?.cancel();
      } catch {
        // Ignore: the utterance is gone anyway.
      }
      this.finish("watchdog");
    }, this.watchdogMs);
  }

  private disarmWatchdog(): void {
    if (this.watchdog !== null) {
      this.timer.clearTimeout(this.watchdog);
      this.watchdog = null;
    }
  }

  private finish(reason: "ended" | "watchdog" | "cancelled" | "unavailable"): void {
    this.disarmWatchdog();
    if (this.simulateTimer !== null) {
      this.timer.clearTimeout(this.simulateTimer);
      this.simulateTimer = null;
    }
    if (!this.speaking && !this.simulating) {
      return;
    }
    this.speaking = false;
    this.simulating = false;
    const handlers = this.handlers;
    this.handlers = null;
    handlers?.onEnd?.(reason);
  }
}

function labelFor(status: VoiceStatus, voiceName: string | null): string {
  switch (status) {
    case "off":
      return "Voce spenta in questa scheda";
    case "locked":
      return "Attiva la voce";
    case "muted":
      return "Voce mutata";
    case "no-voice":
      return "Nessuna voce italiana";
    case "unsupported":
      return "Voce non supportata";
    case "speaking":
      return voiceName === null ? "Sto parlando" : `Sto parlando · ${voiceName}`;
    case "ready":
      return voiceName === null ? "Voce pronta" : `Voce pronta · ${voiceName}`;
  }
}
