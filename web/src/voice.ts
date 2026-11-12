/**
 * Italian voice of the hologram (F-E3 `voice.ts`).
 *
 * Two back ends, chosen at run time:
 * - the local TTS service (`tools/tts`, http://127.0.0.1:8765): the sentence is
 *   fetched as a WAV and played through a WebAudio graph whose `AnalyserNode` is
 *   exposed by `voiceAudioNode()` so the mouth can follow the real amplitude;
 * - the Web Speech API when the service does not answer, keeping Chrome's
 *   asynchronous Italian voices (prefer "Google italiano", fall back to any
 *   `it-*` voice, muted badge when there is none, F-E4 #16).
 *
 * The rules do not change with the back end: speech starts only after the
 * "Attiva la voce" click (#15), only the enabled tab speaks (#18), the mute
 * toggle keeps face and bubble alive (#19), a watchdog cancels after 8 s (#17)
 * and everything is injected so the engine is testable without Chrome.
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
  /** "Kokoro 82M · Sara" when the local service answered, else null. */
  service: string | null;
}

/** What one local TTS sentence is: resolved engine, voice and a readable label. */
export interface TtsServiceInfo {
  url: string;
  engine: string;
  voice: string;
  label: string;
}

/** One selectable voice of the local service (``GET /voices``). */
export interface TtsCatalogVoice {
  id: string;
  name: string;
  lang: string;
  gender: string;
  sample_rate?: number;
}

/** One engine and its voices, as ``GET /voices`` reports them. */
export interface TtsCatalogEngine {
  id: string;
  label: string;
  licence: string;
  device: string;
  available: boolean;
  loaded: boolean;
  error: string | null;
  voices: TtsCatalogVoice[];
}

/** The whole ``GET /voices`` answer: catalog plus the server-side default. */
export interface TtsCatalog {
  defaults: TtsDefaults;
  engines: TtsCatalogEngine[];
}

/** The engine/voice used when a ``POST /tts`` omits them. */
export interface TtsDefaults {
  engine: string;
  voice: string;
}

/** Progress of the background fill behind ``/pregenerate``. */
export interface PregenerateStatus {
  running: boolean;
  engine: string | null;
  voice: string | null;
  total: number;
  done: number;
  synthesized: number;
  cached: number;
  failed: number;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  elapsed_s: number | null;
}

/** The local HTTP service (no DOM: the default client is injected in tests). */
export interface TtsServiceLike {
  probe(): Promise<TtsServiceInfo | null>;
  synthesize(
    text: string,
    engine: string,
    voice: string,
    signal?: AbortSignal,
  ): Promise<ArrayBuffer>;
}

/** What the voice picker needs from the service (catalog, defaults, fill). */
export interface TtsVoiceClient {
  catalog(): Promise<TtsCatalog | null>;
  setDefault(engine: string, voice: string): Promise<TtsDefaults | null>;
  pregenerate(engine: string, voice: string): Promise<PregenerateStatus | null>;
  pregenerateStatus(): Promise<PregenerateStatus | null>;
}

/** The playback graph; `play` resolves true when the audio actually started. */
export interface AudioSinkLike {
  play(
    data: ArrayBuffer,
    handlers: { onStart: () => void; onEnd: () => void },
  ): Promise<boolean>;
  stop(): void;
  /** Create or resume the audio context from a user gesture, when supported. */
  unlock?(): void;
  /** The analyser feeding the mouth, or null before the first gesture. */
  analyserNode(): AnalyserNode | null;
}

export interface VoiceOptions {
  synth?: SpeechSynthLike | null;
  createUtterance?: (text: string) => UtteranceLike;
  storage?: StorageLike | null;
  /** Where the voice picked from the dashboard lives (localStorage). */
  pickStorage?: StorageLike | null;
  timer?: TimerLike | null;
  /** `?voice=1` enables this tab without a click, still needs `unlock()`. */
  enabled?: boolean;
  watchdogMs?: number;
  preferredVoice?: string;
  /** Local TTS service; `null` forces the Web Speech path. */
  tts?: TtsServiceLike | null;
  /** WebAudio playback; `null` forces the Web Speech path. */
  audio?: AudioSinkLike | null;
  serviceUrl?: string;
  /** Called when the local-service probe changes state (badge refresh). */
  onStatusChange?: (snapshot: VoiceSnapshot) => void;
}

export const ENABLED_KEY = "cpsa.voice.enabled.v1";
export const MUTED_KEY = "cpsa.voice.muted.v1";
/** The engine/voice chosen with the picker, per browser profile. */
export const PICK_KEY = "cpsa.voice.pick.v1";
export const TTS_URL = "http://127.0.0.1:8765";
const WATCHDOG_MS = 8_000;
const PROBE_TIMEOUT_MS = 750;

/** One voice chosen from the picker, as written to `PICK_KEY`. */
export interface VoicePick {
  engine: string;
  voice: string;
  label?: string;
}

function defaultStorage(): StorageLike | null {
  try {
    // Per-tab on purpose: another tab must stay silent (F-E4 #18).
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null;
  }
}

function defaultPickStorage(): StorageLike | null {
  try {
    // Shared by the profile: reopening the dashboard keeps the chosen voice.
    return typeof localStorage === "undefined" ? null : localStorage;
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

/** The shared WebAudio graph: `voiceAudioNode()` reads its analyser. */
let sharedSink: WebAudioSink | null = null;

function defaultAudioSink(): AudioSinkLike | null {
  if (typeof window === "undefined" || typeof AudioContext === "undefined") {
    return null;
  }
  sharedSink ??= new WebAudioSink();
  return sharedSink;
}

/**
 * The analyser of the sentence being played locally, or null when the voice is
 * silent, muted, or using the Web Speech API. The face reads its amplitude.
 */
export function voiceAudioNode(): AnalyserNode | null {
  return sharedSink?.analyserNode() ?? null;
}

/** `http://127.0.0.1:8765` client: `/health`, `/voices`, `/tts`, `/default`. */
export class HttpTtsClient implements TtsServiceLike, TtsVoiceClient {
  private readonly base: string;

  constructor(base: string = TTS_URL) {
    this.base = base.replace(/\/$/, "");
  }

  async probe(): Promise<TtsServiceInfo | null> {
    const timeout =
      typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function"
        ? AbortSignal.timeout(PROBE_TIMEOUT_MS)
        : undefined;
    try {
      const health = await fetch(`${this.base}/health`, { signal: timeout, cache: "no-store" });
      if (!health.ok) {
        return null;
      }
      const body = (await health.json()) as {
        status?: string;
        defaults?: { engine?: string; voice?: string };
      };
      if (body.status !== "ok") {
        return null;
      }
      const engine = body.defaults?.engine ?? "";
      const voice = body.defaults?.voice ?? "";
      const label = await this.voiceLabel(engine, voice, timeout);
      return { url: this.base, engine, voice, label };
    } catch {
      return null;
    }
  }

  async synthesize(
    text: string,
    engine: string,
    voice: string,
    signal?: AbortSignal,
  ): Promise<ArrayBuffer> {
    const response = await fetch(`${this.base}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, engine, voice }),
      signal,
    });
    if (!response.ok) {
      throw new Error(`tts ${response.status}`);
    }
    return response.arrayBuffer();
  }

  /** Every engine and voice of the service, for the picker. */
  async catalog(): Promise<TtsCatalog | null> {
    try {
      const response = await fetch(`${this.base}/voices`, { cache: "no-store" });
      if (!response.ok) {
        return null;
      }
      const body = (await response.json()) as TtsCatalog;
      return Array.isArray(body.engines) ? body : null;
    } catch {
      return null;
    }
  }

  /** Say which voice every request that omits one should use from now on. */
  async setDefault(engine: string, voice: string): Promise<TtsDefaults | null> {
    try {
      const response = await fetch(`${this.base}/default`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine, voice }),
      });
      if (!response.ok) {
        return null;
      }
      const body = (await response.json()) as { defaults?: TtsDefaults };
      return body.defaults ?? null;
    } catch {
      return null;
    }
  }

  /** Start filling the cache with the fixed sentences, or report the running job. */
  async pregenerate(engine: string, voice: string): Promise<PregenerateStatus | null> {
    try {
      const response = await fetch(`${this.base}/pregenerate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine, voice }),
      });
      if (response.status === 409) {
        const busy = (await response.json()) as { job?: PregenerateStatus };
        return busy.job ?? null;
      }
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as PregenerateStatus;
    } catch {
      return null;
    }
  }

  async pregenerateStatus(): Promise<PregenerateStatus | null> {
    try {
      const response = await fetch(`${this.base}/pregenerate`, { cache: "no-store" });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as PregenerateStatus;
    } catch {
      return null;
    }
  }

  private async voiceLabel(
    engine: string,
    voice: string,
    signal?: AbortSignal,
  ): Promise<string> {
    try {
      const response = await fetch(`${this.base}/voices`, { cache: "no-store", signal });
      if (!response.ok) {
        return "Voce locale";
      }
      const body = (await response.json()) as {
        engines?: {
          id?: string;
          label?: string;
          voices?: { id?: string; name?: string }[];
        }[];
      };
      const row = body.engines?.find((item) => item.id === engine);
      const name = row?.voices?.find((item) => item.id === voice)?.name;
      const parts = [row?.label, name].filter((part): part is string => Boolean(part));
      return parts.length > 0 ? parts.join(" · ") : "Voce locale";
    } catch {
      return "Voce locale";
    }
  }
}

/** WebAudio playback: source → analyser → gain → speakers. */
export class WebAudioSink implements AudioSinkLike {
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private gain: GainNode | null = null;
  private source: AudioBufferSourceNode | null = null;
  private readonly volume: number;

  constructor(volume = 0.95) {
    this.volume = volume;
  }

  private ensure(): AudioContext {
    if (this.context === null) {
      this.context = new AudioContext();
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.7;
      this.gain = this.context.createGain();
      this.gain.gain.value = this.volume;
      this.analyser.connect(this.gain);
      this.gain.connect(this.context.destination);
    }
    return this.context;
  }

  /** Create or resume the context; call it from a user gesture. */
  unlock(): void {
    try {
      const context = this.ensure();
      if (context.state === "suspended") {
        void context.resume().catch(() => undefined);
      }
    } catch {
      // No audio device: speak() falls back to Web Speech.
    }
  }

  analyserNode(): AnalyserNode | null {
    return this.analyser;
  }

  async play(
    data: ArrayBuffer,
    handlers: { onStart: () => void; onEnd: () => void },
  ): Promise<boolean> {
    let context: AudioContext;
    try {
      context = this.ensure();
      this.unlock();
    } catch {
      return false;
    }
    let buffer: AudioBuffer;
    try {
      // decodeAudioData detaches its argument: keep a copy for the caller.
      buffer = await context.decodeAudioData(data.slice(0));
    } catch {
      return false;
    }
    if (this.analyser === null) {
      return false;
    }
    this.stop();
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser);
    source.onended = () => {
      source.onended = null;
      if (this.source === source) {
        this.source = null;
      }
      handlers.onEnd();
    };
    try {
      source.start();
    } catch {
      return false;
    }
    this.source = source;
    handlers.onStart();
    return true;
  }

  stop(): void {
    const source = this.source;
    this.source = null;
    if (source !== null) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // Already stopped.
      }
      source.disconnect();
    }
  }
}

/** One utterance at a time, gated by click, tab, voice and mute. */
export class VoiceEngine {
  private readonly synth: SpeechSynthLike | null;
  private readonly createUtterance: (text: string) => UtteranceLike;
  private readonly storage: StorageLike | null;
  private readonly pickStorage: StorageLike | null;
  private readonly timer: TimerLike;
  private readonly watchdogMs: number;
  private readonly preferredVoice: string;
  private readonly tts: TtsServiceLike | null;
  private readonly audio: AudioSinkLike | null;
  private readonly serviceUrl: string;
  private readonly onStatusChange: ((snapshot: VoiceSnapshot) => void) | null;

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
  private service: TtsServiceInfo | null = null;
  private abort: AbortController | null = null;

  constructor(options: VoiceOptions = {}) {
    this.synth = options.synth === undefined ? browserSynth() : options.synth;
    this.createUtterance = options.createUtterance ?? browserUtterance;
    this.storage = options.storage === undefined ? defaultStorage() : options.storage;
    this.pickStorage =
      options.pickStorage === undefined ? defaultPickStorage() : options.pickStorage;
    this.timer = options.timer ?? browserTimer();
    this.watchdogMs = options.watchdogMs ?? WATCHDOG_MS;
    this.preferredVoice = (options.preferredVoice ?? "Google italiano").toLowerCase();
    this.serviceUrl = options.serviceUrl ?? TTS_URL;
    this.onStatusChange = options.onStatusChange ?? null;
    this.tts =
      options.tts === undefined
        ? typeof window === "undefined"
          ? null
          : new HttpTtsClient(this.serviceUrl)
        : options.tts;
    this.audio = options.audio === undefined ? defaultAudioSink() : options.audio;
    this.enabled =
      options.enabled === true || this.storage?.getItem(ENABLED_KEY) === "1";
    this.muted = this.storage?.getItem(MUTED_KEY) === "1";
    this.refreshVoices();
    if (this.synth !== null) {
      this.synth.addEventListener?.("voiceschanged", () => this.refreshVoices());
    }
    if (this.tts !== null) {
      void this.probeService();
    }
  }

  /** The click gate: enables this tab and unlocks audio for this page. */
  activate(): void {
    this.enabled = true;
    this.unlocked = true;
    this.storage?.setItem(ENABLED_KEY, "1");
    this.audio?.unlock?.();
    if (this.service === null) {
      void this.probeService();
    }
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
    if (!this.enabled || !this.unlocked || this.muted) {
      return false;
    }
    if (this.service !== null && this.audio !== null) {
      return true;
    }
    return this.selected !== null && this.synth !== null;
  }

  get voiceName(): string | null {
    return this.selected?.name ?? null;
  }

  get available(): boolean {
    return (this.synth !== null && this.selected !== null) || this.service !== null;
  }

  get isEnabled(): boolean {
    return this.enabled;
  }

  get isMuted(): boolean {
    return this.muted;
  }

  /** The label of the local service ("Kokoro 82M · Sara"), null if absent. */
  get serviceLabel(): string | null {
    return this.service === null ? null : this.service.label;
  }

  /** The engine/voice this tab would use for the next local sentence. */
  serviceInfo(): TtsServiceInfo | null {
    return this.service;
  }

  /**
   * Use another voice of the local service for this tab and remember it.
   *
   * The picker calls this; the choice survives a reload through localStorage
   * and is re-applied after every probe.
   */
  chooseVoice(engine: string, voice: string, label?: string): void {
    if (!engine || !voice) {
      return;
    }
    const resolved = label ?? `${engine} · ${voice}`;
    this.service = {
      url: this.service?.url ?? this.serviceUrl,
      engine,
      voice,
      label: resolved,
    };
    const pick: VoicePick = { engine, voice, label: resolved };
    try {
      this.pickStorage?.setItem(PICK_KEY, JSON.stringify(pick));
    } catch {
      // A full or blocked storage only loses the preference, not the voice.
    }
    if (this.speaking) {
      this.cancel();
    }
    this.onStatusChange?.(this.snapshot());
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
    } else if (this.service === null && this.synth === null) {
      status = "unsupported";
    } else if (this.service === null && this.selected === null) {
      status = "no-voice";
    } else {
      status = "ready";
    }
    return {
      status,
      label: this.labelFor(status),
      enabled: this.enabled,
      unlocked: this.unlocked,
      muted: this.muted,
      voiceName: this.selected?.name ?? null,
      available: this.available,
      service: this.serviceLabel,
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

  /** Ask the local service once; safe to call repeatedly. */
  async probeService(): Promise<boolean> {
    if (this.tts === null) {
      this.service = null;
      return false;
    }
    const info = await this.tts.probe();
    // A voice chosen while the service was down survives the probe.
    const pick = this.savedPick();
    this.service =
      info === null
        ? null
        : pick === null
          ? info
          : {
              ...info,
              engine: pick.engine,
              voice: pick.voice,
              label: pick.label ?? info.label,
            };
    this.onStatusChange?.(this.snapshot());
    return info !== null;
  }

  /** The voice chosen by the picker in this browser profile, if any. */
  private savedPick(): VoicePick | null {
    try {
      const raw = this.pickStorage?.getItem(PICK_KEY);
      if (raw === null || raw === undefined) {
        return null;
      }
      const parsed = JSON.parse(raw) as Partial<VoicePick>;
      if (typeof parsed.engine !== "string" || typeof parsed.voice !== "string") {
        return null;
      }
      if (!parsed.engine || !parsed.voice) {
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
    if (this.service !== null && this.audio !== null) {
      void this.speakLocal(text, handlers);
      return true;
    }
    if (this.synth === null || this.selected === null) {
      this.speaking = false;
      this.simulate(text, handlers);
      return false;
    }
    return this.startSynth(text, handlers);
  }

  /** Cancel the utterance in flight (vision lost, mode switch, mute). */
  cancel(): void {
    if (!this.speaking && !this.simulating) {
      return;
    }
    if (this.abort !== null) {
      this.abort.abort();
      this.abort = null;
    }
    this.audio?.stop();
    if (this.speaking) {
      try {
        this.synth?.cancel();
      } catch {
        // The speech stack can be gone (tab closed): nothing to cancel.
      }
    }
    this.finish("cancelled");
  }

  /** Fetch the WAV from the local service and play it; fall back on failure. */
  private async speakLocal(text: string, handlers: VoiceHandlers): Promise<void> {
    const info = this.service;
    if (info === null || this.tts === null || this.audio === null) {
      this.retryWithSynth(text, handlers);
      return;
    }
    const controller = new AbortController();
    this.abort = controller;
    try {
      const wav = await this.tts.synthesize(text, info.engine, info.voice, controller.signal);
      if (!this.speaking || this.abort !== controller) {
        return;
      }
      const started = await this.audio.play(wav, {
        onStart: () => {
          handlers.onStart?.();
          this.armWatchdog();
        },
        onEnd: () => this.finish("ended"),
      });
      if (!started) {
        this.retryWithSynth(text, handlers);
      }
    } catch {
      if (!this.speaking || this.abort !== controller) {
        return;
      }
      // The service died between the probe and the request: Web Speech now.
      this.retryWithSynth(text, handlers);
    } finally {
      if (this.abort === controller) {
        this.abort = null;
      }
    }
  }

  /** The local path failed mid-sentence: speak this one with Web Speech. */
  private retryWithSynth(text: string, handlers: VoiceHandlers): void {
    if (this.synth === null || this.selected === null) {
      this.speaking = false;
      this.simulate(text, handlers);
      return;
    }
    this.speaking = true;
    this.startSynth(text, handlers);
  }

  /** The Web Speech path: one utterance, watchdog, mute and tab gate. */
  private startSynth(text: string, handlers: VoiceHandlers): boolean {
    if (this.synth === null || this.selected === null) {
      this.simulate(text, handlers);
      return false;
    }
    this.handlers = handlers;
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
      this.synth.speak(utterance);
    } catch {
      this.finish("unavailable");
      return false;
    }
    return true;
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
      this.audio?.stop();
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

  private labelFor(status: VoiceStatus): string {
    if (this.service !== null && (status === "ready" || status === "speaking")) {
      return status === "speaking" ? `Sto parlando · ${this.service.label}` : this.service.label;
    }
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
        return this.selected === null
          ? "Sto parlando"
          : `Sto parlando · ${this.selected.name}`;
      case "ready":
        return this.selected === null ? "Voce pronta" : `Voce pronta · ${this.selected.name}`;
    }
  }
}
