/**
 * Voice engine with the local TTS service (R2-V): WAV playback, analyser for
 * the mouth, and the Web Speech fallback when the service is missing or dies.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { memoryStorage } from "./cognitive/testkit";
import {
  FIXED_VOICE,
  HttpTtsClient,
  PICK_KEY,
  VoiceEngine,
  voiceAudioNode,
  type AudioSinkLike,
  type AudioState,
  type PlayResult,
  type SpeechSynthLike,
  type TimerLike,
  type TtsServiceInfo,
  type TtsServiceLike,
  type UtteranceLike,
  type VoiceInfo,
} from "./voice";

const timer: TimerLike = {
  setTimeout: (callback, ms) => setTimeout(callback, ms) as unknown as number,
  clearTimeout: (handle) => clearTimeout(handle as unknown as ReturnType<typeof setTimeout>),
};

const INFO: TtsServiceInfo = {
  url: "http://127.0.0.1:8765",
  engine: "kokoro",
  voice: "if_sara",
  label: "Kokoro 82M · Sara",
};

class FakeTts implements TtsServiceLike {
  info: TtsServiceInfo | null = INFO;
  calls: string[] = [];
  probes = 0;
  fail = false;
  hang = false;
  lastSignal: AbortSignal | undefined;
  readonly wav = new ArrayBuffer(16);

  async probe(): Promise<TtsServiceInfo | null> {
    this.probes += 1;
    return this.info;
  }

  synthesize(
    text: string,
    engine: string,
    voice: string,
    signal?: AbortSignal,
  ): Promise<ArrayBuffer> {
    this.calls.push(`${engine}/${voice}/${text}`);
    this.lastSignal = signal;
    if (this.hang) {
      return new Promise<ArrayBuffer>(() => {});
    }
    if (this.fail) {
      return Promise.reject(new Error("service down"));
    }
    return Promise.resolve(this.wav);
  }
}

class FakeSink implements AudioSinkLike {
  played = 0;
  stopped = 0;
  unlocked = 0;
  startResult: PlayResult = true;
  /** Decoded length reported to `onStart`; undefined keeps the 8 s watchdog. */
  duration: number | undefined;
  hangPlay = false;
  context: AudioState = "running";
  private readonly listeners: (() => void)[] = [];
  handlers: { onStart: (durationS?: number) => void; onEnd: () => void } | null = null;

  async play(
    _data: ArrayBuffer,
    handlers: { onStart: (durationS?: number) => void; onEnd: () => void },
  ): Promise<PlayResult> {
    this.played += 1;
    this.handlers = handlers;
    if (this.hangPlay) {
      return new Promise<PlayResult>(() => {});
    }
    if (this.startResult !== true) {
      return this.startResult;
    }
    handlers.onStart(this.duration);
    return true;
  }

  state(): AudioState {
    return this.context;
  }

  onStateChange(listener: () => void): void {
    this.listeners.push(listener);
  }

  /** The browser suspended or resumed the context (device change). */
  setContext(state: AudioState): void {
    this.context = state;
    for (const listener of this.listeners) {
      listener();
    }
  }

  stop(): void {
    this.stopped += 1;
  }

  unlock(): void {
    this.unlocked += 1;
  }

  analyserNode(): AnalyserNode | null {
    return null;
  }
}

interface FakeSynth extends SpeechSynthLike {
  readonly spoken: UtteranceLike[];
  cancelled: number;
}

function fakeSynth(voices: VoiceInfo[] = [{ name: "Google italiano", lang: "it-IT" }]): FakeSynth {
  const synth: FakeSynth = {
    spoken: [],
    cancelled: 0,
    getVoices: () => voices,
    speak: (utterance) => {
      synth.spoken.push(utterance);
    },
    cancel: () => {
      synth.cancelled += 1;
    },
  };
  return synth;
}

function utterance(text: string): UtteranceLike {
  return {
    text,
    lang: "",
    voice: null,
    rate: 1,
    pitch: 1,
    volume: 1,
    onstart: null,
    onend: null,
    onerror: null,
    onboundary: null,
  };
}

function makeVoice(tts: FakeTts, sink: FakeSink, synth = fakeSynth()): VoiceEngine {
  return new VoiceEngine({
    synth,
    createUtterance: utterance,
    storage: memoryStorage(),
    timer,
    tts,
    audio: sink,
  });
}

/** Let the promise chain of speakLocal settle (fake timers block waitFor). */
async function flush(): Promise<void> {
  for (let index = 0; index < 12; index += 1) {
    await Promise.resolve();
  }
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("R2-V — local service playback", () => {
  it("probes /health, fetches the WAV and plays it through the audio graph", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    await flush();
    expect(tts.probes).toBe(1);
    expect(voice.serviceLabel).toBe("Kokoro 82M · Sara");

    voice.activate();
    expect(sink.unlocked).toBe(1);
    const started = vi.fn();
    const ended = vi.fn();
    expect(voice.speak("Ciao! Che bello vederti.", { onStart: started, onEnd: ended })).toBe(true);
    await flush();

    expect(tts.calls).toEqual(["kokoro/if_sara/Ciao! Che bello vederti."]);
    expect(sink.played).toBe(1);
    expect(started).toHaveBeenCalled();
    expect(synth.spoken).toHaveLength(0);
    expect(voice.isSpeaking).toBe(true);
    expect(voice.snapshot().status).toBe("speaking");
    expect(voice.snapshot().service).toBe("Kokoro 82M · Sara");

    sink.handlers?.onEnd();
    expect(ended).toHaveBeenCalledWith("ended");
    expect(voice.isSpeaking).toBe(false);
  });

  it("keeps the click gate, the single-tab rule and mute for the service", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();

    // No gesture yet: simulated line, no network call.
    const firstEnd = vi.fn();
    expect(voice.speak("Ciao", { onEnd: firstEnd })).toBe(false);
    vi.advanceTimersByTime(2_000);
    expect(firstEnd).toHaveBeenCalled();
    expect(tts.calls).toHaveLength(0);

    voice.activate();
    expect(voice.snapshot().status).toBe("ready");
    voice.setMuted(true);
    expect(voice.snapshot().status).toBe("muted");
    expect(voice.speak("Ciao", {})).toBe(false);
    expect(tts.calls).toHaveLength(0);

    voice.setMuted(false);
    expect(voice.speak("Ciao", {})).toBe(true);
    await flush();
    expect(tts.calls).toHaveLength(1);
  });

  it("cancels a sentence in flight and stops the audio graph", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Una frase lunga.", { onEnd: ended });
    await flush();
    expect(sink.played).toBe(1);

    voice.cancel();
    expect(sink.stopped).toBe(1);
    expect(ended).toHaveBeenCalledWith("cancelled");
    expect(voice.isSpeaking).toBe(false);
  });

  it("aborts the pending fetch when cancelled before playback", async () => {
    const tts = new FakeTts();
    tts.hang = true;
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Frase lenta.", { onEnd: ended });
    await flush();
    expect(sink.played).toBe(0);

    voice.cancel();
    expect(tts.lastSignal?.aborted).toBe(true);
    expect(ended).toHaveBeenCalledWith("cancelled");
  });

  it("arms the 8 s watchdog for the WAV too (F-E4 #17)", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Frase che non finisce.", { onEnd: ended });
    await flush();
    expect(sink.played).toBe(1);

    vi.advanceTimersByTime(8_001);
    expect(ended).toHaveBeenCalledWith("watchdog");
    expect(sink.stopped).toBeGreaterThanOrEqual(1);
    expect(voice.isSpeaking).toBe(false);
  });
});

describe("R2-V — Web Speech fallback", () => {
  it("falls back when the service is absent", async () => {
    const tts = new FakeTts();
    tts.info = null;
    const sink = new FakeSink();
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    await flush();
    expect(voice.serviceLabel).toBeNull();

    voice.activate();
    expect(voice.speak("Ciao", {})).toBe(true);
    expect(synth.spoken).toHaveLength(1);
    expect(tts.calls).toHaveLength(0);
  });

  it("falls back when the synthesis request fails", async () => {
    const tts = new FakeTts();
    tts.fail = true;
    const sink = new FakeSink();
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    voice.activate();
    const started = vi.fn();
    // The local attempt is accepted first, then Web Speech takes over.
    expect(voice.speak("Ciao", { onStart: started })).toBe(true);
    await flush();
    expect(tts.calls).toHaveLength(1);
    expect(synth.spoken).toHaveLength(1);
    expect(synth.spoken[0].text).toBe("Ciao");
    expect(voice.isSpeaking).toBe(true);
  });

  it("falls back when the browser refuses to start the audio", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    sink.startResult = false;
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    voice.activate();
    voice.speak("Ciao", {});
    await flush();
    expect(sink.played).toBe(1);
    expect(synth.spoken).toHaveLength(1);
  });
});

describe("R2-V — analyser for the mouth", () => {
  it("returns null in a context without WebAudio", () => {
    expect(voiceAudioNode()).toBeNull();
  });
});

describe("R2-V — HTTP client", () => {
  it("reads /health and /voices to label the service", async () => {
    const seen: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        seen.push(String(url));
        if (String(url).endsWith("/health")) {
          return new Response(
            JSON.stringify({ status: "ok", defaults: { engine: "kokoro", voice: "if_sara" } }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({
            engines: [
              { id: "kokoro", label: "Kokoro 82M", voices: [{ id: "if_sara", name: "Sara" }] },
            ],
          }),
          { status: 200 },
        );
      }),
    );
    try {
      const info = await new HttpTtsClient().probe();
      expect(info).toEqual({
        url: "http://127.0.0.1:8765",
        engine: "kokoro",
        voice: "if_sara",
        label: "Kokoro 82M · Sara",
      });
      expect(seen).toEqual([
        "http://127.0.0.1:8765/health",
        "http://127.0.0.1:8765/voices",
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("is absent when the health probe fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new Error("ECONNREFUSED"))));
    try {
      expect(await new HttpTtsClient().probe()).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("posts text, engine and voice and returns the WAV bytes", async () => {
    const request = vi.fn(async () => {
      return new Response(new Uint8Array([82, 73, 70, 70]), {
        status: 200,
        headers: { "Content-Type": "audio/wav" },
      });
    });
    vi.stubGlobal("fetch", request);
    try {
      const client = new HttpTtsClient();
      const buffer = await client.synthesize("Ciao", "kokoro", "if_sara");
      expect(new Uint8Array(buffer)).toEqual(new Uint8Array([82, 73, 70, 70]));
      const [url, init] = request.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toBe("http://127.0.0.1:8765/tts");
      expect(init.method).toBe("POST");
      expect(JSON.parse(String(init.body))).toEqual({
        text: "Ciao",
        engine: "kokoro",
        voice: "if_sara",
      });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("reads the catalog, sets the default and drives a fill", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const idle = {
      running: false,
      engine: "edge",
      voice: "it-IT-ElsaNeural",
      total: 3,
      done: 3,
      synthesized: 3,
      cached: 0,
      failed: 0,
      error: null,
      started_at: "2026-11-12T10:00:00+00:00",
      finished_at: "2026-11-12T10:00:02+00:00",
      elapsed_s: 2.1,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        requests.push({ url: String(url), init });
        if (String(url).endsWith("/voices")) {
          return new Response(
            JSON.stringify({
              defaults: { engine: "kokoro", voice: "if_sara" },
              engines: [{ id: "kokoro", label: "Kokoro 82M", voices: [] }],
            }),
            { status: 200 },
          );
        }
        if (String(url).endsWith("/default")) {
          return new Response(JSON.stringify({ defaults: { engine: "edge", voice: "Elsa" } }), {
            status: 200,
          });
        }
        if (String(url).endsWith("/pregenerate") && init?.method === "POST") {
          return new Response(JSON.stringify({ ...idle, running: true, done: 0 }), {
            status: 202,
          });
        }
        return new Response(JSON.stringify(idle), { status: 200 });
      }),
    );
    try {
      const client = new HttpTtsClient();
      const catalog = await client.catalog();
      expect(catalog?.engines[0].id).toBe("kokoro");
      expect(await client.setDefault("edge", "it-IT-ElsaNeural")).toEqual({
        engine: "edge",
        voice: "Elsa",
      });
      const started = await client.pregenerate("edge", "it-IT-ElsaNeural");
      expect(started?.running).toBe(true);
      expect((await client.pregenerateStatus())?.done).toBe(3);

      const put = requests.find((request) => request.url.endsWith("/default"));
      expect(put?.init?.method).toBe("PUT");
      expect(JSON.parse(String(put?.init?.body))).toEqual({
        engine: "edge",
        voice: "it-IT-ElsaNeural",
      });
      const post = requests.find(
        (request) => request.url.endsWith("/pregenerate") && request.init?.method === "POST",
      );
      expect(JSON.parse(String(post?.init?.body))).toEqual({
        engine: "edge",
        voice: "it-IT-ElsaNeural",
      });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("reports a busy service when the fill is already running", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        return new Response(
          JSON.stringify({ error: "already running", job: { running: true, total: 9, done: 2 } }),
          { status: 409 },
        );
      }),
    );
    try {
      const status = await new HttpTtsClient().pregenerate("edge", "it-IT-ElsaNeural");
      expect(status?.running).toBe(true);
      expect(status?.done).toBe(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("R2-V — the voice picked in the dashboard", () => {
  it("stays chosen across a reload, through localStorage", async () => {
    const pick = memoryStorage();
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = new VoiceEngine({
      synth: fakeSynth(),
      createUtterance: utterance,
      storage: memoryStorage(),
      pickStorage: pick,
      timer,
      tts,
      audio: sink,
    });
    await flush();

    voice.chooseVoice("edge", "it-IT-ElsaNeural", "Microsoft Neural (edge-tts) · Elsa");
    expect(voice.serviceLabel).toBe("Microsoft Neural (edge-tts) · Elsa");
    expect(JSON.parse(String(pick.getItem(PICK_KEY)))).toEqual({
      engine: "edge",
      voice: "it-IT-ElsaNeural",
      label: "Microsoft Neural (edge-tts) · Elsa",
    });
    voice.activate();
    voice.speak("Ciao", {});
    await flush();
    expect(tts.calls.at(-1)).toBe("edge/it-IT-ElsaNeural/Ciao");

    // A reload probes /health again (Kokoro) but keeps the picked voice.
    const reloaded = new VoiceEngine({
      synth: fakeSynth(),
      createUtterance: utterance,
      storage: memoryStorage(),
      pickStorage: pick,
      timer,
      tts,
      audio: sink,
    });
    await flush();
    await flush();
    expect(reloaded.serviceInfo()).toMatchObject({
      engine: "edge",
      voice: "it-IT-ElsaNeural",
    });
    reloaded.activate();
    reloaded.speak("Ciao", {});
    await flush();
    expect(tts.calls.at(-1)).toBe("edge/it-IT-ElsaNeural/Ciao");
  });
});

describe("service voice robustness (silent voice after a hung request)", () => {
  it("gives up on a hung POST /tts after 6 s and falls back to Web Speech", async () => {
    const tts = new FakeTts();
    tts.hang = true;
    const sink = new FakeSink();
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    voice.activate();
    expect(voice.speak("Frase appesa.", {})).toBe(true);
    await flush();
    expect(synth.spoken).toHaveLength(0);

    vi.advanceTimersByTime(6_001);
    await flush();
    expect(tts.lastSignal?.aborted).toBe(true);
    expect(synth.spoken).toHaveLength(1);
    expect(synth.spoken[0].text).toBe("Frase appesa.");
  });

  it("never stays speaking after a hung request without Web Speech", async () => {
    const tts = new FakeTts();
    tts.hang = true;
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink, fakeSynth([]));
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Frase appesa.", { onEnd: ended });
    await flush();
    vi.advanceTimersByTime(6_001);
    await flush();
    // The silent line still runs its course for the face and the bubble.
    vi.advanceTimersByTime(8_001);
    expect(ended).toHaveBeenCalledWith("unavailable");
    expect(voice.isSpeaking).toBe(false);

    // The next sentence reaches the service again.
    tts.hang = false;
    expect(voice.speak("Ci sono ancora.", {})).toBe(true);
    await flush();
    expect(tts.calls.at(-1)).toBe("kokoro/if_sara/Ci sono ancora.");
    expect(sink.played).toBe(1);
  });

  it("falls back when the WAV never starts", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    sink.hangPlay = true;
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    voice.activate();
    voice.speak("Decodifica lenta.", {});
    await flush();
    vi.advanceTimersByTime(3_001);
    await flush();
    expect(synth.spoken).toHaveLength(1);
    // A late start of the abandoned attempt is stopped, not played over it.
    const stopsBefore = sink.stopped;
    sink.handlers?.onStart(2);
    expect(sink.stopped).toBe(stopsBefore + 1);
  });

  it("does not cut a long XTTS sentence at 8 s: watchdog = duration + 1.5 s", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    sink.duration = 10;
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Un fatto lungo sui nuraghi.", { onEnd: ended });
    await flush();

    vi.advanceTimersByTime(11_000);
    expect(ended).not.toHaveBeenCalled();
    expect(voice.isSpeaking).toBe(true);
    vi.advanceTimersByTime(600);
    expect(ended).toHaveBeenCalledWith("watchdog");
    expect(voice.isSpeaking).toBe(false);
  });

  it("ends the line and shows the click gate when the context stays suspended", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    sink.startResult = "suspended";
    const synth = fakeSynth();
    const voice = makeVoice(tts, sink, synth);
    await flush();
    voice.activate();
    const ended = vi.fn();
    expect(voice.speak("Ciao", { onEnd: ended })).toBe(true);
    await flush();

    expect(ended).toHaveBeenCalledWith("unavailable");
    expect(voice.isSpeaking).toBe(false);
    // Not a silent Web Speech line either: the gate asks for one click.
    expect(synth.spoken).toHaveLength(0);
    expect(voice.snapshot().status).toBe("locked");
    expect(voice.snapshot().audioSuspended).toBe(true);

    sink.startResult = true;
    voice.activate();
    expect(sink.unlocked).toBe(2);
    sink.setContext("running");
    expect(voice.snapshot().audioSuspended).toBe(false);
    expect(voice.speak("Eccomi", {})).toBe(true);
    await flush();
    expect(sink.played).toBe(2);
  });

  it("follows the context state: suspended locks, running re-unlocks", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();
    const ended = vi.fn();
    voice.speak("Frase interrotta.", { onEnd: ended });
    await flush();
    expect(voice.isSpeaking).toBe(true);

    sink.setContext("suspended");
    expect(ended).toHaveBeenCalledWith("unavailable");
    expect(voice.isSpeaking).toBe(false);
    expect(voice.snapshot().status).toBe("locked");
    expect(voice.snapshot().audioSuspended).toBe(true);

    sink.setContext("running");
    expect(voice.snapshot().status).toBe("ready");
    expect(voice.snapshot().audioSuspended).toBe(false);
  });
});

describe("fixed demo voice", () => {
  it("uses XTTS Ana Florence when the picker never stored a voice", async () => {
    const tts = new FakeTts();
    const voice = new VoiceEngine({
      synth: fakeSynth(),
      createUtterance: utterance,
      storage: memoryStorage(),
      pickStorage: memoryStorage(),
      timer,
      tts,
      audio: new FakeSink(),
      defaultPick: FIXED_VOICE,
    });
    await flush();
    expect(voice.serviceInfo()).toMatchObject({ engine: "xtts", voice: "ana_florence" });
    expect(voice.serviceLabel).toBe("XTTS-v2 (Coqui) · Ana Florence");
    voice.activate();
    voice.speak("Ciao!", {});
    await flush();
    expect(tts.calls).toEqual(["xtts/ana_florence/Ciao!"]);
  });

  it("keeps a voice chosen in debug mode over the fixed one", async () => {
    const pick = memoryStorage();
    pick.setItem(PICK_KEY, JSON.stringify({ engine: "edge", voice: "it-IT-ElsaNeural" }));
    const voice = new VoiceEngine({
      synth: fakeSynth(),
      createUtterance: utterance,
      storage: memoryStorage(),
      pickStorage: pick,
      timer,
      tts: new FakeTts(),
      audio: new FakeSink(),
      defaultPick: FIXED_VOICE,
    });
    await flush();
    expect(voice.serviceInfo()).toMatchObject({ engine: "edge", voice: "it-IT-ElsaNeural" });
  });
});

describe("T1 — prefetch", () => {
  it("requests a line ahead and speaks it without a second request", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();

    voice.prefetch(["Ciao a tutti! Siete in tre.", "Vieni più vicino, non mordo!", "C'è qualcuno?"]);
    // At most two requests in flight: the third line waits for a later call.
    expect(tts.calls).toEqual([
      "kokoro/if_sara/Ciao a tutti! Siete in tre.",
      "kokoro/if_sara/Vieni più vicino, non mordo!",
    ]);
    await flush();

    expect(voice.speak("Ciao a tutti! Siete in tre.")).toBe(true);
    await flush();
    expect(tts.calls).toHaveLength(2);
    expect(sink.played).toBe(1);
  });

  it("asks again when the prefetch failed, and skips a muted tab", async () => {
    const tts = new FakeTts();
    const sink = new FakeSink();
    const voice = makeVoice(tts, sink);
    await flush();
    voice.activate();

    tts.fail = true;
    voice.prefetch(["Ciao!"]);
    await flush();
    tts.fail = false;
    expect(voice.speak("Ciao!")).toBe(true);
    await flush();
    expect(tts.calls).toEqual(["kokoro/if_sara/Ciao!", "kokoro/if_sara/Ciao!"]);
    expect(sink.played).toBe(1);

    voice.setMuted(true);
    voice.prefetch(["Altro"]);
    expect(tts.calls).toHaveLength(2);
  });
});
