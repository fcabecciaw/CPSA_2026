/**
 * Voice engine with the local TTS service (R2-V): WAV playback, analyser for
 * the mouth, and the Web Speech fallback when the service is missing or dies.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { memoryStorage } from "./cognitive/testkit";
import {
  HttpTtsClient,
  VoiceEngine,
  voiceAudioNode,
  type AudioSinkLike,
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
  startResult = true;
  handlers: { onStart: () => void; onEnd: () => void } | null = null;

  async play(
    _data: ArrayBuffer,
    handlers: { onStart: () => void; onEnd: () => void },
  ): Promise<boolean> {
    this.played += 1;
    this.handlers = handlers;
    if (!this.startResult) {
      return false;
    }
    handlers.onStart();
    return true;
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
});
