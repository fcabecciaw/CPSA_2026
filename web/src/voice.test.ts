/**
 * Voice engine: click gate, asynchronous Italian voices, watchdog, mute and
 * per-tab silence.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { memoryStorage } from "./cognitive/testkit";
import {
  ENABLED_KEY,
  VoiceEngine,
  type SpeechSynthLike,
  type TimerLike,
  type UtteranceLike,
  type VoiceInfo,
} from "./voice";

interface FakeSynth extends SpeechSynthLike {
  readonly spoken: UtteranceLike[];
  cancelled: number;
  voices: VoiceInfo[];
  fireVoicesChanged(): void;
}

function fakeSynth(voices: VoiceInfo[] = []): FakeSynth {
  let listener: (() => void) | null = null;
  const synth: FakeSynth = {
    voices,
    spoken: [],
    cancelled: 0,
    getVoices() {
      return synth.voices;
    },
    speak(utterance) {
      synth.spoken.push(utterance);
    },
    cancel() {
      synth.cancelled += 1;
    },
    addEventListener(type, handler) {
      if (type === "voiceschanged") {
        listener = handler;
      }
    },
    removeEventListener() {
      listener = null;
    },
    fireVoicesChanged() {
      listener?.();
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

const timer: TimerLike = {
  setTimeout: (callback, ms) => setTimeout(callback, ms) as unknown as number,
  clearTimeout: (handle) => clearTimeout(handle as unknown as ReturnType<typeof setTimeout>),
};

function makeVoice(synth: FakeSynth, storage = memoryStorage(), enabled = false): VoiceEngine {
  return new VoiceEngine({
    synth,
    createUtterance: utterance,
    storage,
    timer,
    enabled,
  });
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("F-E4 #15 — Chrome autoplay gate", () => {
  it("stays silent until 'Attiva la voce' is clicked, and keeps the state", () => {
    const synth = fakeSynth([{ name: "Google italiano", lang: "it-IT" }]);
    const storage = memoryStorage();
    const voice = makeVoice(synth, storage);
    expect(voice.snapshot().status).toBe("off");

    const ended = vi.fn();
    expect(voice.speak("Ciao!", { onStart: vi.fn(), onEnd: ended })).toBe(false);
    expect(synth.spoken).toHaveLength(0);
    vi.advanceTimersByTime(2_000);
    expect(ended).toHaveBeenCalled();

    voice.activate();
    expect(voice.snapshot().status).toBe("ready");
    expect(voice.speak("Ciao!", {})).toBe(true);
    expect(synth.spoken).toHaveLength(1);
    expect(storage.getItem(ENABLED_KEY)).toBe("1");

    const reloaded = makeVoice(fakeSynth([{ name: "Google italiano", lang: "it-IT" }]), storage);
    expect(reloaded.isEnabled).toBe(true);
    expect(reloaded.snapshot().status).toBe("locked");
  });
});

describe("F-E4 #16 — voices load asynchronously", () => {
  it("prefers Google italiano, then any Italian, then a muted badge", () => {
    const synth = fakeSynth([]);
    const voice = makeVoice(synth);
    voice.activate();
    expect(voice.snapshot().status).toBe("no-voice");
    expect(() => voice.speak("Ciao", {})).not.toThrow();

    synth.voices = [
      { name: "Alice", lang: "en-US" },
      { name: "Luca", lang: "it-IT" },
      { name: "Google italiano", lang: "it-IT" },
    ];
    synth.fireVoicesChanged();
    expect(voice.voiceName).toBe("Google italiano");

    synth.voices = [{ name: "Alice", lang: "en-US" }, { name: "Luca", lang: "it-IT" }];
    synth.fireVoicesChanged();
    expect(voice.voiceName).toBe("Luca");

    synth.voices = [{ name: "Alice", lang: "en-US" }];
    synth.fireVoicesChanged();
    expect(voice.voiceName).toBeNull();
    expect(voice.snapshot().status).toBe("no-voice");
    expect(voice.snapshot().label).toMatch(/Nessuna voce italiana/);
  });
});

describe("F-E4 #17 — stuck speech watchdog", () => {
  it("cancels an utterance that never ends after 8 s", () => {
    const synth = fakeSynth([{ name: "Google italiano", lang: "it-IT" }]);
    const voice = makeVoice(synth);
    voice.activate();
    const ended = vi.fn();
    expect(voice.speak("Una frase lunghissima che non finisce mai", { onEnd: ended })).toBe(true);
    synth.spoken[0].onstart?.();
    vi.advanceTimersByTime(8_001);
    expect(synth.cancelled).toBe(1);
    expect(ended).toHaveBeenCalledWith("watchdog");
    expect(voice.isSpeaking).toBe(false);
  });
});

describe("F-E4 #18 — several dashboard tabs", () => {
  it("only the enabled tab uses the speech synthesizer", () => {
    const silentSynth = fakeSynth([{ name: "Google italiano", lang: "it-IT" }]);
    const silent = makeVoice(silentSynth, memoryStorage());
    const silentEnd = vi.fn();
    expect(silent.speak("Ciao", { onStart: vi.fn(), onEnd: silentEnd })).toBe(false);
    expect(silentSynth.spoken).toHaveLength(0);
    vi.advanceTimersByTime(2_000);
    expect(silentEnd).toHaveBeenCalled();

    const voiceSynth = fakeSynth([{ name: "Google italiano", lang: "it-IT" }]);
    const voice = makeVoice(voiceSynth, memoryStorage(), true);
    expect(voice.speak("Ciao", {})).toBe(false); // ?voice=1 but no gesture yet
    voice.unlock();
    expect(voice.speak("Ciao", {})).toBe(true);
    expect(voiceSynth.spoken).toHaveLength(1);
  });
});

describe("F-E4 #19 — mute keeps the face and bubble alive", () => {
  it("runs the handlers without speaking, then speaks again when unmuted", () => {
    const synth = fakeSynth([{ name: "Google italiano", lang: "it-IT" }]);
    const voice = makeVoice(synth);
    voice.activate();
    expect(voice.toggleMute()).toBe(true);

    const started = vi.fn();
    const ended = vi.fn();
    expect(voice.speak("Ciao", { onStart: started, onEnd: ended })).toBe(false);
    expect(synth.spoken).toHaveLength(0);
    expect(started).toHaveBeenCalled();
    vi.advanceTimersByTime(2_000);
    expect(ended).toHaveBeenCalled();
    expect(voice.snapshot().status).toBe("muted");

    voice.toggleMute();
    expect(voice.speak("Ciao", {})).toBe(true);
    expect(synth.spoken).toHaveLength(1);
  });
});
