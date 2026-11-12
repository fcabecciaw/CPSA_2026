/**
 * Voice picker: catalog, per-tab selection, the `v` comparison loop and the
 * background fill of the fixed sentences (no DOM in Node).
 */

import { describe, expect, it } from "vitest";

import { memoryStorage } from "./cognitive/testkit";
import {
  PICK_KEY,
  type PregenerateStatus,
  type StorageLike,
  type TimerLike,
  type TtsCatalog,
  type TtsDefaults,
  type TtsVoiceClient,
  type VoiceHandlers,
} from "./voice";
import { TEST_SENTENCE, VoicePicker, flattenCatalog, type VoicePickerVoice } from "./voicepicker";

const CATALOG: TtsCatalog = {
  defaults: { engine: "kokoro", voice: "if_sara" },
  engines: [
    {
      id: "kokoro",
      label: "Kokoro 82M",
      licence: "Apache-2.0",
      device: "cpu",
      available: true,
      loaded: false,
      error: null,
      voices: [
        { id: "if_sara", name: "Sara", lang: "it", gender: "female" },
        { id: "im_nicola", name: "Nicola", lang: "it", gender: "male" },
      ],
    },
    {
      id: "edge",
      label: "Microsoft Neural (edge-tts)",
      licence: "Microsoft read-aloud terms",
      device: "online",
      available: true,
      loaded: true,
      error: null,
      voices: [
        { id: "it-IT-ElsaNeural", name: "Elsa", lang: "it-IT", gender: "female" },
        { id: "it-IT-DiegoNeural", name: "Diego", lang: "it-IT", gender: "male" },
      ],
    },
  ],
};

const IDLE: PregenerateStatus = {
  running: false,
  engine: null,
  voice: null,
  total: 0,
  done: 0,
  synthesized: 0,
  cached: 0,
  failed: 0,
  error: null,
  started_at: null,
  finished_at: null,
  elapsed_s: null,
};

class FakeClient implements TtsVoiceClient {
  catalogValue: TtsCatalog | null = CATALOG;
  catalogCalls = 0;
  defaults: TtsDefaults[] = [];
  jobs: string[] = [];
  failDefaults = false;
  status: PregenerateStatus = IDLE;

  async catalog(): Promise<TtsCatalog | null> {
    this.catalogCalls += 1;
    return this.catalogValue;
  }

  async setDefault(engine: string, voice: string): Promise<TtsDefaults | null> {
    if (this.failDefaults) {
      return null;
    }
    this.defaults.push({ engine, voice });
    return { engine, voice };
  }

  async pregenerate(engine: string, voice: string): Promise<PregenerateStatus | null> {
    this.jobs.push(`${engine}/${voice}`);
    this.status = { ...IDLE, running: true, engine, voice, total: 10 };
    return this.status;
  }

  async pregenerateStatus(): Promise<PregenerateStatus | null> {
    return this.status;
  }
}

class FakeVoice implements VoicePickerVoice {
  chosen: string[] = [];
  spoken: string[] = [];
  activations = 0;
  cancels = 0;
  info: { engine: string; voice: string; label: string } | null = {
    engine: "kokoro",
    voice: "if_sara",
    label: "Kokoro 82M · Sara",
  };
  /** Stands in for the engine's localStorage pick. */
  storage: StorageLike = memoryStorage();

  chooseVoice(engine: string, voice: string, label?: string): void {
    this.chosen.push(`${engine}/${voice}`);
    this.info = { engine, voice, label: label ?? `${engine} · ${voice}` };
    this.storage.setItem(PICK_KEY, JSON.stringify({ engine, voice, label: this.info.label }));
  }

  serviceInfo(): { engine: string; voice: string; label: string } | null {
    return this.info;
  }

  activate(): void {
    this.activations += 1;
  }

  cancel(): void {
    this.cancels += 1;
  }

  speak(text: string, _handlers?: VoiceHandlers): boolean {
    this.spoken.push(text);
    return true;
  }
}

interface RecordingTimer extends TimerLike {
  callbacks: Array<() => void>;
  cleared: number;
}

function recordingTimer(): RecordingTimer {
  const timer: RecordingTimer = {
    callbacks: [],
    cleared: 0,
    setTimeout(callback) {
      timer.callbacks.push(callback);
      return timer.callbacks.length;
    },
    clearTimeout() {
      timer.cleared += 1;
    },
  };
  return timer;
}

function makePicker(
  voice: FakeVoice,
  client: FakeClient,
  options: { storage?: StorageLike; pollMs?: number; timer?: TimerLike } = {},
): VoicePicker {
  return new VoicePicker({
    voice,
    client,
    storage: options.storage ?? memoryStorage(),
    timer: options.timer ?? recordingTimer(),
    pollMs: options.pollMs ?? 0,
  });
}

/** Let the async methods (PUT /default, polling) settle. */
async function flush(): Promise<void> {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve();
  }
}

describe("R2-V — voice picker", () => {
  it("lists every voice grouped by engine", () => {
    const entries = flattenCatalog(CATALOG);
    expect(entries.map((entry) => `${entry.engine}/${entry.voice}`)).toEqual([
      "kokoro/if_sara",
      "kokoro/im_nicola",
      "edge/it-IT-ElsaNeural",
      "edge/it-IT-DiegoNeural",
    ]);
    expect(entries[2].label).toBe("Microsoft Neural (edge-tts) · Elsa");
    expect(entries[2].engineLabel).toBe("Microsoft Neural (edge-tts)");
  });

  it("loads the catalog and adopts the voice of the tab", async () => {
    const voice = new FakeVoice();
    voice.info = { engine: "edge", voice: "it-IT-DiegoNeural", label: "Edge · Diego" };
    const picker = makePicker(voice, new FakeClient());
    const state = await picker.load();
    expect(state.ready).toBe(true);
    expect(state.entries).toHaveLength(4);
    expect(state.current?.voice).toBe("it-IT-DiegoNeural");
    expect(voice.chosen).toEqual(["edge/it-IT-DiegoNeural"]);
    expect(state.defaults).toEqual({ engine: "kokoro", voice: "if_sara" });
  });

  it("prefers a saved pick over the server default, then the first voice", async () => {
    const storage = memoryStorage();
    storage.setItem(
      PICK_KEY,
      JSON.stringify({ engine: "edge", voice: "it-IT-ElsaNeural", label: "Edge · Elsa" }),
    );
    const voice = new FakeVoice();
    voice.info = null;
    const picker = makePicker(voice, new FakeClient(), { storage });
    const state = await picker.load();
    // The catalog label is authoritative, not the one saved last time.
    expect(state.current?.label).toBe("Microsoft Neural (edge-tts) · Elsa");
    expect(voice.chosen).toEqual(["edge/it-IT-ElsaNeural"]);

    const fresh = new FakeVoice();
    fresh.info = null;
    const second = await makePicker(fresh, new FakeClient()).load();
    expect(second.current?.engine).toBe("kokoro");
    expect(second.current?.voice).toBe("if_sara");
    expect(second.index).toBe(0);
  });

  it("remembers a selection in this tab and asks the service to default to it", async () => {
    const voice = new FakeVoice();
    const client = new FakeClient();
    const storage = memoryStorage();
    const picker = makePicker(voice, client, { storage });
    voice.storage = storage;
    await picker.load();

    const entry = picker.select("edge", "it-IT-ElsaNeural");
    expect(entry?.name).toBe("Elsa");
    expect(voice.chosen.at(-1)).toBe("edge/it-IT-ElsaNeural");
    expect(JSON.parse(String(storage.getItem(PICK_KEY)))).toMatchObject({
      engine: "edge",
      voice: "it-IT-ElsaNeural",
    });
    expect(client.defaults).toEqual([{ engine: "edge", voice: "it-IT-ElsaNeural" }]);
    await flush();
    expect(picker.state().defaults).toEqual({ engine: "edge", voice: "it-IT-ElsaNeural" });
    expect(picker.select("nope", "nope")).toBeNull();
  });

  it("keeps the tab voice when the service refuses the new default", async () => {
    const voice = new FakeVoice();
    const client = new FakeClient();
    client.failDefaults = true;
    const picker = makePicker(voice, client);
    await picker.load();
    picker.select("edge", "it-IT-DiegoNeural");
    await flush();
    expect(picker.state().message).toMatch(/questa scheda/);
    expect(voice.info?.voice).toBe("it-IT-DiegoNeural");
  });

  it("cycles through the voices and speaks the test sentence each time", async () => {
    const voice = new FakeVoice();
    const client = new FakeClient();
    const picker = makePicker(voice, client);
    await picker.load();

    expect(picker.cycle()?.voice).toBe("im_nicola");
    expect(picker.cycle()?.voice).toBe("it-IT-ElsaNeural");
    expect(picker.cycle()?.voice).toBe("it-IT-DiegoNeural");
    expect(picker.cycle()?.voice).toBe("if_sara"); // wraps around
    expect(voice.spoken).toEqual([TEST_SENTENCE, TEST_SENTENCE, TEST_SENTENCE, TEST_SENTENCE]);
    await flush();
    expect(voice.activations).toBe(4);
    expect(client.defaults).toHaveLength(4);
  });

  it("answers the 'v' shortcut, but not while typing or with a modifier", async () => {
    const voice = new FakeVoice();
    const picker = makePicker(voice, new FakeClient());
    await picker.load();

    expect(picker.handleKey({ key: "v" })).toBe(true);
    expect(picker.state().current?.voice).toBe("im_nicola");
    expect(voice.spoken).toHaveLength(1);

    expect(picker.handleKey({ key: "V", ctrlKey: true })).toBe(false);
    expect(picker.handleKey({ key: "a" })).toBe(false);
    expect(voice.spoken).toHaveLength(1);
  });

  it("does not answer 'v' before the catalog arrives", () => {
    const picker = makePicker(new FakeVoice(), new FakeClient());
    expect(picker.handleKey({ key: "v" })).toBe(false);
    expect(picker.cycle()).toBeNull();
    expect(picker.speakTest()).toBe(false);
  });

  it("starts a fill and follows its progress to the end", async () => {
    const voice = new FakeVoice();
    const client = new FakeClient();
    const timer = recordingTimer();
    const picker = makePicker(voice, client, { timer, pollMs: 10 });
    await picker.load();

    await picker.pregenerate();
    expect(client.jobs).toEqual(["kokoro/if_sara"]);
    expect(picker.state().job?.running).toBe(true);
    expect(picker.state().message).toMatch(/Preparo le frasi/);
    expect(timer.callbacks).toHaveLength(1);

    client.status = { ...client.status, done: 4 };
    timer.callbacks[0]();
    await flush();
    expect(picker.state().job?.done).toBe(4);
    expect(picker.state().message).toContain("4/10");

    client.status = { ...client.status, running: false, done: 10, cached: 6 };
    timer.callbacks[1]();
    await flush();
    expect(picker.state().job?.running).toBe(false);
    expect(picker.state().message).toMatch(/Frasi pronte: 10\/10/);
    expect(timer.callbacks).toHaveLength(2); // polling stopped
  });

  it("says when the fill cannot start and stops polling on dispose", async () => {
    const voice = new FakeVoice();
    const client = new FakeClient();
    const picker = makePicker(voice, client);
    await picker.load();
    client.status = { ...IDLE, running: false, error: "edge-tts synthesis failed: no internet" };
    await picker.poll();
    expect(picker.state().message).toMatch(/interrotta/);

    const timer = recordingTimer();
    const polling = makePicker(voice, client, { timer, pollMs: 10 });
    await polling.load();
    await polling.pregenerate();
    expect(timer.callbacks).toHaveLength(1);
    polling.dispose();
    expect(timer.cleared).toBe(1);
  });

  it("reports a missing service instead of pretending to work", async () => {
    const client = new FakeClient();
    client.catalogValue = null;
    const picker = makePicker(new FakeVoice(), client);
    const state = await picker.load();
    expect(state.ready).toBe(false);
    expect(state.error).toBe("Servizio voce non raggiungibile");

    client.catalogValue = CATALOG;
    await picker.load();
    expect(picker.state().ready).toBe(true);
    expect(picker.state().error).toBeNull();
  });

  it("ignores a corrupt saved pick", async () => {
    const storage = memoryStorage();
    storage.setItem(PICK_KEY, "{broken");
    const voice = new FakeVoice();
    voice.info = null;
    const picker = makePicker(voice, new FakeClient(), { storage });
    const state = await picker.load();
    expect(state.current?.voice).toBe("if_sara");
  });
});
