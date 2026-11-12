/**
 * Wiring of perception → working memory → production rules → voice/face
 * (F-E3). One instance is the cognitive layer of the dashboard: it owns the
 * clock, the debouncing, the warm-up and the lifecycle corners, so `main.ts`
 * only forwards protocol messages and renders what the agent reports.
 */

import type { PoseStatePayload, ShowcaseStatePayload, StatusPayload } from "../protocol";
import type { CognitiveEvent, FaceCueKind, SceneName, WorldModelName } from "./events";
import { describeEvent } from "./events";
import { CognitiveMemory } from "./memory";
import { Perception } from "./perception";
import type { Expression, Utterance } from "./rules";
import { CognitiveEngine } from "./rules";
import { SceneMachine } from "./scene";

export interface MindSnapshot {
  event: string | null;
  ruleId: string | null;
  utility: number | null;
  people: number;
  mode: WorldModelName;
  /** The scene the dialogue is in (R2-D). */
  scene: SceneName;
  /** Whether the laptop face cues are on, for the "mente" strip. */
  face: string;
}

export interface SpeechHandlers {
  onStart?: () => void;
  onBoundary?: () => void;
  onEnd?: (reason: string) => void;
}

export interface AgentOptions {
  engine: CognitiveEngine;
  memory?: CognitiveMemory;
  perception?: Perception;
  /** The dialogue scenes; tests inject a machine with short timings. */
  scene?: SceneMachine;
  /** Speak one sentence; `main.ts` bridges this to `VoiceEngine`. */
  speak?: (text: string, handlers: SpeechHandlers) => void;
  onUtterance?: (utterance: Utterance, mind: MindSnapshot) => void;
  onExpression?: (expression: Expression) => void;
  onMind?: (mind: MindSnapshot) => void;
}

const SELF_FACT_MS = 20_000;

type TimerHandle = ReturnType<typeof setTimeout>;

export class CognitiveAgent {
  readonly engine: CognitiveEngine;
  readonly memory: CognitiveMemory;
  readonly perception: Perception;
  readonly scene: SceneMachine;

  private readonly options: AgentOptions;
  private mode: WorldModelName = "yolox";
  private started = false;
  private visionLost = false;
  private lastClockHour: number | null = null;
  private lastSelfAt = 0;
  private lastEvent: CognitiveEvent | null = null;
  private lastUtterance: Utterance | null = null;
  private speechTimer: TimerHandle | null = null;
  private readonly speechTimers = new Set<TimerHandle>();

  /** Face cues come from the laptop browser, never from the board (R2-D). */
  private faceStatus = "spento";

  constructor(options: AgentOptions) {
    this.options = options;
    this.engine = options.engine;
    this.memory = options.memory ?? new CognitiveMemory();
    this.perception = options.perception ?? new Perception();
    this.scene = options.scene ?? new SceneMachine();
  }

  get isVisionLost(): boolean {
    return this.visionLost;
  }

  get mind(): MindSnapshot {
    return {
      event: this.lastEvent === null ? null : describeEvent(this.lastEvent),
      ruleId: this.lastUtterance?.ruleId ?? null,
      utility: this.lastUtterance?.utility ?? null,
      people: this.memory.working.focus.people,
      mode: this.mode,
      scene: this.scene.scene,
      face: this.faceStatus,
    };
  }

  /** The "mente" strip says where the face cues run. */
  setFaceStatus(status: string): void {
    this.faceStatus = status;
  }

  /**
   * A `showcase.state`: the only place object/people rules see the world.
   * While the vision is lost the frame is ignored; the first frame back
   * announces the recovery and seeds a silent warm-up (F-E4 #20/#21).
   */
  onShowcase(state: ShowcaseStatePayload, now: number): CognitiveEvent[] {
    this.begin(now);
    const model: WorldModelName = state.model === "movenet" ? "movenet" : "yolox";
    const modeEvents: CognitiveEvent[] = [];
    if (model !== this.mode) {
      modeEvents.push(...this.switchMode(model, now));
    }
    if (this.visionLost) {
      this.visionLost = false;
      this.perception.reset(now, this.mode);
      modeEvents.push({ type: "vision", state: "back", now });
      this.options.onExpression?.("happy");
    }
    if (model !== "yolox") {
      this.emit(modeEvents, now);
      return modeEvents;
    }

    const events = this.perception.update(state, now);
    this.prepare(events, now);
    this.emit([...modeEvents, ...events], now);
    return events;
  }

  /** A `pose.state`: only MoveNet feeds it, and only when the pose is real. */
  onPose(pose: PoseStatePayload, now: number): CognitiveEvent[] {
    this.begin(now);
    const modeEvents: CognitiveEvent[] = [];
    if (this.mode !== "movenet") {
      modeEvents.push(...this.switchMode("movenet", now));
    }
    if (this.visionLost) {
      this.visionLost = false;
      this.perception.reset(now, "movenet");
      modeEvents.push({ type: "vision", state: "back", now });
      this.options.onExpression?.("happy");
    }
    const events = this.perception.updatePose(pose, now);
    this.prepare(events, now);
    this.emit([...modeEvents, ...events], now);
    return events;
  }

  /** A face cue confirmed by the laptop face tracker (MediaPipe, R2-D). */
  onFace(kind: FaceCueKind, score: number, faces: number, now: number): CognitiveEvent[] {
    this.begin(now);
    const event: CognitiveEvent = { type: "face", kind, score, faces, now };
    this.prepare([event], now);
    this.emit([event], now);
    return [event];
  }

  /** `status` at 1 Hz: self-facts and camera errors. */
  onStatus(status: StatusPayload, now: number): CognitiveEvent[] {
    this.begin(now);
    if (status.camera !== "ok" || status.state === "error") {
      return this.loseVision(now);
    }
    const events: CognitiveEvent[] = [];
    if (now - this.lastSelfAt >= SELF_FACT_MS) {
      this.lastSelfAt = now;
      const fields: Array<"fps" | "dpu_ms" | "uptime"> = ["fps", "dpu_ms", "uptime"];
      const field = fields[Math.floor(now / SELF_FACT_MS) % fields.length];
      if (field === "fps" && status.fps !== null) {
        events.push({ type: "self", field, value: status.fps, now });
      } else if (field === "dpu_ms" && status.dpu_ms !== null) {
        events.push({ type: "self", field, value: status.dpu_ms, now });
      } else if (field === "uptime") {
        events.push({ type: "self", field, value: Math.round(status.uptime_s / 60), now });
      }
    }
    this.memory.observe(events, now);
    this.engine.consider(events, this.memory, now);
    this.pump(now);
    return events;
  }

  /** Socket lifecycle: a closed socket loses the vision once (#21). */
  onConnection(open: boolean, now: number): CognitiveEvent[] {
    this.begin(now);
    if (!open) {
      return this.loseVision(now, "socket");
    }
    this.perception.reset(now, this.mode);
    return [];
  }

  /** 250 ms heartbeat: dwell/idle/pairs, expression decay, speech pump. */
  tick(now: number): CognitiveEvent[] {
    if (!this.started) {
      this.begin(now);
    }
    this.memory.rollover(now);
    const events = this.perception.tick(now);
    this.prepare(events, now);
    this.emit(events, now);
    this.pump(now);
    return events;
  }

  /** The voice finished (or aborted) the current sentence. */
  speechEnded(_now: number): void {
    this.engine.endSpeech();
    this.options.onExpression?.("neutral");
  }

  /** Interrupt the utterance in flight, e.g. the tab is hidden. */
  interrupt(): void {
    this.engine.interrupt();
    if (this.speechTimer !== null) {
      clearTimeout(this.speechTimer);
      this.speechTimer = null;
    }
    for (const timer of this.speechTimers) {
      clearTimeout(timer);
    }
    this.speechTimers.clear();
  }

  private begin(now: number): void {
    if (this.started) {
      return;
    }
    this.started = true;
    this.memory.rollover(now);
    this.perception.reset(now, this.mode);
  }

  private switchMode(model: WorldModelName, now: number): CognitiveEvent[] {
    // A mode switch must not look like an arrival: the scene keeps the person
    // it can still see and skips the greeting (F-E4 #22).
    const wasPresent =
      this.perception.world.people > 0 || this.perception.world.posePerson;
    this.mode = model;
    this.perception.reset(now, model);
    this.memory.setFocus({ mode: model, people: 0, crowd: false, present: [] });
    this.engine.clear();
    this.scene.reset(now, wasPresent);
    this.scene.silence(now);
    const event: CognitiveEvent = { type: "mode", model, now };
    this.lastEvent = event;
    return [event];
  }

  private loseVision(now: number, reason?: string): CognitiveEvent[] {
    if (this.visionLost) {
      return [];
    }
    this.visionLost = true;
    this.engine.clear();
    this.scene.reset(now);
    const event: CognitiveEvent = { type: "vision", state: "lost", reason, now };
    this.lastEvent = event;
    this.memory.observe([event], now);
    this.engine.consider([event], this.memory, now);
    this.options.onExpression?.("worried");
    this.pump(now);
    return [event];
  }

  private prepare(events: readonly CognitiveEvent[], now: number): void {
    if (events.length > 0) {
      this.lastEvent = events[events.length - 1];
    }
    this.memory.rollover(now);
    this.memory.observe(events, now);
    const world = this.perception.world;
    this.memory.setFocus({
      mode: this.mode,
      people: world.people,
      crowd: world.crowd,
      present: this.perception.presentLabels(),
    });
  }

  private emit(events: readonly CognitiveEvent[], now: number): void {
    this.scene.observe(events);
    if (events.length === 0 && this.started) {
      this.pump(now);
      return;
    }
    this.engine.consider(events, this.memory, now);
    this.pump(now);
  }

  private pump(now: number): void {
    this.memory.rollover(now);
    this.clock(now);
    this.offerScene(now);
    const utterance = this.engine.tick(now, this.memory);
    if (utterance !== null) {
      this.lastUtterance = utterance;
      this.scene.noteSpoken(utterance, this.memory, now);
      this.options.onExpression?.(utterance.expression);
      const handlers: SpeechHandlers = {
        onStart: () => this.options.onExpression?.(utterance.expression),
        onEnd: () => {
          this.engine.endSpeech();
          this.options.onExpression?.(utterance.expression);
        },
      };
      if (this.options.speak !== undefined) {
        this.options.speak(utterance.text, handlers);
      } else {
        handlers.onStart?.();
        // No voice wired (tests): end on the next macrotask.
        this.speechTimer = setTimeout(() => handlers.onEnd?.("unavailable"), 0);
      }
      this.options.onUtterance?.(utterance, this.mind);
    }
    this.options.onMind?.(this.mind);
  }

  /** Ask the scene machine what it wants to say; the engine decides when. */
  private offerScene(now: number): void {
    const world = this.perception.world;
    const line = this.scene.step(now, this.memory, {
      people: world.people,
      crowd: world.crowd,
      posePerson: world.posePerson,
      mode: this.mode,
    });
    if (line !== null) {
      this.engine.offerScene(line, this.memory, now);
    }
  }

  /** Clock greetings: one event at start and one per hour change. */
  private clock(now: number): void {
    const hour = new Date(now).getHours();
    if (hour === this.lastClockHour) {
      return;
    }
    this.lastClockHour = hour;
    this.engine.consider([{ type: "clock", hour, now }], this.memory, now);
  }
}
