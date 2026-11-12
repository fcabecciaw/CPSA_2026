/**
 * The perception events the rules react to (F-E3 "Events").
 *
 * Events are immutable facts with the timestamp they were observed at: the
 * production-rule engine only reads them, never the raw detections. Counts are
 * already settled by `perception.ts` (F-E4 #1/#3), so a rule can trust
 * `count.to` to be the final stable number of people.
 */

/** The model the board is streaming right now (F-E1). */
export type WorldModelName = "yolox" | "movenet";

/** Gestures confirmed by `perception.ts` from `pose.state` (F-E4 #25). */
export type PoseKind =
  | "arms_up"
  | "wave"
  | "t_pose"
  | "crouch"
  | "hands_on_head"
  | "lean";

/**
 * Face cues confirmed by the laptop browser (MediaPipe Face Landmarker): they
 * never come from the board, only from `/video.mjpeg` sampled in the dashboard.
 */
export type FaceCueKind =
  | "smile"
  | "mouth_open"
  | "nod"
  | "head_shake"
  | "looking_at_me";

/** The four scenes of the dialogue (`scene.ts`). */
export type SceneName = "attract" | "engage" | "play" | "farewell";

/** What a scene line is for, shown in the "mente" strip and used by the tests. */
export type SceneLineKind =
  | "fact"
  | "invitation"
  | "greeting"
  | "intro"
  | "prompt"
  | "reaction"
  | "goodbye"
  | "question"
  | "answer"
  | "continuity";

interface EventBase {
  /** Monotonic milliseconds of the dashboard clock. */
  now: number;
}

export type CognitiveEvent =
  | (EventBase & {
      type: "appear";
      label: string;
      count: number;
      score: number;
      /** Box area as a fraction of the frame (F-E4 #10). */
      area: number;
      /** Present since start/reconnect or for over two minutes (F-E4 #9). */
      scenery: boolean;
      /** Implausible indoor class confirmed with a high score (F-E4 #10). */
      doubtful: boolean;
    })
  | (EventBase & { type: "disappear"; label: string; absentS: number })
  | (EventBase & {
      type: "count";
      from: number;
      to: number;
      /** Returned within 10–30 s: no new greeting (F-E4 #4). */
      quiet: boolean;
      /** Absence before a return with >= 1 person, `null` otherwise. */
      returnAfterS: number | null;
    })
  | (EventBase & { type: "return"; label: string; absenceS: number; count: number })
  | (EventBase & { type: "dwell"; label: string; seconds: number; count: number })
  | (EventBase & {
      type: "together";
      a: string;
      b: string;
      countA: number;
      countB: number;
    })
  | (EventBase & { type: "idle"; seconds: number })
  | (EventBase & { type: "milestone"; name: string; value: number })
  | (EventBase & { type: "pose"; kind: PoseKind; score: number })
  | (EventBase & {
      type: "face";
      kind: FaceCueKind;
      /** Blendshape or head-pose confidence of the cue, [0, 1]. */
      score: number;
      /** Faces in the frame when the cue was confirmed. */
      faces: number;
    })
  | (EventBase & {
      type: "scene";
      scene: SceneName;
      kind: SceneLineKind;
      /** The sentence, already rendered by `scene.ts`. */
      text: string;
    })
  | (EventBase & { type: "mode"; model: WorldModelName })
  | (EventBase & { type: "clock"; hour: number })
  | (EventBase & {
      type: "self";
      field: "fps" | "dpu_ms" | "uptime";
      value: number;
    })
  | (EventBase & { type: "vision"; state: "lost" | "back"; reason?: string });

export type CognitiveEventType = CognitiveEvent["type"];

/**
 * Key of the object a rule talks about, for `per_object_cooldown_s`; group and
 * world events have none.
 */
export function objectKeyOf(event: CognitiveEvent): string | null {
  switch (event.type) {
    case "appear":
    case "disappear":
    case "return":
    case "dwell":
      return event.label;
    case "together":
      return `${event.a}+${event.b}`;
    case "count":
      return "persona";
    default:
      return null;
  }
}

/** Class of an event, when there is one, for habituation bookkeeping. */
export function classOf(event: CognitiveEvent): string | null {
  return objectKeyOf(event) === "persona" ? "persona" : objectKeyOf(event);
}

/** Italian names of the face cues, for the "mente" strip. */
const FACE_CUE_LABEL: Record<FaceCueKind, string> = {
  smile: "sorriso",
  mouth_open: "bocca aperta",
  nod: "cenno sì",
  head_shake: "cenno no",
  looking_at_me: "mi guarda",
};

/** One short line for the "mente" strip. */
export function describeEvent(event: CognitiveEvent): string {
  switch (event.type) {
    case "appear":
      return `appare ${event.label} (${event.count})`;
    case "disappear":
      return `sparisce ${event.label}`;
    case "count":
      return `persone ${event.from} → ${event.to}`;
    case "return":
      return `ritorna ${event.label}`;
    case "dwell":
      return `${event.label} da ${Math.round(event.seconds)} s`;
    case "together":
      return `insieme: ${event.a} + ${event.b}`;
    case "idle":
      return `nessuno da ${Math.round(event.seconds)} s`;
    case "milestone":
      return `traguardo: ${event.name}`;
    case "pose":
      return `postura: ${event.kind}`;
    case "face":
      return `volto: ${FACE_CUE_LABEL[event.kind]}`;
    case "scene":
      return `scena ${event.scene}: ${event.kind}`;
    case "mode":
      return `modello: ${event.model}`;
    case "clock":
      return `ore ${event.hour}`;
    case "self":
      return `${event.field}: ${event.value}`;
    case "vision":
      return event.state === "lost" ? "visione persa" : "visione tornata";
  }
}
