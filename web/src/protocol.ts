/**
 * Protocol v1 types and parser, written by hand from
 * `docs/protocol/v1/*.schema.json`.
 *
 * Clients are stateless renderers (IF-504) and MUST ignore unknown message
 * types and unknown fields (IF-103). The parser therefore returns a tagged
 * result: known messages are rebuilt field by field, unknown types are
 * reported as `ignored`, and malformed payloads are reported as `invalid`
 * without taking the dashboard down.
 */

export const PROTOCOL_VERSION = 1 as const;

export const MODES = ["idle", "lab", "medical", "hologram", "rps"] as const;
export type Mode = (typeof MODES)[number];

export const MODE_STATES = ["stopped", "starting", "running", "stopping", "failed"] as const;
export type ModeState = (typeof MODE_STATES)[number];

export const MODE_STEPS = ["stopping", "releasing", "starting", "ready", "failed"] as const;
export type ModeStep = (typeof MODE_STEPS)[number];

export const IMU_STATES = ["n/a", "searching", "connected", "lost"] as const;
export type ImuState = (typeof IMU_STATES)[number];

/**
 * Activation source of the medical mode (IF-451 revision 2026-11-11):
 * `imu` is the wrist-IMU trigger, `vision` the camera-only one.
 */
export const TRIGGER_SOURCES = ["imu", "vision"] as const;
export type TriggerSource = (typeof TRIGGER_SOURCES)[number];

export const TAGS = ["NO_CLASS", "NON_DANGEROUS", "DANGEROUS", "NON_STEREOTIPY"] as const;
export type Tag = (typeof TAGS)[number];

export const VERDICTS = ["CONFIRMED", "REJECTED", "UNDECIDED"] as const;
export type Verdict = (typeof VERDICTS)[number];

export const DECISION_REASONS = [
  "CONFIRMED",
  "UNDECIDED",
  "TIMEOUT",
  "VERIFIER_ERROR",
  "REJECTED",
] as const;
export type DecisionReason = (typeof DECISION_REASONS)[number];

export const ERROR_CODES = [
  "camera_missing",
  "camera_error",
  "model_missing",
  "dpu_error",
  "imu_not_found",
  "asset_invalid",
  "busy",
  "bad_command",
  "internal",
] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

export const CAMERA_STATES = ["ok", "missing", "error"] as const;
export type CameraState = (typeof CAMERA_STATES)[number];

export const LOG_LEVELS = ["info", "warning", "error"] as const;
export type LogLevel = (typeof LOG_LEVELS)[number];

export const HAND_SIDES = ["left", "right"] as const;
export type HandSide = (typeof HAND_SIDES)[number];

export const FACE_REGIONS = [
  "eyes",
  "mouth",
  "nose_cheek",
  "forehead_temple",
  "ear",
  "head_top",
] as const;
export type FaceRegion = (typeof FACE_REGIONS)[number];

export const HAND_SHAPES = ["fist", "point_pinch", "open_palm", "unknown"] as const;
export type HandShape = (typeof HAND_SHAPES)[number];

export const EPISODE_LABELS = ["hand_to_face", "rocking", "hand_flapping"] as const;
export type EpisodeLabel = (typeof EPISODE_LABELS)[number];

export interface ModelInfo {
  id: string;
  name: string;
}

export interface HelloPayload {
  server_version: string;
  protocol_version: 1;
  capabilities: string[];
  modes: Mode[];
  models: ModelInfo[];
}

/**
 * F13 `status.perception.timing`: the verifier's rolling performance window.
 * Every stage is milliseconds (fps for the two rates) and may be `null` when
 * the estimator does not report it; `dropped` is a cumulative integer.
 */
export interface PerceptionTiming {
  analyzed_fps: number | null;
  camera_fps: number | null;
  decode_ms: number | null;
  preprocess_ms: number | null;
  dpu_ms: number | null;
  postprocess_ms: number | null;
  pose_ms: number | null;
  track_ms: number | null;
  latency_ms: number | null;
  dropped: number;
}

export interface Perception {
  fps: number | null;
  dpu_ms: number | null;
  camera: CameraState | null;
  model: string | null;
  /**
   * Optional since F13: an older server omits it, so `null` means "not sent",
   * not an error. The status bar then falls back to `fps` and `dpu_ms`.
   */
  timing: PerceptionTiming | null;
}

export interface StatusPayload {
  mode: Mode;
  mode_state: ModeState;
  perception: Perception;
  imu: ImuState;
  errors: ErrorCode[];
  uptime_s: number;
}

export interface ModeProgressPayload {
  mode: Mode;
  step: ModeStep;
  error: { code: ErrorCode; message: string } | null;
}

export interface ErrorPayload {
  code: ErrorCode;
  message: string;
  ref_seq: number | null;
}

export interface PongPayload {
  id: unknown;
}

export interface LogPayload {
  level: LogLevel;
  source: string;
  message: string;
}

export interface Actuation {
  actuator_id: string;
  kind: string | null;
}

export interface VisionInfo {
  verdict: Verdict;
  severity: Tag | null;
  episode_id: string | null;
  features: Record<string, number | null>;
}

export interface Decision {
  tag: Tag;
  tag_changed: boolean;
  verdict: Verdict | null;
  reason: DecisionReason | null;
  severity: Tag | null;
  vision: VisionInfo | null;
  actuations: Actuation[];
}

export interface EpisodeRef {
  episode_id: string;
  t_start: number;
  active: boolean;
}

export interface MedicalStatePayload {
  imu: ImuState;
  tag: Tag | null;
  episode: EpisodeRef | null;
  last_decision: Decision | null;
  /**
   * Activation source of the medical mode (IF-451 revision 2026-11-11, an
   * optional field). `null` means the server did not report it, as an older
   * server does; the dashboard then renders the `imu` display unchanged.
   */
  trigger: TriggerSource | null;
}

export interface Contact {
  side: HandSide;
  region: FaceRegion;
  t_start: number;
  duration_s: number;
  hand_shape: HandShape;
}

export interface EpisodePayload {
  episode_id: string;
  active: boolean;
  contact_count: number;
  repetition_hz: number | null;
  peak_severity: Tag | null;
  labels: EpisodeLabel[];
  contacts: Contact[];
}

export type Keypoint = [x: number, y: number, conf: number];

export interface HandsPayload {
  layout: string;
  w: number;
  h: number;
  keypoints: Keypoint[];
  subject_box: [number, number, number, number] | null;
  /**
   * F14: the analyzed frame sequence number this pose belongs to, the same
   * value the video stream sends as `X-Frame-Seq`. Additive and optional, so
   * an older server yields `null`.
   */
  seq: number | null;
}

/** MediaPipe hand21 layout: wrist, then four joints for each of the five fingers. */
export const HAND21_POINTS = 21;

/** A lifted joint in metres, IR camera frame (x right, y down, z forward). */
export type Point3D = [x: number, y: number, z: number];

export interface HologramHand {
  score: number;
  box: [number, number, number, number];
  /** hand21 joints in IR pixels: `[u, v, confidence]`. */
  kpts2d: Keypoint[];
  /** The same joints lifted to metres; `null` where the depth was invalid. */
  kpts3d: (Point3D | null)[];
}

export interface HologramTiming {
  dpu_ms: number;
  total_ms: number;
}

/** F-E `hologram.hands`: the hologram view's stream, at most 30 Hz. */
export interface HologramHandsPayload {
  seq: number;
  t: number;
  width: number;
  height: number;
  hands: HologramHand[];
  timing: HologramTiming;
}

export const RPS_PHASES = ["waiting", "ready", "countdown", "shoot", "result"] as const;
export type RpsPhase = (typeof RPS_PHASES)[number];

/** Hand classes of the classifier; `none` means no recognisable gesture. */
export const RPS_GESTURES = ["rock", "paper", "scissors", "none"] as const;
export type RpsGesture = (typeof RPS_GESTURES)[number];

/** What the board can play: always one of the three gestures. */
export const RPS_MOVES = ["rock", "paper", "scissors"] as const;
export type RpsMove = (typeof RPS_MOVES)[number];

export const RPS_OUTCOMES = ["win", "lose", "draw", "unclear"] as const;
export type RpsOutcome = (typeof RPS_OUTCOMES)[number];

export interface RpsScore {
  player: number;
  board: number;
  draws: number;
}

/** F-E `rps.state`: the game state machine, published on each change. */
export interface RpsStatePayload {
  phase: RpsPhase;
  /** 3, 2 or 1 during the countdown; `null` in every other phase. */
  count: number | null;
  player: RpsGesture | null;
  player_p: number | null;
  board: RpsMove | null;
  outcome: RpsOutcome | null;
  score: RpsScore;
  hand: { kpts2d: Keypoint[] } | null;
}

interface Envelope<T extends string, P> {
  v: typeof PROTOCOL_VERSION;
  type: T;
  seq: number;
  t: number;
  wall: string;
  data: P;
}

export type ServerMessage =
  | Envelope<"hello", HelloPayload>
  | Envelope<"status", StatusPayload>
  | Envelope<"mode.progress", ModeProgressPayload>
  | Envelope<"error", ErrorPayload>
  | Envelope<"pong", PongPayload>
  | Envelope<"log", LogPayload>
  | Envelope<"medical.state", MedicalStatePayload>
  | Envelope<"medical.decision", Decision>
  | Envelope<"medical.episode", EpisodePayload>
  | Envelope<"medical.hands", HandsPayload>
  | Envelope<"hologram.hands", HologramHandsPayload>
  | Envelope<"rps.state", RpsStatePayload>;

export type ServerMessageType = ServerMessage["type"];

export interface ClientEnvelope<T extends string, P> {
  v: typeof PROTOCOL_VERSION;
  type: T;
  /**
   * Optional per-connection command counter, starting at 1 and reset on every
   * new connection. The server echoes it as `error.ref_seq` (IF-304), so a
   * rejection can be matched to the command that caused it.
   */
  seq?: number;
  data: P;
}

/** A command as built by the client; `seq` is assigned when it is sent. */
export type ClientCommand =
  | Omit<ClientEnvelope<"ping", { id: number }>, "seq">
  | Omit<ClientEnvelope<"mode.set", { mode: Mode }>, "seq">;

export type ClientMessage =
  | ClientEnvelope<"ping", { id: number }>
  | ClientEnvelope<"mode.set", { mode: Mode }>;

export type ParseResult =
  | { kind: "message"; message: ServerMessage }
  | { kind: "ignored"; type: string }
  | { kind: "invalid"; reason: string };

class PayloadError extends Error {}

type JsonObject = Record<string, unknown>;

const KNOWN_TYPES: readonly ServerMessageType[] = [
  "hello",
  "status",
  "mode.progress",
  "error",
  "pong",
  "log",
  "medical.state",
  "medical.decision",
  "medical.episode",
  "medical.hands",
  "hologram.hands",
  "rps.state",
];

export function isServerMessageType(type: string): type is ServerMessageType {
  return (KNOWN_TYPES as readonly string[]).includes(type);
}

export function parseServerMessage(raw: string): ParseResult {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return { kind: "invalid", reason: "JSON non valido" };
  }

  let envelope: JsonObject;
  try {
    envelope = asObject(value, "messaggio");
  } catch (error) {
    return { kind: "invalid", reason: reasonOf(error) };
  }

  const type = envelope.type;
  if (typeof type !== "string") {
    return { kind: "invalid", reason: "campo type mancante o non stringa" };
  }
  if (!isServerMessageType(type)) {
    return { kind: "ignored", type };
  }
  if (envelope.v !== PROTOCOL_VERSION) {
    return { kind: "invalid", reason: `versione ${String(envelope.v)} non supportata` };
  }

  try {
    const seq = reqInteger(envelope, "seq", 1);
    const t = reqNumber(envelope, "t");
    const wall = reqString(envelope, "wall");
    const data = asObject(envelope.data, "data");

    switch (type) {
      case "hello":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseHello(data) });
      case "status":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseStatus(data) });
      case "mode.progress":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseModeProgress(data) });
      case "error":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseError(data) });
      case "pong":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: { id: data.id } });
      case "log":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseLog(data) });
      case "medical.state":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseMedicalState(data) });
      case "medical.decision":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseDecision(data) });
      case "medical.episode":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseEpisode(data) });
      case "medical.hands":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseHands(data) });
      case "hologram.hands":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseHologramHands(data) });
      case "rps.state":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseRpsState(data) });
    }
  } catch (error) {
    return { kind: "invalid", reason: `${type}: ${reasonOf(error)}` };
  }
}

function message(msg: ServerMessage): ParseResult {
  return { kind: "message", message: msg };
}

function reasonOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// --- Payload parsers -------------------------------------------------------

function parseHello(data: JsonObject): HelloPayload {
  const protocolVersion = reqInteger(data, "protocol_version", 1);
  if (protocolVersion !== PROTOCOL_VERSION) {
    throw new PayloadError("protocol_version non supportata");
  }
  return {
    server_version: reqString(data, "server_version"),
    protocol_version: PROTOCOL_VERSION,
    capabilities: reqArray(data, "capabilities", (value) => stringValue(value, "capabilities[]")),
    modes: reqArray(data, "modes", (value) => enumValue(value, MODES, "modes[]")),
    models: reqArray(data, "models", (value) => {
      const model = asObject(value, "models[]");
      return {
        id: reqString(model, "id"),
        name: reqString(model, "name"),
      };
    }),
  };
}

function parseStatus(data: JsonObject): StatusPayload {
  const perception = asObject(data.perception, "perception");
  return {
    mode: enumField(data, "mode", MODES),
    mode_state: enumField(data, "mode_state", MODE_STATES),
    perception: {
      fps: nullableNumberField(perception, "fps"),
      dpu_ms: nullableNumberField(perception, "dpu_ms"),
      camera: nullableEnumField(perception, "camera", CAMERA_STATES),
      model: nullableStringField(perception, "model"),
      timing: parseTiming(perception.timing),
    },
    imu: enumField(data, "imu", IMU_STATES),
    errors: reqArray(data, "errors", (value) => enumValue(value, ERROR_CODES, "errors[]")),
    uptime_s: reqNumber(data, "uptime_s"),
  };
}

/**
 * F13 `status.perception.timing`, parsed leniently on purpose: the field is
 * additive, so an absent value (older server), a non-object or a single bad
 * entry must never drop the whole `status` message. A bad entry becomes
 * `null` (rendered as "—"); a bad `dropped` becomes 0, since the contract
 * always carries an integer there.
 */
function parseTiming(value: unknown): PerceptionTiming | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const timing = value as JsonObject;
  const stage = (key: string): number | null => {
    const raw = timing[key];
    return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
  };
  const dropped = timing.dropped;
  return {
    analyzed_fps: stage("analyzed_fps"),
    camera_fps: stage("camera_fps"),
    decode_ms: stage("decode_ms"),
    preprocess_ms: stage("preprocess_ms"),
    dpu_ms: stage("dpu_ms"),
    postprocess_ms: stage("postprocess_ms"),
    pose_ms: stage("pose_ms"),
    track_ms: stage("track_ms"),
    latency_ms: stage("latency_ms"),
    dropped:
      typeof dropped === "number" && Number.isInteger(dropped) && dropped >= 0
        ? dropped
        : 0,
  };
}

function parseModeProgress(data: JsonObject): ModeProgressPayload {
  return {
    mode: enumField(data, "mode", MODES),
    step: enumField(data, "step", MODE_STEPS),
    error: optionalError(data.error),
  };
}

function parseError(data: JsonObject): ErrorPayload {
  return {
    code: enumField(data, "code", ERROR_CODES),
    message: reqString(data, "message"),
    ref_seq: optionalInteger(data.ref_seq, "ref_seq"),
  };
}

function parseLog(data: JsonObject): LogPayload {
  return {
    // Lenient: a level added by a newer server shows up as "info", not as a
    // dropped log line (R-03).
    level: enumFieldOr(data, "level", LOG_LEVELS, "info"),
    source: reqString(data, "source"),
    message: reqString(data, "message"),
  };
}

function parseMedicalState(data: JsonObject): MedicalStatePayload {
  return {
    imu: enumField(data, "imu", IMU_STATES),
    tag: nullableEnumField(data, "tag", TAGS),
    episode: data.episode == null ? null : parseEpisodeRef(asObject(data.episode, "episode")),
    last_decision: data.last_decision == null ? null : parseDecision(asObject(data.last_decision, "last_decision")),
    // Optional since the 2026-11-11 revision: an absent field is not an error.
    // Lenient on the value too: a trigger added by a newer server renders like
    // an absent one instead of freezing the dashboard (R-03).
    trigger: optionalEnumField(data, "trigger", TRIGGER_SOURCES),
  };
}

function parseEpisodeRef(data: JsonObject): EpisodeRef {
  return {
    episode_id: reqString(data, "episode_id"),
    t_start: reqNumber(data, "t_start"),
    active: reqBoolean(data, "active"),
  };
}

function parseDecision(data: JsonObject): Decision {
  return {
    tag: enumField(data, "tag", TAGS),
    tag_changed: reqBoolean(data, "tag_changed"),
    verdict: nullableEnumField(data, "verdict", VERDICTS),
    reason: nullableEnumField(data, "reason", DECISION_REASONS),
    severity: nullableEnumField(data, "severity", TAGS),
    vision: data.vision == null ? null : parseVision(asObject(data.vision, "vision")),
    actuations: reqArray(data, "actuations", (value) => {
      const actuation = asObject(value, "actuations[]");
      return {
        actuator_id: reqString(actuation, "actuator_id"),
        kind: nullableStringField(actuation, "kind"),
      };
    }),
  };
}

function parseVision(data: JsonObject): VisionInfo {
  return {
    verdict: enumField(data, "verdict", VERDICTS),
    severity: nullableEnumField(data, "severity", TAGS),
    episode_id: nullableStringField(data, "episode_id"),
    features: parseFeatures(data.features),
  };
}

function parseFeatures(value: unknown): Record<string, number | null> {
  const features = asObject(value, "vision.features");
  const parsed: Record<string, number | null> = {};
  for (const [key, feature] of Object.entries(features)) {
    if (feature === null) {
      parsed[key] = null;
    } else if (typeof feature === "number" && Number.isFinite(feature)) {
      parsed[key] = feature;
    } else {
      throw new PayloadError(`vision.features.${key}: atteso un numero o null`);
    }
  }
  return parsed;
}

function parseEpisode(data: JsonObject): EpisodePayload {
  return {
    episode_id: reqString(data, "episode_id"),
    active: reqBoolean(data, "active"),
    contact_count: reqInteger(data, "contact_count", 0),
    repetition_hz: nullableNumberField(data, "repetition_hz"),
    peak_severity: nullableEnumField(data, "peak_severity", TAGS),
    labels: reqArray(data, "labels", (value) => enumValue(value, EPISODE_LABELS, "labels[]")),
    contacts: reqArray(data, "contacts", (value) => {
      const contact = asObject(value, "contacts[]");
      return {
        side: enumField(contact, "side", HAND_SIDES),
        region: enumField(contact, "region", FACE_REGIONS),
        t_start: reqNumber(contact, "t_start"),
        duration_s: reqNumber(contact, "duration_s"),
        hand_shape: enumField(contact, "hand_shape", HAND_SHAPES),
      };
    }),
  };
}

function parseHands(data: JsonObject): HandsPayload {
  return {
    layout: reqString(data, "layout"),
    w: reqInteger(data, "w", 1),
    h: reqInteger(data, "h", 1),
    keypoints: reqArray(data, "keypoints", (value) => {
      if (!Array.isArray(value) || value.length < 3) {
        throw new PayloadError("keypoints[]: attesa una terna [x, y, conf]");
      }
      const x = numberValue(value[0], "keypoints[][0]");
      const y = numberValue(value[1], "keypoints[][1]");
      const conf = numberValue(value[2], "keypoints[][2]");
      return [x, y, conf];
    }),
    subject_box: parseSubjectBox(data.subject_box),
    // F14: additive frame counter; a value a newer server shapes differently
    // must not cost the pose, so anything but a non-negative integer is null.
    seq: optionalSequence(data.seq),
  };
}

function optionalSequence(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

/**
 * F-E `hologram.hands`. The 2D joints are load-bearing (the view needs them to
 * describe the hand), so a broken `[u, v, c]` rejects the message; a `kpts3d`
 * entry that is not a triple becomes `null`, the same value the lift uses for
 * a joint without depth.
 */
function parseHologramHands(data: JsonObject): HologramHandsPayload {
  const timing = asObject(data.timing, "timing");
  return {
    seq: reqInteger(data, "seq", 0),
    t: reqNumber(data, "t"),
    width: reqInteger(data, "width", 1),
    height: reqInteger(data, "height", 1),
    hands: reqArray(data, "hands", parseHologramHand),
    timing: {
      dpu_ms: reqNumber(timing, "dpu_ms"),
      total_ms: reqNumber(timing, "total_ms"),
    },
  };
}

function parseHologramHand(value: unknown): HologramHand {
  const hand = asObject(value, "hands[]");
  return {
    score: reqNumber(hand, "score"),
    box: parseBox(hand.box, "hands[].box"),
    kpts2d: parseKeypoints(hand.kpts2d, "hands[].kpts2d"),
    kpts3d: parsePoints3D(hand.kpts3d, "hands[].kpts3d"),
  };
}

function parseRpsState(data: JsonObject): RpsStatePayload {
  const score = asObject(data.score, "score");
  return {
    phase: enumField(data, "phase", RPS_PHASES),
    count: parseRpsCount(data.count),
    player: nullableEnumField(data, "player", RPS_GESTURES),
    player_p: nullableNumberField(data, "player_p"),
    board: nullableEnumField(data, "board", RPS_MOVES),
    outcome: nullableEnumField(data, "outcome", RPS_OUTCOMES),
    score: {
      player: reqInteger(score, "player", 0),
      board: reqInteger(score, "board", 0),
      draws: reqInteger(score, "draws", 0),
    },
    hand:
      data.hand == null
        ? null
        : { kpts2d: parseKeypoints(asObject(data.hand, "hand").kpts2d, "hand.kpts2d") },
  };
}

/** The countdown sends 3, 2, 1; every other phase sends `null`. */
function parseRpsCount(value: unknown): number | null {
  if (value == null) {
    return null;
  }
  if (value === 1 || value === 2 || value === 3) {
    return value;
  }
  throw new PayloadError(`count: atteso 1, 2, 3 o null, ricevuto ${String(value)}`);
}

function parseKeypoints(value: unknown, what: string): Keypoint[] {
  if (!Array.isArray(value)) {
    throw new PayloadError(`${what}: attesa una lista`);
  }
  return value.map((item) => {
    if (!Array.isArray(item) || item.length < 3) {
      throw new PayloadError(`${what}[]: attesa una terna [u, v, c]`);
    }
    return [
      numberValue(item[0], `${what}[][0]`),
      numberValue(item[1], `${what}[][1]`),
      numberValue(item[2], `${what}[][2]`),
    ];
  });
}

function parsePoints3D(value: unknown, what: string): (Point3D | null)[] {
  if (!Array.isArray(value)) {
    throw new PayloadError(`${what}: attesa una lista`);
  }
  return value.map((item) => {
    if (item == null) {
      return null;
    }
    if (!Array.isArray(item) || item.length < 3) {
      throw new PayloadError(`${what}[]: attesa una terna [x, y, z] o null`);
    }
    return [
      numberValue(item[0], `${what}[][0]`),
      numberValue(item[1], `${what}[][1]`),
      numberValue(item[2], `${what}[][2]`),
    ];
  });
}

function parseBox(value: unknown, what: string): [number, number, number, number] {
  if (!Array.isArray(value) || value.length < 4) {
    throw new PayloadError(`${what}: attesa una quaterna [x1, y1, x2, y2]`);
  }
  return [
    numberValue(value[0], `${what}[0]`),
    numberValue(value[1], `${what}[1]`),
    numberValue(value[2], `${what}[2]`),
    numberValue(value[3], `${what}[3]`),
  ];
}

function parseSubjectBox(value: unknown): [number, number, number, number] | null {
  if (value == null) {
    return null;
  }
  if (!Array.isArray(value) || value.length < 4) {
    throw new PayloadError("subject_box: attesa una quaterna [x1, y1, x2, y2]");
  }
  return [
    numberValue(value[0], "subject_box[0]"),
    numberValue(value[1], "subject_box[1]"),
    numberValue(value[2], "subject_box[2]"),
    numberValue(value[3], "subject_box[3]"),
  ];
}

function optionalError(value: unknown): { code: ErrorCode; message: string } | null {
  if (value == null) {
    return null;
  }
  const error = asObject(value, "mode.progress.error");
  return {
    code: enumField(error, "code", ERROR_CODES),
    message: reqString(error, "message"),
  };
}

// --- Primitive readers -----------------------------------------------------

function asObject(value: unknown, what: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new PayloadError(`${what}: atteso un oggetto`);
  }
  return value as JsonObject;
}

function stringValue(value: unknown, what: string): string {
  if (typeof value !== "string") {
    throw new PayloadError(`${what}: attesa una stringa`);
  }
  return value;
}

function numberValue(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new PayloadError(`${what}: atteso un numero finito`);
  }
  return value;
}

function enumValue<T extends string>(value: unknown, allowed: readonly T[], what: string): T {
  if (typeof value !== "string" || !allowed.includes(value as T)) {
    throw new PayloadError(`${what}: valore ${String(value)} non ammesso`);
  }
  return value as T;
}

function reqString(data: JsonObject, key: string): string {
  return stringValue(data[key], key);
}

function reqNumber(data: JsonObject, key: string): number {
  return numberValue(data[key], key);
}

function reqInteger(data: JsonObject, key: string, minimum: number): number {
  const value = reqNumber(data, key);
  if (!Number.isInteger(value) || value < minimum) {
    throw new PayloadError(`${key}: atteso un intero >= ${minimum}`);
  }
  return value;
}

function reqBoolean(data: JsonObject, key: string): boolean {
  const value = data[key];
  if (typeof value !== "boolean") {
    throw new PayloadError(`${key}: atteso un booleano`);
  }
  return value;
}

function reqArray<T>(data: JsonObject, key: string, parseItem: (value: unknown) => T): T[] {
  const value = data[key];
  if (!Array.isArray(value)) {
    throw new PayloadError(`${key}: attesa una lista`);
  }
  return value.map((item) => parseItem(item));
}

function nullableStringField(data: JsonObject, key: string): string | null {
  const value = data[key];
  return value == null ? null : stringValue(value, key);
}

function nullableNumberField(data: JsonObject, key: string): number | null {
  const value = data[key];
  return value == null ? null : numberValue(value, key);
}

function optionalInteger(value: unknown, what: string): number | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new PayloadError(`${what}: atteso un intero o null`);
  }
  return value;
}

function enumField<T extends string>(data: JsonObject, key: string, allowed: readonly T[]): T {
  return enumValue(data[key], allowed, key);
}

function nullableEnumField<T extends string>(
  data: JsonObject,
  key: string,
  allowed: readonly T[],
): T | null {
  const value = data[key];
  return value == null ? null : enumValue(value, allowed, key);
}

/**
 * Lenient variant for a value a newer server may extend: an absent, non-string
 * or unlisted value becomes `null` instead of rejecting the whole message, so
 * version skew does not freeze the dashboard (R-03). Strict fields keep
 * `enumField`, which is what makes a malformed load-bearing field visible.
 */
function optionalEnumField<T extends string>(
  data: JsonObject,
  key: string,
  allowed: readonly T[],
): T | null {
  const value = data[key];
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : null;
}

/**
 * Lenient on the value only: a level a newer server adds keeps the message
 * with `fallback`; an absent or non-string value stays an error, so a
 * malformed required field is still visible (R-03).
 */
function enumFieldOr<T extends string>(
  data: JsonObject,
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const value = data[key];
  if (typeof value !== "string") {
    throw new PayloadError(`${key}: attesa una stringa, ricevuto ${String(value)}`);
  }
  return (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}
