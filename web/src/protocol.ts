/**
 * Protocol v1 types and parser of the "Cosa vede l'IA" dashboard, written by
 * hand from `docs/protocol/v1/*.schema.json` (F-K1).
 *
 * The dashboard is a stateless renderer (IF-504): it sends `ping` and nothing
 * else. Clients MUST ignore unknown message types and unknown fields (IF-103),
 * so the parser returns a tagged result: known messages are rebuilt field by
 * field, unknown types are reported as `ignored`, and malformed payloads are
 * reported as `invalid` without taking the dashboard down.
 */

export const PROTOCOL_VERSION = 1 as const;

/** The lifecycle of the one process on the board (F-K1 `status.state`). */
export const RUN_STATES = ["starting", "running", "error"] as const;
export type RunState = (typeof RUN_STATES)[number];

export const CAMERA_STATES = ["ok", "missing", "error"] as const;
export type CameraState = (typeof CAMERA_STATES)[number];

export interface HelloPayload {
  server_version: string;
  protocol_version: 1;
}

/**
 * F-K1 `status`, published at 1 Hz and on change. `fps` and `dpu_ms` are the
 * detector's own rate and the last inference, `null` while it is not measuring;
 * `errors` are the codes of the conditions the process reports.
 */
export interface StatusPayload {
  state: RunState;
  camera: CameraState;
  model: string;
  fps: number | null;
  dpu_ms: number | null;
  errors: string[];
  uptime_s: number;
}

export interface ErrorPayload {
  code: string;
  message: string;
  ref_seq: number | null;
}

export interface PongPayload {
  id: unknown;
}

/** One object of `showcase.state`: a COCO-80 label and its box in camera pixels. */
export interface ShowcaseObject {
  label: string;
  class_id: number;
  score: number;
  box: [number, number, number, number];
}

/** How many objects of one label the detector sees in the current frame. */
export interface ShowcaseCount {
  label: string;
  count: number;
}

/**
 * F-S2 `showcase.state`: what the AI model sees on the RGB camera, at most
 * 10 Hz. `fps` is the processing rate, `dpu_ms` the last inference and
 * `total_ms` the whole chain; `objects` is score-descending (at most 20) and
 * `counts` is count-descending, then by label.
 */
export interface ShowcaseStatePayload {
  model: string;
  width: number;
  height: number;
  fps: number | null;
  dpu_ms: number | null;
  total_ms: number | null;
  objects: ShowcaseObject[];
  counts: ShowcaseCount[];
}

/**
 * F-E1 `pose.state`: the MoveNet skeleton while that model is active, at most
 * 30 Hz. Keypoints are source-pixel `[x, y, score]` in COCO-17 order; `score`
 * is their mean and is below 0.2 when nobody is in the frame.
 */
export interface PoseStatePayload {
  model: string;
  width: number;
  height: number;
  fps: number | null;
  dpu_ms: number | null;
  total_ms: number | null;
  score: number;
  keypoints: Array<[number, number, number]>;
}

interface Envelope<T extends string, P> {
  v: typeof PROTOCOL_VERSION;
  type: T;
  seq: number;
  t: number;
  wall: string;
  data: P;
}

/** Everything the server may publish (F-K1). */
export type ServerMessage =
  | Envelope<"hello", HelloPayload>
  | Envelope<"status", StatusPayload>
  | Envelope<"showcase.state", ShowcaseStatePayload>
  | Envelope<"pose.state", PoseStatePayload>
  | Envelope<"error", ErrorPayload>
  | Envelope<"pong", PongPayload>;

export type ServerMessageType = ServerMessage["type"];

export interface ClientEnvelope<T extends string, P> {
  v: typeof PROTOCOL_VERSION;
  type: T;
  /**
   * Optional per-connection command counter, starting at 1 and reset on every
   * new connection. The server echoes it as `error.ref_seq` (IF-304).
   */
  seq?: number;
  data: P;
}

/** F-K1: `ping` is the only command this dashboard sends. */
export type ClientCommand = Omit<ClientEnvelope<"ping", { id: number }>, "seq">;

export type ClientMessage = ClientEnvelope<"ping", { id: number }>;

export type ParseResult =
  | { kind: "message"; message: ServerMessage }
  | { kind: "ignored"; type: string }
  | { kind: "invalid"; reason: string };

class PayloadError extends Error {}

type JsonObject = Record<string, unknown>;

const KNOWN_TYPES: readonly ServerMessageType[] = [
  "hello",
  "status",
  "showcase.state",
  "pose.state",
  "error",
  "pong",
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
      case "showcase.state":
        return message({
          v: PROTOCOL_VERSION,
          type,
          seq,
          t,
          wall,
          data: parseShowcaseState(data),
        });
      case "pose.state":
        return message({
          v: PROTOCOL_VERSION,
          type,
          seq,
          t,
          wall,
          data: parsePoseState(data),
        });
      case "error":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: parseError(data) });
      case "pong":
        return message({ v: PROTOCOL_VERSION, type, seq, t, wall, data: { id: data.id } });
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
  };
}

function parseStatus(data: JsonObject): StatusPayload {
  return {
    state: enumField(data, "state", RUN_STATES),
    camera: enumField(data, "camera", CAMERA_STATES),
    model: reqString(data, "model"),
    fps: nullableNumberField(data, "fps"),
    dpu_ms: nullableNumberField(data, "dpu_ms"),
    errors: reqArray(data, "errors", (value) => stringValue(value, "errors[]")),
    uptime_s: reqNumber(data, "uptime_s"),
  };
}

function parseError(data: JsonObject): ErrorPayload {
  return {
    code: reqString(data, "code"),
    message: reqString(data, "message"),
    ref_seq: optionalInteger(data.ref_seq, "ref_seq"),
  };
}

/**
 * F-S2 `showcase.state`. The model name, the frame size and the two lists are
 * load-bearing (the panel is built from them); the three timings are nullable,
 * because a frame that failed to time still describes what was seen.
 */
function parseShowcaseState(data: JsonObject): ShowcaseStatePayload {
  return {
    model: reqString(data, "model"),
    width: reqInteger(data, "width", 1),
    height: reqInteger(data, "height", 1),
    fps: nullableNumberField(data, "fps"),
    dpu_ms: nullableNumberField(data, "dpu_ms"),
    total_ms: nullableNumberField(data, "total_ms"),
    objects: reqArray(data, "objects", (value) => {
      const object = asObject(value, "objects[]");
      return {
        label: reqString(object, "label"),
        class_id: reqInteger(object, "class_id", 0),
        score: reqNumber(object, "score"),
        box: parseBox(object.box, "objects[].box"),
      };
    }),
    counts: reqArray(data, "counts", (value) => {
      const count = asObject(value, "counts[]");
      return {
        label: reqString(count, "label"),
        count: reqInteger(count, "count", 0),
      };
    }),
  };
}

/**
 * F-E1 `pose.state`. The keypoints are load-bearing for the gaze and the pose
 * rules; exactly 17 triplets are required, and a malformed one invalidates the
 * message instead of taking the dashboard down (IF-103).
 */
function parsePoseState(data: JsonObject): PoseStatePayload {
  const keypoints = reqArray(data, "keypoints", (value) => {
    if (!Array.isArray(value) || value.length < 3) {
      throw new PayloadError("keypoints[]: attesa una terna [x, y, score]");
    }
    return [
      numberValue(value[0], "keypoints[][0]"),
      numberValue(value[1], "keypoints[][1]"),
      numberValue(value[2], "keypoints[][2]"),
    ] as [number, number, number];
  });
  if (keypoints.length !== 17) {
    throw new PayloadError(`keypoints: attesi 17 punti, ricevuti ${keypoints.length}`);
  }
  return {
    model: reqString(data, "model"),
    width: reqInteger(data, "width", 1),
    height: reqInteger(data, "height", 1),
    fps: nullableNumberField(data, "fps"),
    dpu_ms: nullableNumberField(data, "dpu_ms"),
    total_ms: nullableNumberField(data, "total_ms"),
    score: reqNumber(data, "score"),
    keypoints,
  };
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

function reqArray<T>(data: JsonObject, key: string, parseItem: (value: unknown) => T): T[] {
  const value = data[key];
  if (!Array.isArray(value)) {
    throw new PayloadError(`${key}: attesa una lista`);
  }
  return value.map((item) => parseItem(item));
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
