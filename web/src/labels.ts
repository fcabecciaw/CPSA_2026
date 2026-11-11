/**
 * Italian UI copy and formatters. TECHNICAL enum names stay in the protocol
 * and in `code` elements; everything an onlooker reads is Italian
 * (PRODUCT.md, Brand Commitments).
 */

import type {
  Actuation,
  CameraState,
  ErrorCode,
  FaceRegion,
  HandShape,
  HandSide,
  ImuState,
  LogLevel,
  Mode,
  ModeState,
  ModeStep,
  RpsGesture,
  RpsOutcome,
  RpsPhase,
  Tag,
  Verdict,
  DecisionReason,
  EpisodeLabel,
} from "./protocol";

export const TAG_LABEL: Record<Tag, string> = {
  NO_CLASS: "Nessuna classe",
  NON_DANGEROUS: "Non pericoloso",
  DANGEROUS: "Pericoloso",
  NON_STEREOTIPY: "Non stereotipia",
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  CONFIRMED: "Confermato",
  REJECTED: "Rifiutato",
  UNDECIDED: "Incerto",
};

export const REASON_LABEL: Record<DecisionReason, string> = {
  CONFIRMED: "Confermato",
  REJECTED: "Rifiutato",
  UNDECIDED: "Incerto",
  TIMEOUT: "Tempo scaduto",
  VERIFIER_ERROR: "Errore del verificatore",
};

export const IMU_LABEL: Record<ImuState, string> = {
  "n/a": "Non disponibile",
  searching: "Ricerca",
  connected: "Connesso",
  lost: "Perso",
};

/**
 * Live-tile copy for the medical activation source (`medical.state.trigger`,
 * IF-451 revision 2026-11-11). With `imu` the live tiles are unchanged, so
 * only the camera-only variant has copy.
 */
export const TRIGGER_TILE_LABEL = "Attivazione";
export const TRIGGER_VISION_LABEL = "telecamera";

/** F11: with the camera-only trigger `imu` is "n/a"; it is not an error. */
export const IMU_UNUSED_LABEL = "Non usato (solo telecamera)";

export const MODE_LABEL: Record<Mode, string> = {
  idle: "Idle",
  lab: "Lab",
  medical: "Medicale",
  hologram: "Ologramma",
  rps: "Sasso-carta-forbice",
};

/** T5: the two hand-demo views. */
export const RPS_LABELS = {
  waiting: "Mostra la mano per iniziare",
  ready: "Preparati…",
  shoot: "Via!",
  versus: "contro",
  player: "Tu",
  board: "Tavolo",
  draws: "Pareggi",
  preview: "Anteprima della mano",
} as const;

/** "Sasso… carta… forbice!", one word for each countdown step (3, 2, 1). */
export const RPS_COUNTDOWN_WORDS: readonly string[] = ["Sasso…", "carta…", "forbice!"];

export const RPS_PHASE_LABEL: Record<RpsPhase, string> = {
  waiting: "in attesa",
  ready: "pronto",
  countdown: "conto alla rovescia",
  shoot: "via",
  result: "risultato",
};

export const RPS_GESTURE_LABEL: Record<RpsGesture, string> = {
  rock: "sasso",
  paper: "carta",
  scissors: "forbice",
  none: "gesto non chiaro",
};

/** Paired with the label, never the only cue. */
export const RPS_GESTURE_ICON: Record<RpsGesture, string> = {
  rock: "✊",
  paper: "✋",
  scissors: "✌",
  none: "·",
};

export const RPS_OUTCOME_LABEL: Record<RpsOutcome, string> = {
  win: "Hai vinto!",
  lose: "Vince il tavolo",
  draw: "Pareggio",
  unclear: "Gesto poco chiaro",
};

export const MODE_STATE_LABEL: Record<ModeState, string> = {
  stopped: "ferma",
  starting: "avvio",
  running: "attiva",
  stopping: "arresto",
  failed: "errore",
};

export const MODE_STEP_LABEL: Record<ModeStep, string> = {
  stopping: "arresto della modalità precedente",
  releasing: "rilascio delle risorse",
  starting: "avvio",
  ready: "pronta",
  failed: "fallito",
};

export const CAMERA_LABEL: Record<CameraState, string> = {
  ok: "ok",
  missing: "assente",
  error: "errore",
};

export const LOG_LEVEL_LABEL: Record<LogLevel, string> = {
  info: "info",
  warning: "avviso",
  error: "errore",
};

/** Icons are paired with the level text, never used as the only cue. */
export const LOG_LEVEL_ICON: Record<LogLevel, string> = {
  info: "i",
  warning: "!",
  error: "×",
};

export const ERROR_LABEL: Record<ErrorCode, string> = {
  camera_missing: "telecamera assente",
  camera_error: "errore telecamera",
  model_missing: "modello assente",
  dpu_error: "errore DPU",
  imu_not_found: "IMU non trovata",
  asset_invalid: "asset non valido",
  busy: "sistema occupato",
  bad_command: "comando non valido",
  internal: "errore interno",
};

export const REGION_LABEL: Record<FaceRegion, string> = {
  eyes: "occhi",
  mouth: "bocca",
  nose_cheek: "naso e guancia",
  forehead_temple: "fronte e tempia",
  ear: "orecchio",
  head_top: "sommità della testa",
};

export const SIDE_LABEL: Record<HandSide, string> = {
  left: "sinistra",
  right: "destra",
};

export const HAND_SHAPE_LABEL: Record<HandShape, string> = {
  fist: "pugno",
  point_pinch: "pinza",
  open_palm: "palmo aperto",
  unknown: "sconosciuta",
};

export const EPISODE_LABEL: Record<EpisodeLabel, string> = {
  hand_to_face: "mano al viso",
  rocking: "dondolamento",
  hand_flapping: "sventolio",
};

export function tagLabel(tag: Tag | null): string {
  return tag === null ? "Nessun tag" : TAG_LABEL[tag];
}

export function tagClass(tag: Tag | null): string {
  if (tag === null) {
    return "tag-none";
  }
  return `tag-${tag.toLowerCase().replaceAll("_", "-")}`;
}

export function verdictLabel(verdict: Verdict | null): string {
  return verdict === null ? "Nessun verdetto" : VERDICT_LABEL[verdict];
}

export function reasonLabel(reason: DecisionReason | null): string {
  return reason === null ? "—" : REASON_LABEL[reason];
}

export function actuatorText(actuations: Actuation[]): string {
  if (actuations.length === 0) {
    return "nessuna";
  }
  return actuations
    .map((actuation) =>
      actuation.kind === null || actuation.kind === actuation.actuator_id
        ? actuation.actuator_id
        : `${actuation.actuator_id} (${actuation.kind})`,
    )
    .join(", ");
}

export function formatClock(wall: string): string {
  const date = new Date(wall);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return CLOCK_FORMAT.format(date);
}

const CLOCK_FORMAT = new Intl.DateTimeFormat("it-IT", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

const INTEGER_FORMAT = new Intl.NumberFormat("it-IT", {
  maximumFractionDigits: 0,
});

const ONE_DECIMAL_FORMAT = new Intl.NumberFormat("it-IT", {
  maximumFractionDigits: 1,
});

const TWO_DECIMAL_FORMAT = new Intl.NumberFormat("it-IT", {
  maximumFractionDigits: 2,
});

export function formatNumber(value: number, digits = 1): string {
  const format =
    digits === 0
      ? INTEGER_FORMAT
      : digits >= 2
        ? TWO_DECIMAL_FORMAT
        : ONE_DECIMAL_FORMAT;
  return format.format(value);
}

export function formatHz(value: number | null): string {
  return value === null ? "—" : `${formatNumber(value)} Hz`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }
  return `${formatNumber(seconds)} s`;
}

export function formatUptime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) {
    return `${total} s`;
  }
  const minutes = Math.floor(total / 60);
  if (minutes < 60) {
    const rest = total % 60;
    return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
  }
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest === 0 ? `${hours} h` : `${hours} h ${rest} min`;
}

export function errorCodeText(code: ErrorCode): string {
  return ERROR_LABEL[code];
}

export const STATUS_LABELS = {
  camera: "Telecamera",
  model: "Modello",
  errors: "Errori",
  fps: "FPS",
  dpu: "DPU",
  uptime: "Attivo da",
  timing: "Tempi",
} as const;

/**
 * F13 stage labels of the compact "Tempi" line. Technical keys stay in the
 * protocol; onlookers read Italian.
 */
export const TIMING_LABELS = {
  decode: "Decodifica",
  preprocess: "Pre",
  dpu: "DPU",
  postprocess: "Post",
  track: "Tracking",
  latency: "Latenza",
  dropped: "Scartati",
} as const;
