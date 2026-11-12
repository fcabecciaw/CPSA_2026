/**
 * Italian UI copy and formatters of the hands dashboard.
 *
 * Every word an onlooker reads lives here; the protocol enum names stay in
 * `protocol.ts` (PRODUCT.md: the booth UI is Italian).
 */

import type { ErrorCode, Mode, ModeState, RpsGesture, RpsOutcome } from "./protocol";

export const APP_LABEL = "CPSA";

export const MODE_LABEL: Record<Mode, string> = {
  idle: "Idle",
  lab: "Lab",
  medical: "Medicale",
  hologram: "Ologramma",
  rps: "Sasso-carta-forbice",
};

export const MODE_STATE_LABEL: Record<ModeState, string> = {
  stopped: "ferma",
  starting: "avvio",
  running: "attiva",
  stopping: "arresto",
  failed: "errore",
};

export const CONNECTION_LABEL = {
  connecting: "Connessione…",
  open: "Connesso",
  reconnecting: "Riconnessione…",
  closed: "Disconnesso",
} as const;

export const CAMERA_LABELS = {
  title: "Telecamera infrarossa",
  waiting: "In attesa della telecamera…",
  unavailable: "Video non disponibile",
} as const;

export const VIEW_LABELS = {
  hologramTitle: "Ologramma 3D della mano",
  hologramEmpty: "In attesa della mano…",
  rpsTitle: "Sasso, carta, forbice",
  rpsEmpty: "Mano non visibile",
  idleTitle: "Nessuna demo attiva",
  idleBody: "Scegli Ologramma o Sasso-carta-forbice nella barra in alto.",
} as const;

/** T5: copy of the Sasso-carta-forbice panel. */
export const RPS_LABELS = {
  waiting: "Mostra la mano per iniziare",
  ready: "Preparati…",
  shoot: "Via!",
  versus: "contro",
  player: "Tu",
  board: "Tavolo",
  draws: "Pareggi",
} as const;

/** "Sasso… carta… forbice!", one word for each countdown step (3, 2, 1). */
export const RPS_COUNTDOWN_WORDS: readonly string[] = ["Sasso…", "carta…", "forbice!"];

export const RPS_GESTURE_LABEL: Record<RpsGesture, string> = {
  rock: "sasso",
  paper: "carta",
  scissors: "forbice",
  none: "gesto non chiaro",
};

export const RPS_OUTCOME_LABEL: Record<RpsOutcome, string> = {
  win: "Hai vinto!",
  lose: "Vince il tavolo",
  draw: "Pareggio",
  unclear: "Gesto poco chiaro",
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

export const NOTICE_LABEL = {
  rejected: "Comando rifiutato",
} as const;

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
