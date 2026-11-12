/**
 * Italian UI copy and formatters of the "Cosa vede l'IA" dashboard.
 *
 * Every word an onlooker reads lives here; the protocol enum names stay in
 * `protocol.ts` (PRODUCT.md: the booth UI is Italian).
 */

export const APP_LABEL = "CPSA";

export const CONNECTION_LABEL = {
  connecting: "Connessione…",
  open: "Connesso",
  reconnecting: "Riconnessione…",
  closed: "Disconnesso",
} as const;

/** One camera, the RGB one the detector reads. */
export const CAMERA_LABELS = {
  title: "Telecamera RGB",
  alt: "Video della telecamera RGB",
  badge: "RGB",
  waiting: "In attesa della telecamera…",
  unavailable: "Video non disponibile",
} as const;

/** Copy of the "Cosa vede l'IA" panel: the counts and the detection list. */
export const SHOWCASE_LABELS = {
  countsLabel: "Oggetti nell'inquadratura",
  objectsLabel: "Rilevamenti più sicuri",
} as const;

/** The error codes the process reports; an unknown code shows as itself. */
export const ERROR_LABEL: Record<string, string> = {
  camera_missing: "telecamera assente",
  camera_error: "errore telecamera",
  model_missing: "modello assente",
  dpu_error: "errore DPU",
  asset_invalid: "asset non valido",
  busy: "sistema occupato",
  bad_command: "comando non valido",
  internal: "errore interno",
};

export function errorLabel(code: string): string {
  return ERROR_LABEL[code] ?? code;
}

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
