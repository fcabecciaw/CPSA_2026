/**
 * Client-side view state of the hands dashboard. The server is authoritative
 * (FR-504); this store only keeps the last snapshot of each hand stream.
 *
 * Medical topics stay in the protocol, because the server may still publish
 * them in medical mode, and are ignored here: this dashboard shows only the
 * two hand demos.
 */

import type {
  ErrorPayload,
  HologramHandsPayload,
  Mode,
  ModeProgressPayload,
  RpsStatePayload,
  ServerMessage,
  ShowcaseStatePayload,
  StatusPayload,
} from "./protocol";

export interface ErrorEntry {
  error: ErrorPayload;
  receivedAt: number;
}

export interface DashboardState {
  status: StatusPayload | null;
  modeProgress: ModeProgressPayload | null;
  /** T5: the last `hologram.hands` frame of the hologram mode. */
  hologram: HologramHandsPayload | null;
  /** T5: the last `rps.state` of the Sasso-carta-forbice mode. */
  rps: RpsStatePayload | null;
  /** S3: the last `showcase.state` of the "Cosa vede l'IA" mode. */
  showcase: ShowcaseStatePayload | null;
  lastError: ErrorEntry | null;
  /** `seq` of a sent `mode.set` still waiting for its answer (IF-304). */
  pendingModeSeq: number | null;
  /** Rejection whose `ref_seq` answers the pending `mode.set`. */
  modeError: ErrorEntry | null;
}

/** The parts of `index.html` a message can change. */
export type StateRegion = "mode" | "hologram" | "rps" | "showcase";

export class DashboardStore {
  readonly state: DashboardState = {
    status: null,
    modeProgress: null,
    hologram: null,
    rps: null,
    showcase: null,
    lastError: null,
    pendingModeSeq: null,
    modeError: null,
  };

  /** The mode the server runs or is switching to, or `null` before `hello`. */
  get mode(): Mode | null {
    return this.state.status?.mode ?? this.state.modeProgress?.mode ?? null;
  }

  /**
   * Records the `seq` of a just-sent `mode.set`, so a later `error` can be
   * matched to it (IF-304). A previous rejection no longer applies.
   */
  markModeSent(seq: number): void {
    this.state.pendingModeSeq = seq;
    this.state.modeError = null;
    this.state.lastError = null;
  }

  /** Forgets a pending `mode.set`; used when the connection is replaced. */
  clearModeRequest(): void {
    this.state.pendingModeSeq = null;
    this.state.modeError = null;
  }

  apply(message: ServerMessage, receivedAt: number): StateRegion[] {
    switch (message.type) {
      case "status":
        this.state.status = message.data;
        // T5: the two hand streams belong to their own mode, so leaving it
        // drops the snapshot and the view falls back to its empty copy.
        return ["mode", ...this.dropModeStreams(message.data.mode)];

      case "mode.progress":
        this.state.modeProgress = message.data;
        if (message.data.step === "ready") {
          this.state.lastError = null;
          this.state.modeError = null;
          this.state.pendingModeSeq = null;
        }
        return ["mode"];

      case "error": {
        const entry: ErrorEntry = { error: message.data, receivedAt };
        this.state.lastError = entry;
        if (
          message.data.ref_seq !== null &&
          message.data.ref_seq === this.state.pendingModeSeq
        ) {
          this.state.modeError = entry;
          this.state.pendingModeSeq = null;
        }
        return ["mode"];
      }

      case "hologram.hands":
        this.state.hologram = message.data;
        return ["hologram"];

      case "rps.state":
        this.state.rps = message.data;
        return ["rps"];

      case "showcase.state":
        this.state.showcase = message.data;
        return ["showcase"];

      // `hello` only marks the connection; `pong` keeps the transport alive.
      // `log` and the `medical.*` topics are not shown by this dashboard.
      case "hello":
      case "pong":
      case "log":
      case "medical.state":
      case "medical.decision":
      case "medical.episode":
      case "medical.hands":
        return [];
    }
  }

  /**
   * T5: forgets the stream of a mode the server has left (hologram,
   * Sasso-carta-forbice, showcase), so a view never shows a frozen snapshot.
   */
  private dropModeStreams(mode: Mode): StateRegion[] {
    const regions: StateRegion[] = [];
    if (mode !== "hologram" && this.state.hologram !== null) {
      this.state.hologram = null;
      regions.push("hologram");
    }
    if (mode !== "rps" && this.state.rps !== null) {
      this.state.rps = null;
      regions.push("rps");
    }
    if (mode !== "showcase" && this.state.showcase !== null) {
      this.state.showcase = null;
      regions.push("showcase");
    }
    return regions;
  }
}
