/**
 * Client-side view state. The server is authoritative (FR-504); this store
 * only keeps the last snapshot and the bounded decision history the server
 * does not resend.
 */

import type {
  Decision,
  EpisodePayload,
  ErrorPayload,
  HandsPayload,
  HelloPayload,
  LogLevel,
  MedicalStatePayload,
  ModeProgressPayload,
  ServerMessage,
  StatusPayload,
  Tag,
} from "./protocol";

export const MAX_DECISIONS = 20;
export const MAX_LOGS = 50;

export interface DecisionEntry {
  seq: number;
  wall: string;
  receivedAt: number;
  decision: Decision;
}

export interface ErrorEntry {
  error: ErrorPayload;
  receivedAt: number;
}

export interface LogEntry {
  seq: number;
  wall: string;
  receivedAt: number;
  level: LogLevel;
  source: string;
  message: string;
}

export interface DashboardState {
  hello: HelloPayload | null;
  status: StatusPayload | null;
  medical: MedicalStatePayload | null;
  decisions: DecisionEntry[];
  logs: LogEntry[];
  episode: EpisodePayload | null;
  hands: HandsPayload | null;
  modeProgress: ModeProgressPayload | null;
  lastError: ErrorEntry | null;
  /** `seq` of a sent `mode.set` still waiting for its answer (IF-304). */
  pendingModeSeq: number | null;
  /** Rejection whose `ref_seq` answers the pending `mode.set`. */
  modeError: ErrorEntry | null;
  currentTag: Tag | null;
  episodeActive: boolean;
  episodeId: string | null;
  lastHandsAt: number;
  lastMessageAt: number;
}

export type StateRegion = "mode" | "live" | "feed" | "episode" | "hands" | "logs";

export class DashboardStore {
  readonly state: DashboardState = {
    hello: null,
    status: null,
    medical: null,
    decisions: [],
    logs: [],
    episode: null,
    hands: null,
    modeProgress: null,
    lastError: null,
    pendingModeSeq: null,
    modeError: null,
    currentTag: null,
    episodeActive: false,
    episodeId: null,
    lastHandsAt: 0,
    lastMessageAt: 0,
  };

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
    this.state.lastMessageAt = receivedAt;

    switch (message.type) {
      case "hello":
        this.state.hello = message.data;
        // A new connection gets the server's recent log history again; start
        // from it instead of mixing entries from the previous connection.
        this.state.logs = [];
        return ["mode", "logs"];

      case "status": {
        this.state.status = message.data;
        if (message.data.mode !== "medical") {
          return this.clearMedical(["mode", "live"]);
        }
        return ["mode"];
      }

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

      case "pong":
        return [];

      case "log": {
        const duplicate = this.state.logs.some((item) => item.seq === message.seq);
        if (!duplicate) {
          // Oldest first, bounded: the panel shows the newest MAX_LOGS lines.
          this.state.logs = [
            ...this.state.logs,
            {
              seq: message.seq,
              wall: message.wall,
              receivedAt,
              level: message.data.level,
              source: message.data.source,
              message: message.data.message,
            },
          ].slice(-MAX_LOGS);
        }
        return ["logs"];
      }

      case "medical.state": {
        this.state.medical = message.data;
        this.state.currentTag = message.data.tag;
        this.state.episodeActive = message.data.episode?.active ?? false;
        this.state.episodeId = message.data.episode?.episode_id ?? null;

        const regions: StateRegion[] = ["live"];
        if (message.data.last_decision !== null && this.state.decisions.length === 0) {
          this.state.decisions = [
            {
              seq: message.seq,
              wall: message.wall,
              receivedAt,
              decision: message.data.last_decision,
            },
          ];
          regions.push("feed");
        }
        return regions;
      }

      case "medical.decision": {
        const entry: DecisionEntry = {
          seq: message.seq,
          wall: message.wall,
          receivedAt,
          decision: message.data,
        };
        const duplicate = this.state.decisions.some((item) => item.seq === message.seq);
        if (!duplicate) {
          this.state.decisions = [entry, ...this.state.decisions].slice(0, MAX_DECISIONS);
        }
        this.state.currentTag = message.data.tag;
        return ["live", "feed"];
      }

      case "medical.episode": {
        this.state.episode = message.data;
        this.state.episodeActive = message.data.active;
        this.state.episodeId = message.data.episode_id;
        if (message.data.active) {
          return ["live", "episode"];
        }
        // A finished episode means the live skeleton is stale.
        return this.setHands(null, ["live", "episode"]);
      }

      case "medical.hands":
        this.state.hands = message.data;
        this.state.lastHandsAt = receivedAt;
        return ["hands"];
    }
  }

  /**
   * Forgets the last medical picture when the server leaves `medical` mode
   * (idle, lab, or a failed start). The IMU, tag and episode tiles and the
   * skeleton fall back to their "not active" copy until `medical.*` states
   * arrive again (N-02).
   */
  private clearMedical(regions: StateRegion[]): StateRegion[] {
    const hadEpisode =
      this.state.episode !== null ||
      this.state.episodeActive ||
      this.state.episodeId !== null;
    this.state.medical = null;
    this.state.currentTag = null;
    this.state.episode = null;
    this.state.episodeActive = false;
    this.state.episodeId = null;
    this.state.lastHandsAt = 0;
    return this.setHands(null, hadEpisode ? [...regions, "episode"] : regions);
  }

  private setHands(hands: HandsPayload | null, regions: StateRegion[]): StateRegion[] {
    const changed = this.state.hands !== null;
    this.state.hands = hands;
    return changed ? [...regions, "hands"] : regions;
  }
}
