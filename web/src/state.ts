/**
 * Client-side view state of the "Cosa vede l'IA" dashboard. The server is
 * authoritative (FR-504); this store only keeps the last snapshot of the two
 * topics the dashboard renders.
 */

import type {
  ErrorPayload,
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
  /** S3: the last `showcase.state` of the detector. */
  showcase: ShowcaseStatePayload | null;
  lastError: ErrorEntry | null;
}

/** The parts of `index.html` a message can change. */
export type StateRegion = "bar" | "showcase";

export class DashboardStore {
  readonly state: DashboardState = {
    status: null,
    showcase: null,
    lastError: null,
  };

  apply(message: ServerMessage, receivedAt: number): StateRegion[] {
    switch (message.type) {
      case "status":
        this.state.status = message.data;
        // A process that is not running streams nothing: keeping its last
        // snapshot would leave a frozen picture in the pane.
        if (message.data.state !== "running" && this.state.showcase !== null) {
          this.state.showcase = null;
          return ["bar", "showcase"];
        }
        return ["bar"];

      case "showcase.state":
        this.state.showcase = message.data;
        return ["showcase"];

      case "pose.state":
        // The cognitive layer consumes pose frames directly; the stateless
        // renderer keeps no skeleton snapshot of its own.
        return [];

      case "error":
        this.state.lastError = { error: message.data, receivedAt };
        return ["bar"];

      // `hello` only marks the connection; `pong` keeps the transport alive.
      case "hello":
      case "pong":
        return [];
    }
  }
}
