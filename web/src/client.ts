/**
 * WebSocket client for the dashboard.
 *
 * Connects to the same origin at `/ws` (IF-101), reconnects with exponential
 * backoff, and sends a `ping` every 10 s (IF-203). It does not hold protocol
 * state: every message is handed to the caller, and the server resends a
 * snapshot on reconnect (FR-504). F-K1: `ping` is the only command it sends.
 */

import {
  PROTOCOL_VERSION,
  parseServerMessage,
  type ClientCommand,
  type ModelName,
  type ServerMessage,
} from "./protocol";

export type ConnectionPhase = "connecting" | "open" | "reconnecting" | "closed";

export interface ConnectionInfo {
  phase: ConnectionPhase;
  url: string;
  /** Consecutive failed connection attempts; 0 while connected. */
  attempt: number;
  /** Delay before the next attempt, only while reconnecting. */
  retryInMs: number | null;
  rttMs: number | null;
}

export interface SocketLike {
  send(data: string): void;
  close(): void;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
}

export type SocketFactory = (url: string) => SocketLike;

export interface ClientHandlers {
  onMessage: (message: ServerMessage) => void;
  onConnection: (info: ConnectionInfo) => void;
  /** Protocol or transport problems worth surfacing to the operator. */
  onNotice: (notice: string) => void;
}

export interface DashboardClientOptions {
  handlers: ClientHandlers;
  url?: string;
  factory?: SocketFactory;
  pingIntervalMs?: number;
  minBackoffMs?: number;
  maxBackoffMs?: number;
}

const PING_INTERVAL_MS = 10_000;
const MIN_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 5_000;

export function defaultSocketUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws`;
}

export const browserSocketFactory: SocketFactory = (url) =>
  // WebSocket's DOM handler signatures are wider; the client only uses these
  // four assignments, so the structural cast is safe here.
  new WebSocket(url) as unknown as SocketLike;

export class DashboardClient {
  private readonly handlers: ClientHandlers;
  private readonly url: string;
  private readonly factory: SocketFactory;
  private readonly pingIntervalMs: number;
  private readonly minBackoffMs: number;
  private readonly maxBackoffMs: number;

  private socket: SocketLike | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private pendingPings = new Map<number, number>();
  private pingId = 0;
  private commandSeq = 0;
  private attempt = 0;
  private rttMs: number | null = null;
  private stopped = false;
  private info: ConnectionInfo;

  constructor(options: DashboardClientOptions) {
    this.handlers = options.handlers;
    this.url = options.url ?? defaultSocketUrl();
    this.factory = options.factory ?? browserSocketFactory;
    this.pingIntervalMs = options.pingIntervalMs ?? PING_INTERVAL_MS;
    this.minBackoffMs = options.minBackoffMs ?? MIN_BACKOFF_MS;
    this.maxBackoffMs = options.maxBackoffMs ?? MAX_BACKOFF_MS;
    this.info = {
      phase: "connecting",
      url: this.url,
      attempt: 0,
      retryInMs: null,
      rttMs: null,
    };
  }

  get connection(): ConnectionInfo {
    return this.info;
  }

  start(): void {
    this.stopped = false;
    this.attempt = 0;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    this.clearRetry();
    this.stopPing();
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    this.setInfo({ phase: "closed", retryInMs: null });
  }

  private open(): void {
    this.setInfo({ phase: this.attempt === 0 ? "connecting" : "reconnecting", retryInMs: null });

    let socket: SocketLike;
    try {
      socket = this.factory(this.url);
    } catch (error) {
      this.handlers.onNotice(`Connessione non riuscita: ${errorText(error)}`);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.attempt = 0;
      this.rttMs = null;
      // IF-304: the per-connection command counter starts again at 1.
      this.commandSeq = 0;
      this.setInfo({ phase: "open", attempt: 0, retryInMs: null, rttMs: null });
      this.startPing();
    };

    socket.onmessage = (event) => {
      this.handleRaw(event.data);
    };

    socket.onclose = () => {
      if (this.socket !== socket) {
        return;
      }
      this.socket = null;
      this.stopPing();
      if (!this.stopped) {
        this.scheduleReconnect();
      }
    };

    socket.onerror = () => {
      // `close` follows an error for WebSocket; reconnect is handled there.
    };
  }

  private handleRaw(raw: unknown): void {
    if (typeof raw !== "string") {
      this.handlers.onNotice("Messaggio binario ignorato: il protocollo v1 è JSON testuale.");
      return;
    }

    const result = parseServerMessage(raw);
    switch (result.kind) {
      case "message":
        if (result.message.type === "pong") {
          this.notePong(result.message.data.id);
        }
        this.handlers.onMessage(result.message);
        break;
      case "ignored":
        // IF-103: unknown message types are ignored without side effects.
        break;
      case "invalid":
        console.warn(`[cpsa] messaggio scartato: ${result.reason}`);
        this.handlers.onNotice(`Messaggio scartato: ${result.reason}`);
        break;
    }
  }

  /**
   * F-E1: ask the board for the other model. Returns the command `seq` so the
   * caller can recognise the `error.ref_seq` of a board that does not know the
   * command yet, instead of leaving a red banner on screen.
   */
  setModel(model: ModelName): number | null {
    this.commandSeq += 1;
    if (this.socket === null) {
      return null;
    }
    this.send({
      v: PROTOCOL_VERSION,
      type: "model.set",
      seq: this.commandSeq,
      data: { model },
    });
    return this.commandSeq;
  }

  private send(command: ClientCommand): void {
    const socket = this.socket;
    if (socket === null) {
      return;
    }
    try {
      socket.send(JSON.stringify({ ...command, v: PROTOCOL_VERSION }));
    } catch (error) {
      this.handlers.onNotice(`Invio non riuscito: ${errorText(error)}`);
    }
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => this.sendPing(), this.pingIntervalMs);
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    this.pendingPings.clear();
  }

  private sendPing(): void {
    const id = ++this.pingId;
    const sentAt = Date.now();
    this.pendingPings.set(id, sentAt);
    for (const [pendingId, pendingAt] of this.pendingPings) {
      if (sentAt - pendingAt > this.pingIntervalMs * 3) {
        this.pendingPings.delete(pendingId);
      }
    }
    this.send({ v: PROTOCOL_VERSION, type: "ping", data: { id } });
  }

  private notePong(id: unknown): void {
    if (typeof id !== "number") {
      return;
    }
    const sentAt = this.pendingPings.get(id);
    if (sentAt === undefined) {
      return;
    }
    this.pendingPings.delete(id);
    this.rttMs = Date.now() - sentAt;
    this.setInfo({ rttMs: this.rttMs });
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.retryTimer !== null) {
      return;
    }
    const delay = Math.min(this.minBackoffMs * 2 ** this.attempt, this.maxBackoffMs);
    this.attempt += 1;
    this.setInfo({ phase: "reconnecting", attempt: this.attempt, retryInMs: delay });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      if (!this.stopped) {
        this.open();
      }
    }, delay);
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  private setInfo(patch: Partial<ConnectionInfo>): void {
    this.info = { ...this.info, ...patch };
    this.handlers.onConnection(this.info);
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
