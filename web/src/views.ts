/** DOM rendering for the slim top bar and the status line. */

import type { ConnectionInfo } from "./client";
import { byId, clear, el } from "./dom";
import {
  CAMERA_LABELS,
  CONNECTION_LABEL,
  ERROR_LABEL,
  MODE_LABEL,
  MODE_STATE_LABEL,
  NOTICE_LABEL,
  RPS_LABELS,
  formatNumber,
} from "./labels";
import type { Readout } from "./metrics";
import type { Mode, ModeState } from "./protocol";
import type { DashboardState } from "./state";

export interface DashboardElements {
  connDot: HTMLElement;
  connText: HTMLElement;
  conn: HTMLElement;
  modeButtons: HTMLButtonElement[];
  modeStatus: HTMLElement;
  cameraTitle: HTMLElement;
  cameraBadge: HTMLElement;
  video: HTMLImageElement;
  notice: HTMLElement;
  fps: HTMLElement;
  latency: HTMLElement;
  score: HTMLElement;
}

export function collectElements(): DashboardElements {
  return {
    connDot: byId("conn-dot"),
    connText: byId("conn-text"),
    conn: byId("conn"),
    modeButtons: [
      byId<HTMLButtonElement>("mode-hologram"),
      byId<HTMLButtonElement>("mode-rps"),
      byId<HTMLButtonElement>("mode-showcase"),
    ],
    modeStatus: byId("mode-status"),
    cameraTitle: byId("camera-title"),
    cameraBadge: byId("camera-badge"),
    video: byId<HTMLImageElement>("video"),
    notice: byId("notice"),
    fps: byId("fps"),
    latency: byId("latency"),
    score: byId("rps-score"),
  };
}

export interface RenderInput {
  state: DashboardState;
  connection: ConnectionInfo;
  mode: Mode | null;
  readout: Readout;
  sourceLabel: string | null;
  transportNotice: string | null;
}

export function renderAll(elements: DashboardElements, input: RenderInput): void {
  renderConnection(elements, input.connection);
  renderMode(elements, input);
  renderMetrics(elements, input.readout);
  renderNotice(elements, input);
}

export function renderConnection(elements: DashboardElements, connection: ConnectionInfo): void {
  elements.connDot.className = `dot dot-${connection.phase}`;
  elements.connText.textContent = CONNECTION_LABEL[connection.phase];

  const parts = [`Server ${shortUrl(connection.url)}`];
  if (connection.phase === "open" && connection.rttMs !== null) {
    parts.push(`RTT ${formatNumber(connection.rttMs, 0)} ms`);
  }
  if (connection.phase === "reconnecting" && connection.retryInMs !== null) {
    parts.push(`nuovo tentativo tra ${formatNumber(connection.retryInMs / 1000, 1)} s`);
  }
  elements.conn.title = parts.join(" · ");
}

/**
 * The segmented mode switch, the score while Sasso-carta-forbice runs, and the
 * screen-reader line announcing the mode the server actually runs.
 */
export function renderMode(elements: DashboardElements, input: RenderInput): void {
  const { mode, state, connection } = input;
  const modeState = state.status?.mode_state ?? null;
  const step = state.modeProgress?.step ?? null;
  const switching =
    modeState === "starting" ||
    modeState === "stopping" ||
    step === "starting" ||
    step === "stopping" ||
    step === "releasing";

  for (const button of elements.modeButtons) {
    const buttonMode = button.dataset.mode;
    button.setAttribute("aria-pressed", String(mode !== null && buttonMode === mode));
    button.disabled = connection.phase !== "open" || switching;
  }

  renderScore(elements, mode, state);
  renderCamera(elements, mode);
  elements.modeStatus.textContent = modeAnnouncement(mode, modeState, connection);
}

/**
 * The showcase runs on the RGB camera; the hand demos run on the IR one, so the
 * pane title and badge follow the mode.
 */
function renderCamera(elements: DashboardElements, mode: Mode | null): void {
  const rgb = mode === "showcase";
  elements.cameraTitle.textContent = rgb ? CAMERA_LABELS.rgbTitle : CAMERA_LABELS.title;
  elements.cameraBadge.textContent = rgb ? CAMERA_LABELS.rgbBadge : CAMERA_LABELS.badge;
  elements.video.alt = rgb ? CAMERA_LABELS.rgbAlt : CAMERA_LABELS.alt;
}

function modeAnnouncement(
  mode: Mode | null,
  modeState: ModeState | null,
  connection: ConnectionInfo,
): string {
  if (mode === null) {
    return connection.phase === "open"
      ? "In attesa dello stato dal server…"
      : CONNECTION_LABEL[connection.phase];
  }
  const state = modeState === null ? "" : `, modalità ${MODE_STATE_LABEL[modeState]}`;
  return `${MODE_LABEL[mode]}${state}`;
}

/** "Tu 2 · Tavolo 1", plus the draws once there is at least one. */
function renderScore(elements: DashboardElements, mode: Mode | null, state: DashboardState): void {
  const score = state.rps?.score ?? null;
  const visible = mode === "rps" && score !== null;
  elements.score.hidden = !visible;
  clear(elements.score);
  if (!visible || score === null) {
    return;
  }

  elements.score.append(
    el("span", "score-player", `${RPS_LABELS.player} ${score.player}`),
    el("span", "score-sep", "·"),
    el("span", "score-board", `${RPS_LABELS.board} ${score.board}`),
  );
  if (score.draws > 0) {
    elements.score.append(
      el("span", "score-sep", "·"),
      el("span", "score-draws", `${RPS_LABELS.draws} ${score.draws}`),
    );
  }
}

/** Frames per second and the processing time of one frame, both optional. */
export function renderMetrics(elements: DashboardElements, readout: Readout): void {
  elements.fps.textContent =
    readout.fps === null ? "— fps" : `${formatNumber(readout.fps, 0)} fps`;
  elements.latency.textContent =
    readout.latencyMs === null ? "— ms" : `${formatNumber(readout.latencyMs, 0)} ms`;
}

/**
 * One line under the bar: a rejected command, a protocol error, a transport
 * problem or the mock badge. Empty means the line takes no space.
 */
export function renderNotice(elements: DashboardElements, input: RenderInput): void {
  elements.notice.className = "notice";
  const { state, transportNotice, sourceLabel } = input;

  const modeError = state.modeError;
  if (modeError !== null) {
    const { error } = modeError;
    elements.notice.textContent = `${NOTICE_LABEL.rejected}: ${ERROR_LABEL[error.code]} (${error.code}): ${error.message}`;
    elements.notice.classList.add("notice-error");
    return;
  }

  const progressError = state.modeProgress?.error ?? null;
  if (progressError !== null) {
    elements.notice.textContent = `${ERROR_LABEL[progressError.code]} (${progressError.code}): ${progressError.message}`;
    elements.notice.classList.add("notice-error");
    return;
  }

  if (state.lastError !== null) {
    const { error } = state.lastError;
    elements.notice.textContent = `${ERROR_LABEL[error.code]} (${error.code}): ${error.message}`;
    elements.notice.classList.add("notice-error");
    return;
  }

  if (transportNotice !== null) {
    elements.notice.textContent = transportNotice;
    elements.notice.classList.add("notice-warn");
    return;
  }

  if (sourceLabel !== null) {
    elements.notice.textContent = sourceLabel;
    elements.notice.classList.add("notice-info");
    return;
  }

  elements.notice.textContent = "";
}

function shortUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.host}${parsed.pathname}`;
  } catch {
    return url;
  }
}
