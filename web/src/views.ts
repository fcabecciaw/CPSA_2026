/** DOM rendering for the slim top bar and the status line. */

import type { ConnectionInfo } from "./client";
import { byId } from "./dom";
import { CONNECTION_LABEL, errorLabel, formatNumber } from "./labels";
import type { Readout } from "./metrics";
import type { DashboardState } from "./state";

export interface DashboardElements {
  connDot: HTMLElement;
  connText: HTMLElement;
  conn: HTMLElement;
  video: HTMLImageElement;
  notice: HTMLElement;
  fps: HTMLElement;
  latency: HTMLElement;
}

export function collectElements(): DashboardElements {
  return {
    connDot: byId("conn-dot"),
    connText: byId("conn-text"),
    conn: byId("conn"),
    video: byId<HTMLImageElement>("video"),
    notice: byId("notice"),
    fps: byId("fps"),
    latency: byId("latency"),
  };
}

export interface RenderInput {
  state: DashboardState;
  connection: ConnectionInfo;
  readout: Readout;
  sourceLabel: string | null;
  transportNotice: string | null;
}

export function renderAll(elements: DashboardElements, input: RenderInput): void {
  renderConnection(elements, input.connection);
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

/** Frames per second and the processing time of one frame, both optional. */
export function renderMetrics(elements: DashboardElements, readout: Readout): void {
  elements.fps.textContent =
    readout.fps === null ? "— fps" : `${formatNumber(readout.fps, 0)} fps`;
  elements.latency.textContent =
    readout.latencyMs === null ? "— ms" : `${formatNumber(readout.latencyMs, 0)} ms`;
}

/**
 * One line under the bar: the conditions the process reports, a protocol
 * error, a transport problem or the mock badge. Empty means the line takes no
 * space.
 */
export function renderNotice(elements: DashboardElements, input: RenderInput): void {
  elements.notice.className = "notice";
  const { state, transportNotice, sourceLabel } = input;

  const codes = state.status?.errors ?? [];
  if (codes.length > 0) {
    elements.notice.textContent = codes.map(codeText).join(" · ");
    elements.notice.classList.add("notice-error");
    return;
  }

  if (state.lastError !== null) {
    const { error } = state.lastError;
    elements.notice.textContent = `${codeText(error.code)}: ${error.message}`;
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

function codeText(code: string): string {
  return `${errorLabel(code)} (${code})`;
}

function shortUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.host}${parsed.pathname}`;
  } catch {
    return url;
  }
}
