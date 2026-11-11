/** DOM rendering for the dashboard panels. No innerHTML is used. */

import type { ConnectionInfo } from "./client";
import { byId, chip, clear, code, el } from "./dom";
import {
  CAMERA_LABEL,
  EPISODE_LABEL,
  ERROR_LABEL,
  HAND_SHAPE_LABEL,
  IMU_LABEL,
  MODE_LABEL,
  MODE_STATE_LABEL,
  MODE_STEP_LABEL,
  REGION_LABEL,
  SIDE_LABEL,
  STATUS_LABELS,
  actuatorText,
  errorCodeText,
  formatClock,
  formatHz,
  formatNumber,
  formatUptime,
  formatDuration,
  reasonLabel,
  tagClass,
  tagLabel,
  verdictLabel,
} from "./labels";
import { MAX_DECISIONS, type DashboardState } from "./state";

export interface DashboardElements {
  connDot: HTMLElement;
  connText: HTMLElement;
  connDetail: HTMLElement;
  modeButtons: HTMLButtonElement[];
  modeStatus: HTMLElement;
  modeError: HTMLElement;
  health: HTMLElement;
  notice: HTMLElement;
  liveTiles: HTMLElement;
  episodeTitle: HTMLElement;
  episodeBadge: HTMLElement;
  episodeBody: HTMLElement;
  feed: HTMLElement;
  feedCount: HTMLElement;
}

export function collectElements(): DashboardElements {
  return {
    connDot: byId("conn-dot"),
    connText: byId("conn-text"),
    connDetail: byId("conn-detail"),
    modeButtons: [byId<HTMLButtonElement>("mode-idle"), byId<HTMLButtonElement>("mode-medical")],
    modeStatus: byId("mode-status"),
    modeError: byId("mode-error"),
    health: byId("health"),
    notice: byId("notice"),
    liveTiles: byId("live-tiles"),
    episodeTitle: byId("episode-title"),
    episodeBadge: byId("episode-badge"),
    episodeBody: byId("episode-body"),
    feed: byId("feed"),
    feedCount: byId("feed-count"),
  };
}

export function renderAll(
  elements: DashboardElements,
  state: DashboardState,
  connection: ConnectionInfo,
  sourceLabel: string | null,
  transportNotice: string | null,
): void {
  renderConnection(elements, connection);
  renderMode(elements, state, connection);
  renderNotice(elements, state, sourceLabel, transportNotice);
  renderLive(elements, state);
  renderFeed(elements, state);
  renderEpisode(elements, state);
}

export function renderConnection(elements: DashboardElements, connection: ConnectionInfo): void {
  const { phase } = connection;
  let text = "Connessione…";
  let detail = `a ${shortUrl(connection.url)}`;

  if (phase === "open") {
    text = "Connesso";
    detail = `a ${shortUrl(connection.url)}`;
    if (connection.rttMs !== null) {
      detail += ` · RTT ${formatNumber(connection.rttMs, 0)} ms`;
    }
  } else if (phase === "reconnecting") {
    text = "Riconnessione…";
    if (connection.retryInMs === null) {
      detail = `tentativo ${connection.attempt} in corso…`;
    } else {
      detail = `tentativo ${connection.attempt} · nuovo tentativo tra ${formatNumber(
        connection.retryInMs / 1000,
        1,
      )} s`;
    }
  } else if (phase === "closed") {
    text = "Disconnesso";
    detail = "ricarica la pagina per riprovare";
  }

  elements.connDot.className = `dot dot-${phase}`;
  elements.connText.textContent = text;
  elements.connDetail.textContent = detail;
}

export function renderMode(
  elements: DashboardElements,
  state: DashboardState,
  connection: ConnectionInfo,
): void {
  renderModeError(elements, state);
  const status = state.status;
  const mode = status?.mode ?? state.modeProgress?.mode ?? null;
  const modeState = status?.mode_state ?? null;
  const switching =
    modeState === "starting" ||
    modeState === "stopping" ||
    state.modeProgress?.step === "starting" ||
    state.modeProgress?.step === "stopping" ||
    state.modeProgress?.step === "releasing";

  for (const button of elements.modeButtons) {
    const buttonMode = button.dataset.mode;
    button.setAttribute("aria-pressed", String(mode !== null && buttonMode === mode));
    button.disabled = connection.phase !== "open" || switching;
  }

  if (status === null) {
    elements.modeStatus.textContent =
      connection.phase === "open"
        ? "In attesa dello stato dal server…"
        : "In attesa di connessione…";
    return;
  }

  const parts = [
    `Modalità ${MODE_LABEL[status.mode]}`,
    `stato ${MODE_STATE_LABEL[status.mode_state]}`,
  ];
  if (state.modeProgress !== null && state.modeProgress.step !== "ready") {
    parts.push(MODE_STEP_LABEL[state.modeProgress.step]);
  }
  elements.modeStatus.textContent = parts.join(" · ");
  renderHealth(elements, state);
}

/**
 * Shows the rejection that answered the pending `mode.set` (IF-304) next to
 * the mode buttons. An error with any other `ref_seq` stays in the global
 * notice.
 */
function renderModeError(elements: DashboardElements, state: DashboardState): void {
  const entry = state.modeError;
  if (entry === null) {
    elements.modeError.textContent = "";
    return;
  }
  const { error } = entry;
  elements.modeError.textContent = `Comando rifiutato: ${ERROR_LABEL[error.code]} (${error.code}): ${error.message}`;
}

function renderHealth(elements: DashboardElements, state: DashboardState): void {
  clear(elements.health);
  const status = state.status;
  if (status === null) {
    return;
  }

  const camera = status.perception.camera;
  const cameraTone = camera === "ok" ? "tone-ok" : camera === null ? "tone-muted" : "tone-danger";
  elements.health.appendChild(
    healthItem(
      STATUS_LABELS.camera,
      camera === null ? "—" : CAMERA_LABEL[camera],
      cameraTone,
    ),
  );

  const model = status.perception.model;
  elements.health.appendChild(
    healthItem(
      STATUS_LABELS.model,
      model ?? "assente",
      model === null ? "tone-danger" : "tone-ok",
      model ?? undefined,
    ),
  );

  const errors = status.errors;
  elements.health.appendChild(
    healthItem(
      STATUS_LABELS.errors,
      errors.length === 0 ? "nessuno" : errors.map(errorCodeText).join(", "),
      errors.length === 0 ? "tone-ok" : "tone-danger",
      errors.length === 0 ? undefined : errors.join(", "),
    ),
  );

  const fps = status.perception.fps;
  elements.health.appendChild(
    healthItem(
      STATUS_LABELS.fps,
      fps === null ? "—" : `${formatNumber(fps)} fps`,
      fps === null ? "tone-muted" : "tone-ok",
    ),
  );

  const dpu = status.perception.dpu_ms;
  elements.health.appendChild(
    healthItem(
      STATUS_LABELS.dpu,
      dpu === null ? "—" : `${formatNumber(dpu)} ms`,
      dpu === null ? "tone-muted" : "tone-ok",
    ),
  );

  elements.health.appendChild(
    healthItem(STATUS_LABELS.uptime, formatUptime(status.uptime_s), "tone-muted"),
  );
}

function healthItem(key: string, value: string, tone: string, title?: string): HTMLLIElement {
  const item = el("li", "health-item");
  item.appendChild(el("span", "health-key", key));
  const amount = el("span", `health-value ${tone}`, value);
  if (title !== undefined) {
    amount.title = title;
  }
  item.appendChild(amount);
  return item;
}

export function renderNotice(
  elements: DashboardElements,
  state: DashboardState,
  sourceLabel: string | null,
  transportNotice: string | null,
): void {
  elements.notice.className = "notice";
  if (state.lastError !== null && state.lastError !== state.modeError) {
    const { error } = state.lastError;
    elements.notice.textContent = `${ERROR_LABEL[error.code]} (${error.code}): ${error.message}`;
    elements.notice.classList.add("notice-error");
    return;
  }
  const progressError = state.modeProgress?.error ?? null;
  if (progressError !== null) {
    elements.notice.textContent = `${ERROR_LABEL[progressError.code]} (${progressError.code}): ${progressError.message}`;
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

export function renderLive(elements: DashboardElements, state: DashboardState): void {
  clear(elements.liveTiles);

  const imu = state.medical?.imu ?? state.status?.imu ?? null;
  const imuLabel = imu === null ? "—" : IMU_LABEL[imu];
  const imuTone =
    imu === "connected"
      ? "tone-ok"
      : imu === "searching"
        ? "tone-warn"
        : imu === "lost"
          ? "tone-danger"
          : "tone-muted";
  elements.liveTiles.appendChild(
    tile(
      "IMU L/R",
      el("span", `tile-value ${imuTone}`, imuLabel),
      "polsi sinistro e destro",
    ),
  );

  const tag = state.currentTag;
  const tagTile = tile("Tag corrente", chip(tagLabel(tag), `chip ${tagClass(tag)}`), null);
  elements.liveTiles.appendChild(tagTile);

  const active = state.episodeActive;
  const episodeValue = el(
    "span",
    `tile-value ${active ? "episode-active" : "tone-muted"}`,
    active ? "Attivo" : "Inattivo",
  );
  elements.liveTiles.appendChild(
    tile("Episodio", episodeValue, state.episodeId ?? "nessun episodio corrente"),
  );
}

function tile(label: string, value: Node, note: string | null): HTMLDivElement {
  const box = el("div", "tile");
  box.appendChild(el("span", "tile-label", label));
  const valueWrap = el("div", "tile-value-wrap");
  valueWrap.appendChild(value);
  box.appendChild(valueWrap);
  if (note !== null) {
    box.appendChild(el("span", "tile-note", note));
  }
  return box;
}

export function renderFeed(elements: DashboardElements, state: DashboardState): void {
  elements.feedCount.textContent = `${state.decisions.length} / ${MAX_DECISIONS}`;
  clear(elements.feed);

  if (state.decisions.length === 0) {
    elements.feed.appendChild(el("li", "feed-empty", "Nessuna decisione ricevuta."));
    return;
  }

  state.decisions.forEach((entry, index) => {
    const decision = entry.decision;
    const item = el("li", index === 0 ? "feed-item newest" : "feed-item");

    const head = el("div", "feed-head");
    const time = el("time", "feed-time", formatClock(entry.wall));
    time.dateTime = entry.wall;
    head.appendChild(time);
    head.appendChild(chip(tagLabel(decision.tag), `chip ${tagClass(decision.tag)}`));
    head.appendChild(
      el(
        "span",
        `verdict verdict-${(decision.verdict ?? "none").toLowerCase()}`,
        verdictLabel(decision.verdict),
      ),
    );
    if (decision.tag_changed) {
      head.appendChild(el("span", "mini-flag", "cambio tag"));
    }
    if (index === 0) {
      head.appendChild(el("span", "newest-flag", "ultima"));
    }
    item.appendChild(head);

    const detail = el("p", "feed-detail");
    detail.appendChild(document.createTextNode("Gravità: "));
    detail.appendChild(chip(tagLabel(decision.severity), `chip chip-small ${tagClass(decision.severity)}`));
    detail.appendChild(document.createTextNode(` · Motivo: ${reasonLabel(decision.reason)} · Attuatori: ${actuatorText(decision.actuations)}`));
    item.appendChild(detail);

    elements.feed.appendChild(item);
  });
}

export function renderEpisode(elements: DashboardElements, state: DashboardState): void {
  const episode = state.episode;
  const active = state.episodeActive;
  elements.episodeTitle.textContent = active ? "Episodio attivo" : "Ultimo episodio";

  clear(elements.episodeBadge);
  if (active) {
    elements.episodeBadge.appendChild(el("span", "badge-active", "In corso"));
  } else if (episode !== null) {
    elements.episodeBadge.appendChild(el("span", "badge-ended", "Concluso"));
  }

  clear(elements.episodeBody);

  if (episode === null) {
    if (active && state.episodeId !== null) {
      const running = el("p", "episode-running");
      running.appendChild(document.createTextNode("Episodio "));
      running.appendChild(code(state.episodeId));
      running.appendChild(document.createTextNode(" in corso; il riepilogo arriva alla fine."));
      elements.episodeBody.appendChild(running);
    } else {
      elements.episodeBody.appendChild(el("p", "empty", "Nessun episodio registrato."));
    }
    return;
  }

  const metrics = el("div", "metrics");
  metrics.appendChild(metric("Contatti", String(episode.contact_count)));
  metrics.appendChild(metric("Ripetizioni", formatHz(episode.repetition_hz)));
  metrics.appendChild(
    metricNode(
      "Gravità di picco",
      episode.peak_severity === null
        ? el("strong", "metric-value tone-muted", "—")
        : chip(tagLabel(episode.peak_severity), `chip ${tagClass(episode.peak_severity)}`),
    ),
  );
  metrics.appendChild(
    metric(
      "Etichette",
      episode.labels.length === 0 ? "—" : episode.labels.map((label) => EPISODE_LABEL[label]).join(", "),
    ),
  );
  elements.episodeBody.appendChild(metrics);

  const identity = el("p", "episode-id");
  identity.appendChild(document.createTextNode("ID "));
  identity.appendChild(code(episode.episode_id));
  elements.episodeBody.appendChild(identity);

  if (episode.contacts.length === 0) {
    elements.episodeBody.appendChild(el("p", "empty", "Nessun contatto registrato."));
    return;
  }

  const table = el("table", "contacts");
  const caption = el("caption", "", "Contatti rilevati");
  table.appendChild(caption);
  const thead = el("thead");
  const headRow = el("tr");
  for (const label of ["Lato", "Zona", "Durata", "Forma della mano"]) {
    const th = el("th", "", label);
    th.scope = "col";
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const contact of episode.contacts) {
    const row = el("tr");
    row.appendChild(el("td", "", SIDE_LABEL[contact.side]));
    row.appendChild(el("td", "", REGION_LABEL[contact.region]));
    row.appendChild(el("td", "", formatDuration(contact.duration_s)));
    row.appendChild(el("td", "", HAND_SHAPE_LABEL[contact.hand_shape]));
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  elements.episodeBody.appendChild(table);
}

function metric(label: string, value: string): HTMLDivElement {
  return metricNode(label, el("strong", "metric-value", value));
}

function metricNode(label: string, value: Node): HTMLDivElement {
  const box = el("div", "metric");
  box.appendChild(el("span", "metric-label", label));
  box.appendChild(value);
  return box;
}

function shortUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.host}${parsed.pathname}`;
  } catch {
    return url;
  }
}
