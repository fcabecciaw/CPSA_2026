import "./styles.css";

import {
  DashboardClient,
  defaultSocketUrl,
  type ConnectionInfo,
  type ConnectionPhase,
  type SocketFactory,
} from "./client";
import { byId } from "./dom";
import { HologramView } from "./hologram";
import { MetricsTracker } from "./metrics";
import { MODES, type Mode } from "./protocol";
import { RpsView } from "./rps";
import { DashboardStore, type StateRegion } from "./state";
import {
  collectElements,
  renderAll,
  renderConnection,
  renderMetrics,
  renderMode,
  renderNotice,
  type RenderInput,
} from "./views";
import { VideoFeed, type VideoSource } from "./video";

const elements = collectElements();
const store = new DashboardStore();
const metrics = new MetricsTracker();

const videoImage = byId<HTMLImageElement>("video");
const videoNote = byId("video-note");
let video: VideoSource = new VideoFeed(videoImage, videoNote);

const hologram = new HologramView(
  byId<HTMLCanvasElement>("hologram"),
  byId("hologram-empty"),
  byId("hologram-meta"),
);
const rps = new RpsView({
  count: byId("rps-count"),
  player: byId("rps-player"),
  board: byId("rps-board"),
  duel: byId("rps-duel"),
  canvas: byId<HTMLCanvasElement>("rps-canvas"),
  empty: byId("rps-empty"),
});

/** The three right-hand panes; exactly one is visible at a time (T5). */
const panes = {
  hologram: byId("view-hologram"),
  rps: byId("view-rps"),
  idle: byId("view-idle"),
};

const socketUrl = defaultSocketUrl();
let connection: ConnectionInfo = {
  phase: "connecting",
  url: socketUrl,
  attempt: 0,
  retryInMs: null,
  rttMs: null,
};
let connectionPhase: ConnectionPhase = "connecting";
let sourceLabel: string | null = null;
let transportNotice: string | null = null;
let noticeTimer: ReturnType<typeof setTimeout> | null = null;
let lastMode: Mode | null = null;
let streaming = false;

function input(): RenderInput {
  const mode = store.mode;
  return {
    state: store.state,
    connection,
    mode,
    readout: metrics.read(mode),
    sourceLabel,
    transportNotice,
  };
}

/** Shows the pane of the current mode; idle (and anything else) gets the invitation. */
function refreshView(): void {
  const mode = store.mode;
  const hologramVisible = mode === "hologram";
  panes.hologram.hidden = !hologramVisible;
  panes.rps.hidden = mode !== "rps";
  panes.idle.hidden = hologramVisible || mode === "rps";
  hologram.setActive(hologramVisible);
  rps.setActive(mode === "rps");
}

/**
 * F14: the camera pane follows the mode, not the connection alone. While no
 * hand mode runs, no frames are produced, so the pane shows its placeholder
 * instead of a frozen last frame.
 */
function refreshVideo(): void {
  const mode = store.mode;
  const wanted =
    connection.phase === "open" && (mode === "hologram" || mode === "rps");
  if (wanted === streaming) {
    return;
  }
  streaming = wanted;
  if (wanted) {
    video.start();
  } else {
    video.stop();
  }
}

/** The bar, the pane and the notice line all follow the `status` snapshot. */
function refreshMode(): void {
  const mode = store.mode;
  if (mode !== lastMode) {
    // A new mode starts a new stream: the previous frame arrivals are not a
    // frame rate any more.
    metrics.reset();
    lastMode = mode;
  }
  refreshVideo();
  refreshView();
  renderMode(elements, input());
  renderMetrics(elements, input().readout);
  renderNotice(elements, input());
}

function refreshNotice(): void {
  renderNotice(elements, input());
}

function showTransportNotice(text: string): void {
  transportNotice = text;
  if (noticeTimer !== null) {
    clearTimeout(noticeTimer);
  }
  noticeTimer = setTimeout(() => {
    noticeTimer = null;
    transportNotice = null;
    refreshNotice();
  }, 12_000);
  refreshNotice();
}

function renderRegions(regions: StateRegion[]): void {
  for (const region of regions) {
    switch (region) {
      case "mode":
        refreshMode();
        break;
      case "hologram":
        hologram.setFrame(store.state.hologram);
        renderMetrics(elements, input().readout);
        break;
      case "rps":
        rps.setState(store.state.rps);
        // The score lives in the bar, so the bar follows the round.
        renderMode(elements, input());
        renderMetrics(elements, input().readout);
        break;
    }
  }
}

async function boot(): Promise<void> {
  let factory: SocketFactory | undefined;

  if (
    import.meta.env.DEV &&
    new URLSearchParams(window.location.search).get("mock") === "1"
  ) {
    const mock = await import("./mock");
    factory = mock.createMockSocket;
    sourceLabel = mock.MOCK_BADGE;
    video = mock.createMockVideo(videoImage, videoNote);
  }

  const client = new DashboardClient({
    url: socketUrl,
    factory,
    handlers: {
      onMessage: (message) => {
        const now = Date.now();
        if (message.type === "status") {
          metrics.status(message.data);
        }
        if (message.type === "hologram.hands") {
          metrics.frame(now, message.data.timing.total_ms);
        }
        renderRegions(store.apply(message, now));
      },
      onConnection: (info) => {
        if (info.phase === "open" && connectionPhase !== "open") {
          // A new connection restarts the command counter (IF-304).
          store.clearModeRequest();
        }
        connectionPhase = info.phase;
        connection = info;
        refreshVideo();
        renderConnection(elements, info);
        renderMode(elements, input());
      },
      onNotice: showTransportNotice,
    },
  });

  for (const button of elements.modeButtons) {
    button.addEventListener("click", () => {
      const requested = modeOf(button.dataset.mode);
      if (requested === null) {
        return;
      }
      const seq = client.sendMode(requested);
      if (seq !== null) {
        store.markModeSent(seq);
        renderMode(elements, input());
      }
    });
  }

  renderAll(elements, input());
  refreshView();
  client.start();
}

/** Button `data-mode` to a protocol mode, or `null` for an unknown value. */
function modeOf(value: string | undefined): Mode | null {
  return value !== undefined && (MODES as readonly string[]).includes(value)
    ? (value as Mode)
    : null;
}

boot().catch((error: unknown) => {
  const text = error instanceof Error ? error.message : String(error);
  showTransportNotice(`Avvio non riuscito: ${text}`);
});
