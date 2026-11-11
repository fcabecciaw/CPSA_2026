import "./styles.css";

import {
  DashboardClient,
  defaultSocketUrl,
  type ConnectionInfo,
  type ConnectionPhase,
  type SocketFactory,
} from "./client";
import { byId } from "./dom";
import { SkeletonView } from "./skeleton";
import { DashboardStore, type StateRegion } from "./state";
import {
  collectElements,
  renderAll,
  renderConnection,
  renderEpisode,
  renderFeed,
  renderLive,
  renderLog,
  renderMode,
  renderNotice,
  shouldStreamVideo,
} from "./views";

const elements = collectElements();
const store = new DashboardStore();
const skeleton = new SkeletonView(
  byId<HTMLCanvasElement>("skeleton"),
  byId("canvas-empty"),
  byId("skeleton-meta"),
  byId<HTMLImageElement>("video"),
  byId("video-note"),
);

/** `?mock=1` has no backend, so it never opens the F14 stream. */
let mockActive = false;

/**
 * F14: the stream follows the status snapshot (medical mode, camera ok) and is
 * always off in the mock, where the panel keeps the placeholder.
 */
function refreshVideo(): void {
  skeleton.setVideo(!mockActive && shouldStreamVideo(store.state));
}

const socketUrl = defaultSocketUrl();
let connection: ConnectionInfo = {
  phase: "connecting",
  url: socketUrl,
  attempt: 0,
  retryInMs: null,
  rttMs: null,
};
let sourceLabel: string | null = null;
let transportNotice: string | null = null;
let noticeTimer: ReturnType<typeof setTimeout> | null = null;
let connectionPhase: ConnectionPhase = "connecting";

function refreshNotice(): void {
  renderNotice(elements, store.state, sourceLabel, transportNotice);
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
        renderMode(elements, store.state, connection);
        refreshNotice();
        break;
      case "live":
        renderLive(elements, store.state);
        break;
      case "feed":
        renderFeed(elements, store.state);
        break;
      case "episode":
        renderEpisode(elements, store.state);
        break;
      case "hands":
        skeleton.setHands(store.state.hands);
        break;
      case "logs":
        renderLog(elements, store.state);
        break;
    }
  }
  // The stream follows the mode/camera state, not the hands stream: a camera
  // that stops reporting must take the video away too.
  refreshVideo();
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
    mockActive = true;
  }

  const client = new DashboardClient({
    url: socketUrl,
    factory,
    handlers: {
      onMessage: (message) => {
        renderRegions(store.apply(message, Date.now()));
      },
      onConnection: (info) => {
        if (info.phase === "open" && connectionPhase !== "open") {
          // A new connection restarts the command counter (IF-304).
          store.clearModeRequest();
        }
        connectionPhase = info.phase;
        connection = info;
        renderConnection(elements, info);
        renderMode(elements, store.state, info);
      },
      onNotice: showTransportNotice,
    },
  });

  for (const button of elements.modeButtons) {
    button.addEventListener("click", () => {
      const requested = button.dataset.mode;
      if (requested === "idle" || requested === "medical") {
        const seq = client.sendMode(requested);
        if (seq !== null) {
          store.markModeSent(seq);
          renderMode(elements, store.state, connection);
        }
      }
    });
  }

  renderAll(elements, store.state, connection, sourceLabel, transportNotice);
  refreshVideo();
  client.start();
}

boot().catch((error: unknown) => {
  const text = error instanceof Error ? error.message : String(error);
  showTransportNotice(`Avvio non riuscito: ${text}`);
});
