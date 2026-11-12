import "./styles.css";

import {
  DashboardClient,
  defaultSocketUrl,
  type ConnectionInfo,
  type SocketFactory,
} from "./client";
import { byId } from "./dom";
import { MetricsTracker } from "./metrics";
import { ShowcaseView } from "./showcase";
import { DashboardStore } from "./state";
import { collectElements, renderAll, renderConnection, renderMetrics, renderNotice } from "./views";
import { VideoFeed, type VideoSource } from "./video";

const elements = collectElements();
const store = new DashboardStore();
const metrics = new MetricsTracker();

const videoImage = byId<HTMLImageElement>("video");
const videoNote = byId("video-note");
let video: VideoSource = new VideoFeed(videoImage, videoNote);

const showcase = new ShowcaseView({
  countsGroup: byId("showcase-counts-group"),
  counts: byId("showcase-counts"),
  objectsGroup: byId("showcase-objects-group"),
  objects: byId("showcase-objects"),
  empty: byId("showcase-empty"),
  meta: byId("showcase-meta"),
});

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
let streaming = false;

function input() {
  return {
    state: store.state,
    connection,
    readout: metrics.read(),
    sourceLabel,
    transportNotice,
  };
}

/** F-K2: the one camera stream lives as long as the socket is open. */
function refreshVideo(): void {
  const wanted = connection.phase === "open";
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

function refreshBar(): void {
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
    refreshBar();
  }, 12_000);
  refreshBar();
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

  new DashboardClient({
    url: socketUrl,
    factory,
    handlers: {
      onMessage: (message) => {
        const now = Date.now();
        if (message.type === "status") {
          metrics.status(message.data);
        }
        if (message.type === "showcase.state") {
          metrics.showcase(message.data);
        }
        for (const region of store.apply(message, now)) {
          if (region === "showcase") {
            showcase.setState(store.state.showcase);
          }
          if (region === "bar") {
            renderNotice(elements, input());
          }
          renderMetrics(elements, metrics.read());
        }
      },
      onConnection: (info) => {
        connection = info;
        refreshVideo();
        renderConnection(elements, info);
        renderMetrics(elements, metrics.read());
        renderNotice(elements, input());
      },
      onNotice: showTransportNotice,
    },
  }).start();

  renderAll(elements, input());
  refreshVideo();
}

boot().catch((error: unknown) => {
  const text = error instanceof Error ? error.message : String(error);
  showTransportNotice(`Avvio non riuscito: ${text}`);
});
