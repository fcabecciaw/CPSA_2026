import "./styles.css";

import {
  DashboardClient,
  defaultSocketUrl,
  type ConnectionInfo,
  type SocketFactory,
} from "./client";
import { CognitiveAgent } from "./cognitive/agent";
import { CognitiveMemory } from "./cognitive/memory";
import { Perception } from "./cognitive/perception";
import { RULEBOOK } from "./cognitive/rulebook";
import { CognitiveEngine, type Expression } from "./cognitive/rules";
import { byId } from "./dom";
import { FaceView } from "./face";
import { FaceCueTracker } from "./facepipeline";
import { formatNumber } from "./labels";
import { MetricsTracker } from "./metrics";
import { ShowcaseView } from "./showcase";
import { DashboardStore } from "./state";
import { VoiceEngine } from "./voice";
import { collectElements, renderAll, renderConnection, renderMetrics, renderNotice } from "./views";
import { VideoFeed, type VideoSource } from "./video";

const params = new URLSearchParams(window.location.search);

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

const face = new FaceView(byId("face"), byId("bubble"));
// `?voice=1` marks this tab as the speaking one; the click gate still applies.
const voice = new VoiceEngine({ enabled: params.get("voice") === "1" });

const voiceBadge = byId("voice-badge");
const voiceGate = byId<HTMLButtonElement>("voice-gate");
const muteToggle = byId<HTMLButtonElement>("mute-toggle");
const showcaseEmpty = byId("showcase-empty");
const mente = byId("mente");
const menteEvent = byId("mente-event");
const menteRule = byId("mente-rule");
const menteUtility = byId("mente-utility");
const menteScene = byId("mente-scene");
const menteFace = byId("mente-face");

const mind = new CognitiveAgent({
  engine: new CognitiveEngine(RULEBOOK),
  memory: new CognitiveMemory(),
  perception: new Perception(),
  speak: (text, handlers) => {
    if (bubbleTimer !== null) {
      clearTimeout(bubbleTimer);
      bubbleTimer = null;
    }
    face.setBubble(text);
    face.beginUtterance(expression, Date.now());
    voice.speak(text, {
      onStart: () => {
        handlers.onStart?.();
        face.setSpeaking(true);
      },
      onBoundary: () => handlers.onBoundary?.(),
      onEnd: (reason) => {
        handlers.onEnd?.(reason);
        face.endUtterance(Date.now());
        if (bubbleTimer !== null) {
          clearTimeout(bubbleTimer);
        }
        bubbleTimer = window.setTimeout(() => {
          bubbleTimer = null;
          face.clearBubble();
        }, 4_500);
      },
    });
    renderVoice();
  },
  onExpression: (next) => {
    expression = next;
    face.hold(next, Date.now());
  },
  onMind: () => renderMind(),
});

let expression: Expression = "neutral";
let bubbleTimer: number | null = null;
let frameWidth = 640;
let frameHeight = 480;
let lastActivityAt = Date.now();
/** In MoveNet the objects panel would freeze on its last frame: clear it. */
let poseMode = false;

/**
 * Face cues run in this laptop browser (MediaPipe on `/video.mjpeg`), never on
 * the board; `faceCues` is the tracker or the scripted mock of `?mock=1`.
 */
const faceTracker = new FaceCueTracker({
  onCue: (cue) => {
    mind.onFace(cue.kind, cue.score, cue.faces, Date.now());
  },
  onStatus: () => {
    mind.setFaceStatus(faceTracker.label);
    renderMind();
  },
});

let faceCues: VideoSource = {
  start: () => {
    void faceTracker.start(videoImage).then(() => {
      mind.setFaceStatus(faceTracker.label);
      renderMind();
    });
  },
  stop: () => {
    faceTracker.stop();
    mind.setFaceStatus(faceTracker.label);
    renderMind();
  },
};

function renderMind(): void {
  const snapshot = mind.mind;
  mente.hidden =
    snapshot.event === null && snapshot.ruleId === null && snapshot.scene === "attract";
  menteEvent.textContent = snapshot.event ?? "—";
  menteRule.textContent = snapshot.ruleId ?? "—";
  menteUtility.textContent =
    snapshot.utility === null ? "—" : formatNumber(snapshot.utility, 2);
  menteScene.textContent = snapshot.scene;
  menteFace.textContent = snapshot.face;
}

/** The badge, the click gate and the mute button, from the voice snapshot. */
function renderVoice(): void {
  const snapshot = voice.snapshot();
  voiceBadge.textContent = snapshot.label;
  voiceBadge.dataset.status = snapshot.status;
  const gate = !snapshot.enabled || snapshot.status === "locked";
  voiceGate.hidden = !gate;
  muteToggle.hidden = !snapshot.enabled;
  muteToggle.setAttribute("aria-pressed", String(snapshot.muted));
  muteToggle.textContent = snapshot.muted ? "Muto" : "Voce";
}

/** One gesture is enough for this page: audio and the tab's voice switch. */
voiceGate.addEventListener("click", () => {
  voice.activate();
  renderVoice();
});

muteToggle.addEventListener("click", () => {
  voice.toggleMute();
  renderVoice();
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
    faceCues.start();
  } else {
    video.stop();
    faceCues.stop();
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

  if (import.meta.env.DEV && params.get("mock") === "1") {
    const mock = await import("./mock");
    factory = mock.createMockSocket;
    sourceLabel = mock.MOCK_BADGE;
    video = mock.createMockVideo(videoImage, videoNote);
    faceCues = mock.createMockFaceCues((cue) => {
      mind.onFace(cue.kind, cue.score, cue.faces, Date.now());
    });
    mind.setFaceStatus("mock");
    renderMind();
  }

  new DashboardClient({
    url: socketUrl,
    factory,
    handlers: {
      onMessage: (message) => {
        const now = Date.now();
        if (message.type === "status") {
          metrics.status(message.data);
          mind.onStatus(message.data, now);
        }
        if (message.type === "showcase.state") {
          metrics.showcase(message.data);
          poseMode = false;
          frameWidth = message.data.width;
          frameHeight = message.data.height;
          if (message.data.objects.length > 0) {
            lastActivityAt = now;
          }
          mind.onShowcase(message.data, now);
        }
        if (message.type === "pose.state") {
          lastActivityAt = now;
          if (!poseMode) {
            poseMode = true;
            showcase.setState(null);
          }
          mind.onPose(message.data, now);
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
        mind.onConnection(info.phase === "open", Date.now());
        refreshVideo();
        renderConnection(elements, info);
        renderMetrics(elements, metrics.read());
        renderNotice(elements, input());
      },
      onNotice: showTransportNotice,
    },
  }).start();

  // The cognitive heartbeat: rule candidates, face, calm panel, voice label.
  window.setInterval(() => {
    const now = Date.now();
    mind.tick(now);
    const world = mind.perception.world;
    const idleS = (now - lastActivityAt) / 1_000;
    face.tick(now, idleS);
    face.updateGaze(
      world.largestPerson === null ? [] : [world.largestPerson],
      { width: frameWidth, height: frameHeight },
      now,
      world.largestPerson === null,
    );
    showcase.tick();
    // The empty invitation changes meaning while the pose model is active.
    const note =
      world.mode === "movenet"
        ? "Modalità postura: guardo i gesti"
        : "Mostra un oggetto alla telecamera";
    if (showcaseEmpty.textContent !== note) {
      showcaseEmpty.textContent = note;
    }
    renderVoice();
  }, 250);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      voice.cancel();
      mind.interrupt();
      faceCues.stop();
    } else if (streaming) {
      faceCues.start();
    }
  });

  renderAll(elements, input());
  renderVoice();
  renderMind();
  refreshVideo();
}

boot().catch((error: unknown) => {
  const text = error instanceof Error ? error.message : String(error);
  showTransportNotice(`Avvio non riuscito: ${text}`);
});
