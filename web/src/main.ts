import "./styles.css";

import {
  DashboardClient,
  defaultSocketUrl,
  type ConnectionInfo,
  type SocketFactory,
} from "./client";
import { CognitiveAgent, HEARTBEAT_MS, type MindSnapshot } from "./cognitive/agent";
import { CognitiveMemory } from "./cognitive/memory";
import { Perception } from "./cognitive/perception";
import { RULEBOOK } from "./cognitive/rulebook";
import { CognitiveEngine, type Expression } from "./cognitive/rules";
import { AttractView, type LiveNumbers } from "./attract";
import { AvatarView } from "./avatar";
import { CalibrationPanel } from "./calibrationpanel";
import { byId } from "./dom";
import { FaceCueTracker } from "./facepipeline";
import { formatNumber } from "./labels";
import { MetricsTracker } from "./metrics";
import { ShowcaseView } from "./showcase";
import { RoomView } from "./room";
import { DashboardStore } from "./state";
import { Subtitles } from "./subtitles";
import { SHOW_BRAND_LINE } from "./uiconfig";
import { UiModeController } from "./uimode";
import { FIXED_VOICE, VoiceEngine, voiceAudioNode } from "./voice";
import { attachVoicePicker } from "./voicepicker";
import { collectElements, renderAll, renderConnection, renderMetrics, renderNotice } from "./views";
import type { PoseStatePayload } from "./protocol";
import { VideoFeed, type VideoSource } from "./video";

const params = new URLSearchParams(window.location.search);

const elements = collectElements();
const store = new DashboardStore();
const metrics = new MetricsTracker();

const videoImage = byId<HTMLImageElement>("video");
const videoNote = byId("video-note");
let video: VideoSource = new VideoFeed(videoImage, videoNote);

const showcase = new ShowcaseView({
  chips: byId("showcase-chips"),
  countsGroup: byId("showcase-counts-group"),
  counts: byId("showcase-counts"),
  objectsGroup: byId("showcase-objects-group"),
  objects: byId("showcase-objects"),
  meta: byId("showcase-meta"),
});

/**
 * The avatar replaces the SVG face. The dev-only query parameters pin one
 * frame for the screenshots under `build/web-shots/`: a mood, a speaking
 * level, the attention point and presence.
 */
const debugGaze = params.get("gaze");
const avatarDebug = import.meta.env.DEV
  ? {
      expression: params.get("expression") ?? undefined,
      mouth: params.has("mouth") ? Number(params.get("mouth")) : undefined,
      gaze:
        debugGaze === "left"
          ? { x: -1, y: 0 }
          : debugGaze === "right"
            ? { x: 1, y: 0 }
            : debugGaze === "up"
              ? { x: 0, y: 0.8 }
              : undefined,
      presence: params.has("presence") ? Number(params.get("presence")) : undefined,
    }
  : undefined;
const avatar = new AvatarView(byId("avatar"), {
  debug: avatarDebug,
  // The sphere speaks with the amplitude of the local voice (R2-V); Web Speech
  // has no audio graph and falls back to the voice driver's own envelope.
  audio: voiceAudioNode,
});
if (import.meta.env.DEV) {
  // Runtime handle for the dev screenshot harness and the perf probes.
  (window as unknown as { __cpsaAvatar?: AvatarView }).__cpsaAvatar = avatar;
}
const room = new RoomView(byId("room"));
/** ARI's line: its own node, written only when the spoken line changes. */
const subtitles = new Subtitles(byId("bubble"));
byId("brand-line").hidden = !SHOW_BRAND_LINE;

/**
 * Presentation mode by default; the top-right button and `d` show the debug
 * readouts (scores, "mente", fps, voice picker), remembered per browser.
 */
new UiModeController({
  root: document.documentElement,
  button: byId<HTMLButtonElement>("debug-toggle"),
  params,
});
/**
 * Debug "Calibrazione" (button or `c`): YOLOX zoom/centre and the camera image,
 * applied live on the board and stored there; the dashed outline on the video
 * shows the crop while the card is open.
 */
const calibration = new CalibrationPanel({
  root: byId("calibration"),
  rect: byId("analysis-rect"),
  frame: videoImage.parentElement ?? videoImage,
  send: (values) => client.setCalibration(values),
  reset: () => client.resetCalibration(),
  isDebug: () => document.documentElement.dataset.ui === "debug",
});
if (params.get("calibration") === "1") {
  // `?debug=1&calibration=1`: open the card at load (screenshots, calibration day).
  calibration.setOpen(true);
}
// `?voice=1` marks this tab as the speaking one; the click gate still applies.
const voice = new VoiceEngine({
  enabled: params.get("voice") === "1",
  // XTTS Ana Florence is the demo's voice; the debug picker can still override it.
  defaultPick: FIXED_VOICE,
  onStatusChange: () => renderVoice(),
});
// The voice picker keeps its own module: see `voicepicker.ts`.
attachVoicePicker(byId("voice-picker"), voice);

const voiceBadge = byId("voice-badge");
const voiceGate = byId<HTMLButtonElement>("voice-gate");
const muteToggle = byId<HTMLButtonElement>("mute-toggle");
const muteLabel = byId("mute-label");
const voicePaused = byId("voice-paused");
const showcaseEmpty = byId("showcase-empty");
const mente = byId("mente");
const menteEvent = byId("mente-event");
const menteRule = byId("mente-rule");
const menteUtility = byId("mente-utility");
const menteScene = byId("mente-scene");
const menteFace = byId("mente-face");
const presenceYolo = byId("presence-yolo");
const presenceFace = byId("presence-face");
const presencePose = byId("presence-pose");
const presenceState = byId("presence-state");
const presenceSmile = byId("presence-smile");
const presenceCue = byId("presence-cue");

const mind = new CognitiveAgent({
  engine: new CognitiveEngine(RULEBOOK),
  // Warm the voice cache with the lines the engine is about to choose from.
  prefetch: (texts) => voice.prefetch(texts),
  memory: new CognitiveMemory(),
  perception: new Perception(),
  speak: (text, handlers) => {
    if (bubbleTimer !== null) {
      clearTimeout(bubbleTimer);
      bubbleTimer = null;
    }
    subtitles.show(text);
    avatar.beginUtterance(expression, Date.now());
    voice.speak(text, {
      onStart: () => {
        handlers.onStart?.();
        avatar.setSpeaking(true);
      },
      onBoundary: () => handlers.onBoundary?.(),
      onEnd: (reason) => {
        handlers.onEnd?.(reason);
        avatar.endUtterance(Date.now());
        if (bubbleTimer !== null) {
          clearTimeout(bubbleTimer);
        }
        bubbleTimer = window.setTimeout(() => {
          bubbleTimer = null;
          subtitles.clear();
        }, 4_500);
      },
    });
    renderVoice();
  },
  onExpression: (next) => {
    expression = next;
    avatar.hold(next, Date.now());
  },
  onMind: () => renderMind(),
});

let expression: Expression = "neutral";
let bubbleTimer: number | null = null;
let frameWidth = 640;
let frameHeight = 480;
let lastActivityAt = Date.now();
let lastPose: PoseStatePayload | null = null;
let lastPoseAt = 0;
/** Which model view the top-bar switch asks for (F-E1). */
let viewMode: "yolox" | "movenet" = params.get("view") === "skeleton" ? "movenet" : "yolox";
/** The `seq` of the last `model.set`, to recognise a board that ignores it. */
let modelCommandSeq: number | null = null;
/**
 * The board runs one model at a time: a swap unloads one `.xmodel` and loads
 * the other in about half a second, and `status` may say `switching`. Until the
 * first frame of the new view arrives the right panel says what is happening.
 */
let switching: { mode: "yolox" | "movenet"; since: number } | null = null;
/** No board is that slow: after this the transition note gives way to the pane. */
const SWITCH_TIMEOUT_MS = 3_000;
/** The swap takes ~0.5 s: keep the note up long enough to be read. */
const SWITCH_MIN_MS = 450;

const modeObjects = byId<HTMLButtonElement>("mode-objects");
const modeSkeleton = byId<HTMLButtonElement>("mode-skeleton");
const showcasePanel = byId("showcase-panel");
const switchingNote = byId("switching-note");

/** The transition note and the empty invitation, kept in sync with the mode. */
function syncViewNotes(): void {
  switchingNote.hidden = switching === null;
  if (switching !== null) {
    switchingNote.textContent =
      switching.mode === "movenet" ? "Cambio modello: scheletro…" : "Cambio modello: oggetti…";
    showcaseEmpty.hidden = true;
    return;
  }
  const skeleton = viewMode === "movenet";
  showcaseEmpty.hidden = skeleton;
}

function renderMode(): void {
  const skeleton = viewMode === "movenet";
  modeObjects.classList.toggle("mode-active", !skeleton);
  modeSkeleton.classList.toggle("mode-active", skeleton);
  modeObjects.setAttribute("aria-pressed", String(!skeleton));
  modeSkeleton.setAttribute("aria-pressed", String(skeleton));
  showcasePanel.hidden = skeleton;
  room.setVisible(skeleton);
  syncViewNotes();
}

function setViewMode(mode: "yolox" | "movenet"): void {
  viewMode = mode;
  switching = { mode, since: Date.now() };
  modelCommandSeq = client.setModel(mode);
  renderMode();
}

/** The new view published a frame (or the board says it is running again). */
function settleSwitch(mode: "yolox" | "movenet"): void {
  if (switching === null) {
    return;
  }
  if (switching.mode !== mode) {
    // Another switch happened while this one was in flight: keep the newest.
    return;
  }
  // A frame already in flight from the old model must not end the swap the
  // board just announced.
  if (store.state.status?.state === "switching") {
    return;
  }
  if (Date.now() - switching.since < SWITCH_MIN_MS) {
    return;
  }
  switching = null;
  renderMode();
}

modeObjects.addEventListener("click", () => setViewMode("yolox"));
modeSkeleton.addEventListener("click", () => {
  setViewMode("movenet");
  if (lastPose !== null) {
    room.update(lastPose, Date.now());
  }
});

/**
 * Face cues run in this laptop browser (MediaPipe on `/video.mjpeg`), never on
 * the board; `faceCues` is the tracker or the scripted mock of `?mock=1`.
 */
const faceTracker = new FaceCueTracker({
  onCue: (cue) => {
    mind.onFace(cue.kind, cue.score, cue.faces, Date.now());
  },
  // Faces are a presence source too (C3): a close visitor YOLOX misses.
  onFaces: (sample) => mind.onFaces(sample.count, Date.now()),
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
  renderPresence(snapshot);
}

/** Write a node only when its text changes: this runs at the 10 Hz heartbeat. */
function setText(node: HTMLElement, text: string): void {
  if (node.textContent !== text) {
    node.textContent = text;
  }
}

function setDot(node: HTMLElement, on: boolean): void {
  const value = on ? "on" : "off";
  if (node.dataset.state !== value) {
    node.dataset.state = value;
  }
}

/**
 * Debug readout (C3): why ARI greets, says goodbye or reacts to a smile —
 * the three presence sources, the fused state, the live smile score against
 * its per-face threshold and the last face cue.
 */
function renderPresence(snapshot: MindSnapshot): void {
  if (document.documentElement.dataset.ui !== "debug") {
    return;
  }
  const presence = snapshot.presence;
  setDot(presenceYolo, presence.sources.yolo);
  setDot(presenceFace, presence.sources.face);
  setDot(presencePose, presence.sources.pose);
  const state = presence.present
    ? presence.missingMs !== null
      ? `presente · nessuna fonte da ${formatNumber(presence.missingMs / 1000, 1)} s`
      : `presente · ${presence.people} ${presence.people === 1 ? "persona" : "persone"}`
    : "assente";
  const visit = presence.greeted ? " · salutato" : "";
  setText(presenceState, `${state}${visit}`);
  const smile = faceTracker.smile;
  setText(
    presenceSmile,
    smile === null
      ? "—"
      : `${formatNumber(smile.value, 2)} / ${formatNumber(smile.threshold, 2)}` +
          (smile.baseline === null ? " (base …)" : ` (base ${formatNumber(smile.baseline, 2)})`),
  );
  const cue = snapshot.lastFace;
  setText(
    presenceCue,
    cue === null
      ? "—"
      : `${cue.kind} ${formatNumber(cue.score, 2)} · ${Math.round((Date.now() - cue.at) / 1000)} s fa`,
  );
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
  muteLabel.textContent = snapshot.muted ? "Muto" : "Voce";
  muteToggle.setAttribute("aria-label", snapshot.muted ? "Riattiva la voce" : "Muta la voce");
  voicePaused.hidden = !snapshot.audioSuspended;
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
let client: DashboardClient;
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
/** The model of the current connection is in sync with the switch. */
let modelSynced = false;

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

  client = new DashboardClient({
    url: socketUrl,
    factory,
    handlers: {
      onMessage: (message) => {
        const now = Date.now();
        if (message.type === "status") {
          metrics.status(message.data);
          calibration.setStatus(message.data.calibration);
          mind.onStatus(message.data, now);
          if (message.data.state === "switching") {
            if (switching === null) {
              switching = { mode: viewMode, since: now };
              renderMode();
            }
          } else if (message.data.state === "running") {
            settleSwitch(viewMode);
          }
        }
        if (message.type === "showcase.state") {
          metrics.showcase(message.data);
          calibration.setFrameSize(message.data.width, message.data.height);
          frameWidth = message.data.width;
          frameHeight = message.data.height;
          if (message.data.objects.length > 0) {
            lastActivityAt = now;
          }
          mind.onShowcase(message.data, now);
          settleSwitch("yolox");
        }
        if (message.type === "pose.state") {
          lastActivityAt = now;
          frameWidth = message.data.width;
          frameHeight = message.data.height;
          lastPose = message.data;
          lastPoseAt = now;
          mind.onPose(message.data, now);
          if (viewMode === "movenet") {
            room.update(message.data, now);
            settleSwitch("movenet");
          }
        }
        if (
          message.type === "error" &&
          message.data.code === "bad_command" &&
          modelCommandSeq !== null &&
          message.data.ref_seq === modelCommandSeq
        ) {
          // A board that predates F-E1: the switch keeps working as a view
          // toggle, without a red banner for a command it does not know yet.
          modelCommandSeq = null;
          showTransportNotice("Il server non supporta ancora il cambio modello: vista solo locale.");
          return;
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
        calibration.setConnected(info.phase === "open");
        if (info.phase === "open" && !modelSynced) {
          // A deep link to the skeleton view still has to tell the board (and
          // the mock) which model to run; `seq` restarts at 1 per connection.
          modelSynced = true;
          if (viewMode === "movenet") {
            modelCommandSeq = client.setModel("movenet");
          }
        }
        if (info.phase !== "open") {
          modelSynced = false;
        }
        mind.onConnection(info.phase === "open", Date.now());
        refreshVideo();
        renderConnection(elements, info);
        renderMetrics(elements, metrics.read());
        renderNotice(elements, input());
      },
      onNotice: showTransportNotice,
    },
  });
  client.start();

  // The cognitive heartbeat: rule candidates, face, calm panel, voice label.
  window.setInterval(() => {
    const now = Date.now();
    mind.tick(now);
    const world = mind.perception.world;
    const idleS = (now - lastActivityAt) / 1_000;
    avatar.tick(now, idleS);
    avatar.updateGaze(
      world.largestPerson === null ? [] : [world.largestPerson],
      { width: frameWidth, height: frameHeight },
      now,
      world.largestPerson === null,
    );
    // Face when someone is there — a YOLO person box or a MoveNet pose with a
    // real body in it — and back to the sphere after 3 s alone.
    const posePresent = lastPose !== null && now - lastPoseAt < 2_000 && lastPose.score >= 0.3;
    avatar.setEngaged(
      world.largestPerson !== null || posePresent || mind.presence.isPresent,
      now,
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
    if (switching !== null && now - switching.since > SWITCH_TIMEOUT_MS) {
      switching = null;
      renderMode();
    }
    // `showcase-empty` belongs to the mode alone (ShowcaseView never touches it).
    syncViewNotes();
    renderVoice();
  }, HEARTBEAT_MS);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      voice.cancel();
      mind.interrupt();
      faceCues.stop();
    } else if (streaming) {
      faceCues.start();
    }
  });

  /*
   * "Come funziona" runs from load until the operator starts the demo: that
   * one gesture also unlocks the audio (it is the voice gate). `i` or the
   * debug button bring the loop back and silence ARI.
   */
  const attract = new AttractView({
    root: byId("attract"),
    video: videoImage,
    live: liveNumbers,
    params,
    returnButton: byId<HTMLButtonElement>("attract-return"),
    onActivate: () => {
      voice.activate();
      renderVoice();
      avatar.setPaused(false);
      room.setPaused(false);
    },
    onReturn: () => {
      avatar.setPaused(true);
      room.setPaused(true);
      voice.disable();
      mind.interrupt();
      subtitles.clear();
      renderVoice();
    },
  });
  attract.start();
  // The loop covers the whole page: the sphere and the room need no frames.
  avatar.setPaused(attract.phase === "loop");
  room.setPaused(attract.phase === "loop");

  renderAll(elements, input());
  renderVoice();
  renderMind();
  renderMode();
  refreshVideo();
  startBench();
}

/** The numbers the attract loop shows live; null when not connected. */
function liveNumbers(): LiveNumbers {
  if (connection.phase !== "open") {
    return { fps: null, dpuMs: null, labels: [] };
  }
  const showcaseState = store.state.showcase;
  const counts = new Map<string, number>();
  for (const object of showcaseState?.objects ?? []) {
    counts.set(object.label, (counts.get(object.label) ?? 0) + 1);
  }
  const labels = [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .map(([name]) => name);
  return {
    fps: metrics.read().fps,
    dpuMs: showcaseState?.dpu_ms ?? store.state.status?.dpu_ms ?? null,
    labels,
  };
}

/**
 * Dev-only `?bench=1`: frames per second of the whole page once the scene has
 * settled, published in the title and on `window` for the screenshot harness.
 *
 * The first three seconds are warm-up and excluded: shader compilation, the
 * head asset decode and the mock camera loop all land there, and counting them
 * turns a 60 fps page into a 20 fps average. Steady state is reported as the
 * median and 95th percentile frame time, which survive a busy laptop better
 * than a mean.
 */
function startBench(): void {
  if (!import.meta.env.DEV || params.get("bench") !== "1") {
    return;
  }
  const warmupMs = 3_000;
  const sampleMs = 6_000;
  const startedAt = performance.now();
  const samples: number[] = [];
  let previous = startedAt;
  const step = (now: number): void => {
    const delta = now - previous;
    previous = now;
    const elapsed = now - startedAt;
    if (elapsed < warmupMs) {
      requestAnimationFrame(step);
      return;
    }
    samples.push(delta);
    if (elapsed < warmupMs + sampleMs) {
      requestAnimationFrame(step);
      return;
    }
    const sorted = [...samples].sort((left, right) => left - right);
    const at = (fraction: number): number =>
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] ?? 0;
    const fps = samples.length / ((elapsed - warmupMs) / 1_000);
    const report = {
      fps: Math.round(fps * 10) / 10,
      medianFrameMs: Math.round(at(0.5) * 10) / 10,
      p95FrameMs: Math.round(at(0.95) * 10) / 10,
      worstFrameMs: Math.round((sorted[sorted.length - 1] ?? 0) * 10) / 10,
      frames: samples.length,
    };
    (window as unknown as { __cpsaBench?: typeof report }).__cpsaBench = report;
    document.title = `bench ${report.fps} fps · ${report.medianFrameMs} ms median`;
    console.info("[cpsa] bench", report);
  };
  requestAnimationFrame(step);
}

boot().catch((error: unknown) => {
  const text = error instanceof Error ? error.message : String(error);
  showTransportNotice(`Avvio non riuscito: ${text}`);
});
