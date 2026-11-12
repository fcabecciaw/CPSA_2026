#!/usr/bin/env node
/**
 * Screenshots of the particle avatar and the skeleton room (R2-F).
 *
 * It starts the Vite dev server, drives headless Chrome over the DevTools
 * protocol (no extra dependency: `fetch` and `WebSocket` are in Node 22) and
 * writes the PNGs to `../../build/web-shots/`, which is git-ignored.
 *
 *   node scripts/shots.mjs                 # every shot at 1280x720
 *   node scripts/shots.mjs --only mood-happy
 *   node scripts/shots.mjs --width 1920 --height 1080
 *   node scripts/shots.mjs --only demo-present-objects,demo-debug-skeleton --suffix
 *
 * The URL parameters are the dev-only hooks of `main.ts` and `avatar/view.ts`:
 * `?mock=1`, `expression`, `mouth`, `gaze=left|right|up`, `presence`,
 * `view=skeleton`, `attract=0` (skip "Come funziona"), `scene=N` (pin one of its
 * scenes), `debug=1|0` (presentation or debug mode for that load) and
 * `bench=1`. `--suffix` appends the viewport (`-1280x720`) to the file names.
 */

import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const WIDTH = Number(args.width ?? 1280);
const HEIGHT = Number(args.height ?? 720);
const PORT = Number(args.port ?? 5173);
const OUT = resolve(WEB, "..", "build", "web-shots");
const BASE = `http://127.0.0.1:${PORT}/`;

/** One shot: URL suffix, how long to let the scene settle, and the file name. */
const SHOTS = [
  // "Come funziona", the attract loop, one shot per scene (pinned).
  ...[1, 2, 3, 4, 5, 6, 7].map((scene) => ({
    name: `attract-${scene}`,
    query: `mock=1&scene=${scene}`,
    settleMs: 3200,
  })),
  // The demo itself, presentation (default) and debug modes, both views.
  { name: "demo-present-objects", query: "mock=1&attract=0&debug=0", settleMs: 7000 },
  {
    name: "demo-present-voice",
    query: "mock=1&attract=0&debug=0",
    click: "#voice-gate",
    settleMs: 9000,
  },
  { name: "demo-present-skeleton", query: "mock=1&attract=0&debug=0&view=skeleton", settleMs: 7000 },
  { name: "demo-debug-objects", query: "mock=1&attract=0&debug=1", settleMs: 7000 },
  { name: "demo-debug-skeleton", query: "mock=1&attract=0&debug=1&view=skeleton", settleMs: 7000 },
  // The seven moods of the sphere, each pinned through the dev-only hook.
  { name: "mood-neutral", query: "mock=1&attract=0&expression=neutral&presence=1", settleMs: 2000 },
  { name: "mood-happy", query: "mock=1&attract=0&expression=happy&presence=1", settleMs: 2000 },
  { name: "mood-surprised", query: "mock=1&attract=0&expression=surprised&presence=1", settleMs: 2000 },
  { name: "mood-curious", query: "mock=1&attract=0&expression=curious&presence=1", settleMs: 2000 },
  { name: "mood-thinking", query: "mock=1&attract=0&expression=thinking&presence=1", settleMs: 2000 },
  { name: "mood-sleepy", query: "mock=1&attract=0&expression=sleepy&presence=1", settleMs: 2000 },
  { name: "mood-worried", query: "mock=1&attract=0&expression=worried&presence=1", settleMs: 2000 },
  // The voice: the surface swells on a syllable as the analyser reports it.
  { name: "speaking", query: "mock=1&attract=0&expression=happy&mouth=0.9&presence=1", settleMs: 2200 },
  // Attention: the cloud leans, bulges and brightens toward the person.
  { name: "gaze-left", query: "mock=1&attract=0&expression=curious&gaze=left&presence=1", settleMs: 2200 },
  { name: "gaze-right", query: "mock=1&attract=0&expression=curious&gaze=right&presence=1", settleMs: 2200 },
  {
    // The ~0.5 s single-model swap: the note that replaces the empty pane.
    name: "switching-note",
    query: "mock=1&attract=0",
    click: "#mode-skeleton",
    settleMs: 60,
  },
  {
    name: "room-skeleton",
    query: "mock=1&attract=0&view=skeleton",
    settleMs: 6000,
  },
];

const selected = args.query
  ? [
      {
        name: args.name ?? "adhoc",
        query: args.query,
        settleMs: Number(args.settle ?? 3500),
      },
    ]
  : args.only
    ? SHOTS.filter((shot) => args.only.split(",").includes(shot.name))
    : SHOTS;
if (selected.length === 0) {
  console.error(`nessuna schermata: ${args.only}`);
  process.exit(2);
}

const workdir = join(tmpdir(), `cpsa-shots-${process.pid}`);
await mkdir(OUT, { recursive: true });
await mkdir(workdir, { recursive: true });

const vite = spawn("npx", ["vite", "--port", String(PORT), "--strictPort"], {
  cwd: WEB,
  stdio: ["ignore", "pipe", "pipe"],
});
const chromeArgs = [
  "--remote-debugging-port=0",
  `--user-data-dir=${join(workdir, "chrome")}`,
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-extensions",
  "--disable-background-networking",
  "--hide-scrollbars",
  "--mute-audio",
  "--force-device-scale-factor=1",
  `--window-size=${WIDTH},${HEIGHT}`,
  "--window-position=0,0",
];
if (args.gpu) {
  // A real window on $DISPLAY: the honest frame rate of the laptop GPU.
  chromeArgs.push("--use-gl=angle", "--use-angle=gl");
} else {
  // SwiftShader keeps WebGL available on a headless machine.
  chromeArgs.push(
    "--headless=new",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  );
}
const chrome = spawn("google-chrome", chromeArgs, { stdio: ["ignore", "ignore", "pipe"] });

let exitCode = 0;
try {
  await waitForHttp(BASE, 30_000);
  const devtools = await waitForDevtools(join(workdir, "chrome"), 20_000);
  const session = connect(devtools.webSocketDebuggerUrl);
  const page = await session.newPage();
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: WIDTH,
    height: HEIGHT,
    deviceScaleFactor: 1,
    mobile: false,
  });

  for (const shot of selected) {
    const url = `${BASE}?${shot.query}`;
    await page.goto(url);
    await page.waitFor("document.querySelector('#avatar')?.dataset.status === 'ready'", 20_000);
    if (shot.click) {
      await page.evaluate(`document.querySelector(${JSON.stringify(shot.click)})?.click()`);
    }
    await sleep(shot.settleMs);
    const png = await page.screenshot();
    const suffix = args.suffix ? `-${WIDTH}x${HEIGHT}` : "";
    const file = join(OUT, `${shot.name}${suffix}.png`);
    await writeFile(file, png);
    const bench = await page.evaluate("JSON.stringify(window.__cpsaBench ?? null)");
    console.log(`${shot.name}: ${file}${bench && bench !== "null" ? ` · bench ${bench}` : ""}`);
  }
  await page.close();
} catch (error) {
  console.error(`schermate non riuscite: ${error instanceof Error ? error.message : error}`);
  exitCode = 1;
} finally {
  chrome.kill("SIGKILL");
  vite.kill("SIGTERM");
  await sleep(300);
  await rm(workdir, { recursive: true, force: true, maxRetries: 5, retryDelay: 150 });
}
process.exit(exitCode);

function parseArgs(list) {
  const parsed = {};
  for (let index = 0; index < list.length; index += 1) {
    const key = list[index];
    if (!key.startsWith("--")) {
      continue;
    }
    const value = list[index + 1];
    if (value === undefined || value.startsWith("--")) {
      parsed[key.slice(2)] = true;
      continue;
    }
    parsed[key.slice(2)] = value;
    index += 1;
  }
  return parsed;
}

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function waitForHttp(url, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (response.ok) {
        return;
      }
    } catch {
      // The dev server is still starting.
    }
    await sleep(150);
  }
  throw new Error(`vite non risponde su ${url}`);
}

async function waitForDevtools(profile, timeoutMs) {
  const file = join(profile, "DevToolsActivePort");
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const text = await readText(file);
      const [port] = text.split("\n");
      const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
      return version;
    } catch {
      // Chrome is still starting.
    }
    await sleep(120);
  }
  throw new Error("Chrome non espone DevTools");
}

async function readText(path) {
  const { readFile } = await import("node:fs/promises");
  return readFile(path, "utf8");
}

/** A tiny CDP client: one WebSocket per page target and request/reply by id. */
function connect(webSocketDebuggerUrl) {
  const targetUrl = new URL(webSocketDebuggerUrl);
  const httpBase = `http://${targetUrl.host}`;

  return {
    async newPage() {
      const response = await fetch(`${httpBase}/json/new?about:blank`, { method: "PUT" });
      const target = await response.json();
      const socket = await openSocket(target.webSocketDebuggerUrl);
      await socket.send("Page.enable");
      await socket.send("Runtime.enable");
      return {
        send: socket.send,
        async goto(url) {
          await socket.send("Page.navigate", { url });
        },
        async waitFor(expression, timeoutMs) {
          const started = Date.now();
          while (Date.now() - started < timeoutMs) {
            const result = await socket.send("Runtime.evaluate", {
              expression,
              returnByValue: true,
            });
            if (result.result?.value === true) {
              return;
            }
            await sleep(120);
          }
          throw new Error(`condizione non soddisfatta: ${expression}`);
        },
        async evaluate(expression) {
          const result = await socket.send("Runtime.evaluate", {
            expression,
            returnByValue: true,
          });
          return result.result?.value;
        },
        async screenshot() {
          const result = await socket.send("Page.captureScreenshot", {
            format: "png",
            captureBeyondViewport: false,
          });
          return Buffer.from(result.data, "base64");
        },
        close() {
          return socket.close();
        },
      };
    },
  };
}

function openSocket(url) {
  return new Promise((resolveSocket, reject) => {
    const socket = new WebSocket(url);
    let nextId = 0;
    const pending = new Map();
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const resolver = pending.get(message.id);
      if (resolver !== undefined) {
        pending.delete(message.id);
        if (message.error) {
          resolver.reject(new Error(message.error.message));
        } else {
          resolver.resolve(message.result);
        }
      }
    });
    socket.addEventListener("error", (event) => reject(event.error ?? new Error("WebSocket")));
    socket.addEventListener("open", () => {
      resolveSocket({
        send(method, params = {}) {
          const id = ++nextId;
          socket.send(JSON.stringify({ id, method, params }));
          return new Promise((resolveSend, rejectSend) => {
            pending.set(id, { resolve: resolveSend, reject: rejectSend });
            setTimeout(() => {
              if (pending.delete(id)) {
                rejectSend(new Error(`timeout CDP ${method}`));
              }
            }, 30_000);
          });
        },
        close() {
          socket.close();
          return Promise.resolve();
        },
      });
    });
  });
}
