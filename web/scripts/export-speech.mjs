#!/usr/bin/env node
/**
 * Export every pre-renderable sentence of ARI to JSON for the local TTS cache.
 *
 * Loads `src/cognitive/speechcatalog.ts` through Vite's SSR loader, so the
 * rendering uses the very same grammar and rulebook as the dashboard, then
 * writes `{generated_at, counts, lines: [{text, group, source}]}`.
 *
 *   node scripts/export-speech.mjs [out.json]   (default ../build/speech-catalog.json)
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = resolve(process.argv[2] ?? resolve(web, "..", "build", "speech-catalog.json"));

const server = await createServer({
  root: web,
  configFile: false,
  logLevel: "error",
  appType: "custom",
  server: { middlewareMode: true, hmr: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: [] },
});
try {
  const { speechCatalog, SPEECH_GROUPS } = await server.ssrLoadModule(
    "/src/cognitive/speechcatalog.ts",
  );
  const lines = speechCatalog();
  const counts = Object.fromEntries(
    SPEECH_GROUPS.map((group) => [group, lines.filter((line) => line.group === group).length]),
  );
  counts.total = lines.length;
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(
    out,
    `${JSON.stringify({ generated_at: new Date().toISOString(), counts, lines }, null, 1)}\n`,
    "utf8",
  );
  console.log(`${lines.length} sentences -> ${out}`, counts);
} finally {
  await server.close();
}
