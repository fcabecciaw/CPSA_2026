import { defineConfig } from "vite";

// The repository root holds `docs/protocol/v1/examples`, which the dev-only
// mock imports. The production build never includes the mock.
const repoRoot = decodeURIComponent(new URL("..", import.meta.url).pathname);

export default defineConfig({
  base: "./",
  server: {
    fs: {
      allow: [repoRoot],
    },
  },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
});
