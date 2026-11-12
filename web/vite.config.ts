import { defineConfig } from "vitest/config";

// One static app, no imports from outside `web/`: the dev-only mock is
// self-contained, and the production build never includes it.
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
