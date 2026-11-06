import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000", "/results": "http://localhost:8000", "/v1": "http://localhost:8000" } },
  test: { include: ["src/**/*.test.ts"] },
});
