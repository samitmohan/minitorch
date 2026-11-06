import { defineConfig } from "@playwright/test";

// Runs against the FastAPI server serving the built UI: npm run build, then this.
export default defineConfig({
  testDir: "e2e",
  use: { baseURL: "http://localhost:8000" },
  webServer: {
    command: "cd .. && uv run uvicorn minitorch.serve.api:app --port 8000",
    url: "http://localhost:8000/api/runs",
    reuseExistingServer: true,
  },
});
