import { expect, test } from "@playwright/test";

// The site is one post. These drive its live figures against the real FastAPI server and engine.

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Deep learning framework from scratch");
});

test("the training figure draws the run's loss curve", async ({ page }) => {
  await expect(page.getByLabel("Loss curve").first().locator(".recharts-line path").first()).toBeVisible({ timeout: 30_000 });
});

test("the autograd graph shows the gradients backward() computed", async ({ page }) => {
  // relu(x*w + b)^2 at x=2, w=-0.5, b=3: d/dx = 2 * 2 * w = -2, d/dw = 2 * 2 * x = 8
  const node = (label: string) =>
    page.getByLabel("Autograd graph").locator(".op-node").filter({ has: page.locator("strong", { hasText: new RegExp(`^${label}$`) }) });
  await expect(node("x").locator(".g")).toHaveText("∂ -2");
  await expect(node("w").locator(".g")).toHaveText("∂ 8");
});

test("bad expressions are refused with a reason", async ({ page }) => {
  await page.getByLabel("Expression").fill("__import__('os')");
  await page.getByRole("button", { name: "Run backward" }).click();
  await expect(page.getByText(/unsupported/)).toBeVisible();
});

test("the BPE figure steps from bytes to tokens", async ({ page }) => {
  await expect(page.getByLabel("BPE tokenizer").locator(".ladder li").first()).toBeVisible();
});

test("gradient checks all pass from the browser", async ({ page }) => {
  await page.getByRole("button", { name: "Run gradient checks" }).click();
  await expect(page.getByText(/(\d+) of \1 match/)).toBeVisible({ timeout: 30_000 });
});

test("benchmark figures and result tables render from the result files", async ({ page }) => {
  await expect(page.getByLabel("Tokens per second, higher is faster").locator("li").first()).toBeVisible();
  await expect(page.getByLabel("Attention kernels").getByRole("columnheader", { name: "v3 simdgroup matmul" })).toBeVisible();
  await expect(page.locator(".katex").first()).toBeVisible();
  await expect(page.locator("#table-4")).toContainText("CPU (NumPy), float32");
  await expect(page.locator("#table-A1")).toContainText("Gradient checks that agree");
  await expect(page.locator("#figure-7 > figcaption")).toContainText("ends at a validation loss of", { timeout: 30_000 });
});

test("chat with Qwen3 streams an answer through the engine", async ({ page }) => {
  const chat = page.getByLabel("Chat with Qwen3");
  await chat.getByLabel("Message").fill("What is 2 + 3? Reply with only the number.");
  await chat.getByRole("button", { name: "Send" }).click();
  await expect(chat.locator(".turn.assistant small")).toContainText("tokens/s", { timeout: 120_000 });
  await expect(chat.locator(".turn.assistant p")).toContainText("5");
});

test("the attention figure draws every head and enlarges one", async ({ page }) => {
  const fig = page.getByLabel("Attention heads");
  await expect(fig.getByRole("button", { name: "layer 1, head 1", exact: true })).toBeVisible({ timeout: 30_000 });
  await fig.getByRole("button", { name: "layer 2, head 3", exact: true }).click();
  await expect(fig.getByText("Layer 2, head 3.")).toBeVisible();
  await expect(fig.locator("svg rect").first()).toBeVisible();
});

test("retrieval finds three paragraphs and answers from them", async ({ page }) => {
  test.setTimeout(240_000);  // the first question loads Qwen3-Embedding and Qwen3
  const demo = page.getByLabel("Retrieval demo");
  await demo.getByRole("button", { name: "Search and answer" }).click();
  await expect(demo.locator(".passage")).toHaveCount(3, { timeout: 220_000 });
  await expect(demo.locator(".answer")).toContainText("German");  // apps/rag.py's own answer to this SQuAD question
});

test("the agent calls the calculator and answers", async ({ page }) => {
  test.setTimeout(240_000);
  const demo = page.getByLabel("Agent demo");
  await demo.getByRole("button", { name: "Solve with the calculator" }).click();
  await expect(demo.locator(".trace li").last()).toContainText("Final answer", { timeout: 220_000 });
  await expect(demo.locator(".trace code").first()).toBeVisible();  // at least one calculator call
});
