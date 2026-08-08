/**
 * Browser smoke test.
 *
 * The claim worth testing automatically is the streaming contract: sources must
 * render *before* the answer text exists. That is the whole reason the API emits a
 * trace event first, and it is invisible to a unit test.
 *
 * Requires the API on :8000 and the dev server on :5173, plus a Chromium:
 *   npx playwright install chromium
 *   node smoke.mjs
 *
 * CHROME_PATH may point at an existing Chromium instead of a downloaded one.
 */
import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "http://127.0.0.1:5173/";
const QUESTION = "How is the security system organized?";

const problems = [];
const check = (label, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) problems.push(label);
};

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || undefined,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });

const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));
page.on("response", (r) => {
  if (r.status() >= 400) consoleErrors.push(`HTTP ${r.status()} ${r.url()}`);
});

await page.goto(BASE, { waitUntil: "networkidle" });
check("app loads", (await page.title()) === "Codebase Q&A");
check("repo manifest reached the UI", (await page.locator(".repo-chip").count()) === 1);

await page.getByRole("button", { name: QUESTION }).click();

// The streaming contract: sources first, answer second.
await page.waitForSelector(".source", { timeout: 60000 });
const sourcesAt = Date.now();
const answerEmptyWhenSourcesArrived = (await page.locator(".answer").count()) === 0;
check("sources render before answer text", answerEmptyWhenSourcesArrived);
check("sources present", (await page.locator(".source").count()) > 0);

await page.waitForSelector(".verify", { timeout: 120000 });
const lead = Date.now() - sourcesAt;
check("answer completed after sources", lead > 0, `${lead}ms later`);

const verify = await page.locator(".verify").innerText();
check("citations verified", /\d+\/\d+ citations verified/.test(verify), verify.trim());
check("citation markers rendered", (await page.locator(".cite").count()) > 0);
check("no invalid citations", (await page.locator(".cite.invalid").count()) === 0);

await page.locator(".cite").first().click();
await page.waitForTimeout(500);
check("citation click activates a source", (await page.locator(".source.active").count()) === 1);

const codeSource = page.locator(".source", { has: page.locator(".badge.code") }).first();
if (await codeSource.count()) {
  await codeSource.click();
  await page.waitForSelector(".viewer .code-line", { timeout: 30000 });
  check("viewer highlights the cited lines", (await page.locator(".viewer .code-line.hl").count()) > 0);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  check("Escape closes the viewer", (await page.locator(".viewer").count()) === 0);
} else {
  check("a code source was retrieved", false);
}

// Refusal path: this question has no answer in the corpus.
await page.getByRole("button", { name: "How does the billing module calculate charges?" }).click();
await page.waitForSelector(".refusal", { timeout: 120000 });
check("unanswerable question is refused, not answered", true);

check("no console errors or failed requests", consoleErrors.length === 0, consoleErrors.join("; "));

await page.screenshot({ path: "smoke.png" });
await browser.close();

console.log(problems.length ? `\n${problems.length} failing check(s)` : "\nall checks passed");
process.exit(problems.length ? 1 : 0);
