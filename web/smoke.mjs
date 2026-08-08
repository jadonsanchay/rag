/**
 * Browser smoke test.
 *
 * Asserts the things a unit test cannot: that sources render *before* the answer
 * exists (the trace-first streaming contract), that a follow-up containing a
 * pronoun is rewritten into a standalone query before retrieval, that citation
 * spans verify against the working tree, and that an unanswerable question is
 * refused rather than invented.
 *
 * Needs the API on :8000 and the dev server on :5173, plus a Chromium:
 *   npx playwright install chromium && node smoke.mjs
 */
import { chromium } from "playwright";

const problems = [];
const check = (label, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) problems.push(label);
};

const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
check("app loads", (await page.title()) === "Codebase Q&A");
check("repo tabs rendered from the registry", (await page.locator(".repo-tab").count()) > 0);

// Target a known repo: "first ready tab" silently changes meaning as the
// registry changes, and the questions below are specific to this codebase.
const REPO = process.env.SMOKE_REPO ?? "psf-requests";
const tab = page.locator(".repo-tab", { hasText: REPO }).first();
check(`repo '${REPO}' is present`, (await tab.count()) > 0);
await tab.click();
await page.waitForTimeout(1200);
check("conversation input enabled for a ready repo",
      await page.locator(".ask.sticky input").isEnabled());

// First turn.
await page.locator(".ask.sticky input").fill("Where is the Session class defined?");
await page.locator(".ask.sticky button[type=submit]").click();
await page.waitForSelector(".turn-sources .source", { timeout: 90000 });
check("sources arrive before the answer", (await page.locator(".answer").count()) === 0);
await page.waitForSelector(".verify", { timeout: 120000 });
check("first answer verified",
      /citations verified/.test(await page.locator(".verify").first().innerText()));

// Follow-up with a pronoun — this is the step 12 claim.
await page.locator(".ask.sticky input").fill("and what does its request method do?");
await page.locator(".ask.sticky button[type=submit]").click();
await page.waitForSelector(".rewrite-note", { timeout: 120000 });
const rewrite = await page.locator(".rewrite-note").first().innerText();
check("follow-up was rewritten into a standalone query",
      /Session/i.test(rewrite) && !/^\s*searched for:\s*and\b/i.test(rewrite),
      rewrite.replace(/\n/g, " "));

await page.waitForFunction(() => document.querySelectorAll(".verify").length >= 2, null,
                           { timeout: 120000 });
check("two turns in the transcript", (await page.locator(".turn").count()) === 2);
await page.screenshot({ path: "smoke.png" });

// Source viewer still works inside a turn.
const codeSource = page.locator(".turn-sources .source", { has: page.locator(".badge.code") }).first();
if (await codeSource.count()) {
  await codeSource.click();
  await page.waitForSelector(".viewer .code-line", { timeout: 30000 });
  check("viewer opens from a turn's sources", (await page.locator(".viewer .code-line.hl").count()) > 0);
  await page.keyboard.press("Escape");
}

// Refusal: a question the corpus cannot answer must be declined, not invented.
await page.locator(".ask.sticky input").fill("How does the billing module calculate monthly charges?");
await page.locator(".ask.sticky button[type=submit]").click();
await page.waitForSelector(".refusal", { timeout: 120000 });
check("unanswerable question is refused, not answered", true);

check("no page errors", errors.length === 0, errors.slice(0, 2).join("; "));
await browser.close();
console.log(problems.length ? `\n${problems.length} failing check(s)` : "\nall checks passed");
process.exit(problems.length ? 1 : 0);
