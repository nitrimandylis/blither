import { chromium } from "playwright-core";
const exe = `${process.env.HOME}/Library/Caches/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-mac-arm64/chrome-headless-shell`;
const b = await chromium.launch({ executablePath: exe });
const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
await p.goto("file://${import.meta.dir}/card.html");
await p.evaluate(() => document.fonts.ready);
await p.screenshot({ path: "../public/og.png", type: "png" });
await b.close();
// regenerate: bun add -d playwright-core && bun og/shot.ts
