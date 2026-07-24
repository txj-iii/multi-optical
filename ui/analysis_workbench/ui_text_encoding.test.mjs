import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const files = [
  "ui/analysis_workbench/app.js",
  "ui/analysis_workbench/curve_helpers.js",
  "ui/analysis_workbench/curve_helpers.mjs",
  "ui/analysis_workbench/hover_helpers.js",
  "ui/analysis_workbench/hover_helpers.mjs",
  "ui/analysis_workbench/index.html",
];

for (const file of files) {
  const text = readFileSync(new URL(`../${file.split("/").slice(1).join("/")}`, import.meta.url), "utf8");
  assert.equal(
    /\?{3,}/.test(text),
    false,
    `${file} still contains placeholder question-mark text`,
  );
}

console.log("ui_text_encoding ok");
