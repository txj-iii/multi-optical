import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("./app.js", import.meta.url), "utf8");
const server = readFileSync(new URL("./server.mjs", import.meta.url), "utf8");
const workflow = readFileSync(new URL("../../train/workbench_workflow.py", import.meta.url), "utf8");

for (const forbidden of ["adjust", "retrain-main"]) {
  assert.equal(
    server.includes(`/${forbidden}`) || server.includes(`"${forbidden}"`),
    false,
    `server must not expose ${forbidden}`,
  );
}

assert.equal(app.includes("/adjust"), false, "UI must not call adjust API");
assert.equal(app.includes("train_dir"), false, "UI must not display training directories");
assert.equal(app.includes("last_train_stdout"), false, "UI must not display training logs");

for (const forbidden of [
  "run_training.py",
  "CANDIDATE_",
  "MAIN_BLEND_",
  "retrain-main",
  "def _run_candidate_training",
  "def retrain_with_main_dataset",
]) {
  assert.equal(workflow.includes(forbidden), false, `workflow must not contain training entry ${forbidden}`);
}

console.log("cpu_workflow ok");
