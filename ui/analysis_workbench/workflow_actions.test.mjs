import assert from "node:assert/strict";
import { describeHoldAction } from "./workflow_actions.mjs";

const pending = describeHoldAction({ status: "pending_review" }, "SAMPLE_052");
assert.equal(pending.label, "暂不采用");
assert.equal(pending.pendingText, "SAMPLE_052 暂不采用当前标注...");
assert.equal(pending.endpointAction, "hold");

const approved = describeHoldAction({ status: "approved" }, "SAMPLE_052");
assert.equal(approved.label, "取消采用");
assert.equal(approved.pendingText, "SAMPLE_052 取消采用并回到审核队列...");
assert.equal(approved.endpointAction, "hold");

const held = describeHoldAction({ status: "held" }, "SAMPLE_052");
assert.equal(held.label, "暂不采用");
assert.equal(held.pendingText, "SAMPLE_052 暂不采用当前标注...");

console.log("workflow_actions ok");
