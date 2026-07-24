import assert from "node:assert/strict";
import { getAuditOverlay, getDefaultAuditLayer, listAuditEntries } from "./audit_helpers.mjs";

const annotationAudit = {
  heads: {
    paint: {
      missing: { positive_pixels: 42, component_count: 2, overlay: "paint_missing_overlay.png" },
      overmark: { positive_pixels: 7, component_count: 1, overlay: "paint_overmark_overlay.png" },
    },
    aging: {
      missing: { positive_pixels: 12, component_count: 1, overlay: "aging_missing_overlay.png" },
    },
  },
};

const entries = listAuditEntries(annotationAudit);
assert.equal(entries.length, 3);
assert.equal(entries[0].layerId, "audit:paint:missing");
assert.equal(getAuditOverlay(annotationAudit, "audit:paint:overmark"), "paint_overmark_overlay.png");
assert.equal(getDefaultAuditLayer(annotationAudit), "audit:paint:missing");
assert.equal(getDefaultAuditLayer({ heads: { paint: { missing: { positive_pixels: 0, overlay: null } } } }), null);

console.log("audit_helpers ok");
