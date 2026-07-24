import assert from "node:assert/strict";
import { buildCurvePanelPayload } from "./curve_helpers.mjs";

const sample = {
  review_heads: {
    paint: { values: [0.12, 0.24, 0.35, 0.31, 0.28], normalized: [0.0, 0.52, 1.0, 0.83, 0.69], positive_pixels: 420, peak_wavelength: 600 },
    pollution: { values: [0.18, 0.16, 0.13, 0.11, 0.09], normalized: [1.0, 0.78, 0.44, 0.18, 0.0], positive_pixels: 120, peak_wavelength: 450 },
    aging: { values: [0, 0, 0, 0, 0], normalized: [0, 0, 0, 0, 0], positive_pixels: 0, peak_wavelength: 450 },
  },
  pigment_analysis: {
    cluster_analysis: {
      region_summaries: [
        { cluster_id: 1, label: "\u77f3\u9752", curve_values: [0.11, 0.2, 0.31, 0.29, 0.25], curve_normalized: [0.0, 0.45, 1.0, 0.9, 0.7], positive_pixels: 210, peak_wavelength: 600 },
        { cluster_id: 2, label: "\u6731\u7802", curve_values: [0.09, 0.12, 0.22, 0.33, 0.35], curve_normalized: [0.0, 0.12, 0.5, 0.92, 1.0], positive_pixels: 160, peak_wavelength: 700 },
      ],
    },
  },
};

const payload = buildCurvePanelPayload(sample, sample.review_heads);
assert.equal(payload.sections.length, 1);
assert.equal(payload.sections[0].id, "pigment-subregions");
assert.deepEqual(payload.sections[0].series.map((item) => item.label), ["\u5b50\u533a 1 \u00b7 \u77f3\u9752", "\u5b50\u533a 2 \u00b7 \u6731\u7802"]);
assert.deepEqual(payload.sections[0].series[1].values, [0.09, 0.12, 0.22, 0.33, 0.35]);

const empty = buildCurvePanelPayload({ review_heads: { paint: { values: [0,0,0,0,0], normalized: [0,0,0,0,0], positive_pixels: 0 } } }, { paint: { values: [0,0,0,0,0], normalized: [0,0,0,0,0], positive_pixels: 0 } });
assert.equal(empty.sections.length, 0);

console.log("curve_helpers ok");
