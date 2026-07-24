import assert from "node:assert/strict";
import {
  getHoverRasterSource,
  getRegionLabelMapUrl,
  summarizeHoverRegion,
  summarizeUnavailableHover,
} from "./hover_helpers.mjs";

const sample = {
  pigment_prediction: {
    enabled: true,
    predicted_label: "blue+red",
    predicted_score: 0.884,
    top_candidates: [
      { name: "blue+red", score: 0.884 },
      { name: "blue+ochre+red", score: 0.061 },
    ],
    margin: 0.823,
  },
  mixed_pigment_analysis: {
    triggered: true,
    subregions: [
      {
        cluster_id: 1,
        label: "blue",
        raw_label: "green",
        label_source: "preferred_candidates",
        score: 0.6004589,
        margin: 0.0377717,
        positive_pixels: 104472,
        peak_wavelength: 450,
        preferred_top_candidates: [
          { name: "blue", score: 0.592 },
          { name: "red", score: 0.401 },
        ],
        top_candidates: [
          { name: "green", score: 0.6004589 },
          { name: "blue", score: 0.592 },
        ],
      },
      {
        cluster_id: 2,
        label: "red",
        raw_label: "red",
        label_source: "preferred_candidates",
        score: 0.6823690,
        margin: 0.0429547,
        positive_pixels: 56482,
        peak_wavelength: 650,
        preferred_top_candidates: [
          { name: "red", score: 0.6823690 },
          { name: "blue", score: 0.411 },
        ],
      },
    ],
  },
};

assert.equal(summarizeHoverRegion(sample, 0).active, false);
const first = summarizeHoverRegion(sample, 1);
assert.equal(first.active, true);
assert.equal(first.text, "paint + blue");
assert.equal(first.label, "blue");
assert.equal(first.rawLabel, "green");
assert.equal(first.wasConstrained, true);
assert.equal(first.labelSource, "preferred_candidates");
assert.equal(first.scoreText, "0.600");
assert.equal(first.candidateText, "blue / red");

const second = summarizeHoverRegion(sample, 2);
assert.equal(second.active, true);
assert.equal(second.text, "paint + red");
assert.equal(second.label, "red");
assert.equal(second.candidateText, "red / blue");

assert.equal(
  getRegionLabelMapUrl({
    pigment_analysis: {
      cluster_analysis: {
        label_map: "dual_pigment_labels.png",
        region_summaries: sample.mixed_pigment_analysis.subregions,
      },
    },
  }),
  "dual_pigment_labels.png",
);
assert.equal(getRegionLabelMapUrl(sample), null);

const unavailableSample = {
  review_heads: {
    paint: {
      mask: "paint_pred.png",
      positive_pixels: 3210,
      peak_wavelength: 450,
    },
  },
};

assert.deepEqual(
  getHoverRasterSource({
    review_heads: {
      paint: { mask: "paint_pred.png" },
    },
    pigment_analysis: {
      cluster_analysis: {
        label_map: "dual_pigment_labels.png",
        region_summaries: sample.mixed_pigment_analysis.subregions,
      },
    },
    mixed_pigment_analysis: sample.mixed_pigment_analysis,
  }, "review"),
  { kind: "region-label-map", url: "dual_pigment_labels.png" },
);
assert.deepEqual(
  getHoverRasterSource(unavailableSample, "review"),
  { kind: "paint-mask-unavailable", url: "paint_pred.png" },
);

const unavailable = summarizeUnavailableHover(unavailableSample);
assert.equal(unavailable.active, true);
assert.equal(unavailable.text, "\u65e0\u53ef\u7528\u5b50\u533a\u5206\u6790");
assert.equal(unavailable.detailText, "\u5f53\u524d\u6837\u672c\u5c1a\u672a\u4ea7\u51fa\u53ef\u7528\u7684\u989c\u6599\u5b50\u533a\u6807\u7b7e\u56fe");
assert.equal(unavailable.positivePixels, 3210);
assert.equal(unavailable.peakWavelength, 450);
assert.equal(unavailable.candidateText, "");

console.log("hover_helpers ok");
