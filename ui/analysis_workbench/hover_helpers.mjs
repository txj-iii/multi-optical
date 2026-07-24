const PIGMENT_SPLIT_RE = /\s*\+\s*/;
const CONSTRAINED_LABEL_FALLBACKS = {
  "\u77f3\u7eff": "\u77f3\u9752",
};

function extractAllowedPigments(sample) {
  const labels = [];
  const predictedLabel = sample?.pigment_prediction?.predicted_label;
  if (predictedLabel) labels.push(predictedLabel);
  const topCandidates = sample?.pigment_prediction?.top_candidates ?? [];
  for (const candidate of topCandidates.slice(0, 2)) {
    if (candidate?.name) labels.push(candidate.name);
  }
  const sampleLabel = sample?.pigment_analysis?.sample_label;
  if (sampleLabel) labels.push(sampleLabel);

  const allowed = new Set();
  for (const label of labels) {
    String(label)
      .split(PIGMENT_SPLIT_RE)
      .map((item) => item.trim())
      .filter(Boolean)
      .forEach((item) => allowed.add(item));
  }
  allowed.delete("\u65e0\u989c\u6599");
  return allowed;
}

function constrainRegionLabel(sample, rawLabel) {
  const allowed = extractAllowedPigments(sample);
  if (!rawLabel || allowed.size === 0 || allowed.has(rawLabel)) {
    return { label: rawLabel, wasConstrained: false };
  }
  const fallback = CONSTRAINED_LABEL_FALLBACKS[rawLabel];
  if (fallback && allowed.has(fallback)) {
    return { label: fallback, wasConstrained: true };
  }
  return { label: rawLabel, wasConstrained: false };
}

function getRegionSummaries(sample) {
  const direct = sample?.pigment_analysis?.cluster_analysis?.region_summaries;
  if (Array.isArray(direct) && direct.length) return direct;
  const mixed = sample?.mixed_pigment_analysis?.subregions;
  if (Array.isArray(mixed) && mixed.length) return mixed;
  return [];
}

function getSourceHeads(sample, sourceId = "review") {
  if (sourceId === "annotation") {
    return sample?.heads ?? sample?.review_heads ?? null;
  }
  return sample?.review_heads ?? sample?.heads ?? null;
}

function getPaintMaskUrl(sample, sourceId = "review") {
  return getSourceHeads(sample, sourceId)?.paint?.mask ?? null;
}

export function getRegionLabelMapUrl(sample) {
  const summaries = getRegionSummaries(sample);
  if (!summaries.length) return null;
  return sample?.pigment_analysis?.cluster_analysis?.label_map ?? null;
}

function summarizeCandidateNames(candidates) {
  if (!Array.isArray(candidates) || candidates.length === 0) return "";
  return candidates.slice(0, 2).map((item) => item?.name).filter(Boolean).join(" / ");
}

export function summarizeUnavailableHover(sample) {
  const paintHead = sample?.review_heads?.paint ?? sample?.heads?.paint ?? {};
  return {
    active: true,
    text: "\u65e0\u53ef\u7528\u5b50\u533a\u5206\u6790",
    detailText: "\u5f53\u524d\u6837\u672c\u5c1a\u672a\u4ea7\u51fa\u53ef\u7528\u7684\u989c\u6599\u5b50\u533a\u6807\u7b7e\u56fe",
    scoreText: "",
    marginText: "",
    candidateText: "",
    positivePixels: Number(paintHead.positive_pixels ?? 0),
    peakWavelength: paintHead.peak_wavelength ?? null,
  };
}

export function getHoverRasterSource(sample, sourceId = "review") {
  const labelMap = getRegionLabelMapUrl(sample);
  if (labelMap) {
    return { kind: "region-label-map", url: labelMap };
  }
  const paintMask = getPaintMaskUrl(sample, sourceId);
  if (!paintMask) return null;
  return { kind: "paint-mask-unavailable", url: paintMask };
}

export function summarizeHoverRegion(sample, clusterId) {
  const summaries = getRegionSummaries(sample);
  const region = summaries.find((item) => Number(item.cluster_id) === Number(clusterId));
  if (!region) {
    return {
      active: false,
      clusterId: Number(clusterId) || 0,
      label: null,
      rawLabel: null,
      wasConstrained: false,
      labelSource: null,
      text: "",
      scoreText: "",
      marginText: "",
      candidateText: "",
      detailText: "",
    };
  }

  const backendLabel = region.label ?? null;
  const rawLabel = region.raw_label ?? backendLabel ?? "??";
  let finalLabel = backendLabel;
  let wasConstrained = Boolean(region.label_source === "preferred_candidates" && rawLabel && backendLabel && rawLabel !== backendLabel);

  if (!finalLabel) {
    const constrained = constrainRegionLabel(sample, rawLabel);
    finalLabel = constrained.label;
    wasConstrained = constrained.wasConstrained;
  }

  const candidateText = summarizeCandidateNames(region.preferred_top_candidates) || summarizeCandidateNames(region.top_candidates);
  return {
    active: true,
    clusterId: Number(clusterId),
    label: finalLabel ?? "??",
    rawLabel,
    wasConstrained,
    labelSource: region.label_source ?? (wasConstrained ? "preferred_candidates" : "global_candidates"),
    text: `paint + ${finalLabel ?? "??"}`,
    detailText: "",
    scoreText: Number(region.score ?? 0).toFixed(3),
    marginText: Number(region.margin ?? 0).toFixed(3),
    positivePixels: Number(region.positive_pixels ?? 0),
    peakWavelength: region.peak_wavelength ?? null,
    candidateText,
  };
}

export function resolveContainViewport(naturalWidth, naturalHeight, clientWidth, clientHeight) {
  if (!naturalWidth || !naturalHeight || !clientWidth || !clientHeight) {
    return { drawWidth: 0, drawHeight: 0, offsetX: 0, offsetY: 0, scaleX: 1, scaleY: 1 };
  }
  const imageAspect = naturalWidth / naturalHeight;
  const boxAspect = clientWidth / clientHeight;
  let drawWidth = clientWidth;
  let drawHeight = clientHeight;
  if (imageAspect > boxAspect) {
    drawHeight = clientWidth / imageAspect;
  } else {
    drawWidth = clientHeight * imageAspect;
  }
  const offsetX = (clientWidth - drawWidth) / 2;
  const offsetY = (clientHeight - drawHeight) / 2;
  return {
    drawWidth,
    drawHeight,
    offsetX,
    offsetY,
    scaleX: naturalWidth / drawWidth,
    scaleY: naturalHeight / drawHeight,
  };
}

export function mapPointerToImagePixel({ clientX, clientY, rect, naturalWidth, naturalHeight }) {
  const viewport = resolveContainViewport(naturalWidth, naturalHeight, rect.width, rect.height);
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  const inBounds = (
    localX >= viewport.offsetX
    && localX <= viewport.offsetX + viewport.drawWidth
    && localY >= viewport.offsetY
    && localY <= viewport.offsetY + viewport.drawHeight
  );
  if (!inBounds || viewport.drawWidth <= 0 || viewport.drawHeight <= 0) {
    return { inside: false, x: -1, y: -1 };
  }
  const x = Math.max(0, Math.min(naturalWidth - 1, Math.floor((localX - viewport.offsetX) * viewport.scaleX)));
  const y = Math.max(0, Math.min(naturalHeight - 1, Math.floor((localY - viewport.offsetY) * viewport.scaleY)));
  return { inside: true, x, y };
}
