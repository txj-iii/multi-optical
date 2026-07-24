const HEADS = [
  { id: "paint", label: "颜料/绘制主体区域", color: "#e84b4b", alpha: 0.28 },
  { id: "pollution", label: "污染/粉末附着区域", color: "#d8b72a", alpha: 0.42 },
  { id: "aging", label: "老化/褪变区域", color: "#2f8bd8", alpha: 0.46 },
];

const TEXT = {
  noSamples: "\u6682\u65e0\u6837\u672c",
  waiting: "\u7b49\u5f85\u4e2d",
  workflowUnavailable: "\u5de5\u4f5c\u6d41\u72b6\u6001\u6682\u4e0d\u53ef\u7528",
  importRunning: "\u6b63\u5728\u5bfc\u5165\u6700\u65b0 5 \u6ce2\u6bb5\u6837\u672c\u5e76\u751f\u6210 review \u7ed3\u679c...",
  imported: "\u5df2\u5bfc\u5165\uff0creview \u7ed3\u679c\u5df2\u751f\u6210",
  reused: "\u68c0\u6d4b\u5230\u5df2\u6709\u6837\u672c\uff0c\u5df2\u590d\u7528\u73b0\u6709\u8bb0\u5f55",
  review: "\u5f85\u5ba1\u6838",
  approved: "\u5df2\u901a\u8fc7",
  held: "\u5df2\u6401\u7f6e",
  needsAdjustment: "\u5f85\u4fee\u6539",
  error: "\u5f02\u5e38",
  awaitingBackground: "待选择背景",
  sourceAnnotationLabel: "\u6807\u6ce8\u7ed3\u679c",
  sourceReviewLabel: "\u9884\u6d4b\u7ed3\u679c",
};

const PAINT_MASK_THRESHOLD = 16;

const state = {
  manifest: null,
  workflow: null,
  selectedVersionId: null,
  selectedBackground4VersionId: "background4_v2",
  selectedSampleId: null,
  layer: "combined",
  overlaySource: "review",
  busy: false,
  workflowError: null,
  hover: {
    tooltip: null,
    hoverAssetKey: null,
    hoverAssetKind: null,
    hoverAssetData: null,
    hoverAssetPromise: null,
  },
  editor: {
    active: false,
    head: "paint",
    pigmentClass: 1,
    mode: "add",
    brushSize: 24,
    width: 0,
    height: 0,
    maskBuffers: null,
    pointerDown: false,
    lastPoint: null,
    sourceLabel: "review",
  },
};

const els = {
  sourceAnnotationButton: document.querySelector('#sourceControls button[data-source="annotation"]'),
  sourceReviewButton: document.querySelector('#sourceControls button[data-source="review"]'),
  sampleList: document.getElementById("sampleList"),
  imageStage: document.getElementById("imageStage"),
  previewImage: document.getElementById("previewImage"),
  overlayImage: document.getElementById("overlayImage"),
  editorCanvas: document.getElementById("editorCanvas"),
  opacityInput: document.getElementById("opacityInput"),
  versionSelect: document.getElementById("versionSelect"),
  background4VersionSelect: document.getElementById("background4VersionSelect"),
  versionBadge: document.getElementById("versionBadge"),
  versionSummary: document.getElementById("versionSummary"),
  versionProvenance: document.getElementById("versionProvenance"),
  reviewStatus: document.getElementById("reviewStatus"),
  workflowMeta: document.getElementById("workflowMeta"),
  annotationMeta: document.getElementById("annotationMeta"),
  auditPanel: document.getElementById("auditPanel"),
  headCards: document.getElementById("headCards"),
  sampleMeta: document.getElementById("sampleMeta"),
  importLatestButton: document.getElementById("importLatestButton"),
  importBackground4Button: document.getElementById("importBackground4Button"),
  runReviewSeedButton: document.getElementById("runReviewSeedButton"),
  runBackground4PredictButton: document.getElementById("runBackground4PredictButton"),
  pigmentControls: document.getElementById("pigmentControls"),
  reloadButton: document.getElementById("reloadButton"),
  approveButton: document.getElementById("approveButton"),
  rejectButton: document.getElementById("rejectButton"),
  editButton: document.getElementById("editButton"),
  editorPanel: document.getElementById("editorPanel"),
  editorBrushInput: document.getElementById("editorBrushInput"),
  editorBrushValue: document.getElementById("editorBrushValue"),
  saveAnnotationButton: document.getElementById("saveAnnotationButton"),
  discardAnnotationButton: document.getElementById("discardAnnotationButton"),
  backgroundPanel: document.getElementById("backgroundPanel"),
  backgroundControls: document.getElementById("backgroundControls"),
  lightControls: document.getElementById("lightControls"),
  confirmBackgroundButton: document.getElementById("confirmBackgroundButton"),
  curveLegend: document.getElementById("curveLegend"),
  curvePlaceholder: document.getElementById("curvePlaceholder"),
  curvePanelBody: document.getElementById("curvePanelBody"),
};

function safeText(value, fallback = "-") {
  return value == null || value === "" ? fallback : String(value);
}

function shortTaskSpecificPath(value, fallback = "-") {
  const text = safeText(value, "");
  if (!text) return fallback;
  const normalized = text.replace(/\\/g, "/");
  const marker = "/task_specific/";
  const markerIndex = normalized.lastIndexOf(marker);
  if (markerIndex >= 0) {
    const rest = normalized.slice(markerIndex + marker.length).replace(/^\/+|\/+$/g, "");
    return rest || fallback;
  }
  const parts = normalized.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : fallback;
}

function withCacheBust(url) {
  if (!url) return url;
  const stamp = `ts=${Date.now()}`;
  return `${url}${url.includes("?") ? "&" : "?"}${stamp}`;
}

function normalizeManifest(raw) {
  const versions = Array.isArray(raw?.versions) && raw.versions.length
    ? raw.versions
    : [{
        id: raw?.current_version_id ?? raw?.active_version?.id ?? "default",
        label: raw?.active_version?.label ?? "default",
        provenance: raw?.active_version?.provenance ?? {},
        sample_count: Array.isArray(raw?.samples) ? raw.samples.length : 0,
        samples: raw?.samples ?? raw?.active_version?.samples ?? [],
      }];
  const currentVersionId = raw?.current_version_id ?? versions[0]?.id ?? null;
  return { versions, current_version_id: currentVersionId };
}

function getVersions() {
  return state.manifest?.versions ?? [];
}

function getSelectedVersion() {
  return getVersions().find((item) => item.id === state.selectedVersionId) ?? getVersions()[0] ?? null;
}

function getSamples() {
  return getSelectedVersion()?.samples ?? [];
}

function getSelectedSample() {
  return getSamples().find((item) => item.id === state.selectedSampleId) ?? null;
}

function getWorkflowRecord(sceneId) {
  return state.workflow?.samples?.find((item) => item.scene_id === sceneId) ?? null;
}

function getStatusLabel(status) {
  return ({
    pending_review: TEXT.review,
    approved: TEXT.approved,
    held: TEXT.held,
    awaiting_background: TEXT.awaitingBackground,
    needs_adjustment: TEXT.needsAdjustment,
    error: TEXT.error,
  })[status] ?? safeText(status, TEXT.waiting);
}

function getCurrentSourceId() {
  if (state.overlaySource === "annotation" && !getSelectedSample()?.annotation_available) {
    return "review";
  }
  return state.overlaySource === "annotation" ? "annotation" : "review";
}

function getSourceAssets(sample, sourceId = getCurrentSourceId()) {
  if (!sample) return {};
  if (sourceId === "annotation") {
    return sample.annotation_assets ?? sample.assets ?? {};
  }
  return sample.review_assets ?? sample.assets ?? {};
}

function getSourceHeads(sample, sourceId = getCurrentSourceId()) {
  if (!sample) return {};
  if (sourceId === "annotation") {
    return sample.annotation_available ? (sample.heads ?? {}) : {};
  }
  return sample.review_heads ?? sample.heads ?? {};
}

function setBusy(flag) {
  state.busy = flag;
  const disabled = flag;
  [
    els.importLatestButton,
    els.importBackground4Button,
    els.runReviewSeedButton,
    els.runBackground4PredictButton,
    els.approveButton,
    els.rejectButton,
    els.editButton,
    els.reloadButton,
    els.saveAnnotationButton,
    els.discardAnnotationButton,
    els.confirmBackgroundButton,
  ].forEach((item) => {
    if (item) item.disabled = disabled;
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${url} ${response.status}`);
  return data;
}

async function loadManifest() {
  const data = await fetchJson(`./workbench_manifest.json?ts=${Date.now()}`);
  state.manifest = normalizeManifest(data);
  state.selectedVersionId = state.selectedVersionId ?? state.manifest.current_version_id;
}

async function loadWorkflow() {
  try {
    state.workflow = await fetchJson(`/api/workbench/state?ts=${Date.now()}`);
    state.workflowError = null;
  } catch (error) {
    state.workflow = { samples: [], count: 0, updated_at: null };
    state.workflowError = String(error.message || error);
  }
}

async function loadSelectedBackground4Prediction() {
  const sample = getSelectedSample();
  if (!sample) return null;
  const query = new URLSearchParams({
    scene_id: sample.id,
    version_id: state.selectedBackground4VersionId,
  });
  const result = await fetchJson(`/api/workbench/background4-prediction?${query.toString()}`);
  sample.review_assets = result.available
    ? { ...(sample.review_assets ?? {}), ...result.assets }
    : { preview: sample.assets?.preview ?? sample.review_assets?.preview ?? "" };
  sample.review_heads = result.available ? result.heads : {};
  sample.pigment_prediction = result.available ? result.pigment_prediction : null;
  sample.active_background4_prediction = result;
  return result;
}

function pickPreferredSampleId() {
  const workflowRecords = Array.isArray(state.workflow?.samples) ? state.workflow.samples.slice() : [];
  if (workflowRecords.length) {
    workflowRecords.sort((a, b) => safeText(b.updated_at).localeCompare(safeText(a.updated_at)));
    const preferred = workflowRecords[0]?.scene_id;
    if (preferred && getSamples().some((sample) => sample.id === preferred)) {
      return preferred;
    }
  }
  const samples = getSamples();
  return samples.length ? samples[samples.length - 1].id : null;
}

function renderVersion() {
  const versions = getVersions();
  const selected = getSelectedVersion();
  els.versionSelect.innerHTML = "";
  versions.forEach((version) => {
    const option = document.createElement("option");
    option.value = version.id;
    option.textContent = shortTaskSpecificPath(version.label ?? version.id, "default");
    option.selected = version.id === state.selectedVersionId;
    els.versionSelect.appendChild(option);
  });
  els.versionBadge.textContent = `${getSamples().length}`;
  if (els.background4VersionSelect) els.background4VersionSelect.value = state.selectedBackground4VersionId;
  if (els.runBackground4PredictButton) els.runBackground4PredictButton.textContent = `运行 ${state.selectedBackground4VersionId} 预测`;
  els.versionSummary.textContent = selected
    ? `当前样本数：${getSamples().length}，最新：${safeText(getSamples()[getSamples().length - 1]?.id, "-")}`
    : TEXT.noSamples;
  const provenance = selected?.provenance ?? {};
  const rows = [
    ["prediction", shortTaskSpecificPath(provenance.prediction_root)],
    ["paint", shortTaskSpecificPath(provenance.paint_root)],
    ["pollution", shortTaskSpecificPath(provenance.pollution_root)],
    ["aging", shortTaskSpecificPath(provenance.aging_root)],
  ];
  els.versionProvenance.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderSamples() {
  const samples = getSamples();
  if (!samples.length) {
    els.sampleList.innerHTML = `<div class="empty-state">${TEXT.noSamples}</div>`;
    return;
  }
  els.sampleList.innerHTML = "";
  samples.slice().reverse().forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-item${sample.id === state.selectedSampleId ? " active" : ""}`;
    const record = getWorkflowRecord(sample.id);
    const reviewHeads = getSourceHeads(sample, "review");
    const headCount = HEADS.filter((head) => Number(reviewHeads?.[head.id]?.area_ratio ?? 0) > 0).length;
    button.innerHTML = `${sample.id}<span class="sample-meta">${headCount} heads</span>${record ? `<span class="sample-status">${getStatusLabel(record.status)}</span>` : ""}`;
    button.addEventListener("click", async () => {
      if (state.editor.active) {
        exitEditorMode({ preserveStatus: true });
      }
      state.selectedSampleId = sample.id;
      // A saved annotation is the useful default.  A new/imported sample has
      // no final masks yet, so it continues to open on the prediction layer.
      state.overlaySource = sample.annotation_available ? "annotation" : "review";
      resetHoverState();
      const prediction = await loadSelectedBackground4Prediction();
      render();
      if (state.overlaySource === "review" && !prediction?.available) {
        els.reviewStatus.textContent = `${sample.id} 暂无 ${state.selectedBackground4VersionId} 预测`;
      }
    });
    els.sampleList.appendChild(button);
  });
}

function resolveOverlay(sample) {
  if (!sample || state.layer === "preview" || state.editor.active) return null;
  const assets = getSourceAssets(sample);
  const heads = getSourceHeads(sample);
  if (state.layer === "combined") return assets?.combined_overlay ?? null;
  return heads?.[state.layer]?.overlay ?? null;
}

function syncSourceControlLabels() {
  if (els.sourceAnnotationButton) {
    els.sourceAnnotationButton.textContent = TEXT.sourceAnnotationLabel;
    els.sourceAnnotationButton.title = "\u663e\u793a\u4f60\u4fdd\u5b58\u5230 masks \u7684\u6700\u7ec8\u6807\u6ce8\u7ed3\u679c";
  }
  if (els.sourceReviewButton) {
    els.sourceReviewButton.textContent = TEXT.sourceReviewLabel;
    els.sourceReviewButton.title = "\u663e\u793a\u6a21\u578b\u5f53\u524d\u5bfc\u51fa\u7684\u9884\u6d4b\u7ed3\u679c";
  }
}

function renderSourceControls() {
  syncSourceControlLabels();
  document.querySelectorAll("#sourceControls button").forEach((item) => {
    const isAnnotation = item.dataset.source === "annotation";
    const sourceExists = !isAnnotation || Boolean(getSelectedSample()?.annotation_available);
    const enabled = sourceExists && !state.editor.active;
    item.disabled = !enabled;
    item.classList.toggle("source-button-disabled", !enabled);
    item.classList.toggle("active", item.dataset.source === getCurrentSourceId());
  });
}

function renderLayerControls() {
  document.querySelectorAll("#layerControls button").forEach((item) => {
    item.classList.toggle("active", item.dataset.layer === state.layer);
  });
}

function renderEditorControls() {
  document.querySelectorAll("#editorHeadControls button").forEach((item) => {
    item.classList.toggle("active", item.dataset.head === state.editor.head);
  });
  document.querySelectorAll("#editorModeControls button").forEach((item) => {
    item.classList.toggle("active", item.dataset.mode === state.editor.mode);
  });
  els.editorBrushValue.textContent = `${state.editor.brushSize} px`;
  els.editorPanel.classList.toggle("hidden", !state.editor.active);
}

function syncEditorCanvasFrame() {
  const stageRect = els.imageStage.getBoundingClientRect();
  const previewRect = els.previewImage.getBoundingClientRect();
  if (!stageRect.width || !stageRect.height || !previewRect.width || !previewRect.height) return;
  const left = previewRect.left - stageRect.left;
  const top = previewRect.top - stageRect.top;
  els.editorCanvas.style.left = `${left}px`;
  els.editorCanvas.style.top = `${top}px`;
  els.editorCanvas.style.width = `${previewRect.width}px`;
  els.editorCanvas.style.height = `${previewRect.height}px`;
}

function renderImages() {
  const sample = getSelectedSample();
  renderSourceControls();
  renderLayerControls();
  if (!sample) {
    els.previewImage.removeAttribute("src");
    els.overlayImage.removeAttribute("src");
    return;
  }
  const previewUrl = sample.assets?.preview ?? sample.review_assets?.preview ?? sample.annotation_assets?.preview ?? "";
  els.previewImage.src = withCacheBust(previewUrl);
  const overlay = resolveOverlay(sample);
  if (!overlay) {
    els.overlayImage.removeAttribute("src");
  } else {
    els.overlayImage.src = withCacheBust(overlay);
    els.overlayImage.style.opacity = els.opacityInput.value;
  }
  els.overlayImage.classList.toggle("hidden", state.editor.active);
  els.editorCanvas.classList.toggle("hidden", !state.editor.active);
  els.editorCanvas.classList.toggle("editing", state.editor.active);
  requestAnimationFrame(() => {
    syncEditorCanvasFrame();
    if (state.editor.active) {
      drawEditorOverlay();
    }
  });
  primeHoverAssets(sample);
}

function formatPercent(value) {
  return `${(Number(value ?? 0) * 100).toFixed(1)}%`;
}

function getCurveHelpers() {
  return window.CurveHelpers ?? {};
}

function buildCurvePath(values, width, height, padding, minValue, maxValue) {
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const span = Math.max(maxValue - minValue, 1e-6);
  return values.map((value, index) => {
    const x = padding.left + (innerWidth * index) / Math.max(values.length - 1, 1);
    const normalized = (Number(value ?? 0) - minValue) / span;
    const y = padding.top + innerHeight - normalized * innerHeight;
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function renderCurveSection(section, bandLabels) {
  const width = 920;
  const height = 220;
  const padding = { top: 18, right: 18, bottom: 36, left: 48 };
  const allValues = section.series.flatMap((series) => series.values.map((value) => Number(value ?? 0)));
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const yTicks = 4;
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const tickLines = Array.from({ length: yTicks + 1 }, (_, index) => {
    const ratio = index / yTicks;
    const y = padding.top + innerHeight - ratio * innerHeight;
    const value = minValue + (maxValue - minValue) * ratio;
    return `<g><line x1="${padding.left}" y1="${y.toFixed(1)}" x2="${width - padding.right}" y2="${y.toFixed(1)}" class="curve-grid-line"></line><text x="${padding.left - 10}" y="${(y + 4).toFixed(1)}" class="curve-axis-text" text-anchor="end">${value.toFixed(3)}</text></g>`;
  }).join('');
  const xTicks = bandLabels.map((band, index) => {
    const x = padding.left + (innerWidth * index) / Math.max(bandLabels.length - 1, 1);
    return `<g><line x1="${x.toFixed(1)}" y1="${padding.top}" x2="${x.toFixed(1)}" y2="${height - padding.bottom}" class="curve-grid-band"></line><text x="${x.toFixed(1)}" y="${height - 12}" class="curve-axis-text" text-anchor="middle">${band}</text></g>`;
  }).join('');
  const paths = section.series.map((series) => {
    const path = buildCurvePath(series.values, width, height, padding, minValue, maxValue);
    const markers = series.values.map((value, index) => {
      const x = padding.left + (innerWidth * index) / Math.max(series.values.length - 1, 1);
      const normalized = (Number(value ?? 0) - minValue) / Math.max(maxValue - minValue, 1e-6);
      const y = padding.top + innerHeight - normalized * innerHeight;
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.5" fill="${series.color}"></circle>`;
    }).join('');
    return `<g><path d="${path}" fill="none" stroke="${series.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>${markers}</g>`;
  }).join('');
  const legend = section.series.map((series) => `<div class="curve-series-chip"><span class="curve-chip-dot" style="background:${series.color}"></span><span>${series.label}</span><span class="curve-chip-meta">peak ${safeText(series.peakWavelength, '-')} nm · ${safeText(series.positivePixels, 0)} px</span></div>`).join('');
  return `
    <section class="curve-section" data-section="${section.id}">
      <div class="curve-section-head">
        <div>
          <h3>${section.title}</h3>
          <p>${section.description}</p>
        </div>
      </div>
      <div class="curve-series-list">${legend}</div>
      <svg class="curve-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${section.title}">${tickLines}${xTicks}${paths}</svg>
    </section>
  `;
}

function renderCurvePanel() {
  const sample = getSelectedSample();
  const sourceId = getCurrentSourceId();
  const sourceHeads = sample ? getSourceHeads(sample, sourceId) : null;
  const helpers = getCurveHelpers();
  if (!sample || !helpers.buildCurvePanelPayload) {
    els.curveLegend.innerHTML = '';
    els.curvePlaceholder.innerHTML = '\u5f53\u524d\u6837\u672c\u6682\u65e0\u53ef\u5c55\u793a\u7684\u6ce2\u6bb5\u66f2\u7ebf\u3002';
    return;
  }
  const payload = helpers.buildCurvePanelPayload(sample, sourceHeads);
  const sourceLabel = sourceId === "annotation" ? TEXT.sourceAnnotationLabel : TEXT.sourceReviewLabel;
  els.curveLegend.innerHTML = payload.bandLabels.map((band) => `<span class="curve-band-chip">${band} nm</span>`).join('') + `<span class="tag muted">\u6765\u6e90\uff1a${sourceLabel}</span>`;
  if (!payload.sections.length) {
    els.curvePlaceholder.innerHTML = '\u5f53\u524d\u6765\u6e90\u4e0b\u6682\u65e0\u5934\u90e8\u6216\u989c\u6599\u5b50\u533a\u66f2\u7ebf\u6570\u636e\u3002';
    return;
  }
  els.curvePlaceholder.innerHTML = payload.sections.map((section) => renderCurveSection(section, payload.bandLabels)).join('');
}

function renderHeadCards() {
  const sample = getSelectedSample();
  els.headCards.innerHTML = "";
  if (!sample) return;
  HEADS.forEach((head) => {
    const sourceHeads = getSourceHeads(sample);
    const data = sourceHeads?.[head.id] ?? {};
    const card = document.createElement("article");
    card.className = "head-card";
    card.style.borderLeftColor = head.color;
    card.innerHTML = `
      <h3>${head.label}</h3>
      <div class="ratio-bar"><span style="width:${Math.max(0, Math.min(100, Number(data.area_ratio ?? 0) * 100))}%; background:${head.color}"></span></div>
      <dl>
        <dt>ratio</dt><dd>${formatPercent(data.area_ratio)}</dd>
        <dt>pixels</dt><dd>${safeText(data.positive_pixels, 0)} / ${safeText(data.total_pixels, 0)}</dd>
        <dt>peak</dt><dd>${safeText(data.peak_wavelength, "-")} nm</dd>
      </dl>
    `;
    els.headCards.appendChild(card);
  });
}

function renderWorkflow() {
  const sample = getSelectedSample();
  const record = sample ? getWorkflowRecord(sample.id) : null;
  els.auditPanel.classList.add("hidden");
  if (!sample) {
    els.reviewStatus.textContent = TEXT.noSamples;
    els.workflowMeta.textContent = state.workflowError || TEXT.workflowUnavailable;
    els.annotationMeta.textContent = "标注来源未加载";
    els.sampleMeta.textContent = "-";
    renderEditorControls();
    return;
  }

  const hasFinal = Boolean(sample.annotation_available);
  const awaitingBackground = record?.status === "awaiting_background";
  if (els.backgroundPanel) els.backgroundPanel.classList.toggle("hidden", !awaitingBackground);
  if (awaitingBackground) {
    const selectedBackground = record?.background_role;
    els.backgroundControls?.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.background === selectedBackground));
    const selectedLight = record?.light_level;
    els.lightControls?.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.light === String(selectedLight ?? "")));
  }
  const sourceLabel = hasFinal
    ? safeText(record?.annotation_source_label, "已保存最终标注")
    : "当前仅有 review 底稿，尚未保存最终标注";
  const ruleLabel = safeText(record?.annotation_rule_label, "review 用于人工确认；保存后才写入最终标注 masks");
  els.annotationMeta.innerHTML = `<strong>标注结果：</strong>你保存到 masks 的最终标注<br /><strong>预测结果：</strong>模型当前导出的预测图层<br /><strong>当前标注状态：</strong>${sourceLabel}<br /><strong>说明：</strong>${ruleLabel}`;

  if (!record) {
    els.reviewStatus.textContent = TEXT.waiting;
    els.workflowMeta.textContent = state.workflowError || `当前样本：${sample.id}`;
  } else {
    els.reviewStatus.textContent = `${sample.id} ${getStatusLabel(record.status)}`;
    const parts = [];
    if (record.stage) parts.push(`stage: ${record.stage}`);
    if (record.updated_at) parts.push(`updated: ${record.updated_at}`);
    if (record.annotation_decision) parts.push(`annotation: ${record.annotation_decision}`);
    els.workflowMeta.textContent = parts.join(" | ") || sample.id;
  }
  els.sampleMeta.textContent = sample.id;
  renderEditorControls();
}

function render() {
  renderVersion();
  renderSamples();
  renderImages();
  renderHeadCards();
  renderWorkflow();
  renderCurvePanel();
}

async function refreshAll(preferredSceneId = null) {
  await loadManifest();
  await loadWorkflow();
  state.selectedSampleId = preferredSceneId && getSamples().some((sample) => sample.id === preferredSceneId)
    ? preferredSceneId
    : pickPreferredSampleId();
  await loadSelectedBackground4Prediction();
  render();
}

async function postWorkflow(url, body = {}) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function runAction(url, pendingText, body = {}) {
  if (url.includes("/run-background4-prediction") && body.version_id == null) {
    body = { ...body, version_id: state.selectedBackground4VersionId };
  }
  if (url.includes("/run-background4-prediction")) {
    pendingText = `${getSelectedSample()?.id ?? "当前样本"} 正在运行 ${body.version_id} 像素级预测...`;
  }
  try {
    setBusy(true);
    els.reviewStatus.textContent = pendingText;
    const result = await postWorkflow(url, body);
    await refreshAll(result.scene_id ?? null);
    return result;
  } catch (error) {
    els.reviewStatus.textContent = String(error.message || error);
    throw error;
  } finally {
    setBusy(false);
  }
}

function getHoverHelpers() {
  return window.HoverHelpers ?? {};
}

function getHoverTooltip() {
  if (!state.hover.tooltip) {
    const tooltip = document.createElement("div");
    tooltip.className = "image-hover-tooltip hidden";
    els.imageStage.appendChild(tooltip);
    state.hover.tooltip = tooltip;
  }
  return state.hover.tooltip;
}

function hideHoverTooltip() {
  const tooltip = state.hover.tooltip;
  if (tooltip) tooltip.classList.add("hidden");
}

function resetHoverState() {
  state.hover.hoverAssetKey = null;
  state.hover.hoverAssetKind = null;
  state.hover.hoverAssetData = null;
  state.hover.hoverAssetPromise = null;
  hideHoverTooltip();
}

async function loadRasterData(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0);
      resolve({ canvas, context, width: canvas.width, height: canvas.height, src: url });
    };
    image.onerror = () => reject(new Error(`failed to load ${url}`));
    image.src = withCacheBust(url);
  });
}

function primeHoverAssets(sample) {
  // The legacy hover panel is based on old model/curve candidates.  It is
  // deliberately disabled for the four-background manual-label workflow so
  // that it cannot be mistaken for a saved pigment class.
  const hasPixelPigmentMap = Boolean(sample?.pigment_prediction?.pixel_map);
  if (getWorkflowRecord(sample?.id)?.background_role && !hasPixelPigmentMap) {
    state.hover.hoverAssetData = null;
    state.hover.hoverAssetKey = null;
    state.hover.hoverAssetKind = null;
    state.hover.hoverAssetPromise = null;
    hideHoverTooltip();
    return;
  }
  const helpers = getHoverHelpers();
  const hoverSource = helpers.getHoverRasterSource ? helpers.getHoverRasterSource(sample, getCurrentSourceId()) : null;
  if (!hoverSource?.url) {
    state.hover.hoverAssetData = null;
    state.hover.hoverAssetKey = null;
    state.hover.hoverAssetKind = null;
    state.hover.hoverAssetPromise = null;
    return;
  }
  if ((state.hover.hoverAssetKey === hoverSource.url && state.hover.hoverAssetKind === hoverSource.kind) || state.hover.hoverAssetPromise) {
    return;
  }
  state.hover.hoverAssetKey = hoverSource.url;
  state.hover.hoverAssetKind = hoverSource.kind;
  state.hover.hoverAssetPromise = loadRasterData(hoverSource.url)
    .then((data) => {
      state.hover.hoverAssetData = data;
      return data;
    })
    .catch(() => {
      state.hover.hoverAssetData = null;
      state.hover.hoverAssetKey = null;
      state.hover.hoverAssetKind = null;
      return null;
    })
    .finally(() => {
      state.hover.hoverAssetPromise = null;
    });
}

function readMaskValue(maskData, x, y) {
  if (!maskData?.context) return 0;
  if (x < 0 || y < 0 || x >= maskData.width || y >= maskData.height) return 0;
  return maskData.context.getImageData(x, y, 1, 1).data[0] ?? 0;
}

function showHoverTooltip(event, summary) {
  const tooltip = getHoverTooltip();
  const lines = [];
  lines.push(`<strong>${summary.text}</strong>`);
  if (summary.detailText) lines.push(`<span>${summary.detailText}</span>`);
  if (summary.scoreText) lines.push(`<span>score ${summary.scoreText}</span>`);
  if (summary.marginText) lines.push(`<span>margin ${summary.marginText}</span>`);
  if (summary.candidateText) lines.push(`<span>candidates ${summary.candidateText}</span>`);
  if (summary.positivePixels) lines.push(`<span>pixels ${summary.positivePixels}</span>`);
  if (summary.peakWavelength) lines.push(`<span>peak ${summary.peakWavelength} nm</span>`);
  tooltip.innerHTML = lines.join("");
  tooltip.classList.remove("hidden");
  const stageRect = els.imageStage.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  let left = event.clientX - stageRect.left + 16;
  let top = event.clientY - stageRect.top + 16;
  if (left + tooltipRect.width > stageRect.width - 8) left = stageRect.width - tooltipRect.width - 8;
  if (top + tooltipRect.height > stageRect.height - 8) top = stageRect.height - tooltipRect.height - 8;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function handleImageHover(event) {
  if (state.editor.active) {
    hideHoverTooltip();
    return;
  }
  const sample = getSelectedSample();
  const hasPixelPigmentMap = Boolean(sample?.pigment_prediction?.pixel_map);
  if (getWorkflowRecord(sample?.id)?.background_role && !hasPixelPigmentMap) {
    hideHoverTooltip();
    return;
  }
  const helpers = getHoverHelpers();
  if (!sample || !helpers.summarizeHoverRegion || !helpers.summarizeUnavailableHover || !helpers.getHoverRasterSource) {
    hideHoverTooltip();
    return;
  }
  if (!els.previewImage.complete || !els.previewImage.naturalWidth) {
    hideHoverTooltip();
    return;
  }
  const hoverAsset = state.hover.hoverAssetData;
  const hoverKind = state.hover.hoverAssetKind;
  if (!hoverAsset || !hoverKind) {
    primeHoverAssets(sample);
    hideHoverTooltip();
    return;
  }
  const point = mapClientToImagePixel(event.clientX, event.clientY);
  if (!point) {
    hideHoverTooltip();
    return;
  }
  const maskValue = readMaskValue(hoverAsset, point.x, point.y);
  let summary = null;
  if (hoverKind === "region-label-map") {
    if (maskValue < 1) {
      hideHoverTooltip();
      return;
    }
    summary = helpers.summarizeHoverRegion(sample, maskValue);
  } else if (hoverKind === "pigment-pixel-map") {
    if (maskValue < 1) {
      hideHoverTooltip();
      return;
    }
    summary = helpers.summarizePixelPigment?.(sample, maskValue);
  } else if (hoverKind === "paint-mask-unavailable") {
    if (maskValue < PAINT_MASK_THRESHOLD) {
      hideHoverTooltip();
      return;
    }
    summary = helpers.summarizeUnavailableHover(sample);
  }
  if (!summary?.active) {
    hideHoverTooltip();
    return;
  }
  showHoverTooltip(event, summary);
}

function mapClientToImagePixel(clientX, clientY) {
  const rect = els.previewImage.getBoundingClientRect();
  if (!rect.width || !rect.height || !els.previewImage.naturalWidth || !els.previewImage.naturalHeight) return null;
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) return null;
  const scaleX = els.previewImage.naturalWidth / rect.width;
  const scaleY = els.previewImage.naturalHeight / rect.height;
  const x = Math.max(0, Math.min(els.previewImage.naturalWidth - 1, Math.floor(localX * scaleX)));
  const y = Math.max(0, Math.min(els.previewImage.naturalHeight - 1, Math.floor(localY * scaleY)));
  return { x, y };
}

function getEditorSourceId(sample) {
  if (!sample) return "review";
  return "review";
}

function maskUrlForEditor(sample, headId) {
  const reviewMask = sample.review_heads?.[headId]?.mask ?? null;
  if (reviewMask) return reviewMask;
  return sample.heads?.[headId]?.mask ?? null;
}

async function loadMaskBuffer(url, width, height) {
  if (!url) return new Uint8ClampedArray(width * height);
  const raster = await loadRasterData(url);
  const data = raster.context.getImageData(0, 0, raster.width, raster.height).data;
  const buffer = new Uint8ClampedArray(width * height);
  for (let index = 0; index < width * height; index += 1) {
    buffer[index] = data[index * 4] > 127 ? 255 : 0;
  }
  return buffer;
}

function getHeadColor(headId) {
  return HEADS.find((item) => item.id === headId) ?? HEADS[0];
}

async function enterEditorMode() {
  const sample = getSelectedSample();
  if (!sample) return;
  if (!els.previewImage.complete || !els.previewImage.naturalWidth) {
    els.reviewStatus.textContent = "请等待原图加载完成后再进入人工修改";
    return;
  }
  state.editor.active = true;
  state.editor.width = els.previewImage.naturalWidth;
  state.editor.height = els.previewImage.naturalHeight;
  state.editor.sourceLabel = sample.annotation_available ? "annotation" : "review";
  state.editor.maskBuffers = {};
  for (const head of HEADS) {
    state.editor.maskBuffers[head.id] = await loadMaskBuffer(maskUrlForEditor(sample, head.id), state.editor.width, state.editor.height);
  }
  state.editor.maskBuffers.pigment = await loadPigmentBuffer(sample, state.editor.width, state.editor.height);
  els.editorCanvas.width = state.editor.width;
  els.editorCanvas.height = state.editor.height;
  renderEditorControls();
  renderImages();
  drawEditorOverlay();
  els.reviewStatus.textContent = `${sample.id} 已进入人工修改，当前底稿：${state.editor.sourceLabel}`;
}

async function loadPigmentBuffer(sample, width, height) {
  const url = `../../train/camera_eval_workspace/${encodeURIComponent(sample.id)}/masks/pigment.png`;
  try {
    const raster = await loadRasterData(url);
    const data = raster.context.getImageData(0, 0, raster.width, raster.height).data;
    const buffer = new Uint8ClampedArray(width * height);
    for (let index = 0; index < buffer.length; index += 1) buffer[index] = Math.min(4, data[index * 4]);
    return buffer;
  } catch { return new Uint8ClampedArray(width * height); }
}

function exitEditorMode({ preserveStatus = false } = {}) {
  state.editor.active = false;
  state.editor.pointerDown = false;
  state.editor.lastPoint = null;
  state.editor.maskBuffers = null;
  renderEditorControls();
  renderImages();
  if (!preserveStatus) {
    const sample = getSelectedSample();
    if (sample) els.reviewStatus.textContent = `${sample.id} 已退出人工修改`;
  }
}

function blendPixel(base, color, alpha) {
  return Math.round((1 - alpha) * base + alpha * color);
}

function drawEditorOverlay() {
  if (!state.editor.active || !state.editor.maskBuffers) return;
  const context = els.editorCanvas.getContext("2d", { willReadFrequently: true });
  const width = state.editor.width;
  const height = state.editor.height;
  const imageData = context.createImageData(width, height);
  const pixels = imageData.data;
  for (let index = 0; index < width * height; index += 1) {
    let r = 0;
    let g = 0;
    let b = 0;
    let a = 0;
    for (const head of HEADS) {
      if (state.editor.maskBuffers[head.id][index] > 0) {
        const color = hexToRgb(head.color);
        if (a === 0) {
          r = color[0];
          g = color[1];
          b = color[2];
          a = Math.round(head.alpha * 255);
        } else {
          r = blendPixel(r, color[0], head.alpha);
          g = blendPixel(g, color[1], head.alpha);
          b = blendPixel(b, color[2], head.alpha);
          a = Math.min(255, a + Math.round(head.alpha * 255 * 0.7));
        }
      }
    }
    const pigment = state.editor.maskBuffers.pigment?.[index] ?? 0;
    if (pigment > 0) {
      const colors = [null, [214, 54, 48], [157, 85, 42], [45, 116, 181], [51, 145, 91]];
      [r, g, b] = colors[pigment]; a = 180;
    }
    const offset = index * 4;
    pixels[offset] = r;
    pixels[offset + 1] = g;
    pixels[offset + 2] = b;
    pixels[offset + 3] = a;
  }
  context.putImageData(imageData, 0, 0);
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
  ];
}

function drawCircle(buffer, width, height, cx, cy, radius, value) {
  const r2 = radius * radius;
  const minX = Math.max(0, Math.floor(cx - radius));
  const maxX = Math.min(width - 1, Math.ceil(cx + radius));
  const minY = Math.max(0, Math.floor(cy - radius));
  const maxY = Math.min(height - 1, Math.ceil(cy + radius));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      if ((dx * dx) + (dy * dy) <= r2) {
        buffer[(y * width) + x] = value;
      }
    }
  }
}

function paintStroke(from, to) {
  if (!state.editor.maskBuffers) return;
  const buffer = state.editor.maskBuffers[state.editor.head];
  const value = state.editor.mode === "add" ? (state.editor.head === "pigment" ? state.editor.pigmentClass : 255) : 0;
  const radius = Math.max(1, Math.round(state.editor.brushSize / 2));
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
  for (let step = 0; step <= steps; step += 1) {
    const x = from.x + (dx * step / steps);
    const y = from.y + (dy * step / steps);
    drawCircle(buffer, state.editor.width, state.editor.height, x, y, radius, value);
    // Pigment labels are valid only inside paint.  Make the common workflow
    // one stroke: choosing a pigment both labels the subregion and marks it
    // as paint, instead of silently discarding the pigment label on save.
    if (state.editor.head === "pigment" && state.editor.mode === "add") {
      drawCircle(state.editor.maskBuffers.paint, state.editor.width, state.editor.height, x, y, radius, 255);
    }
  }
  drawEditorOverlay();
}

function editorPointerPoint(event) {
  if (!state.editor.active) return null;
  return mapClientToImagePixel(event.clientX, event.clientY);
}

function handleEditorPointerDown(event) {
  if (!state.editor.active) return;
  const point = editorPointerPoint(event);
  if (!point) return;
  state.editor.pointerDown = true;
  state.editor.lastPoint = point;
  paintStroke(point, point);
  event.preventDefault();
}

function handleEditorPointerMove(event) {
  if (!state.editor.active || !state.editor.pointerDown) return;
  const point = editorPointerPoint(event);
  if (!point || !state.editor.lastPoint) return;
  paintStroke(state.editor.lastPoint, point);
  state.editor.lastPoint = point;
  event.preventDefault();
}

function handleEditorPointerUp() {
  state.editor.pointerDown = false;
  state.editor.lastPoint = null;
}

function maskBufferToDataUrl(buffer, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = context.createImageData(width, height);
  for (let index = 0; index < width * height; index += 1) {
    const value = buffer[index];
    const offset = index * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

async function saveAnnotationMasks() {
  const sample = getSelectedSample();
  if (!sample || !state.editor.active || !state.editor.maskBuffers) return;
  const markedPixels = HEADS.reduce((total, head) => total + state.editor.maskBuffers[head.id].reduce((count, value) => count + (value > 0 ? 1 : 0), 0), 0);
  if (markedPixels === 0) {
    els.reviewStatus.textContent = "尚未画出任何标注；请先在图上涂画后再保存。";
    return;
  }
  const payload = { masks: {} };
  for (const head of HEADS) {
    payload.masks[head.id] = maskBufferToDataUrl(state.editor.maskBuffers[head.id], state.editor.width, state.editor.height);
  }
  payload.masks.pigment = maskBufferToDataUrl(state.editor.maskBuffers.pigment, state.editor.width, state.editor.height);
  const result = await runAction(
    `/api/workbench/sample/${encodeURIComponent(sample.id)}/save-annotation`,
    `${sample.id} 正在保存最终标注...`,
    payload,
  );
  state.overlaySource = "annotation";
  exitEditorMode({ preserveStatus: true });
  await refreshAll(result.scene_id ?? sample.id);
  els.reviewStatus.textContent = `${sample.id} 已保存最终标注`;
}

document.getElementById("sourceControls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-source]");
  if (!button || button.disabled) return;
  state.overlaySource = button.dataset.source === "annotation" ? "annotation" : "review";
  renderImages();
  renderHeadCards();
});

document.getElementById("layerControls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-layer]");
  if (!button) return;
  state.layer = button.dataset.layer;
  renderImages();
});

document.getElementById("editorHeadControls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-head]");
  if (!button) return;
  state.editor.head = button.dataset.head;
  renderEditorControls();
});

document.getElementById("editorModeControls").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) return;
  state.editor.mode = button.dataset.mode;
  renderEditorControls();
});

els.editorBrushInput.addEventListener("input", () => {
  state.editor.brushSize = Number(els.editorBrushInput.value || 24);
  renderEditorControls();
});


els.previewImage.addEventListener("load", () => {
  syncEditorCanvasFrame();
  if (state.editor.active) {
    drawEditorOverlay();
  }
});

window.addEventListener("resize", () => {
  syncEditorCanvasFrame();
  if (state.editor.active) {
    drawEditorOverlay();
  }
});

els.imageStage.addEventListener("mousemove", handleImageHover);
els.imageStage.addEventListener("mouseleave", hideHoverTooltip);
els.editorCanvas.addEventListener("pointerdown", handleEditorPointerDown);
els.editorCanvas.addEventListener("pointermove", handleEditorPointerMove);
window.addEventListener("pointerup", handleEditorPointerUp);
window.addEventListener("pointercancel", handleEditorPointerUp);

els.opacityInput.addEventListener("input", () => {
  els.overlayImage.style.opacity = els.opacityInput.value;
});

els.background4VersionSelect?.addEventListener("change", async () => {
  state.selectedBackground4VersionId = els.background4VersionSelect.value;
  if (els.runBackground4PredictButton) {
    els.runBackground4PredictButton.textContent = `运行 ${state.selectedBackground4VersionId} 预测`;
  }
  state.overlaySource = "review";
  resetHoverState();
  const prediction = await loadSelectedBackground4Prediction();
  render();
  els.reviewStatus.textContent = prediction?.available
    ? `${prediction.scene_id} 当前显示 ${prediction.version_id} 预测`
    : `${getSelectedSample()?.id ?? "当前样本"} 暂无 ${state.selectedBackground4VersionId} 预测`;
});

els.versionSelect.addEventListener("change", () => {
  if (state.editor.active) exitEditorMode({ preserveStatus: true });
  state.selectedVersionId = els.versionSelect.value;
  state.selectedSampleId = pickPreferredSampleId();
  resetHoverState();
  render();
});

els.importLatestButton.addEventListener("click", async () => {
  const result = await runAction("/api/workbench/import-latest", "正在导入最新 5 波段样本；请选择背景后才会预测...");
  state.overlaySource = "review";
  resetHoverState();
  const extra = result.reused ? ` ${TEXT.reused}` : "";
  els.reviewStatus.textContent = `${result.scene_id} 已导入${extra}；请选择背景并确认后开始预测`;
});

els.pigmentControls?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-pigment]");
  if (!button) return;
  state.editor.head = "pigment";
  state.editor.pigmentClass = Number(button.dataset.pigment);
  els.pigmentControls.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
  renderEditorControls();
});

els.importBackground4Button?.addEventListener("click", async () => {
  const result = await runAction("/api/workbench/import-background4", "正在导入四背景板正向、光照 5 母本...");
  els.reviewStatus.textContent = `已导入/复用 ${result.samples?.length ?? 0} 个母本；可直接标注或点击 softcomp 起标。`;
});

els.runReviewSeedButton?.addEventListener("click", async () => {
  const sample = getSelectedSample();
  if (!sample) return;
  await runAction(`/api/workbench/sample/${encodeURIComponent(sample.id)}/run-review-seed`, `${sample.id} 正在运行 softcomp 起标...`);
});

els.runBackground4PredictButton?.addEventListener("click", async () => {
  const sample = getSelectedSample();
  if (!sample) return;
  await runAction(
    `/api/workbench/sample/${encodeURIComponent(sample.id)}/run-background4-prediction`,
    `${sample.id} 正在运行 ${state.selectedBackground4VersionId} 像素级预测...`,
  );
});

els.backgroundControls?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-background]");
  if (!button) return;
  els.backgroundControls.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
});
els.lightControls?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-light]");
  if (!button) return;
  els.lightControls.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
});
els.confirmBackgroundButton?.addEventListener("click", async () => {
  const sample = getSelectedSample();
  const background = els.backgroundControls?.querySelector("button.active")?.dataset.background;
  const light = els.lightControls?.querySelector("button.active")?.dataset.light;
  if (!sample || !background) { els.reviewStatus.textContent = "请先选择一种背景板。"; return; }
  await runAction(`/api/workbench/sample/${encodeURIComponent(sample.id)}/confirm-background`, `${sample.id} 正在以 ${background} 背景启动 softcomp 起标...`, { background_role: background, light_level: light });
});

els.reloadButton.addEventListener("click", async () => {
  try {
    setBusy(true);
    resetHoverState();
    await refreshAll(state.selectedSampleId);
  } finally {
    setBusy(false);
  }
});

els.approveButton.addEventListener("click", async () => {
  const sample = getSelectedSample();
  if (!sample) return;
  if (state.editor.active) {
    els.reviewStatus.textContent = "请先保存或放弃当前人工修改，再执行通过";
    return;
  }
  await runAction(`/api/workbench/sample/${encodeURIComponent(sample.id)}/approve`, `${sample.id} 正在采用当前标注...`);
  state.overlaySource = "review";
  renderImages();
  renderHeadCards();
  els.reviewStatus.textContent = `${sample.id} 已采用当前标注`;
});

els.rejectButton.addEventListener("click", async () => {
  const sample = getSelectedSample();
  if (!sample) return;
  if (state.editor.active) {
    exitEditorMode({ preserveStatus: true });
  }
  await runAction(`/api/workbench/sample/${encodeURIComponent(sample.id)}/hold`, `${sample.id} 暂不采用当前结果...`);
});

els.editButton.addEventListener("click", async () => {
  if (state.editor.active) {
    exitEditorMode();
    return;
  }
  await enterEditorMode();
});

els.saveAnnotationButton.addEventListener("click", async () => {
  await saveAnnotationMasks();
});

els.discardAnnotationButton.addEventListener("click", () => {
  exitEditorMode();
});



refreshAll().catch((error) => {
  els.sampleList.innerHTML = `<div class="empty-state">${String(error.message || error)}</div>`;
  els.reviewStatus.textContent = String(error.message || error);
});
