const BAND_LABELS = [450, 550, 600, 650, 700];
const SUBREGION_COLORS = ["#1c6fb8", "#d9485f", "#2b8a3e", "#a269ff"];

function hasCurveValues(values) {
  return Array.isArray(values) && values.length === BAND_LABELS.length && values.some((value) => Number(value ?? 0) !== 0);
}

function getSubregionRows(sample) {
  const direct = sample?.pigment_analysis?.cluster_analysis?.region_summaries;
  if (Array.isArray(direct) && direct.length) return direct;
  const mixed = sample?.mixed_pigment_analysis?.subregions;
  if (Array.isArray(mixed) && mixed.length) return mixed;
  return [];
}

function buildSubregionSeries(sample) {
  return getSubregionRows(sample)
    .map((region, index) => {
      if (Number(region?.positive_pixels ?? 0) <= 0 || !hasCurveValues(region?.curve_values)) return null;
      const clusterId = Number(region.cluster_id ?? index + 1);
      const regionLabel = String(region.label ?? region.raw_label ?? "\u672a\u547d\u540d");
      return {
        id: `cluster-${clusterId}`,
        label: `\u5b50\u533a ${clusterId} \u00b7 ${regionLabel}`,
        color: SUBREGION_COLORS[index % SUBREGION_COLORS.length],
        values: region.curve_values.map((value) => Number(value ?? 0)),
        normalized: Array.isArray(region.curve_normalized) ? region.curve_normalized.map((value) => Number(value ?? 0)) : [],
        positivePixels: Number(region.positive_pixels ?? 0),
        peakWavelength: Number(region.peak_wavelength ?? 0) || null,
      };
    })
    .filter(Boolean);
}

export function buildCurvePanelPayload(sample) {
  const sections = [];
  const subregionSeries = buildSubregionSeries(sample);
  if (subregionSeries.length) {
    sections.push({
      id: 'pigment-subregions',
      title: '\u989c\u6599\u5b50\u533a\u6ce2\u6bb5\u66f2\u7ebf',
      description: '\u5c55\u793a\u4e3b\u56fe\u4e0b\u65b9\u5404\u989c\u6599\u5b50\u533a\u7684\u4e94\u6ce2\u6bb5\u54cd\u5e94\uff0c\u4fbf\u4e8e\u4eba\u5de5\u6bd4\u5bf9',
      series: subregionSeries,
    });
  }

  return {
    bandLabels: BAND_LABELS.slice(),
    sections,
  };
}
