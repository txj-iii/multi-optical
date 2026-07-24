function buildAuditLayerId(headName, candidateKind) {
  return `audit:${headName}:${candidateKind}`;
}

function listAuditEntries(annotationAudit) {
  const heads = annotationAudit?.heads;
  if (!heads || typeof heads !== "object") return [];
  const entries = [];
  for (const headName of ["paint", "pollution", "aging"]) {
    const headPayload = heads[headName];
    if (!headPayload || typeof headPayload !== "object") continue;
    for (const candidateKind of ["missing", "overmark"]) {
      const item = headPayload[candidateKind];
      if (!item || typeof item !== "object") continue;
      entries.push({
        head: headName,
        kind: candidateKind,
        layerId: buildAuditLayerId(headName, candidateKind),
        positivePixels: Number(item.positive_pixels ?? 0),
        componentCount: Number(item.component_count ?? 0),
        overlay: item.overlay ?? null,
        mask: item.mask ?? null,
      });
    }
  }
  return entries;
}

function getAuditOverlay(annotationAudit, layerId) {
  const entries = listAuditEntries(annotationAudit);
  return entries.find((item) => item.layerId === layerId)?.overlay ?? null;
}

function getDefaultAuditLayer(annotationAudit) {
  const entries = listAuditEntries(annotationAudit).filter((item) => item.positivePixels > 0 && item.overlay);
  if (!entries.length) return null;
  entries.sort((a, b) => b.positivePixels - a.positivePixels || a.layerId.localeCompare(b.layerId));
  return entries[0].layerId;
}

export { buildAuditLayerId, getAuditOverlay, getDefaultAuditLayer, listAuditEntries };
