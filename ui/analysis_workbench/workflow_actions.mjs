export function describeHoldAction(record, sampleId = "") {
  const status = String(record?.status ?? "");
  const isApproved = status === "approved";
  return {
    label: isApproved ? "取消采用" : "暂不采用",
    pendingText: isApproved
      ? `${sampleId} 取消采用并回到审核队列...`
      : `${sampleId} 暂不采用当前标注...`,
    endpointAction: "hold",
  };
}
