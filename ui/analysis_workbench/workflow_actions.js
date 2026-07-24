function describeHoldAction(record, sampleId = "") {
  const status = String(record?.status ?? "");
  const isApproved = status === "approved";
  return {
    label: isApproved ? "\u53d6\u6d88\u91c7\u7528" : "\u6682\u4e0d\u91c7\u7528",
    pendingText: isApproved
      ? `${sampleId} \u53d6\u6d88\u91c7\u7528\u5e76\u56de\u5230\u5ba1\u6838\u961f\u5217...`
      : `${sampleId} \u6682\u4e0d\u91c7\u7528\u5f53\u524d\u6807\u6ce8...`,
    endpointAction: "hold",
  };
}

window.WorkflowActions = { describeHoldAction };
