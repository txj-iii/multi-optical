import http from "node:http";
import { createReadStream, existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import { tmpdir } from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const root = resolve(process.argv[2] ?? ".");
const port = Number(process.argv[3] ?? 8768);
const host = "127.0.0.1";
// Prefer an explicitly configured interpreter. Otherwise use the Python from
// the currently activated Conda environment, which keeps the UI portable.
const workflowPython = process.env.WORKBENCH_PYTHON || process.env.PYTHON || "python";
const workflowScript = resolve(join(root, "train", "workbench_workflow.py"));
const workflowStatePath = resolve(join(root, "ui", "analysis_workbench", "workflow_state.json"));
const workflowStatusLabels = {
  awaiting_background: "待选择背景",
  pending_review: "待审核",
  approved: "已采用",
  held: "暂不采用",
  error: "执行失败",
};

const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".bmp": "image/bmp",
};

function resolveRequestPath(url) {
  const pathname = decodeURIComponent(new URL(url, `http://${host}:${port}`).pathname);
  const safePath = normalize(pathname).replace(/^([/\\])+/, "");
  const fullPath = resolve(join(root, safePath || "ui/analysis_workbench/index.html"));
  const normalizedRoot = resolve(root).toLowerCase();
  const normalizedFullPath = resolve(fullPath).toLowerCase();
  if (!normalizedFullPath.startsWith(normalizedRoot)) {
    return null;
  }
  return fullPath;
}

function writeJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store, no-cache, must-revalidate",
  });
  response.end(JSON.stringify(payload));
}

function readWorkflowStatePayload() {
  let state = { samples: {}, updated_at: null };
  try {
    state = JSON.parse(readFileSync(workflowStatePath, "utf-8"));
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }
  const sampleRecords = state?.samples && typeof state.samples === "object" ? state.samples : {};
  const samples = Object.keys(sampleRecords)
    .sort()
    .map((sceneId) => {
      const record = { ...sampleRecords[sceneId] };
      const status = String(record.status || "");
      return {
        ...record,
        scene_id: sceneId,
        status_label: workflowStatusLabels[status] || status || "未知",
      };
    });
  return {
    samples,
    updated_at: state?.updated_at ?? null,
    count: samples.length,
  };
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(Buffer.from(chunk));
  }
  if (chunks.length === 0) return {};
  const text = Buffer.concat(chunks).toString("utf-8").trim();
  if (!text) return {};
  return JSON.parse(text);
}

function parseWorkflowOutput(stdout) {
  const lines = String(stdout ?? "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
  if (!lines.length) return {};
  return JSON.parse(lines[lines.length - 1]);
}

async function runWorkflowCommand(args) {
  const { stdout, stderr } = await execFileAsync(workflowPython, [workflowScript, ...args], {
    cwd: root,
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUTF8: "1",
    },
    encoding: "utf8",
    maxBuffer: 8 * 1024 * 1024,
    windowsHide: true,
  });
  return {
    payload: parseWorkflowOutput(stdout),
    stdout,
    stderr,
  };
}

async function handleApiRequest(request, response) {
  const url = new URL(request.url ?? "/", `http://${host}:${port}`);
  if (request.method === "GET" && url.pathname === "/api/workbench/state") {
    try {
      writeJson(response, 200, readWorkflowStatePayload());
    } catch (error) {
      writeJson(response, 500, { error: String(error?.message ?? error) });
    }
    return true;
  }
  if (request.method === "GET" && url.pathname === "/api/workbench/background4-prediction") {
    try {
      const sceneId = String(url.searchParams.get("scene_id") || "");
      const versionId = String(url.searchParams.get("version_id") || "");
      if (!/^SAMPLE_\d+$/.test(sceneId)) throw new Error("Invalid scene_id.");
      if (!["background4_v2", "background4_v3", "background4_v3_agingfix_v1_best"].includes(versionId)) {
        throw new Error("Unsupported background4 version.");
      }
      const relativeRoot = `train/experiments/five_band_predictions/task_specific/${versionId}/${sceneId}`;
      const predictionRoot = resolve(join(root, relativeRoot));
      const assetUrl = (name) => `/${relativeRoot.replaceAll("\\", "/")}/${name}`;
      const available = existsSync(join(predictionRoot, "combined_overlay.png"));
      writeJson(response, 200, {
        scene_id: sceneId,
        version_id: versionId,
        available,
        assets: available ? {
          combined_overlay: assetUrl("combined_overlay.png"),
        } : {},
        heads: available ? Object.fromEntries(["paint", "pollution", "aging"].map((head) => [head, {
          mask: assetUrl(`${head}_pred.png`),
          overlay: assetUrl(`${head}_overlay.png`),
        }])) : {},
        pigment_prediction: available && existsSync(join(predictionRoot, "pigment_pred.png")) ? {
          pixel_map: assetUrl("pigment_pred.png"),
          class_names: ["朱砂", "代赭", "石青", "石绿"],
          version_id: versionId,
        } : null,
      });
    } catch (error) {
      writeJson(response, 400, { error: String(error?.message ?? error) });
    }
    return true;
  }

  if (request.method === "POST" && url.pathname === "/api/workbench/import-latest") {
    try {
      await readJsonBody(request);
      const result = await runWorkflowCommand(["import-latest"]);
      writeJson(response, 200, result.payload);
    } catch (error) {
      writeJson(response, 500, { error: String(error?.message ?? error) });
    }
    return true;
  }
  if (request.method === "POST" && url.pathname === "/api/workbench/import-background4") {
    try {
      await readJsonBody(request);
      const result = await runWorkflowCommand(["import-background4-mothers"]);
      writeJson(response, 200, result.payload);
    } catch (error) { writeJson(response, 500, { error: String(error?.message ?? error) }); }
    return true;
  }
  const seedMatch = url.pathname.match(/^\/api\/workbench\/sample\/([^/]+)\/run-review-seed$/);
  if (request.method === "POST" && seedMatch) {
    try {
      const body = await readJsonBody(request);
      const args = ["run-review-seed", "--scene-id", decodeURIComponent(seedMatch[1])];
      if (body.reset) args.push("--reset");
      const result = await runWorkflowCommand(args);
      writeJson(response, 200, result.payload);
    } catch (error) { writeJson(response, 500, { error: String(error?.message ?? error) }); }
    return true;
  }
  const background4PredictMatch = url.pathname.match(/^\/api\/workbench\/sample\/([^/]+)\/run-background4-prediction$/);
  if (request.method === "POST" && background4PredictMatch) {
    try {
      const body = await readJsonBody(request);
      const versionId = body.version_id ?? "background4_v2";
      if (!["background4_v2", "background4_v3", "background4_v3_agingfix_v1_best"].includes(versionId)) {
        throw new Error("Unsupported background4 version.");
      }
      const result = await runWorkflowCommand(["run-background4-prediction", "--scene-id", decodeURIComponent(background4PredictMatch[1]), "--version-id", versionId]);
      writeJson(response, 200, result.payload);
    } catch (error) {
      const message = String(error?.message ?? error);
      if (message.includes("尚未完成正式训练")) {
        writeJson(response, 409, { error: message });
      } else {
        writeJson(response, 500, { error: message });
      }
    }
    return true;
  }

  const backgroundMatch = url.pathname.match(/^\/api\/workbench\/sample\/([^/]+)\/confirm-background$/);
  if (request.method === "POST" && backgroundMatch) {
    try {
      const body = await readJsonBody(request);
      const sceneId = decodeURIComponent(backgroundMatch[1]);
      const allowed = new Set(["代赭", "石青", "石绿", "朱砂"]);
      if (!allowed.has(body.background_role)) throw new Error("请选择有效背景：代赭、石青、石绿或朱砂。");
      const args = ["confirm-background", "--scene-id", sceneId, "--background-role", body.background_role];
      if (body.light_level) args.push("--light-level", String(body.light_level));
      const result = await runWorkflowCommand(args);
      writeJson(response, 200, result.payload);
    } catch (error) {
      writeJson(response, 500, { error: String(error?.message ?? error) });
    }
    return true;
  }


  const saveMatch = url.pathname.match(/^\/api\/workbench\/sample\/([^/]+)\/save-annotation$/);
  if (request.method === "POST" && saveMatch) {
    let tempDir = null;
    try {
      const body = await readJsonBody(request);
      const sceneId = decodeURIComponent(saveMatch[1]);
      tempDir = mkdtempSync(join(tmpdir(), "workbench-annotation-"));
      const payloadPath = join(tempDir, "masks.json");
      writeFileSync(payloadPath, JSON.stringify(body), "utf-8");
      const result = await runWorkflowCommand(["save-annotation", "--scene-id", sceneId, "--masks-json-path", payloadPath]);
      writeJson(response, 200, result.payload);
    } catch (error) {
      writeJson(response, 500, { error: String(error?.message ?? error) });
    } finally {
      if (tempDir) {
        rmSync(tempDir, { recursive: true, force: true });
      }
    }
    return true;
  }

  const sampleMatch = url.pathname.match(/^\/api\/workbench\/sample\/([^/]+)\/(approve|rerun|hold)$/);
  if (request.method === "POST" && sampleMatch) {
    try {
      const body = await readJsonBody(request);
      const sceneId = decodeURIComponent(sampleMatch[1]);
      const action = sampleMatch[2];
      const command = action === "rerun" ? "rerun-review" : action;
      const args = [command, "--scene-id", sceneId];
      const result = await runWorkflowCommand(args);
      writeJson(response, 200, result.payload);
    } catch (error) {
      writeJson(response, 500, { error: String(error?.message ?? error) });
    }
    return true;
  }

  return false;
}

const server = http.createServer(async (request, response) => {
  if (await handleApiRequest(request, response)) {
    return;
  }
  const fullPath = resolveRequestPath(request.url ?? "/");
  if (!fullPath) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }
  try {
    const stats = statSync(fullPath);
    const filePath = stats.isDirectory() ? join(fullPath, "index.html") : fullPath;
    response.writeHead(200, {
      "Content-Type": types[extname(filePath).toLowerCase()] ?? "application/octet-stream",
      "Cache-Control": "no-store, no-cache, must-revalidate",
    });
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404);
    response.end("Not found");
  }
});

server.listen(port, host, () => {
  console.log(`analysis workbench: http://${host}:${port}/ui/analysis_workbench/index.html`);
});
