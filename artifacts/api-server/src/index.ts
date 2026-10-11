import app from "./app";
import { logger } from "./lib/logger";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const apiDir = path.dirname(fileURLToPath(import.meta.url));
const pythonServiceDir = path.resolve(apiDir, "../python");
const pythonInterpreter =
  process.env["ARGUS_PYTHON_BIN"] ??
  path.resolve(pythonServiceDir, "../../../.pythonlibs/bin/python");
const visionPort = Number(process.env["ARGUS_VISION_PORT"] ?? "8765");
const visionWorker = spawn(
  pythonInterpreter,
  [
    "-m",
    "uvicorn",
    "argus_cv.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    String(visionPort),
    "--log-level",
    "warning",
  ],
  {
    cwd: pythonServiceDir,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  },
);

visionWorker.on("spawn", () => {
  logger.info({ port: visionPort }, "OpenCV service process started");
});
visionWorker.on("error", (err) => {
  logger.error({ err }, "Could not start the Python OpenCV service");
});
visionWorker.on("exit", (code, signal) => {
  logger.warn({ code, signal }, "Python OpenCV service stopped");
});
visionWorker.stdout.on("data", (chunk: Buffer) => {
  logger.info({ output: chunk.toString().trim() }, "OpenCV service");
});
visionWorker.stderr.on("data", (chunk: Buffer) => {
  logger.warn({ output: chunk.toString().trim() }, "OpenCV service");
});

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});

process.on("SIGTERM", () => {
  server.close(() => {
    visionWorker.kill("SIGTERM");
    process.exit(0);
  });
});

process.on("SIGINT", () => {
  server.close(() => {
    visionWorker.kill("SIGTERM");
    process.exit(0);
  });
});
