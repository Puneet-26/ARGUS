import { Router, type IRouter } from "express";
import {
  InspectImageResponse,
  ListOperationsResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();
const visionServiceUrl = `http://127.0.0.1:${process.env["ARGUS_VISION_PORT"] ?? "8765"}`;

router.get("/operations", async (req, res): Promise<void> => {
  try {
    const response = await fetch(`${visionServiceUrl}/operations`, {
      signal: AbortSignal.timeout(2_000),
    });
    if (!response.ok) {
      res.status(503).json({ error: "OpenCV operations are unavailable." });
      return;
    }

    const operations = ListOperationsResponse.parse(await response.json());
    res.json(operations);
  } catch (err) {
    req.log.warn({ err }, "Could not retrieve OpenCV operations");
    res.status(503).json({ error: "OpenCV operations are unavailable." });
  }
});

router.post("/image/inspect", async (req, res): Promise<void> => {
  const contentType = req.get("content-type");
  if (!contentType?.toLowerCase().startsWith("multipart/form-data;")) {
    res.status(400).json({ error: "Upload an image using multipart form data." });
    return;
  }

  if (!Buffer.isBuffer(req.body)) {
    res.status(400).json({ error: "The image upload could not be read." });
    return;
  }

  try {
    const response = await fetch(`${visionServiceUrl}/image/inspect`, {
      method: "POST",
      headers: { "content-type": contentType },
      body: req.body,
      signal: AbortSignal.timeout(10_000),
    });
    const payload = await response.arrayBuffer();
    const upstreamType = response.headers.get("content-type");

    res.status(response.status);
    if (upstreamType) {
      res.setHeader("content-type", upstreamType);
    }
    res.setHeader("cache-control", "no-store");
    if (response.ok) {
      const inspection = InspectImageResponse.parse(JSON.parse(Buffer.from(payload).toString("utf8")));
      res.json(inspection);
      return;
    }
    res.send(Buffer.from(payload));
  } catch (err) {
    req.log.error({ err }, "OpenCV image inspection request failed");
    res.status(503).json({ error: "The OpenCV processing service is unavailable." });
  }
});

router.post("/image/process", async (req, res): Promise<void> => {
  const contentType = req.get("content-type");
  if (!contentType?.toLowerCase().startsWith("multipart/form-data;")) {
    res.status(400).json({ error: "Upload an image using multipart form data." });
    return;
  }

  if (!Buffer.isBuffer(req.body)) {
    res.status(400).json({ error: "The image upload could not be read." });
    return;
  }

  try {
    const response = await fetch(`${visionServiceUrl}/image/process`, {
      method: "POST",
      headers: { "content-type": contentType },
      body: req.body,
      signal: AbortSignal.timeout(30_000),
    });
    const payload = Buffer.from(await response.arrayBuffer());
    const upstreamType = response.headers.get("content-type");

    res.status(response.status);
    if (upstreamType) {
      res.setHeader("content-type", upstreamType);
    }
    res.setHeader("cache-control", "no-store");
    res.send(payload);
  } catch (err) {
    req.log.error({ err }, "OpenCV image processing request failed");
    res.status(503).json({ error: "The OpenCV processing service is unavailable." });
  }
});

export default router;
