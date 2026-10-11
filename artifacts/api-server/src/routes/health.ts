import { Router, type IRouter } from "express";
import { HealthCheckResponse } from "@workspace/api-zod";

const router: IRouter = Router();
const visionServiceUrl = `http://127.0.0.1:${process.env["ARGUS_VISION_PORT"] ?? "8765"}`;

router.get("/healthz", async (req, res): Promise<void> => {
  let data: unknown;

  try {
    const response = await fetch(`${visionServiceUrl}/healthz`, {
      signal: AbortSignal.timeout(1_500),
    });
    if (!response.ok) {
      throw new Error(`OpenCV service returned ${response.status}`);
    }
    data = await response.json();
  } catch (err) {
    req.log.warn({ err }, "OpenCV health check failed");
    data = {
      status: "unavailable",
      pythonAvailable: false,
      opencvVersion: null,
    };
  }

  const parsed = HealthCheckResponse.parse(data);
  res.json(parsed);
});

export default router;
