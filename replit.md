# ARGUS

ARGUS is an educational computer-vision workshop where participants investigate image evidence using real OpenCV operations and record their observations.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the Express API gateway and its FastAPI/OpenCV worker
- `pnpm --filter @workspace/argus run dev` — run the React workshop app
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `uv sync --locked` — install the locked Python/OpenCV dependencies from the workspace root

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React, TypeScript, Vite
- API gateway: Express 5
- Processing: Python FastAPI, NumPy, `opencv-python-headless`
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/argus/` — ARGUS Level 1 workshop interface
- `artifacts/api-server/src/` — Express API gateway at `/api`
- `artifacts/api-server/python/argus_cv/` — FastAPI service and modular OpenCV functions
- `lib/api-spec/openapi.yaml` — API contract source of truth
- `lib/api-client-react/` and `lib/api-zod/` — generated client hooks and validation schemas

## Architecture decisions

- Uploaded images are decoded and processed in Python/OpenCV; browser filters and simulated image results are not acceptable substitutes.
- The API workflow starts FastAPI as a loopback worker and proxies it through the shared `/api` path.
- The evidence notebook stores operation metadata and participant observations in browser local storage; uploaded image bytes are not persisted.
- The workshop currently implements Level 1 only. Participants supply their own image because no CCTV evidence image was provided with the spec.
- Never reveal or hardcode the intended timestamp clue before participants analyze the evidence.

## Product

Participants upload an image, inspect its OpenCV-measured dimensions and channel layout, apply grayscale, brightness/contrast, histogram equalization, CLAHE, Gaussian blur, thresholding, adaptive thresholding, crop, or resize, compare original and processed images, review session history, and save observations in a persistent evidence notebook.

## User preferences

- Use dark charcoal surfaces, muted red and amber accents, and condensed typography.
- Do not use green backgrounds.
- Keep algorithm output, model predictions, prepared assets, and fictional narrative records clearly distinguished.
- Do not imply that image processing proves identity, ownership, intent, or guilt.

## Gotchas

- Change `lib/api-spec/openapi.yaml` before generated API code, then rerun codegen before importing new hooks or schemas.
- Express must preserve multipart boundaries when forwarding uploads to FastAPI; keep its raw-body middleware ahead of JSON parsing.
- OpenCV's headless package is intentional for server use; do not install it alongside `opencv-python`.
- Do not persist uploaded source images unless the user explicitly asks for that behavior.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
