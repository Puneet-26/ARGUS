# ARGUS OpenCV service

The Express API service starts this FastAPI worker on loopback and proxies its
health, operation-list, and image-processing routes to the browser-facing `/api`
path. The worker uses `opencv-python-headless`; it does not need a desktop GUI.

From the workspace root, install the locked Python dependencies with:

```sh
uv sync --locked
```

The API workflow starts the worker automatically. To run it directly for local
development:

```sh
cd artifacts/api-server/python
../../../.pythonlibs/bin/python -m uvicorn argus_cv.main:app --host 127.0.0.1 --port 8765
```

Supported uploads are JPEG, PNG, WebP, and BMP, up to 10 MB. OpenCV decodes the
uploaded bytes in memory; the worker does not persist the image.
