# Level 2 — Reconstruct the Lab Scene

**Task:** use actual pretrained YOLO predictions to determine which supported classes can be established in a laboratory image or baseline clip, and what the model does not establish.

**Learning objectives:** load local weights, read class labels/confidence/boxes, compare class occurrence across actual frames, and separate model output from identity or event claims.

The manifest currently lists `assets/evidence/level2_lab_before.png`; no video filename is present in this checkout. When the supplied lab clip is available, select it in ARGUS or pass its real path to the script. Do not create replacement footage.

1. Open `workshop_code/level2/` in VS Code and edit `starter.py`.
2. Complete its TODOs. The code accepts an image or video and reports missing files, weights, and dependencies.
3. From the project root, run `python workshop_code/level2/starter.py` or `python workshop_code/level2/starter.py --source PATH`.
4. Inspect the actual annotated frames and JSON under `outputs/level2/`.

The instructor reference is `solution.py`. Video inference evaluates every decodable frame and saves a bounded set of actual review frames plus class-frame counts. It does not track object identity. An intermittent or absent detection does not prove an object is absent; this baseline scene-analysis exercise is not proof of theft.
