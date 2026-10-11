import { useRef, useState } from "react";
import {
  Aperture, ArrowDownToLine, BookOpen, Braces, Check, ChevronDown,
  CircleAlert, Cpu, FileImage, History, ImagePlus, RotateCcw, ScanEye,
  SlidersHorizontal, Upload,
} from "lucide-react";
import "./argusdeck.css";

const recipes = [
  { id: "brightness_contrast", label: "Brightness / contrast", call: "cv.convertScaleAbs(image, alpha=1.0, beta=0)", description: "Rebalance tonal range and exposure." },
  { id: "gaussian_blur", label: "Gaussian blur", call: "cv.GaussianBlur(image, (5, 5), 0)", description: "Soften detail with a Gaussian kernel." },
  { id: "threshold", label: "Threshold", call: "cv.threshold(image, 127, 255, cv.THRESH_BINARY)", description: "Separate foreground from background." },
  { id: "adaptive_threshold", label: "Adaptive threshold", call: "cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)", description: "Set local thresholds across uneven light." },
  { id: "resize", label: "Resize", call: "cv.resize(image, None, fx=1.0, fy=1.0)", description: "Rescale the evidence frame." },
  { id: "clahe", label: "CLAHE", call: "cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))", description: "Recover local contrast in muted regions." },
  { id: "crop", label: "Crop", call: "image[y:y+h, x:x+w]", description: "Isolate a bounded area of interest." },
];

export default function ArgusDeck() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [image, setImage] = useState("");
  const [fileName, setFileName] = useState("");
  const [dims, setDims] = useState("");
  const [recipe, setRecipe] = useState(recipes[0]);
  const [observation, setObservation] = useState("");
  const [notes, setNotes] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<"history" | "notebook">("history");

  const chooseFile = (file?: File) => {
    if (!file || !file.type.startsWith("image/")) return;
    const nextUrl = URL.createObjectURL(file);
    if (image) URL.revokeObjectURL(image);
    setImage(nextUrl);
    setFileName(file.name);
    setDims("");
  };

  const clear = () => {
    if (image) URL.revokeObjectURL(image);
    setImage("");
    setFileName("");
    setDims("");
  };

  const saveNote = () => {
    const text = observation.trim();
    if (text) {
      setNotes((current) => [text, ...current]);
      setObservation("");
      setActiveTab("notebook");
    }
  };

  return (
    <div className="argus-deck">
      <header className="deck-topbar">
        <div className="deck-brand">
          <span className="deck-mark"><Aperture size={20} strokeWidth={1.7} /></span>
          <span className="deck-brand-copy">
            <strong>ARGUS<span> / LAB</span></strong>
            <small>COMPUTER VISION WORKBENCH</small>
          </span>
        </div>
        <nav className="deck-nav" aria-label="Workbench sections">
          <a href="#deck-workbench" className="deck-nav-active"><ScanEye size={14} />Workbench</a>
          <a href="#deck-records"><History size={14} />Records</a>
          <a href="#deck-observation"><BookOpen size={14} />Field notes</a>
        </nav>
        <div className="deck-status"><i /> PROCESSOR OFFLINE</div>
      </header>

      <main className="deck-main">
        <section className="deck-intro">
          <div>
            <div className="deck-eyebrow"><span /> LEVEL 01 <b>/</b> FIELD EXERCISE</div>
            <h1>Eyes <em>Online</em></h1>
            <p>Every transformation leaves evidence. Start with your own.</p>
          </div>
          <div className="deck-session"><small>SESSION</small><strong>CV—01</strong><span>FIELD SYSTEMS / 01</span></div>
        </section>

        <section id="deck-workbench" className="deck-bench">
          <div className="deck-bench-head">
            <div className="deck-section-label"><b>01</b> IMAGE WORKBENCH</div>
            <div className="deck-file-tools">
              {fileName && <span className="deck-file-name"><FileImage size={13} />{fileName}</span>}
              <button className="deck-quiet-button" onClick={clear} disabled={!image}><RotateCcw size={13} />Reset</button>
            </div>
          </div>

          <div className="deck-recipe">
            <div className="deck-recipe-select">
              <label htmlFor="deck-recipe">PROCESSING BAY <span>01 / SELECT</span></label>
              <div className="deck-select-wrap">
                <select id="deck-recipe" value={recipe.id} onChange={(event) => setRecipe(recipes.find((item) => item.id === event.target.value) ?? recipes[0])}>
                  {recipes.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
                </select>
                <ChevronDown size={14} />
              </div>
              <p>{recipe.description}</p>
            </div>
            <div className="deck-call">
              <span><Braces size={12} /> OPENCV CALL</span>
              <code>{recipe.call}</code>
            </div>
            <div className="deck-parameter">
              <div><span>CONTRAST</span><strong>1.0</strong></div>
              <input aria-label="Contrast" type="range" min="0" max="3" step=".1" defaultValue="1" />
              <small><span>0</span><span>3</span></small>
            </div>
            <div className="deck-parameter deck-brightness">
              <div><span>BRIGHTNESS</span><strong>0 levels</strong></div>
              <input aria-label="Brightness" type="range" min="-100" max="100" defaultValue="0" />
              <small><span>-100</span><span>100</span></small>
            </div>
            <button className="deck-run" disabled title="OpenCV operations unlock when the processor is available">
              <Cpu size={15} /> Run operation <span>↗</span>
            </button>
          </div>

          <div className="deck-comparison">
            <div className="deck-compare-head">
              <div><span className="deck-red-square" /> SOURCE <i>↔</i> RESULT</div>
              <span>{image ? "SOURCE LOADED · AWAITING PROCESSOR" : "AWAITING EVIDENCE"}</span>
            </div>
            <div className="deck-plates">
              <article className="deck-plate deck-source">
                <header><span><i className="plate-dot plate-red" /> ORIGINAL</span><small>{image ? "INPUT" : "01 / INPUT"}</small></header>
                <div className={`deck-stage ${image ? "deck-stage-loaded" : ""}`}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}>
                  {image ? <>
                    <img src={image} alt={`Original evidence: ${fileName}`} onLoad={(event) => setDims(`${event.currentTarget.naturalWidth} × ${event.currentTarget.naturalHeight}`)} />
                    <span className="deck-image-chip">{dims || "READING FRAME"}</span>
                  </> : <div className="deck-upload">
                    <div className="deck-upload-icon"><ImagePlus size={22} strokeWidth={1.4} /></div>
                    <span>NO SOURCE LOADED</span>
                    <h2>Bring your own<br /><em>evidence.</em></h2>
                    <p>Choose a real image from your device, or drop it here to begin.</p>
                    <button onClick={() => inputRef.current?.click()}><Upload size={13} /> Select image</button>
                    <small>PNG · JPG · WEBP · BMP</small>
                  </div>}
                </div>
                <footer><span>{image ? "SOURCE READY" : "IMAGE INPUT"}</span><span>{image ? fileName : "DROP IMAGE TO LOAD"}</span></footer>
              </article>
              <article className="deck-plate deck-result">
                <header><span><i className="plate-dot" /> PROCESSED</span><small>02 / OUTPUT</small></header>
                <div className="deck-stage deck-result-stage">
                  <div className="deck-reticle"><i /><i /><i /><i /></div>
                  <span className="deck-result-title">RESULT BAY / STANDBY</span>
                  <p>{image ? "Choose an OpenCV operation and run it to see the actual output." : "Your processed image will appear here after a real operation."}</p>
                </div>
                <footer><span>SERVER PROCESSED</span><span>— / —</span></footer>
              </article>
            </div>
            <div className="deck-inspection">
              <span className="deck-inspection-state"><i /> SOURCE INSPECTION</span>
              <span>DIMENSIONS <b>{dims || "—"}</b></span>
              <span>CHANNELS <b>—</b></span>
              <span>COLOR MODEL <b>—</b></span>
              <span>BIT DEPTH <b>—</b></span>
              <span>MIME <b>{image ? "LOCAL IMAGE" : "—"}</b></span>
            </div>
          </div>
          <div className="deck-offline"><CircleAlert size={14} /><span><b>PROCESSOR UNAVAILABLE</b> — OpenCV operations unlock when the workshop processor reports healthy.</span><button onClick={() => window.location.reload()}>Retry connection</button></div>
        </section>

        <div className="deck-lower">
          <section id="deck-observation" className="deck-observation">
            <header><div className="deck-section-label"><b>02</b> OBSERVATION</div><span>WRITE WHAT THE OPERATION REVEALS</span></header>
            <div className="deck-note-body">
              <div className="deck-note-prompt"><span>FIELD NOTE</span><p>What changed in the image? What detail became clearer, weaker, or disappeared?</p></div>
              <div className="deck-note-editor">
                <textarea value={observation} onChange={(event) => setObservation(event.target.value)}
                  disabled title="Run an operation before recording an observation"
                  placeholder="Run an operation before recording an observation." maxLength={1200} />
                <div><span>{observation.length} / 1200</span><button onClick={saveNote} disabled={!observation.trim()}><BookOpen size={13} /> Save to notebook</button></div>
              </div>
            </div>
          </section>
          <section id="deck-records" className="deck-records">
            <header className="deck-record-head">
              <div className="deck-section-label"><b>03—04</b> FIELD RECORDS</div>
              <div className="deck-tabs">
                <button className={activeTab === "history" ? "is-active" : ""} onClick={() => setActiveTab("history")}><History size={12} /> HISTORY <i>00</i></button>
                <button className={activeTab === "notebook" ? "is-active" : ""} onClick={() => setActiveTab("notebook")}><BookOpen size={12} /> NOTEBOOK <i>{String(notes.length).padStart(2, "0")}</i></button>
              </div>
            </header>
            {activeTab === "history" ? <div className="deck-record-empty"><span>A</span><div><b>No processing history yet.</b><small>Completed OpenCV runs will collect here for this session.</small></div><History size={16} /></div>
              : notes.length === 0 ? <div className="deck-record-empty"><span>B</span><div><b>Your notebook is still blank.</b><small>Save an observation after processing an image. Entries persist in this browser.</small></div><BookOpen size={16} /></div>
                : <div className="deck-note-list">{notes.map((note, index) => <div className="deck-saved-note" key={`${index}-${note}`}><b>{String(notes.length - index).padStart(2, "0")}</b><p>{note}</p><Check size={14} /></div>)}</div>}
          </section>
        </div>
        <footer className="deck-footer"><span>ARGUS / FIELD SYSTEMS</span><span>LEVEL 1 — EYES ONLINE</span><span>RESULTS ARE SERVER-PROCESSED</span></footer>
      </main>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={(event) => chooseFile(event.target.files?.[0])} />
      {image && <button className="deck-mobile-upload" onClick={() => inputRef.current?.click()}><Upload size={14} /> Replace image</button>}
    </div>
  );
}
