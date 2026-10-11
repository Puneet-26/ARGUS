import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import {
  Activity, Aperture, ArrowDownToLine, BookOpen, Braces, Check, ChevronDown,
  CircleAlert, Cpu, FileImage, Focus, History, ImagePlus, LoaderCircle,
  RotateCcw, ScanEye, SlidersHorizontal, Trash2, Upload, X,
} from 'lucide-react';
import {
  getListOperationsQueryKey, useHealthCheck, useInspectImage, useListOperations, useProcessImage,
} from '@workspace/api-client-react';
import type { ImageOperation } from '@workspace/api-client-react';

const queryClient = new QueryClient();
const NOTEBOOK_KEY = 'argus-evidence-notebook-v1';

type NotebookEntry = {
  id: string;
  createdAt: string;
  sourceName: string;
  operationId: string;
  operationLabel: string;
  cvFunction: string;
  parameters: Record<string, number>;
  observation: string;
};
type ParameterDef = { key: string; label: string; min: number; max: number; step: number; initial: number; unit?: string };
type HistoryItem = {
  id: string; operation: ImageOperation; parameters: Record<string, number>; blob: Blob; sourceName: string;
};

const OPERATION_PARAMETERS: Record<string, ParameterDef[]> = {
  brightness_contrast: [
    { key: 'contrast', label: 'Contrast', min: 0, max: 3, step: 0.1, initial: 1 },
    { key: 'brightness', label: 'Brightness', min: -100, max: 100, step: 1, initial: 0, unit: 'levels' },
  ],
  gaussian_blur: [{ key: 'kernel_size', label: 'Kernel size', min: 1, max: 51, step: 2, initial: 5, unit: 'px' }],
  threshold: [{ key: 'threshold', label: 'Threshold', min: 0, max: 255, step: 1, initial: 127 }],
  adaptive_threshold: [
    { key: 'block_size', label: 'Neighborhood · block size', min: 3, max: 99, step: 2, initial: 11, unit: 'px' },
    { key: 'constant', label: 'Constant', min: -20, max: 20, step: 1, initial: 2 },
  ],
  resize: [{ key: 'scale', label: 'Scale', min: 0.1, max: 2.5, step: 0.1, initial: 1 }],
  clahe: [
    { key: 'clip_limit', label: 'Clip limit', min: 1, max: 8, step: 0.5, initial: 2 },
    { key: 'tile_grid_size', label: 'Tile grid size', min: 2, max: 32, step: 1, initial: 8 },
  ],
};

function parameterDefinitions(operation?: ImageOperation, imageSize?: { width: number; height: number } | null, values: Record<string, number> = {}) {
  if (operation?.id !== 'crop') return OPERATION_PARAMETERS[operation?.id ?? ''] ?? [];
  const imageWidth = imageSize?.width ?? 2048;
  const imageHeight = imageSize?.height ?? 2048;
  const width = Math.min(Math.max(1, values.width ?? Math.min(320, imageWidth)), imageWidth);
  const height = Math.min(Math.max(1, values.height ?? Math.min(240, imageHeight)), imageHeight);
  const x = Math.min(Math.max(0, values.x ?? 0), Math.max(0, imageWidth - width));
  const y = Math.min(Math.max(0, values.y ?? 0), Math.max(0, imageHeight - height));
  return [
    { key: 'x', label: 'Left · x', min: 0, max: Math.max(0, imageWidth - width), step: 1, initial: 0, unit: 'px' },
    { key: 'y', label: 'Top · y', min: 0, max: Math.max(0, imageHeight - height), step: 1, initial: 0, unit: 'px' },
    { key: 'width', label: 'Crop width', min: 1, max: Math.max(1, imageWidth - x), step: 1, initial: Math.min(320, imageWidth), unit: 'px' },
    { key: 'height', label: 'Crop height', min: 1, max: Math.max(1, imageHeight - y), step: 1, initial: Math.min(240, imageHeight), unit: 'px' },
  ];
}

function defaultsFor(operation?: ImageOperation, imageSize?: { width: number; height: number } | null) {
  return Object.fromEntries(parameterDefinitions(operation, imageSize).map((item) => [item.key, item.initial]));
}

function clampCropParameters(values: Record<string, number>, imageSize?: { width: number; height: number } | null) {
  const maxWidth = Math.max(1, imageSize?.width ?? values.width ?? 1);
  const maxHeight = Math.max(1, imageSize?.height ?? values.height ?? 1);
  const x = Math.min(Math.max(0, Math.round(values.x ?? 0)), maxWidth - 1);
  const y = Math.min(Math.max(0, Math.round(values.y ?? 0)), maxHeight - 1);
  const width = Math.min(Math.max(1, Math.round(values.width ?? maxWidth)), maxWidth - x);
  const height = Math.min(Math.max(1, Math.round(values.height ?? maxHeight)), maxHeight - y);
  return { x, y, width, height };
}

function valuesForOperation(operation: ImageOperation, values: Record<string, number>, imageSize?: { width: number; height: number } | null) {
  if (operation.id === 'crop') return clampCropParameters(values, imageSize);
  return Object.fromEntries(parameterDefinitions(operation).map(({ key, min, max, step, initial }) => {
    const raw = Math.min(max, Math.max(min, Number.isFinite(values[key]) ? values[key] : initial));
    const snapped = min + Math.round((raw - min) / step) * step;
    return [key, Number(snapped.toFixed(3))];
  }));
}

function readNotebook(): NotebookEntry[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(NOTEBOOK_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed as NotebookEntry[] : [];
  } catch {
    return [];
  }
}

function Home() {
  const health = useHealthCheck();
  const inspectImage = useInspectImage();
  const isHealthy = health.isSuccess && health.data?.pythonAvailable === true;
  const operationsQuery = useListOperations({
    query: { enabled: isHealthy, queryKey: getListOperationsQueryKey() },
  });
  const processImage = useProcessImage();
  const fileInput = useRef<HTMLInputElement>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState('');
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [resultUrl, setResultUrl] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [parameters, setParameters] = useState<Record<string, number>>({});
  const [notebook, setNotebook] = useState<NotebookEntry[]>(readNotebook);
  const [observation, setObservation] = useState('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentOperation, setCurrentOperation] = useState<ImageOperation | null>(null);
  const [currentParameters, setCurrentParameters] = useState<Record<string, number>>({});
  const [dragActive, setDragActive] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const operations = useMemo(() => operationsQuery.data ?? [], [operationsQuery.data]);
  const selectedOperation = operations.find((item) => item.id === selectedId) ?? operations[0];
  const parameterDefs = useMemo(
    () => parameterDefinitions(selectedOperation, imageSize, parameters),
    [selectedOperation?.id, imageSize, parameters],
  );
  const hasResult = Boolean(resultUrl && currentOperation);

  useEffect(() => {
    if (operations.length && !operations.some((operation) => operation.id === selectedId)) {
      setSelectedId(operations[0].id);
      setParameters(defaultsFor(operations[0], imageSize));
    }
  }, [operations, selectedId, imageSize]);

  useEffect(() => {
    if (selectedOperation?.id === 'crop' && imageSize) {
      setParameters((current) => Object.keys(current).length
        ? clampCropParameters(current, imageSize)
        : defaultsFor(selectedOperation, imageSize));
    }
  }, [selectedOperation?.id, imageSize]);

  useEffect(() => () => {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  }, [sourceUrl]);
  useEffect(() => () => {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
  }, [resultUrl]);
  useEffect(() => {
    localStorage.setItem(NOTEBOOK_KEY, JSON.stringify(notebook));
  }, [notebook]);

  function loadFile(file?: File) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setErrorMessage('Choose an image file to begin an investigation.');
      return;
    }
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSourceFile(file);
    setSourceUrl(URL.createObjectURL(file));
    setImageSize(null);
    inspectImage.reset();
    inspectImage.mutate({ data: { image: file } });
    setResultUrl('');
    setCurrentOperation(null);
    setHistory([]);
    setObservation('');
    setErrorMessage('');
  }

  async function runOperation() {
    if (!sourceFile || !selectedOperation || !isHealthy) return;
    if (selectedOperation.id === 'crop' && !imageSize) return;
    setErrorMessage('');
    const operationValues = valuesForOperation(selectedOperation, parameters, imageSize);
    try {
      const blob = await processImage.mutateAsync({
        data: {
          image: sourceFile,
          operation: selectedOperation.id,
          parameters: JSON.stringify(operationValues),
        },
      });
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      setResultUrl(URL.createObjectURL(blob));
      setCurrentOperation(selectedOperation);
      setCurrentParameters(operationValues);
      setHistory((items) => [{
        id: `${Date.now()}-${items.length}`,
        operation: selectedOperation,
        parameters: operationValues,
        blob,
        sourceName: sourceFile.name,
      }, ...items].slice(0, 8));
      setObservation('');
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'The OpenCV operation could not be completed.');
    }
  }

  function resetWorkspace() {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl('');
    setCurrentOperation(null);
    setCurrentParameters({});
    setObservation('');
    setErrorMessage('');
    if (selectedOperation) setParameters(defaultsFor(selectedOperation, imageSize));
  }

  function saveObservation() {
    if (!sourceFile || !currentOperation || !hasResult || !observation.trim()) return;
    const entry: NotebookEntry = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      sourceName: sourceFile.name,
      operationId: currentOperation.id,
      operationLabel: currentOperation.label,
      cvFunction: currentOperation.cvFunction,
      parameters: currentParameters,
      observation: observation.trim(),
    };
    setNotebook((items) => [entry, ...items]);
    setObservation('');
  }

  function showHistory(item: HistoryItem) {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl(URL.createObjectURL(item.blob));
    setCurrentOperation(item.operation);
    setCurrentParameters(item.parameters);
    setObservation('');
  }

  function downloadResult() {
    if (!resultUrl) return;
    const link = document.createElement('a');
    link.href = resultUrl;
    link.download = `${sourceFile?.name.replace(/\.[^.]+$/, '') ?? 'evidence'}-${currentOperation?.id ?? 'result'}.png`;
    link.click();
  }

  const serviceLabel = health.isLoading ? 'CHECKING PROCESSOR' : isHealthy ? `OPENCV ${health.data?.opencvVersion ?? 'READY'}` : 'PROCESSOR OFFLINE';

  return (
    <div className="argus-app min-h-[100dvh]">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><Aperture size={22} strokeWidth={1.7} /></div>
          <div>
            <div className="brand-name">ARGUS<span> / LAB</span></div>
            <div className="brand-caption">COMPUTER VISION WORKBENCH</div>
          </div>
        </div>
        <div className="topbar-center"><span className="topbar-segment">WORKSHOP</span><span className="slash">/</span><span>LEVEL 01</span></div>
        <div className={`service-pill ${isHealthy ? 'is-ready' : ''}`} data-testid="status-processor">
          <span className="status-dot" />{serviceLabel}
        </div>
      </header>

      <div className="shell">
        <aside className="sidebar">
          <div className="side-section-label">INVESTIGATION</div>
          <a href="#workbench" className="side-link selected" data-testid="link-workbench"><ScanEye size={16} /><span>Workbench</span><span className="side-index">01</span></a>
          <a href="#history" className="side-link" data-testid="link-history"><History size={16} /><span>Processing history</span><span className="side-count">{history.length.toString().padStart(2, '0')}</span></a>
          <a href="#notebook" className="side-link" data-testid="link-notebook"><BookOpen size={16} /><span>Evidence notebook</span><span className="side-count">{notebook.length.toString().padStart(2, '0')}</span></a>
          <div className="side-divider" />
          <div className="level-card">
            <div className="level-number">01</div>
            <div className="level-eyebrow">FIELD EXERCISE</div>
            <div className="level-title">Eyes<br />Online</div>
            <div className="level-rule" />
            <p>Upload an image. Apply a real OpenCV operation. Record what changes.</p>
          </div>
          <div className="sidebar-bottom">
            <div className="mini-mark"><Focus size={15} /> <span>OPTICAL SYSTEMS</span></div>
            <span className="side-code">ARG / CV–01</span>
          </div>
        </aside>

        <main className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> LEVEL 1 <span className="heading-slash">/</span> FIELD NOTES</div>
              <h1>Eyes <em>Online</em></h1>
              <p className="heading-subtitle">Every transformation leaves evidence. Start with your own.</p>
            </div>
            <div className="heading-meta"><span className="meta-label">SESSION</span><span className="meta-value">CV–01</span></div>
          </div>

          <section id="workbench" className="workbench-section">
            <div className="section-bar">
              <div className="section-title"><span className="section-number">01</span><span>IMAGE WORKBENCH</span></div>
              <div className="section-tools">
                {sourceFile && <div className="source-tag"><FileImage size={13} /><span>{sourceFile.name}</span></div>}
                <button className="text-button" onClick={resetWorkspace} disabled={!sourceFile} data-testid="button-reset"><RotateCcw size={13} /> Reset</button>
              </div>
            </div>

            <div className="workbench-grid">
              <div className="visual-column">
                <div className="comparison-heading">
                  <div className="compare-title"><span className="compare-marker" /> SOURCE <span className="compare-vs">↔</span> RESULT</div>
                  <div className="comparison-state">{hasResult ? 'COMPARISON READY' : 'AWAITING EVIDENCE'}</div>
                </div>
                <div className={`image-pair ${sourceFile ? 'has-source' : ''}`}>
                  <div className="image-frame source-frame">
                    <div className="frame-head"><span><span className="frame-dot red-dot" /> ORIGINAL</span>{sourceFile && <span className="frame-label">INPUT</span>}</div>
                    <div className={`image-stage ${dragActive && !sourceFile ? 'drag-active' : ''} ${!sourceFile ? 'empty-stage' : ''}`}
                      onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
                      onDragLeave={() => setDragActive(false)}
                      onDrop={(event) => { event.preventDefault(); setDragActive(false); loadFile(event.dataTransfer.files[0]); }}
                      data-testid="panel-original">
                      {sourceFile ? (
                        <>
                      <img src={sourceUrl} alt={`Original evidence: ${sourceFile.name}`} className="evidence-image" data-testid="img-original"
                        onLoad={(event) => {
                          const size = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
                          setImageSize(size);
                          if (selectedOperation?.id === 'crop') {
                            setParameters((current) => Object.keys(current).length
                              ? clampCropParameters(current, size)
                              : defaultsFor(selectedOperation, size));
                          }
                        }} />
                        </>
                      ) : (
                        <div className="upload-prompt">
                          <div className="upload-symbol"><ImagePlus size={25} strokeWidth={1.4} /></div>
                          <div className="upload-kicker">NO SOURCE LOADED</div>
                          <h2>Bring your own<br /><em>evidence.</em></h2>
                          <p>Choose a real image from your device, or drop it here to begin.</p>
                          <button className="upload-button" onClick={() => fileInput.current?.click()} data-testid="button-upload"><Upload size={14} /> Select image</button>
                          <span className="format-note">PNG · JPG · WEBP · BMP</span>
                        </div>
                      )}
                    </div>
                    {sourceFile && (
                      <div className="inspection-strip" data-testid="panel-inspection">
                        {inspectImage.isPending ? (
                          <div className="inspection-state inspection-loading"><LoaderCircle className="spin" size={13} /><span>INSPECTING SOURCE WITH OPENCV</span><span className="inspection-wait">READING PIXEL STRUCTURE</span></div>
                        ) : inspectImage.isError ? (
                          <div className="inspection-state inspection-error" role="alert" data-testid="status-inspection-error">
                            <CircleAlert size={14} />
                            <span>{inspectImage.error instanceof Error ? inspectImage.error.message : 'OpenCV could not inspect this source image.'}</span>
                            <button onClick={() => { inspectImage.reset(); inspectImage.mutate({ data: { image: sourceFile } }); }} data-testid="button-retry-inspection">Retry inspection</button>
                          </div>
                        ) : inspectImage.data ? (
                          <div className="inspection-values" data-testid="text-inspection-properties">
                            <div className="inspection-origin"><span className="inspection-status-mark" /><span>SOURCE INSPECTED</span></div>
                            <div><span>DIMENSIONS</span><strong>{inspectImage.data.width} × {inspectImage.data.height}</strong></div>
                            <div><span>CHANNELS</span><strong>{inspectImage.data.channels}</strong></div>
                            <div><span>COLOR MODEL</span><strong>{inspectImage.data.colorModel}</strong></div>
                            <div><span>BIT DEPTH</span><strong>{inspectImage.data.bitDepth}</strong></div>
                            <div className="inspection-mime"><span>MIME</span><strong>{inspectImage.data.mimeType}</strong></div>
                          </div>
                        ) : (
                          <div className="inspection-state"><span>WAITING FOR INSPECTION RESULT</span></div>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="image-frame result-frame">
                    <div className="frame-head"><span><span className={`frame-dot ${hasResult ? 'amber-dot' : ''}`} /> PROCESSED</span>{hasResult && <span className="frame-label amber-label">OPENCV OUTPUT</span>}</div>
                    <div className="image-stage result-stage" data-testid="panel-result">
                      {hasResult ? (
                        <>
                          <img src={resultUrl} alt={`OpenCV result for ${sourceFile?.name}`} className="evidence-image" data-testid="img-result" />
                          <div className="image-dimensions">PNG · SERVER PROCESSED</div>
                          <button className="download-overlay" onClick={downloadResult} aria-label="Download processed PNG" data-testid="button-download"><ArrowDownToLine size={15} /></button>
                        </>
                      ) : (
                        <div className="result-empty">
                          <div className="result-target"><span /><span /><span /><span /></div>
                          <div className="result-empty-title">{sourceFile ? 'RESULT NOT YET GENERATED' : 'RESULT BAY / STANDBY'}</div>
                          <p>{sourceFile ? 'Choose an OpenCV operation and run it to see the actual output.' : 'Your processed image will appear here after a real operation.'}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                {errorMessage && <div className="error-banner" role="alert" data-testid="status-error"><CircleAlert size={15} /> <span>{errorMessage}</span><button onClick={() => setErrorMessage('')} aria-label="Dismiss error"><X size={14} /></button></div>}
                {hasResult && currentOperation && (
                  <div className="result-caption" data-testid="text-current-operation">
                    <span className="caption-label">CURRENT PIPELINE</span>
                    <span className="caption-operation">{currentOperation.label}</span>
                    <code>{currentOperation.cvFunction}</code>
                    <button className="caption-download" onClick={downloadResult}><ArrowDownToLine size={13} /> Save PNG</button>
                  </div>
                )}
              </div>

              <aside className="control-panel">
                <div className="panel-heading"><div className="panel-icon"><SlidersHorizontal size={15} /></div><div><div className="panel-kicker">PROCESSING BAY</div><h2>Operation setup</h2></div></div>
                {!isHealthy ? (
                  <div className="service-notice" data-testid="status-operations-gated">
                    <div className="notice-icon"><Cpu size={20} /></div>
                    <div className="notice-title">{health.isLoading ? 'Checking the CV service' : 'Processor unavailable'}</div>
                    <p>{health.isLoading ? 'Waiting for the workshop processor to report healthy.' : 'OpenCV operations unlock when the backend reports a healthy Python processor.'}</p>
                    {health.isError && <button className="retry-button" onClick={() => health.refetch()} data-testid="button-retry-health">Retry connection</button>}
                  </div>
                ) : operationsQuery.isLoading ? (
                  <div className="skeleton-controls" aria-label="Loading operations"><span /><span /><span /></div>
                ) : operationsQuery.isError ? (
                  <div className="service-notice"><div className="notice-title">Operations failed to load</div><p>Check the service connection, then retry.</p><button className="retry-button" onClick={() => operationsQuery.refetch()} data-testid="button-retry-operations">Retry operations</button></div>
                ) : operations.length === 0 ? (
                  <div className="service-notice"><div className="notice-title">No operations reported</div><p>The healthy processor has not listed any Level 1 operations.</p></div>
                ) : (
                  <>
                    <label className="field-label" htmlFor="operation-select">OpenCV operation</label>
                    <div className="select-wrap">
                      <select id="operation-select" value={selectedOperation?.id ?? ''} onChange={(event) => {
                        const next = operations.find((item) => item.id === event.target.value);
                        if (next) { setSelectedId(next.id); setParameters(defaultsFor(next, imageSize)); setErrorMessage(''); }
                      }} data-testid="select-operation">
                        {operations.map((operation) => <option value={operation.id} key={operation.id}>{operation.label}</option>)}
                      </select>
                      <ChevronDown size={14} />
                    </div>
                    {selectedOperation && <div className="operation-description">{selectedOperation.description}</div>}
                    <div className="callout-code">
                      <div className="code-label"><Braces size={12} /> OPENCV CALL</div>
                      <code data-testid="text-opencv-call">{selectedOperation?.cvFunction ?? '—'}</code>
                    </div>
                    <div className="parameter-heading"><span>PARAMETERS</span><span>{parameterDefs.length.toString().padStart(2, '0')} ACTIVE</span></div>
                    {parameterDefs.length ? (
                      <div className="parameter-list">
                        {parameterDefs.map((definition) => {
                          const value = parameters[definition.key] ?? definition.initial;
                          return <label className="parameter-control" key={definition.key}>
                            <span className="parameter-label">{definition.label}</span>
                            <span className="parameter-readout">{value}{definition.unit ? ` ${definition.unit}` : ''}</span>
                            <input type="range" min={definition.min} max={definition.max} step={definition.step} value={value}
                      disabled={selectedOperation?.id === 'crop' && !imageSize}
                              onChange={(event) => setParameters((current) => {
                                const next = { ...current, [definition.key]: Number(event.target.value) };
                                return selectedOperation?.id === 'crop' ? clampCropParameters(next, imageSize) : next;
                              })}
                              data-testid={`input-param-${definition.key}`} />
                            <span className="range-ends"><span>{definition.min}</span><span>{definition.max}</span></span>
                          </label>;
                        })}
                      </div>
                    ) : (
                      <div className="no-parameters"><Check size={13} /> This operation uses its default behavior.</div>
                    )}
                    <div className="parameters-json">
                      <span>PARAMETER PAYLOAD</span>
                      <code data-testid="text-parameters">{selectedOperation ? JSON.stringify(valuesForOperation(selectedOperation, parameters, imageSize)) : '{}'}</code>
                    </div>
                    <button className="process-button" onClick={runOperation} disabled={!sourceFile || processImage.isPending || !isHealthy || (selectedOperation?.id === 'crop' && !imageSize)} data-testid="button-process">
                      {processImage.isPending ? <><LoaderCircle className="spin" size={16} /> Processing image…</> : <><Activity size={16} /> Run OpenCV operation <span className="button-arrow">↗</span></>}
                    </button>
                    {!sourceFile ? <div className="control-hint">Load an image to enable processing.</div> : selectedOperation?.id === 'crop' && !imageSize ? <div className="control-hint">Reading image bounds…</div> : null}
                  </>
                )}
                {isHealthy && !operationsQuery.isLoading && operations.length > 0 && (
                  <div className="processor-footer"><span className="status-dot" /><span>PROCESSING SERVICE READY</span><span className="processor-version">PYTHON / CV</span></div>
                )}
              </aside>
            </div>
          </section>

          <section className="observation-section" aria-label="Evidence observation">
            <div className="observation-header">
              <div className="section-title"><span className="section-number">02</span><span>OBSERVATION</span></div>
              <span className="observation-guidance">Write what the operation reveals—not what you expect to see.</span>
            </div>
            <div className="observation-content">
              <div className="observation-prompt"><span className="prompt-index">FIELD NOTE</span><p>What changed in the image? What detail became clearer, weaker, or disappeared?</p></div>
              <div className="observation-input-wrap">
                <textarea value={observation} onChange={(event) => setObservation(event.target.value)}
                  placeholder={hasResult ? 'Describe what you can actually see…' : 'Run an operation before recording an observation.'}
                  disabled={!hasResult} maxLength={1200} data-testid="input-observation" />
                <div className="observation-foot"><span>{observation.length} / 1200</span>
                  <button className="save-note-button" onClick={saveObservation} disabled={!hasResult || !observation.trim()} data-testid="button-save-observation"><BookOpen size={14} /> Save to notebook</button>
                </div>
              </div>
            </div>
          </section>

          <section id="history" className="records-section">
            <div className="records-header"><div className="section-title"><span className="section-number">03</span><span>PROCESSING HISTORY</span></div><span className="records-count">{history.length.toString().padStart(2, '0')} RUNS</span></div>
            {history.length === 0 ? (
              <div className="quiet-empty"><div className="empty-index">A</div><div><strong>No processing history yet.</strong><p>Completed OpenCV runs will collect here for this session.</p></div><History size={18} /></div>
            ) : (
              <div className="history-list">
                {history.map((item, index) => <button key={item.id} className={`history-row ${currentOperation?.id === item.operation.id && index === 0 ? 'history-current' : ''}`}
                  onClick={() => showHistory(item)} data-testid={`button-history-${item.id}`}>
                  <span className="history-seq">{String(history.length - index).padStart(2, '0')}</span>
                  <span className="history-operation">{item.operation.label}<small>{item.sourceName}</small></span>
                  <code>{item.operation.cvFunction}</code>
                  <span className="history-params">{JSON.stringify(item.parameters)}</span>
                  <span className="history-open">Load result ↗</span>
                </button>)}
              </div>
            )}
          </section>

          <section id="notebook" className="records-section notebook-section">
            <div className="records-header"><div className="section-title"><span className="section-number">04</span><span>EVIDENCE NOTEBOOK</span></div>
              <span className="records-count">{notebook.length.toString().padStart(2, '0')} ENTRIES</span>
              {notebook.length > 0 && <button className="text-button clear-notes" onClick={() => {
                if (window.confirm('Remove every saved notebook entry from this browser?')) setNotebook([]);
              }} data-testid="button-clear-notebook"><Trash2 size={13} /> Clear notebook</button>}
            </div>
            {notebook.length === 0 ? (
              <div className="quiet-empty notebook-empty"><div className="empty-index">B</div><div><strong>Your notebook is still blank.</strong><p>Save an observation after processing an image. Entries persist in this browser.</p></div><BookOpen size={18} /></div>
            ) : (
              <div className="notebook-list">
                {notebook.map((entry, index) => <article className="notebook-entry" key={entry.id} data-testid={`card-notebook-${entry.id}`}>
                  <div className="entry-index">{String(notebook.length - index).padStart(2, '0')}</div>
                  <div className="entry-main">
                    <div className="entry-heading"><span className="entry-operation">{entry.operationLabel}</span><span className="entry-source"><FileImage size={12} />{entry.sourceName}</span></div>
                    <p className="entry-observation">{entry.observation}</p>
                    <div className="entry-call"><code>{entry.cvFunction}</code><span>{JSON.stringify(entry.parameters)}</span></div>
                  </div>
                  <button className="remove-entry" onClick={() => setNotebook((items) => items.filter((item) => item.id !== entry.id))} aria-label={`Delete notebook entry for ${entry.operationLabel}`} data-testid={`button-delete-entry-${entry.id}`}><Trash2 size={14} /></button>
                </article>)}
              </div>
            )}
          </section>
          <footer className="page-footer"><span>ARGUS / FIELD SYSTEMS</span><span>LEVEL 1 — EYES ONLINE</span><span>RESULTS ARE SERVER-PROCESSED</span></footer>
        </main>
      </div>
      <input ref={fileInput} type="file" accept="image/*" className="visually-hidden" onChange={(event) => loadFile(event.target.files?.[0])} data-testid="input-image" />
      {sourceFile && <button className="mobile-change-image" onClick={() => fileInput.current?.click()} data-testid="button-change-image"><Upload size={14} /> Replace image</button>}
    </div>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
