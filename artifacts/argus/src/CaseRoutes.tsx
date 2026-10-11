import { useMemo, useState } from 'react';
import { Link, useLocation } from 'wouter';
import {
  useGetCaseOverview, getGetCaseOverviewQueryKey, useCompareImages, useSegmentColor,
  useListTrailFrames, getListTrailFramesQueryKey, useAnalyzeTrail,
  useSubmitLevelTwo, useSubmitLevelThree, useSubmitLevelFour, useSubmitAccusation,
} from '@workspace/api-client-react';
import type { EvidenceClaim, ImageCandidate } from '@workspace/api-client-react';
import {
  Aperture, ArrowRight, BookOpen, Check, ChevronRight, CircleAlert, Clock3, FileImage,
  Fingerprint, LockKeyhole, Map, ScanSearch, ShieldAlert, Upload, X,
} from 'lucide-react';

export type Finding = {
  id: string;
  sourceLevel: number;
  sourceAsset: string;
  method: string;
  observation: string;
  limitations: string;
};
type LocalCaseState = { completed: number[]; findings: Finding[]; attempts: Array<{ suspectId: string; explanation: string; at: string; result?: unknown }> };
const STATE_KEY = 'argus-case-state-v2';
const emptyState: LocalCaseState = { completed: [], findings: [], attempts: [] };
export function readCaseState(): LocalCaseState {
  try {
    const value = JSON.parse(localStorage.getItem(STATE_KEY) || '{}');
    return { ...emptyState, ...value, completed: Array.isArray(value.completed) ? value.completed : [], findings: Array.isArray(value.findings) ? value.findings : [], attempts: Array.isArray(value.attempts) ? value.attempts : [] };
  } catch { return emptyState; }
}
export function saveFinding(finding: Finding) {
  const state = readCaseState();
  localStorage.setItem(STATE_KEY, JSON.stringify({ ...state, findings: [finding, ...state.findings.filter((item) => item.id !== finding.id)] }));
}
export function markComplete(level: number) {
  const state = readCaseState();
  localStorage.setItem(STATE_KEY, JSON.stringify({ ...state, completed: [...new Set([...state.completed, level])] }));
  window.dispatchEvent(new Event('argus-state-change'));
}
function saveAttempt(attempt: LocalCaseState['attempts'][number]) {
  const state = readCaseState();
  localStorage.setItem(STATE_KEY, JSON.stringify({ ...state, attempts: [attempt, ...state.attempts] }));
  window.dispatchEvent(new Event('argus-state-change'));
}
function useCaseState() {
  const [, refresh] = useState(0);
  useMemo(() => {
    const listener = () => refresh((value) => value + 1);
    window.addEventListener('argus-state-change', listener);
    window.addEventListener('storage', listener);
    return () => { window.removeEventListener('argus-state-change', listener); window.removeEventListener('storage', listener); };
  }, []);
  return readCaseState();
}
const assetUrl = (path: string) => {
  if (/^(https?:|data:|blob:)/i.test(path)) return path;
  return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
};
async function assetBlob(path: string): Promise<Blob> {
  const response = await fetch(assetUrl(path));
  if (!response.ok) throw new Error(`Evidence asset could not be loaded (${response.status}).`);
  return response.blob();
}
async function assetData(path: string): Promise<string> {
  const blob = await assetBlob(path);
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Could not read the prepared frame.'));
    reader.readAsDataURL(blob);
  });
}
function Header() {
  const [location] = useLocation();
  const state = useCaseState();
  const links = [['/', 'Case file'], ['/levels/1', '01 / Eyes Online'], ['/levels/2', '02 / Missing Clue'], ['/levels/3', '03 / Red Bag'], ['/levels/4', '04 / Follow Trail'], ['/notebook', 'Notebook'], ['/accusation', 'Conclusion']];
  return <header className="case-nav">
    <Link href="/" className="case-brand" data-testid="link-home"><span className="brand-emblem"><Aperture size={18} /></span><span>ARGUS<small>CASEWORK / 04</small></span></Link>
    <nav aria-label="Case navigation">{links.map(([href, label]) => {
      const level = Number(href.match(/levels\/(\d)/)?.[1]);
      const locked = level > 1 && !state.completed.includes(level - 1);
      return <Link key={href} href={locked ? '#' : href} onClick={(event) => { if (locked) event.preventDefault(); }} className={`case-nav-link ${location === href ? 'active' : ''} ${locked ? 'locked' : ''}`} data-testid={`link-nav-${href.replaceAll('/', '') || 'dashboard'}`}>{locked && <LockKeyhole size={12} />}{label}</Link>;
    })}</nav>
    <span className="case-status"><i /> FIELD FILE ACTIVE</span>
  </header>;
}
export function CaseShell({ children }: { children: React.ReactNode }) {
  return <div className="case-app"><Header />{children}</div>;
}
function PageTop({ index, eyebrow, title, summary }: { index: string; eyebrow: string; title: string; summary: string }) {
  return <div className="case-page-top"><div className="mono-label"><span>{index}</span> / {eyebrow}</div><h1>{title}</h1><p>{summary}</p></div>;
}
function LoadingBlock({ label = 'Retrieving case material' }: { label?: string }) { return <div className="case-loading"><span className="skeleton-bar" /><span className="skeleton-bar short" /><small>{label.toUpperCase()}</small></div>; }
function ErrorBlock({ message, retry }: { message: string; retry?: () => void }) { return <div className="case-error" role="alert"><CircleAlert size={17} /><span>{message}</span>{retry && <button className="quiet-button" onClick={retry}>Retry</button>}</div>; }

export function CaseDashboard() {
  const query = useGetCaseOverview({ query: { queryKey: getGetCaseOverviewQueryKey() } });
  const state = useCaseState();
  if (query.isLoading) return <main className="case-main"><LoadingBlock /></main>;
  if (query.isError || !query.data) return <main className="case-main"><ErrorBlock message="Case briefing could not be retrieved." retry={() => query.refetch()} /></main>;
  const overview = query.data;
  const completeCount = state.completed.length;
  return <main className="case-main dashboard-page reveal">
    <section className="case-hero">
      <div className="hero-copy"><div className="mono-label"><span>ACTIVE INVESTIGATION</span><b>{overview.caseId}</b></div><p className="hero-kicker">Evidence before inference.</p><h1>{overview.title}</h1><p className="hero-brief">{overview.briefing}</p><div className="hero-meta"><span><Clock3 size={14} /> INCIDENT / {overview.incidentDate}</span><span><ShieldAlert size={14} /> STATUS / {overview.status}</span></div></div>
      <div className="case-stamp"><Aperture size={42} strokeWidth={1} /><span>ARGUS</span><small>IMAGE ANALYSIS UNIT</small><i>FILE 07—A</i></div>
    </section>
    <section className="progress-strip" aria-label="Investigation progress">
      <div><span className="mono-label">CHECKPOINTS</span><strong data-testid="text-progress">{String(completeCount).padStart(2, '0')}<small> / 04</small></strong></div>
      <div className="progress-track">{[1,2,3,4].map((n) => <i key={n} className={state.completed.includes(n) ? 'done' : ''} />)}</div>
      <Link href={completeCount === 4 ? '/accusation' : `/levels/${Math.max(1, completeCount + 1)}`} className="case-button" data-testid="link-continue">{completeCount === 4 ? 'Review conclusion' : 'Continue case'} <ArrowRight size={15} /></Link>
    </section>
    <div className="dashboard-columns">
      <section className="case-panel timeline-panel"><div className="panel-title"><span className="mono-label">01 / INCIDENT RECORD</span><h2>Known sequence</h2></div><div className="timeline-list">{overview.timeline.map((event) => <article className="timeline-item" key={event.id} data-testid={`timeline-${event.id}`}><time>{event.time}</time><div><strong>{event.title}</strong><p>{event.detail}</p><small>SOURCE / {event.source}</small></div></article>)}</div></section>
      <section className="case-panel"><div className="panel-title"><span className="mono-label">02 / PERSONS OF INTEREST</span><h2>Access & statements</h2></div><div className="persons-list">{overview.personsOfInterest.map((person, i) => <article className="person-row" key={person.id} data-testid={`person-${person.id}`}><span className="person-index">0{i + 1}</span><div><strong>{person.name}</strong><small>{person.role} · ACCESS {person.accessCode}</small><p>“{person.statement}”</p></div></article>)}</div></section>
    </div>
    <section className="level-overview"><div className="panel-title"><span className="mono-label">03 / IMAGE WORKSHOP</span><h2>Four questions. No shortcuts.</h2></div><div className="level-list">{overview.levels.map((level) => {
      const unlocked = level.id === 1 || state.completed.includes(level.id - 1);
      const done = state.completed.includes(level.id);
      return <Link href={unlocked ? `/levels/${level.id}` : '#'} onClick={(e) => { if (!unlocked) e.preventDefault(); }} key={level.id} className={`level-row ${!unlocked ? 'is-locked' : ''}`} data-testid={`card-level-${level.id}`}><span className="level-num">{String(level.id).padStart(2, '0')}</span><div><small>{done ? 'CHECKPOINT FILED' : unlocked ? 'AVAILABLE' : 'LOCKED · COMPLETE PREVIOUS CHECKPOINT'}</small><h3>{level.title}</h3><p>{level.objective}</p></div><ChevronRight size={17} /></Link>;
    })}</div></section>
    <footer className="case-footer"><span>ARGUS / EVIDENCE-LED COMPUTER VISION</span><span>OBSERVATION IS NOT IDENTIFICATION</span></footer>
  </main>;
}

function LevelLayout({ level, title, subtitle, children }: { level: number; title: string; subtitle: string; children: React.ReactNode }) {
  const state = useCaseState();
  const locked = level > 1 && !state.completed.includes(level - 1);
  return <main className="case-main level-page reveal"><PageTop index={`LEVEL ${String(level).padStart(2, '0')}`} eyebrow="FIELD EXERCISE" title={title} summary={subtitle} />{locked ? <section className="locked-panel"><LockKeyhole size={24} /><h2>Checkpoint required</h2><p>Complete the previous exercise before opening this evidence workspace.</p><Link href={`/levels/${level - 1}`} className="case-button">Return to prior level <ArrowRight size={15} /></Link></section> : children}</main>;
}
const imageFile = (file: File | null, path: string) => file ?? null;
const b64 = (value: string) => value.startsWith('data:') ? value : `data:image/png;base64,${value}`;

export function LevelTwo() {
  const overview = useGetCaseOverview({ query: { queryKey: getGetCaseOverviewQueryKey() } });
  const level = overview.data?.levels.find((item) => item.id === 2);
  const [before, setBefore] = useState<File | null>(null), [after, setAfter] = useState<File | null>(null);
  const [candidate, setCandidate] = useState(''), [observation, setObservation] = useState('');
  const [threshold, setThreshold] = useState(28);
  const compare = useCompareImages(), submit = useSubmitLevelTwo();
  const [error, setError] = useState('');
  async function run() {
    if (!level || (!before && !level.assetUrls[0]) || (!after && !level.assetUrls[1])) return;
    setError('');
    try {
      const [a, b] = await Promise.all([before ?? assetBlob(level.assetUrls[0]), after ?? assetBlob(level.assetUrls[1])]);
      const result = await compare.mutateAsync({ data: { imageBefore: a, imageAfter: b, threshold } });
      setCandidate(result.candidates[0]?.id ?? '');
    } catch (e) { setError(e instanceof Error ? e.message : 'Comparison could not be completed.'); }
  }
  async function send() {
    if (!candidate || observation.trim().length < 12) return;
    try {
      const response = await submit.mutateAsync({ data: { candidateId: candidate, observation: observation.trim() } });
      if (response.completed) { markComplete(2); saveFinding({ id: crypto.randomUUID(), sourceLevel: 2, sourceAsset: `${before?.name ?? level?.assetUrls[0]} ↔ ${after?.name ?? level?.assetUrls[1]}`, method: `OpenCV image comparison · threshold ${threshold}`, observation: observation.trim(), limitations: 'A changed-region candidate identifies pixel difference, not a person, object owner, or cause.' }); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Checkpoint submission failed.'); }
  }
  const result = compare.data;
  return <LevelLayout level={2} title="The Missing Clue" subtitle={level?.objective ?? 'Compare two source images and inspect where pixels differ.'}>
    <section className="exercise-panel"><div className="exercise-head"><span className="mono-label">DIFFERENCE REVIEW / TWO SOURCES</span><span className="evidence-tag">SERVER-SIDE OPENCV</span></div>
      <div className="source-pair">{[['BEFORE',before,setBefore,level?.assetUrls[0]],['AFTER',after,setAfter,level?.assetUrls[1]]].map(([label,file,setter,path]) => <label className="source-drop" key={String(label)}><span>{String(label)} / SOURCE IMAGE</span>{(file as File | null) ? <img src={URL.createObjectURL(file as File)} alt={`${label} evidence`} /> : path ? <img src={assetUrl(String(path))} alt={`${label} prepared evidence`} /> : <div><Upload size={22} /><small>Choose image</small></div>}<input type="file" accept="image/*" onChange={(e) => (setter as (f: File | null) => void)(e.target.files?.[0] ?? null)} data-testid={`input-source-${String(label).toLowerCase()}`} /></label>)}</div>
      <div className="tool-row"><label className="range-field">DIFFERENCE THRESHOLD <strong>{threshold}</strong><input type="range" min="0" max="120" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} data-testid="input-threshold" /></label><button className="case-button" onClick={run} disabled={compare.isPending} data-testid="button-run-comparison"><ScanSearch size={15} />{compare.isPending ? 'Comparing…' : 'Compare with OpenCV'}</button></div>
      {compare.isPending && <LoadingBlock label="Comparing source pixels with OpenCV" />}{error && <ErrorBlock message={error} />}
      {result && <><div className="comparison-results"><figure><figcaption>PIXEL DIFFERENCE</figcaption><img src={b64(result.differencePng)} alt="OpenCV pixel difference map" /></figure><figure><figcaption>REGION CANDIDATES / COMPUTER VISION ONLY</figcaption><img src={b64(result.overlayPng)} alt="OpenCV changed region candidates" /></figure></div><div className="metric-line"><span><b>{result.changedPixels.toLocaleString()}</b> changed pixels</span><span><b>{result.changedPercent.toFixed(2)}%</b> of image</span><span><b>{result.candidates.length}</b> candidate regions</span></div><div className="candidate-list">{result.candidates.map((item) => <button className={`candidate ${candidate === item.id ? 'selected' : ''}`} key={item.id} onClick={() => setCandidate(item.id)} data-testid={`button-candidate-${item.id}`}><span>{item.id}</span><small>{item.width} × {item.height} · area {item.area}</small>{candidate === item.id && <Check size={14} />}</button>)}</div><p className="limitation-note">Candidates describe changed image regions only. They do not establish what changed, who moved it, or why.</p>
      <div className="checkpoint-form"><label>YOUR OBSERVATION<textarea value={observation} onChange={(e) => setObservation(e.target.value)} minLength={12} maxLength={1200} placeholder="Describe a visible difference, and separate it from your interpretation." data-testid="input-level-observation" /></label><button className="case-button critical" onClick={send} disabled={!candidate || observation.trim().length < 12 || submit.isPending} data-testid="button-submit-level">{submit.isPending ? 'Submitting…' : 'Submit checkpoint'} <ArrowRight size={15} /></button>{submit.data?.message && <p className={submit.data.completed ? 'feedback-success' : 'feedback-error'} data-testid="status-level-feedback">{submit.data.feedback || submit.data.message}</p>}</div></>}
    </section>
  </LevelLayout>;
}

export function LevelThree() {
  const overview = useGetCaseOverview({ query: { queryKey: getGetCaseOverviewQueryKey() } });
  const level = overview.data?.levels.find((item) => item.id === 3);
  const [file, setFile] = useState<File | null>(null), [candidate, setCandidate] = useState(''), [observation, setObservation] = useState('');
  const [params, setParams] = useState({ lower_h: 0, lower_s: 90, lower_v: 70, upper_h: 12, upper_s: 255, upper_v: 255, opening: 1, closing: 2 });
  const segmentation = useSegmentColor(), submit = useSubmitLevelThree();
  const [error, setError] = useState('');
  async function run() {
    if (!file && !level?.assetUrls[0]) return;
    setError('');
    try { const image = file ?? await assetBlob(level!.assetUrls[0]); const result = await segmentation.mutateAsync({ data: { image, parameters: JSON.stringify(params) } }); setCandidate(result.candidates[0]?.id ?? ''); }
    catch (e) { setError(e instanceof Error ? e.message : 'Segmentation failed.'); }
  }
  async function send() {
    if (!candidate || observation.trim().length < 12) return;
    try {
      const result = await submit.mutateAsync({ data: { candidateId: candidate, observation: observation.trim(), parameters: params } });
      if (result.completed) { markComplete(3); saveFinding({ id: crypto.randomUUID(), sourceLevel: 3, sourceAsset: file?.name ?? level?.assetUrls[0] ?? 'Prepared image', method: `OpenCV HSV color segmentation · ${JSON.stringify(params)}`, observation: observation.trim(), limitations: 'A color mask marks pixels within the selected HSV range; it does not prove an object’s identity, ownership, or continuity.' }); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Checkpoint submission failed.'); }
  }
  const result = segmentation.data;
  return <LevelLayout level={3} title="The Red Bag" subtitle={level?.objective ?? 'Test a color hypothesis against the source image.'}>
    <section className="exercise-panel"><div className="exercise-head"><span className="mono-label">COLOR SEGMENTATION / HSV RANGE</span><span className="evidence-tag">SERVER-SIDE OPENCV</span></div>
      <div className="triptych"><figure><figcaption>01 / ORIGINAL</figcaption>{file ? <img src={URL.createObjectURL(file)} alt="Original uploaded image" /> : level?.assetUrls[0] ? <img src={assetUrl(level.assetUrls[0])} alt="Original prepared evidence" /> : <div className="empty-image">Select source image</div>}<label className="mini-upload"><Upload size={13} /> Change source<input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} data-testid="input-level3-image" /></label></figure>{result ? <><figure><figcaption>02 / MASK · {result.selectedPixels.toLocaleString()} PIXELS</figcaption><img src={b64(result.maskPng)} alt="OpenCV HSV mask" /></figure><figure><figcaption>03 / SEGMENTED VIEW</figcaption><img src={b64(result.segmentedPng)} alt="OpenCV segmented source image" /></figure></> : <div className="triptych-empty"><ScanSearch size={22} /><span>Run segmentation to inspect mask and result</span></div>}</div>
      <div className="seg-controls"><div className="seg-title"><span className="mono-label">HSV PARAMETERS</span><p>Adjust a reproducible range; avoid treating color as identity.</p></div><div className="seg-range-grid">{(['lower_h','lower_s','lower_v','upper_h','upper_s','upper_v'] as const).map((key) => <label key={key}>{key.replace('_',' ').toUpperCase()} <b>{params[key]}</b><input type="range" min="0" max={key.endsWith('h') ? 179 : 255} value={params[key]} onChange={(e) => setParams({ ...params, [key]: Number(e.target.value) })} data-testid={`input-${key}`} /></label>)}</div><button className="case-button" onClick={run} disabled={segmentation.isPending} data-testid="button-run-segmentation">{segmentation.isPending ? 'Segmenting…' : 'Run color segmentation'} <ArrowRight size={15} /></button></div>
      {error && <ErrorBlock message={error} />}{result && <><div className="candidate-list">{result.candidates.map((item) => <button key={item.id} className={`candidate ${candidate === item.id ? 'selected' : ''}`} onClick={() => setCandidate(item.id)} data-testid={`button-segment-candidate-${item.id}`}><span>{item.id}</span><small>{item.width} × {item.height} · {item.area.toLocaleString()} px</small>{candidate === item.id && <Check size={14} />}</button>)}</div><div className="checkpoint-form"><label>WHAT DOES THE MASK SUPPORT?<textarea value={observation} onChange={(e) => setObservation(e.target.value)} minLength={12} placeholder="Record the visible result and its limits." data-testid="input-level-observation" /></label><button className="case-button critical" onClick={send} disabled={!candidate || observation.trim().length < 12 || submit.isPending} data-testid="button-submit-level">{submit.isPending ? 'Submitting…' : 'Submit checkpoint'} <ArrowRight size={15} /></button>{submit.data?.message && <p className={submit.data.completed ? 'feedback-success' : 'feedback-error'} data-testid="status-level-feedback">{submit.data.feedback || submit.data.message}</p>}</div></>}
    </section>
  </LevelLayout>;
}

export function LevelFour() {
  const framesQuery = useListTrailFrames({ query: { queryKey: getListTrailFramesQueryKey() } });
  const submit = useSubmitLevelFour(), analyze = useAnalyzeTrail();
  const [threshold, setThreshold] = useState(22), [entry, setEntry] = useState(.22), [exit, setExit] = useState(.78);
  const [selected, setSelected] = useState(''), [observation, setObservation] = useState(''), [error, setError] = useState('');
  const frames = framesQuery.data?.frames ?? [];
  async function run() {
    if (frames.length < 2) return;
    setError('');
    try {
      const imageData = await Promise.all(frames.map((frame) => assetData(frame.url)));
      await analyze.mutateAsync({ data: { frames: imageData, timestamps: frames.map((frame) => frame.timestamp), motionThreshold: threshold, entryLineFraction: entry, exitLineFraction: exit } });
    } catch (e) { setError(e instanceof Error ? e.message : 'Sequence analysis failed.'); }
  }
  async function send() {
    if (!selected || observation.trim().length < 12) return;
    try { const result = await submit.mutateAsync({ data: { eventId: selected, observation: observation.trim() } }); if (result.completed) { markComplete(4); saveFinding({ id: crypto.randomUUID(), sourceLevel: 4, sourceAsset: framesQuery.data?.label ?? 'Prepared frame sequence', method: 'OpenCV sequence motion analysis', observation: observation.trim(), limitations: 'Motion events and centroids describe detected movement, not a person’s identity or intent.' }); } }
    catch (e) { setError(e instanceof Error ? e.message : 'Checkpoint submission failed.'); }
  }
  const result = analyze.data;
  return <LevelLayout level={4} title="Follow the Trail" subtitle="Read motion across a prepared sequence. A centroid is a location estimate, not a person.">
    <section className="exercise-panel"><div className="exercise-head"><span className="mono-label">SEQUENCE VIEWER / {framesQuery.data?.label ?? 'PREPARED FRAMES'}</span><span className="evidence-tag">PROVENANCE / {framesQuery.data?.provenance ?? 'PENDING'}</span></div>
      {framesQuery.isLoading ? <LoadingBlock label="Loading prepared sequence" /> : framesQuery.isError ? <ErrorBlock message="Frame sequence could not be loaded." retry={() => framesQuery.refetch()} /> : <>
        <div className="sequence-grid">{frames.map((frame, i) => { const analyzed = result?.frames.find((item) => item.frameIndex === frame.frameIndex); return <figure key={frame.frameIndex} data-testid={`frame-${frame.frameIndex}`}><figcaption>FRAME {String(frame.frameIndex).padStart(2,'0')} <time>{frame.timestamp}</time></figcaption><img src={analyzed ? b64(analyzed.overlayPng) : assetUrl(frame.url)} alt={`Frame ${frame.frameIndex} ${analyzed?.hasMotion ? 'with motion overlay' : ''}`} /><small>{analyzed ? analyzed.hasMotion ? 'MOTION DETECTED' : 'NO MOTION DETECTED' : 'SOURCE FRAME'}</small></figure>; })}</div>
        <div className="trail-controls"><div className="trail-tuning"><label>MOTION THRESHOLD <b>{threshold}</b><input type="range" min="5" max="100" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} /></label><label>ENTRY LINE <b>{entry.toFixed(2)}</b><input type="range" min=".05" max=".45" step=".01" value={entry} onChange={(e) => setEntry(Number(e.target.value))} /></label><label>EXIT LINE <b>{exit.toFixed(2)}</b><input type="range" min=".55" max=".95" step=".01" value={exit} onChange={(e) => setExit(Number(e.target.value))} /></label></div><button className="case-button" onClick={run} disabled={analyze.isPending || frames.length < 2} data-testid="button-analyze-trail">{analyze.isPending ? 'Analyzing sequence…' : 'Analyze sequence'} <ArrowRight size={15} /></button></div>
        {error && <ErrorBlock message={error} />}
        {result && <div className="trail-results"><div className="trail-timeline"><div className="mono-label">DETECTED EVENT TIMELINE</div><div className="event-track">{result.frames.map((frame) => <span key={frame.frameIndex} className={frame.hasMotion ? 'motion' : ''} title={`${frame.timestamp}: ${frame.hasMotion ? 'motion' : 'no motion'}`} />)}</div><div className="event-labels"><span>{frames[0]?.timestamp}</span><span>{frames.at(-1)?.timestamp}</span></div><p>{result.explanation}</p></div><div className="event-log"><div className="mono-label">EVENT LOG · COMPUTER VISION CANDIDATES</div>{result.events.length ? result.events.map((event) => <button key={event.id} className={`event-row ${selected === event.id ? 'selected' : ''}`} onClick={() => setSelected(event.id)} data-testid={`button-event-${event.id}`}><span className="event-marker" /><strong>{event.kind.replaceAll('-', ' ')}</strong><time>{event.timestamp}</time><small>FRAME {event.frameIndex} · CENTROID {event.centroidX.toFixed(2)}, {event.centroidY.toFixed(2)}</small></button>) : <div className="quiet-empty"><span>No events returned for these parameters.</span></div>}</div></div>}
        {result && <div className="checkpoint-form"><label>INTERPRET THE MOVEMENT<textarea value={observation} onChange={(e) => setObservation(e.target.value)} minLength={12} placeholder="What does the event sequence support—and what remains unknown?" data-testid="input-level-observation" /></label><button className="case-button critical" onClick={send} disabled={!selected || observation.trim().length < 12 || submit.isPending} data-testid="button-submit-level">{submit.isPending ? 'Submitting…' : 'Submit checkpoint'} <ArrowRight size={15} /></button>{submit.data?.message && <p className={submit.data.completed ? 'feedback-success' : 'feedback-error'} data-testid="status-level-feedback">{submit.data.feedback || submit.data.message}</p>}</div>}
      </>}
    </section>
  </LevelLayout>;
}

export function NotebookPage() {
  const state = useCaseState();
  const [legacy, setLegacy] = useState(() => { try { return JSON.parse(localStorage.getItem('argus-evidence-notebook-v1') || '[]') as Array<{id:string;operationLabel:string;sourceName:string;observation:string;cvFunction:string;parameters:Record<string,number>}>; } catch { return []; } });
  function remove(id: string) { const next = state.findings.filter((item) => item.id !== id); localStorage.setItem(STATE_KEY, JSON.stringify({ ...state, findings: next })); window.dispatchEvent(new Event('argus-state-change')); setLegacy(legacy.filter((item) => item.id !== id)); }
  return <main className="case-main reveal"><PageTop index="SHARED RECORD" eyebrow="EVIDENCE NOTEBOOK" title="Field notes." summary="Observations saved from performed operations. Methods, sources, and limitations stay attached." /><div className="notebook-toolbar"><span className="mono-label">{state.findings.length + legacy.length} ENTRIES · LOCAL TO THIS DEVICE</span><BookOpen size={17} /></div>{state.findings.length + legacy.length === 0 ? <div className="notebook-blank"><BookOpen size={27} /><h2>No findings filed yet.</h2><p>Complete a real server-side analysis and save a reasoned observation to begin the shared record.</p><Link href="/levels/1" className="case-button">Open Eyes Online <ArrowRight size={15} /></Link></div> : <div className="finding-list">{state.findings.map((f) => <article key={f.id} className="finding-card" data-testid={`finding-${f.id}`}><span className="finding-level">LEVEL {String(f.sourceLevel).padStart(2,'0')}</span><div className="finding-body"><h2>{f.method}</h2><small>SOURCE / {f.sourceAsset}</small><p>{f.observation}</p><div className="limitation-note"><ShieldAlert size={14} /> LIMITATION / {f.limitations}</div></div><button className="icon-button" onClick={() => remove(f.id)} aria-label="Delete finding" data-testid={`button-delete-finding-${f.id}`}><X size={15} /></button></article>)}{legacy.map((f) => <article className="finding-card" key={f.id}><span className="finding-level">LEVEL 01</span><div className="finding-body"><h2>{f.operationLabel}</h2><small>SOURCE / {f.sourceName}</small><p>{f.observation}</p><div className="limitation-note"><ShieldAlert size={14} /> METHOD / {f.cvFunction} · PARAMETERS / {JSON.stringify(f.parameters)}</div></div><button className="icon-button" onClick={() => remove(f.id)} aria-label="Delete finding" data-testid={`button-delete-finding-${f.id}`}><X size={15} /></button></article>)}</div>}</main>;
}

export function AccusationPage() {
  const overview = useGetCaseOverview({ query: { queryKey: getGetCaseOverviewQueryKey() } });
  const state = useCaseState();
  const submit = useSubmitAccusation();
  const [suspectId, setSuspectId] = useState(''), [explanation, setExplanation] = useState('');
  const [selectedEvidence, setSelectedEvidence] = useState<string[]>([]);
  const [error, setError] = useState('');
  const suspects = overview.data?.personsOfInterest ?? [];
  const complete = [1,2,3,4].every((n) => state.completed.includes(n));
  const evidence: EvidenceClaim[] = state.findings.map((f) => ({ id: f.id, sourceLevel: f.sourceLevel, sourceAsset: f.sourceAsset, method: f.method, observation: f.observation, limitations: f.limitations }));
  async function send() {
    if (!complete || !suspectId || explanation.trim().length < 20) return;
    setError('');
    try {
      const result = await submit.mutateAsync({ data: { suspectId, explanation: explanation.trim(), evidence: evidence.filter((item) => selectedEvidence.includes(item.id)) } });
      saveAttempt({ suspectId, explanation: explanation.trim(), at: new Date().toISOString(), result });
    } catch (e) { setError(e instanceof Error ? e.message : 'Conclusion could not be submitted.'); saveAttempt({ suspectId, explanation: explanation.trim(), at: new Date().toISOString() }); }
  }
  return <main className="case-main reveal"><PageTop index="FINAL REVIEW" eyebrow="CASE CONCLUSION" title="Make the case." summary="State who you believe is responsible, connect the evidence, and name what it cannot establish." />{!complete && <div className="locked-panel compact"><LockKeyhole size={20} /><div><strong>Four checkpoints required</strong><p>Complete each workshop level before submitting a conclusion.</p></div><Link href={`/levels/${Math.max(1,state.completed.length + 1)}`} className="quiet-link">Continue investigation <ArrowRight size={14} /></Link></div>}
    {overview.isLoading ? <LoadingBlock /> : overview.isError ? <ErrorBlock message="Person-of-interest list unavailable." retry={() => overview.refetch()} /> : <div className="accusation-layout"><section className="exercise-panel"><div className="exercise-head"><span className="mono-label">01 / PERSON OF INTEREST</span><span className="evidence-tag">{complete ? 'CONCLUSION OPEN' : 'CHECKPOINTS INCOMPLETE'}</span></div><div className="suspect-grid">{suspects.map((person, i) => <button type="button" key={person.id} className={`suspect-choice ${suspectId === person.id ? 'selected' : ''}`} onClick={() => setSuspectId(person.id)} data-testid={`button-suspect-${person.id}`}><span>0{i+1} / {person.accessCode}</span><strong>{person.name}</strong><small>{person.role}</small>{suspectId === person.id && <Check size={17} />}</button>)}</div><label className="form-label">02 / EXPLANATION<textarea value={explanation} maxLength={1200} minLength={20} onChange={(e) => setExplanation(e.target.value)} placeholder="Build a reasoned account. Explain which evidence supports it, how the sources relate, and where uncertainty remains." data-testid="input-accusation-explanation" /><small>{explanation.length} / 1200 · minimum 20 characters</small></label><div className="evidence-picker"><div className="mono-label">03 / EVIDENCE CLAIMS ({evidence.length})</div>{evidence.length ? evidence.map((item) => <label key={item.id} className="evidence-check"><input type="checkbox" checked={selectedEvidence.includes(item.id)} onChange={(e) => setSelectedEvidence(e.target.checked ? [...selectedEvidence,item.id] : selectedEvidence.filter((id) => id !== item.id))} data-testid={`input-evidence-${item.id}`} /><span><b>LEVEL {String(item.sourceLevel).padStart(2,'0')} · {item.method}</b><small>{item.observation}</small></span></label>) : <p className="muted-copy">No workshop findings available. Notebook observations from your own Level 1 processing remain available in the notebook; only structured checkpoint findings can be submitted here.</p>}</div><button className="case-button critical" onClick={send} disabled={!complete || !suspectId || explanation.trim().length < 20 || submit.isPending} data-testid="button-submit-accusation">{submit.isPending ? 'Reviewing conclusion…' : 'Submit reasoned conclusion'} <ArrowRight size={15} /></button>{error && <ErrorBlock message={error} />}</section>
      <aside className="case-panel conclusion-aside"><span className="mono-label">EVIDENCE STANDARD</span><h2>Inference, not certainty.</h2><p>A computer-vision candidate is not an identity. A matching color is not ownership. A movement event does not identify the person who caused it.</p><div className="aside-rule" /><div className="mono-label">FILED CHECKPOINTS</div>{[1,2,3,4].map((n) => <div className="checkpoint-status" key={n}><span className={state.completed.includes(n) ? 'status-filed' : ''}>{state.completed.includes(n) ? <Check size={13} /> : <span>{String(n).padStart(2,'0')}</span>}</span>LEVEL {String(n).padStart(2,'0')}<b>{state.completed.includes(n) ? 'FILED' : 'PENDING'}</b></div>)}</aside></div>}
    {submit.data && <section className="result-review" data-testid="panel-accusation-result"><div className="result-score"><span className="mono-label">CASE REVIEW / SCORE</span><strong>{submit.data.score}</strong><small>{submit.data.correct ? 'CONCLUSION SUPPORTED' : 'CONCLUSION NOT SUPPORTED'}</small></div><div className="review-content"><h2>{submit.data.summary}</h2><p>Reference conclusion: <strong>{submit.data.correctSuspectName}</strong></p><div className="review-list">{submit.data.evidenceReview.map((item) => <article key={item.evidenceId}><span>{item.relation.toUpperCase()}</span><p>{item.feedback}</p></article>)}</div>{submit.data.limitations.length > 0 && <div className="limitation-note"><ShieldAlert size={14} />{submit.data.limitations.join(' ')}</div>}</div></section>}
    {state.attempts.length > 0 && <section className="attempt-history"><span className="mono-label">PRIOR SUBMISSIONS</span>{state.attempts.map((attempt, i) => <div key={`${attempt.at}-${i}`}><time>{new Date(attempt.at).toLocaleString()}</time><span>{suspects.find((s) => s.id === attempt.suspectId)?.name ?? attempt.suspectId}</span><small>{attempt.result ? 'REVIEW RECEIVED' : 'SUBMISSION RECORD SAVED'}</small></div>)}</section>}
  </main>;
}
