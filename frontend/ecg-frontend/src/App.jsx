// src/App.jsx
import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const API_URL_2 = "http://127.0.0.1:8000/predict";
const CLASS_NAMES = ["Normal (N)", "LBBB (L)", "RBBB (R)", "APB (A)", "VPB (V)"];

function makeRandomSignal(n = 2000) {
  return Array.from({ length: n }, () =>
    Number((Math.random() * 0.2 - 0.1).toFixed(6))
  );
}

function downsample(arr, step) {
  const out = [];
  for (let i = 0; i < arr.length; i += step) out.push(arr[i]);
  return out;
}

function ECGPlot({
  signal,
  attribution = null, // length 2000, values 0..1
  start = 0,
  windowSize = 700,
  thresh = 0.55, // how much attribution to turn red
}) {
  const width = 980;
  const height = 280;

  const end = Math.min(start + windowSize, signal.length);
  const slice = signal.slice(start, end);

  const MAX_POINTS = 900;
  const step = Math.max(1, Math.floor(slice.length / MAX_POINTS));
  const dsSignal = downsample(slice, step);

  const dsAttr =
    attribution && Array.isArray(attribution) && attribution.length === signal.length
      ? downsample(attribution.slice(start, end), step)
      : null;

  // Signal min/max for scaling
  let min = Infinity,
    max = -Infinity;
  for (const v of dsSignal) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    min = -1;
    max = 1;
  }

  const pad = 14;
  const w = width - pad * 2;
  const h = height - pad * 2;

  // Coordinates for segment-by-segment drawing
  const coords = dsSignal.map((v, i) => {
    const x = pad + (i / (dsSignal.length - 1)) * w;
    const yNorm = (v - min) / (max - min);
    const y = pad + (1 - yNorm) * h;
    return { x, y };
  });

  // Grid lines
  const gridLines = [];
  const gridCount = 6;
  for (let i = 0; i <= gridCount; i++) {
    gridLines.push(pad + (i / gridCount) * h);
  }

  return (
    <div className="plotWrap">
      <div className="plotTitle">
        ECG Waveform (samples {start}–{end - 1})
        <span className="plotMeta">
          min={min.toFixed(3)} max={max.toFixed(3)}
          {dsAttr ? <span className="xaiTag">XAI ON</span> : null}
        </span>
      </div>

      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        className="plotSvg"
        preserveAspectRatio="none"
      >
        <rect x="0" y="0" width={width} height={height} rx="14" className="plotBg" />

        {gridLines.map((y, idx) => (
          <line
            key={idx}
            x1={pad}
            x2={width - pad}
            y1={y}
            y2={y}
            className="plotGrid"
          />
        ))}

        {/* ECG line as colored segments (red where attribution is high) */}
        {coords.slice(0, -1).map((p, i) => {
          const p2 = coords[i + 1];
          const a = dsAttr ? dsAttr[i] : 0;
          const hot = dsAttr ? a >= thresh : false;

          return (
            <line
              key={i}
              x1={p.x}
              y1={p.y}
              x2={p2.x}
              y2={p2.y}
              className={hot ? "plotLineHot" : "plotLine"}
              strokeWidth="2"
              strokeLinecap="round"
            />
          );
        })}
      </svg>

      {dsAttr && (
        <div className="xaiLegend">
          <span className="xaiSwatchHot" />
          <span>Highlighted segments indicate important regions (threshold ≥ {thresh.toFixed(2)})</span>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [signal, setSignal] = useState(() => makeRandomSignal(2000));
  const [loading, setLoading] = useState(false);
  const [loadingXai, setLoadingXai] = useState(false);
  const [err, setErr] = useState("");
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState({ ok: false, model_loaded: false });

  const [viewStart, setViewStart] = useState(0);
  const [viewSize, setViewSize] = useState(700);

  const [xai, setXai] = useState(null);
  const [xaiMethod, setXaiMethod] = useState("integrated_gradients");
  const [xaiThresh, setXaiThresh] = useState(0.55);

  const signalText = useMemo(() => JSON.stringify(signal), [signal]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        setHealth({ ok: true, model_loaded: !!data?.model_loaded });
      } catch {
        setHealth({ ok: false, model_loaded: false });
      }
    })();
  }, []);

  const regenerate = () => {
    setErr("");
    setResult(null);
    setXai(null);
    setSignal(makeRandomSignal(2000));
    setViewStart(0);
  };

  const onSignalChange = (text) => {
    setErr("");
    setResult(null);
    setXai(null);
    try {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) throw new Error("Signal must be a JSON array.");
      if (parsed.length !== 2000)
        throw new Error(`Signal length must be 2000, got ${parsed.length}.`);

      const nums = parsed.map((v, i) => {
        const num = Number(v);
        if (!Number.isFinite(num)) throw new Error(`Invalid number at index ${i}`);
        return num;
      });

      setSignal(nums);
      setViewStart(0);
    } catch (e) {
      setErr(e?.message || "Invalid JSON array.");
    }
  };

  const predict = async () => {
    setLoading(true);
    setErr("");
    setResult(null);
    setXai(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ signal, return_probabilities: true }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Request failed");
      setResult(data);
    } catch (e) {
      setErr(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const explain = async () => {
    setLoadingXai(true);
    setErr("");
    setXai(null);

    try {
      const res = await fetch(`${API_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signal,
          method: xaiMethod,
          steps: 50,
          baseline: "zeros",
          return_probabilities: true,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Explain request failed");

      if (!Array.isArray(data?.attribution) || data.attribution.length !== 2000) {
        throw new Error("Invalid XAI attribution returned by API.");
      }

      setXai(data);
      setResult((prev) => prev || data);
    } catch (e) {
      setErr(e?.message || "XAI failed");
    } finally {
      setLoadingXai(false);
    }
  };

  const topLabel = result?.predicted_label ?? "—";
  const topConf = typeof result?.confidence === "number" ? result.confidence : null;

  return (
    <div className="page">
      <div className="bgGlow" />
      <header className="header">
        <div className="titleBlock">
          <div className="badge">Federated ECG Demo</div>
          <h1>Arrhythmia Classification</h1>
          <p>
            Paste a 2000-sample signal or generate a random one. Predict calls FastAPI. Explain (XAI) highlights important
            regions in red on the ECG curve.
          </p>
        </div>

        <div className="statusRow">
          <div className={`chip ${health.ok ? "chipOk" : "chipBad"}`}>
            API: {health.ok ? "Online" : "Offline"}
          </div>
          <div className={`chip ${health.model_loaded ? "chipOk" : "chipWarn"}`}>
            Model: {health.model_loaded ? "Loaded" : "Not loaded"}
          </div>
          {/* <div className="chip chipNeutral">Endpoint: {API_URL_2}</div> */}
        </div>
      </header>

      <main className="grid cardFull">
        {/* Left: Input */}
        <section className="card">
          <div className="cardHeader">
            <div>
              <h2>Signal Input</h2>
              {/* <div className="subtext">JSON array with exactly 2000 numbers</div> */}
            </div>

            <div className="actions">
              <button className="btn btnGhost" onClick={regenerate} disabled={loading || loadingXai}>
                Generate
              </button>
              {/* <button className="btn btnPrimary" onClick={predict} disabled={loading || loadingXai || !health.ok}>
                {loading ? "Predicting…" : "Predict"}
              </button> */}
              <button className="btn btnPrimary" onClick={explain} disabled={loadingXai || loading || !health.ok}>
                {loadingXai ? "Predicting…" : "Predict"}
              </button>
              {/* <button className="btn btnGhost" onClick={() => setXai(null)} disabled={!xai}>
              Clear XAI
            </button> */}
            {xai && (
  <button
    className="btn btnGhost"
    onClick={() => setXai(null)}
  >
    Clear XAI
  </button>
)}

            </div>
          </div>

          {/* XAI controls */}
          <div className="xaiControls">
            <label className="xaiLabel">XAI method</label>
            <select
              className="xaiSelect"
              value={xaiMethod}
              onChange={(e) => setXaiMethod(e.target.value)}
              disabled={loading || loadingXai}
            >
              <option value="integrated_gradients">Integrated Gradients (recommended)</option>
              <option value="saliency">Saliency (fast)</option>
            </select>

            <label className="xaiLabel">Threshold</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={xaiThresh}
              onChange={(e) => setXaiThresh(Number(e.target.value))}
              disabled={loading || loadingXai}
            />
            <span className="plotVal">{xaiThresh.toFixed(2)}</span>

            
          </div>

          {/* ECG Plot */}
          <ECGPlot
            signal={signal}
            attribution={xai?.attribution ?? null}
            start={viewStart}
            windowSize={viewSize}
            thresh={xaiThresh}
          />

          <div className="plotControls">
            <div className="plotControlRow">
              <label>Start</label>
              <input
                type="range"
                min="0"
                max={Math.max(0, 2000 - viewSize)}
                value={viewStart}
                onChange={(e) => setViewStart(Number(e.target.value))}
              />
              <span className="plotVal">{viewStart}</span>
            </div>

            <div className="plotControlRow">
              <label>Window</label>
              <input
                type="range"
                min="200"
                max="2000"
                step="50"
                value={viewSize}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setViewSize(v);
                  setViewStart((s) => Math.min(s, Math.max(0, 2000 - v)));
                }}
              />
              <span className="plotVal">{viewSize}</span>
            </div>
          </div>

          <div className="editorWrap">
            <textarea
              className="editor"
              value={signalText}
              onChange={(e) => onSignalChange(e.target.value)}
              spellCheck={false}
            />
          </div>

          {err && (
            <div className="alert">
              <div className="alertTitle">Error</div>
              <div className="alertBody">{err}</div>
            </div>
          )}

          <div className="hint">
            Tip: For meaningful XAI, use a real ECG segment (2000 samples). Random signals are only for API testing.
          </div>
        </section>

        {/* Right: Output */}
        <section className="card">
          {/* <div className="cardHeader">
            <div>
              <h2>Prediction</h2>
              <div className="subtext">Top class + probability breakdown</div>
            </div>
          </div> */}

          <div className="hero">
            <div className="heroLabel">{topLabel}</div>
            <div className="heroMeta">
              <span className="heroPill">
                Confidence: <b>{topConf == null ? "—" : `${(topConf * 100).toFixed(2)}%`}</b>
              </span>
              <span className="heroPill">
                Output: <b>{result ? "OK" : "—"}</b>
              </span>
              <span className="heroPill">
                XAI: <b>{xai ? xai.method : "—"}</b>
              </span>
            </div>
          </div>

          {!result && <div className="empty">No prediction yet. Click <b>Predict</b>.</div>}

          {result?.probabilities && Array.isArray(result.probabilities) && (
            <div className="probs">
              {result.probabilities.map((p, i) => {
                const pct = Math.max(0, Math.min(100, p * 100));
                const label = CLASS_NAMES[i] ?? `Class ${i}`;
                const isTop = i === result.predicted_index;

                return (
                  <div key={i} className={`probRow ${isTop ? "probTop" : ""}`}>
                    <div className="probTopLine">
                      <div className="probName">
                        {label} {isTop && <span className="miniTag">TOP</span>}
                      </div>
                      <div className="probVal">{pct.toFixed(2)}%</div>
                    </div>

                    <div className="bar">
                      <div className="barFill" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <footer className="footerNote">
            Classes: N=Normal, L=LBBB, R=RBBB, A=Atrial premature, V=Ventricular premature
          </footer>
        </section>
      </main>
    </div>
  );
}
