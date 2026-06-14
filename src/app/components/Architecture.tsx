"use client";

/**
 * Architecture tab — a technical walkthrough, aimed at senior AI engineers, of:
 *   1. how the LangGraph runtime compiles the agent fleet into a Pregel-style state machine,
 *   2. what actually happens inside a single LLM node (tokenize → transformer decode → sample → tool-use),
 *   3. the LangChain / LangGraph abstraction stack this project sits on.
 *
 * Diagrams are hand-built SVG so they render with no extra deps and match the dark theme.
 */

const NODES = [
  { id: "orchestrator", label: "orchestrator", sub: "parse metric + scope", kind: "code" },
  { id: "sql_analyst", label: "sql_analyst", sub: "LLM → SQL → guard → run", kind: "llm" },
  { id: "anomaly_detector", label: "anomaly_detector", sub: "seasonal z-score (EVT/POT)", kind: "compute" },
  { id: "forecaster", label: "forecaster", sub: "Chronos-Bolt / Holt-Winters", kind: "compute" },
  { id: "root_cause", label: "root_cause", sub: "LLM hypotheses + deploy join", kind: "llm" },
  { id: "narrator", label: "narrator", sub: "LLM executive briefing", kind: "llm" },
] as const;

const KIND_COLOR: Record<string, string> = {
  llm: "#7c5cff",
  compute: "#3ddc97",
  code: "#4f9cf9",
};

export default function Architecture() {
  return (
    <div className="arch">
      <div className="arch-intro card">
        <h3>How Prima works under the hood</h3>
        <p className="summary" style={{ marginTop: 4 }}>
          Prima is a <b>LangGraph</b> agent fleet. At runtime the graph is compiled into a{" "}
          <b>Pregel-style state machine</b>: each agent is a pure node{" "}
          <code>(state) → Partial&lt;state&gt;</code>, edges define the super-step schedule, and a typed
          state object with per-channel <b>reducers</b> is the only shared memory. Three of the six nodes
          call a real Claude model; the rest are deterministic numerics. Below: the orchestration layer,
          then a drill-down into what a single LLM node actually does at the token level.
        </p>
        <div className="legend">
          <span><i style={{ background: KIND_COLOR.llm }} /> LLM node (live Claude)</span>
          <span><i style={{ background: KIND_COLOR.compute }} /> Numerical / ML compute</span>
          <span><i style={{ background: KIND_COLOR.code }} /> Deterministic code</span>
        </div>
      </div>

      {/* ---------------- 1. LangGraph runtime ---------------- */}
      <div className="arch-section-title">1 · LangGraph runtime — the agent fleet as a compiled graph</div>
      <div className="card">
        <GraphDiagram />
        <div className="arch-notes">
          <div>
            <b>StateGraph&lt;PrimaState&gt;</b> — channels are declared with reducers.{" "}
            <code>trace</code> uses a concat reducer, <code>tokensIn/Out</code> &amp; <code>costUsd</code> sum,
            scalar channels are last-write-wins. Nodes never mutate; they return a partial update that the
            runtime <i>folds</i> into the channel via its reducer.
          </div>
          <div>
            <b>Super-step execution (Pregel/BSP).</b> Each tick: active nodes run concurrently → their partial
            updates are reduced into the state → the next frontier of nodes is scheduled. This linear pipeline
            is one node per step; branching graphs run siblings in parallel within a step.
          </div>
          <div>
            <b>compile() → checkpointer.</b> <code>graph.compile()</code> freezes the topology into a runnable.
            With a checkpointer each super-step snapshots the channels, enabling resume, time-travel and
            human-in-the-loop interrupts — the same mechanism that powers streaming the trace you see live.
          </div>
        </div>
      </div>

      {/* ---------------- 2. Inside one LLM node ---------------- */}
      <div className="arch-section-title">2 · Inside one LLM node — how the model actually produces a token</div>
      <div className="card">
        <ModelDiagram />
        <div className="arch-notes">
          <div>
            <b>Prompt → tokens.</b> The node renders a message list (system + few-shot schema + user metric),
            which the <b>BPE tokenizer</b> splits into ~sub-word integer IDs. Each ID indexes a learned
            <code>d_model</code>-dim embedding; rotary/positional information is mixed in.
          </div>
          <div>
            <b>N decoder blocks.</b> Every block = <b>causal multi-head self-attention</b> (each token attends
            only to earlier tokens via the QKᵀ/√d softmax, masked) → residual + RMSNorm → <b>MLP / SwiGLU</b> →
            residual. Stacked <code>L</code> times this builds the contextual representation.
          </div>
          <div>
            <b>Logits → sampling.</b> The final hidden state is projected to a vocab-sized logit vector,
            softmax-normalised, then sampled (temperature, top-p). Decoding is <b>autoregressive</b>: the new
            token is appended and fed back; a <b>KV-cache</b> stores past keys/values so each step is O(seq), not
            O(seq²) recomputed.
          </div>
          <div>
            <b>Back to the graph.</b> For <code>sql_analyst</code> the decoded text is a SQL statement → passed
            through a read-only guard → executed. This <i>structured / tool-use</i> contract is what turns raw
            next-token prediction into a reliable agent action.
          </div>
        </div>
      </div>

      {/* ---------------- 3. Abstraction stack + tool loop ---------------- */}
      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>3 · The abstraction stack</h3>
          <StackDiagram />
          <p className="muted" style={{ fontSize: 12.5, marginTop: 12, lineHeight: 1.6 }}>
            Everything is a <code>Runnable</code> (LCEL): models, prompts and tools share one{" "}
            <code>invoke / stream / batch</code> interface. LangGraph adds <i>stateful, cyclic</i> orchestration
            on top of that linear chain primitive — which is what an agent (loop until done) requires.
          </p>
        </div>
        <div className="card">
          <h3>4 · The agentic tool-use loop</h3>
          <ToolLoopDiagram />
          <p className="muted" style={{ fontSize: 12.5, marginTop: 12, lineHeight: 1.6 }}>
            A node may loop: the model emits a <b>tool call</b> (JSON args), the runtime executes the tool,
            appends an observation, and re-invokes the model. The loop exits when the model returns a final
            message with no tool call — the LangGraph state machine externalises this so each hop is
            checkpointed and auditable.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Engineering properties this buys us</h3>
        <div className="arch-props">
          <div><b>Determinism boundary</b><span>Numerics (anomaly/forecast/segments) never depend on model output — reproducible regardless of sampling.</span></div>
          <div><b>Observability</b><span>Every super-step appends to <code>trace[]</code> + token/cost reducers → the live execution trace and eval scorecard.</span></div>
          <div><b>Typed state</b><span>One <code>PrimaState</code> schema; nodes are pure functions of it — trivially unit-testable in isolation.</span></div>
          <div><b>Swappable engines</b><span>LLM, VAE service and forecaster sit behind interfaces; offline fallbacks keep the graph runnable without a network.</span></div>
        </div>
      </div>
    </div>
  );
}

/* ============================ Diagram 1: the graph ============================ */
function GraphDiagram() {
  const W = 1080;
  const colW = W / NODES.length;
  const nodeW = 150;
  const nodeH = 64;
  const y = 150;
  const stateY = 36;
  return (
    <svg viewBox={`0 0 ${W} 250`} className="arch-svg" role="img" aria-label="LangGraph state graph">
      <defs>
        <marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
          <path d="M0,0 L7,3 L0,6 Z" fill="#5a6b8c" />
        </marker>
      </defs>

      {/* shared state bus */}
      <rect x={20} y={stateY - 14} width={W - 40} height={30} rx={8} fill="#0c1120" stroke="#243049" />
      <text x={34} y={stateY + 5} className="arch-t-bus">PrimaState  ·  channels: {`{ metric, series, anomalies, forecast, rootCause, summary, trace⊕, tokens∑, cost∑ }`}</text>

      {NODES.map((n, i) => {
        const cx = i * colW + colW / 2;
        const x = cx - nodeW / 2;
        const color = KIND_COLOR[n.kind];
        const nextX = (i + 1) * colW + colW / 2 - nodeW / 2;
        return (
          <g key={n.id}>
            {/* edge to next */}
            {i < NODES.length - 1 && (
              <line x1={x + nodeW} y1={y + nodeH / 2} x2={nextX} y2={y + nodeH / 2} stroke="#5a6b8c" strokeWidth={1.6} markerEnd="url(#arr)" />
            )}
            {/* state read/write link */}
            <line x1={cx} y1={stateY + 16} x2={cx} y2={y} stroke={color} strokeWidth={1.2} strokeDasharray="3 3" opacity={0.6} />
            <circle cx={cx} cy={stateY + 16} r={2.6} fill={color} />

            <rect x={x} y={y} width={nodeW} height={nodeH} rx={10} fill="#161d31" stroke={color} strokeWidth={1.4} />
            <rect x={x} y={y} width={4} height={nodeH} rx={2} fill={color} />
            <text x={cx} y={y + 26} className="arch-t-node" textAnchor="middle">{n.label}</text>
            <text x={cx} y={y + 45} className="arch-t-sub" textAnchor="middle">{n.sub}</text>
            <text x={x + 10} y={y - 8} className="arch-t-step">super-step {i + 1}</text>
          </g>
        );
      })}

      {/* START / END */}
      <text x={24} y={y + nodeH / 2 + 4} className="arch-t-end">START ▸</text>
      <text x={W - 20} y={y + nodeH / 2 + 4} className="arch-t-end" textAnchor="end">▸ END</text>
    </svg>
  );
}

/* ============================ Diagram 2: the model ============================ */
function ModelDiagram() {
  return (
    <svg viewBox="0 0 1080 300" className="arch-svg" role="img" aria-label="Transformer decode path">
      <defs>
        <marker id="arr2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
          <path d="M0,0 L7,3 L0,6 Z" fill="#5a6b8c" />
        </marker>
        <linearGradient id="block" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1c2440" />
          <stop offset="1" stopColor="#141a2e" />
        </linearGradient>
      </defs>

      {/* horizontal pipeline */}
      <Stage x={20} title="messages" lines={["system + schema", "user: metric Q"]} color="#4f9cf9" />
      <Arrow x1={172} x2={208} y={70} />
      <Stage x={208} title="BPE tokenizer" lines={["text → int IDs", "sub-word vocab"]} color="#4f9cf9" />
      <Arrow x1={360} x2={396} y={70} />
      <Stage x={396} title="embed + pos" lines={["IDs → d_model", "+ rotary pos"]} color="#4f9cf9" />
      <Arrow x1={548} x2={584} y={70} />

      {/* the transformer stack box */}
      <rect x={584} y={20} width={300} height={170} rx={12} fill="url(#block)" stroke="#7c5cff" strokeWidth={1.5} />
      <text x={734} y={40} className="arch-t-node" textAnchor="middle">decoder block × L</text>

      <rect x={602} y={52} width={264} height={34} rx={8} fill="#161d31" stroke="#7c5cff" opacity={0.9} />
      <text x={734} y={73} className="arch-t-sub2" textAnchor="middle">causal multi-head self-attention · softmax(QKᵀ/√d)·V</text>

      <rect x={602} y={92} width={264} height={28} rx={8} fill="#161d31" stroke="#2c3550" />
      <text x={734} y={110} className="arch-t-sub2" textAnchor="middle">+ residual · RMSNorm</text>

      <rect x={602} y={126} width={264} height={30} rx={8} fill="#161d31" stroke="#7c5cff" opacity={0.9} />
      <text x={734} y={145} className="arch-t-sub2" textAnchor="middle">MLP / SwiGLU feed-forward</text>

      <text x={734} y={178} className="arch-t-mini" textAnchor="middle">KV-cache reused every decode step → O(seq) per token</text>

      <Arrow x1={884} x2={920} />
      <Stage x={920} title="logits → sample" lines={["softmax over vocab", "temp · top-p"]} color="#3ddc97" />

      {/* autoregressive feedback loop */}
      <path d="M996,150 L996,250 L228,250 L228,124" fill="none" stroke="#7c5cff" strokeWidth={1.4} strokeDasharray="5 4" markerEnd="url(#arr2)" />
      <text x={612} y={268} className="arch-t-loop" textAnchor="middle">autoregressive loop — append sampled token, decode next  (until stop / tool-call)</text>
    </svg>
  );
}

/* ============================ Diagram 3: stack ============================ */
function StackDiagram() {
  const layers = [
    { t: "LangGraph", s: "StateGraph · super-steps · checkpoints · cycles", c: "#7c5cff" },
    { t: "LangChain Runnable (LCEL)", s: "invoke / stream / batch · prompt | model | parser", c: "#4f9cf9" },
    { t: "ChatModel adapter", s: "@langchain/anthropic → messages + tool schema", c: "#4f9cf9" },
    { t: "Anthropic API", s: "Claude · tokenizer + transformer decode", c: "#3ddc97" },
  ];
  return (
    <div className="stack">
      {layers.map((l) => (
        <div key={l.t} className="stack-layer" style={{ borderLeftColor: l.c }}>
          <span className="stack-t">{l.t}</span>
          <span className="stack-s">{l.s}</span>
        </div>
      ))}
    </div>
  );
}

/* ============================ Diagram 4: tool loop ============================ */
function ToolLoopDiagram() {
  return (
    <svg viewBox="0 0 480 200" className="arch-svg" role="img" aria-label="Tool-use loop">
      <defs>
        <marker id="arr3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
          <path d="M0,0 L7,3 L0,6 Z" fill="#5a6b8c" />
        </marker>
      </defs>
      <LoopBox x={60} y={20} w={150} label="Claude model" sub="emit tool_call(args)" color="#7c5cff" />
      <LoopBox x={270} y={20} w={150} label="tool / SQL guard" sub="execute → observation" color="#3ddc97" />
      <LoopBox x={165} y={130} w={150} label="final message" sub="no tool_call → exit" color="#4f9cf9" />

      <line x1={210} y1={45} x2={266} y2={45} stroke="#5a6b8c" strokeWidth={1.5} markerEnd="url(#arr3)" />
      <text x={238} y={38} className="arch-t-mini" textAnchor="middle">call</text>
      <path d="M300,75 L300,110 L246,130" fill="none" stroke="#5a6b8c" strokeWidth={1.5} markerEnd="url(#arr3)" />
      <path d="M270,55 C150,90 150,55 135,75" fill="none" stroke="#5a6b8c" strokeWidth={1.5} markerEnd="url(#arr3)" />
      <text x={150} y={108} className="arch-t-mini" textAnchor="middle">observation ↺</text>
    </svg>
  );
}

/* ---- small SVG helpers ---- */
function Stage({ x, title, lines, color }: { x: number; title: string; lines: string[]; color: string }) {
  return (
    <g>
      <rect x={x} y={36} width={152} height={68} rx={10} fill="#161d31" stroke={color} strokeWidth={1.3} />
      <rect x={x} y={36} width={4} height={68} rx={2} fill={color} />
      <text x={x + 78} y={58} className="arch-t-node" textAnchor="middle">{title}</text>
      {lines.map((l, i) => (
        <text key={i} x={x + 78} y={76 + i * 15} className="arch-t-sub" textAnchor="middle">{l}</text>
      ))}
    </g>
  );
}
function Arrow({ x1, x2, y = 70 }: { x1: number; x2: number; y?: number }) {
  return <line x1={x1} y1={y} x2={x2} y2={y} stroke="#5a6b8c" strokeWidth={1.6} markerEnd="url(#arr2)" />;
}
function LoopBox({ x, y, w, label, sub, color }: { x: number; y: number; w: number; label: string; sub: string; color: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={50} rx={9} fill="#161d31" stroke={color} strokeWidth={1.3} />
      <text x={x + w / 2} y={y + 22} className="arch-t-node" textAnchor="middle">{label}</text>
      <text x={x + w / 2} y={y + 39} className="arch-t-sub" textAnchor="middle">{sub}</text>
    </g>
  );
}
