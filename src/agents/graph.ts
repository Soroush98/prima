import { END, START, StateGraph } from "@langchain/langgraph";
import { PrimaState } from "./state";
import { anomalyDetector, forecaster, narrator, orchestrator, rootCause, sqlAnalyst } from "./nodes";
import type { PrimaResult } from "./types";

/**
 * The Prima agent graph:
 *
 *   START → orchestrator → sql_analyst → anomaly_detector
 *         → forecaster → root_cause → narrator → END
 *
 * Each node is an autonomous agent that reads shared state and contributes its
 * slice (SQL, anomalies, forecast, diagnosis, narrative) plus an observability
 * trace entry.
 */
function buildGraph() {
  return new StateGraph(PrimaState)
    .addNode("orchestrator", orchestrator)
    .addNode("sql_analyst", sqlAnalyst)
    .addNode("anomaly_detector", anomalyDetector)
    .addNode("forecaster", forecaster)
    .addNode("root_cause", rootCause)
    .addNode("narrator", narrator)
    .addEdge(START, "orchestrator")
    .addEdge("orchestrator", "sql_analyst")
    .addEdge("sql_analyst", "anomaly_detector")
    .addEdge("anomaly_detector", "forecaster")
    .addEdge("forecaster", "root_cause")
    .addEdge("root_cause", "narrator")
    .addEdge("narrator", END)
    .compile();
}

let _graph: ReturnType<typeof buildGraph> | null = null;
function getGraph() {
  if (!_graph) _graph = buildGraph();
  return _graph;
}

export async function runPrima(question: string): Promise<PrimaResult> {
  const start = performance.now();
  const final = await getGraph().invoke({ question });
  const totalMs = +(performance.now() - start).toFixed(1);

  return {
    question,
    metric: final.metric,
    filter: final.filter,
    sql: final.sql,
    series: final.series,
    anomalies: final.anomalies,
    forecast: final.forecast,
    rootCause: final.rootCause ?? null,
    summary: final.summary,
    trace: final.trace,
    telemetry: {
      totalMs,
      tokensIn: final.tokensIn,
      tokensOut: final.tokensOut,
      costUsd: +final.costUsd.toFixed(6),
      llmMode: "live",
    },
  };
}
