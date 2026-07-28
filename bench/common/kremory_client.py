"""Shared kremory HTTP client for every benchmark harness.

EXTRACTED 2026-07-28 from bench/locomo/harness.py — the only copy that has ever
actually run against kremory. bench/longmemeval/harness.py carried an
independently-adapted FORK of the same class (7 methods vs 14, two stubbed),
and that fork silently missed every hardening the LoCoMo side had learned:

  * `store_timeout = 90.0` — kremory's ingest fans out to ~20 sequential LLM
    calls per store, so httpx's 30s default times out MID-STORE. The fork kept
    30.0 AND had no try/except, so an ingest timeout raised an uncaught
    httpx.ReadTimeout and killed the run outright. Reproduced 2026-07-28 on the
    first real call ever made through it: a 13,229-char LongMemEval session was
    still running extraction passes 42s in.
  * `KremoryStalled` — fail fast on a stalled ingest instead of grinding.
  * `total_http_errors` — the per-source counter the circuit breaker reads.
  * `scrape_metrics` — server-side token / cost / latency observability.

Three further pieces had ALSO been duplicated and were already drifting
(`normalize_text`, `list_item_overlap_score`, the crash-safe JSONL writer),
which is why the shared surface is being pulled out rather than the timeout fix
being ported a fourth time.

Import this. Do not re-fork it.
"""
from __future__ import annotations

import os as _os
import sys

import httpx

# Override via KREMORY_INGEST_BUDGET_S; tighten the hang guard via KREMORY_INGEST_STALL_S.
INGEST_BUDGET_S = float(_os.environ.get("KREMORY_INGEST_BUDGET_S", "7200"))


class KremoryStalled(RuntimeError):
    """Raised when kremory ingest stalls (store timeout) or blows the wall-clock
    budget. Bubbles to the harness's fail-fast handler — never swallowed."""


# ---------------------------------------------------------------------------
# Codemem API client
# ---------------------------------------------------------------------------

class CodememClient:
    def __init__(self, base_url: str, timeout: float = 30.0, server_mode: str = "recall"):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)
        # kremory-http GET /search server-side retrieval mode (recall/content/
        # hybrid) — see Config.server_mode docstring above for why this is a
        # separate axis from --mode.
        self.server_mode = server_mode
        # Raw wire rows from the most recent `recall()` — see the comment there.
        self.last_results: list[dict] = []
        # kremory's POST /memories runs a SYNCHRONOUS 3-stage LLM extraction
        # pipeline (entities -> relations -> triplets, then per-entity
        # ResolutionVerdict dedup calls) on every store — this is not a
        # cheap embed-only write. Measured against gemma4:e4b (the fast
        # default model): kremory.ingest.completed total_ms observed at
        # 15.7s / 25.1s / 21.6s for successful stores in a smoke run, and
        # one store still had ~6 pending ResolutionVerdict calls after 30s
        # of stage time — tripping this client's default 30.0s timeout
        # with an httpx.ReadTimeout mid-store. 120s gives headroom for a
        # slow store plus one ladder-arm fallback retry.
        # Env-overridable because the right value is a property of the CORPUS,
        # not a constant. 90s was tuned on LoCoMo episodes; LongMemEval stores
        # whole ~9.7K-char SESSIONS, measured at 75.2s and 175.7s (mean 125s,
        # ~40 LLM calls each), so 90s aborts on the first question. Default is
        # unchanged so LoCoMo behaviour is byte-identical; raise it per corpus
        # via KREMORY_STORE_TIMEOUT_S.
        self.store_timeout = float(_os.environ.get("KREMORY_STORE_TIMEOUT_S", "90"))
        # kremory's POST /consolidation/{cycle} runs the full dream()
        # reconciliation (discover/aliases/reclassify/consistency_check/
        # canonicalize passes) over every episode in the namespace — scales
        # with namespace size, not per-call content size. Generous ceiling
        # so a 19-session conversation's worth of memories doesn't trip the
        # same class of timeout.
        # Raised from 300s (conv0's dream exceeded it → ReadTimeout aborted
        # consolidation → multi-hop under-served). Env-overridable.
        self.consolidate_timeout = float(_os.environ.get("KREMORY_CONSOLIDATE_TIMEOUT_S", "1800"))
        # Per-source HTTP-error counter (observability-first-class: an
        # aggregate accuracy score alone can't tell you WHY it's low —
        # retrieval-broken vs model-too-weak vs scoring-bug. This is the
        # cheap half of that signal; recall_empty tracked alongside it in
        # run_benchmark() is the other half).
        self.total_http_errors = 0

    def health(self) -> bool:
        try:
            r = self.http.get("/health")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def store_memory(
        self,
        content: str,
        namespace: str,
        memory_type: str = "Context",
        importance: float = 0.5,
        tags: list[str] | None = None,
        published_at: str | None = None,
    ) -> str | None:
        # kremory's POST /memories accepts {content, namespace, published_at?}
        # — memory_type/importance/tags are codemem-only fields kremory
        # ignores; kept in the signature so callers below don't need to
        # change, just not sent over the wire.
        #
        # `published_at` (RFC3339) sets the episode's WORLD-time, distinct from
        # `recorded_at` (system time, stamped at ingest). The server has always
        # accepted it and NEITHER harness fork has ever sent it: LoCoMo has no
        # per-episode date to send, and LongMemEval embedded its
        # `haystack_dates` value into the content STRING only. Wiring it here
        # is the first time any benchmark exercises kremory's temporal axis.
        body: dict = {"content": content, "namespace": namespace}
        if published_at:
            body["published_at"] = published_at
        try:
            r = self.http.post(
                "/memories",
                json=body,
                timeout=self.store_timeout,
            )
        except httpx.TimeoutException as e:
            # A store TIMEOUT means kremory's ingest stalled (per-remember() fans
            # out to ~20 sequential LLM calls). Per fail-fast-and-loud: do NOT
            # grind or swallow — abort the whole run immediately with a clear
            # diagnostic. Bubbles to run_benchmark's KremoryStalled handler.
            raise KremoryStalled(
                f"POST /memories timed out after {self.store_timeout:.0f}s "
                f"(namespace={namespace}, content_len={len(content)}) — "
                f"kremory ingest stalled"
            ) from e
        except httpx.HTTPError as e:
            raise KremoryStalled(
                f"POST /memories transport error ({type(e).__name__}: {e}) "
                f"(namespace={namespace})"
            ) from e
        if r.status_code == 201:
            return r.json().get("id")
        self.total_http_errors += 1
        print(f"  [warn] store failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
        return None

    def recall(self, query: str, namespace: str, limit: int = 15) -> list[dict]:
        r = self.http.get(
            "/search",
            params={
                "q": query,
                "namespace": namespace,
                "k": limit,
                # ALWAYS send `mode` explicitly (fixed 2026-07-28, ADR-078).
                # This previously omitted the param whenever `server_mode ==
                # "recall"`, on the belief that the server's default was also
                # `recall`. It is NOT: `SearchMode`'s `#[default]` is `Hybrid`
                # (`kremory-http.rs`), deliberately, because the entity-graph
                # `recall` surface judges 40.2% vs hybrid's 71.4%. So every run
                # this harness has ever labelled `server_mode: recall` was in
                # fact measuring HYBRID — the label lied, and `--server-mode
                # recall` was unreachable over the wire. Sending it explicitly
                # makes the provenance stamp true and the flag actually work.
                "mode": self.server_mode,
            },
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            # ADR-078 Phase A: stash the RAW wire rows so the caller can persist
            # per-item provenance (`kind`, `source_episode_id` — TD-139's
            # `SearchResultWire`) alongside the flattened content strings. The
            # flattening below is lossy: it turns entity summaries, facts and
            # verbatim episode turns into one anonymous list, which is exactly
            # the format the answerer has always been fed. Keeping the kinds
            # lets us test whether LABELLED context beats the flat blob.
            self.last_results = results
            return results
        self.last_results = []
        self.total_http_errors += 1
        print(f"  [warn] recall failed ({r.status_code}): {r.text[:200]}", file=sys.stderr)
        return []

    def scrape_metrics(self) -> dict[str, float]:
        """Scrape GET /metrics (Prometheus text) and sum each metric across its
        label sets → {metric_name: total}. TD-132: surfaces the server-side
        `kremory_core_*` counters (tokens, cost, request latency, dream duration)
        the kremory-http `prometheus` feature exposes. Returns {} if the endpoint
        is absent (server built without the feature) — bench degrades gracefully.
        """
        try:
            r = self.http.get("/metrics", timeout=10.0)
        except Exception:
            return {}
        if r.status_code != 200:
            return {}
        totals: dict[str, float] = {}
        for line in r.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Prometheus text: `name{labels} value`  OR  `name value`.
            try:
                left, val = line.rsplit(" ", 1)
                value = float(val)
            except ValueError:
                continue
            name = left.split("{", 1)[0]
            totals[name] = totals.get(name, 0.0) + value
        return totals

    def graph_neighbors(self, node_id: str, depth: int = 2) -> list[dict]:
        # kremory-http has no graph-traversal REST tool yet — stub so
        # --mode codemem-graph degrades to plain recall instead of 404ing.
        return []

    def get_memory(self, memory_id: str) -> dict | None:
        # No GET /memories/{id} route on kremory-http yet — stub.
        return None

    def consolidate(self, cycle: str, namespace: str) -> bool:
        # kremory's POST /consolidation/{cycle} REQUIRES ?namespace= — dream()
        # is always namespace-scoped (unlike codemem's global consolidation);
        # omitting it is a loud 422, not a silent no-op.
        try:
            r = self.http.post(
                f"/consolidation/{cycle}",
                params={"namespace": namespace},
                timeout=self.consolidate_timeout,
            )
        except httpx.HTTPError as e:
            # Consolidation is best-effort enrichment (SHARES_THEME edges) — a
            # timeout must NOT nuke the run, but MUST be loud (not a raw crash).
            self.total_http_errors += 1
            print(
                f"  [warn] consolidation '{cycle}' failed after "
                f"{self.consolidate_timeout:.0f}s: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return False
        return r.status_code == 200

    def start_session(self, namespace: str) -> str | None:
        # kremory-http has no session concept/route — no-op (matches the
        # already-no-op contract: return None, ingest_conversation() treats a
        # falsy session_id as "skip end_session").
        return None

    def end_session(self, session_id: str, summary: str = "") -> bool:
        # No-op — see start_session.
        return True

    def delete_namespace(self, namespace: str) -> bool:
        r = self.http.delete(f"/namespaces/{namespace}")
        return r.status_code in (200, 404)

    def get_namespaces(self) -> list[dict]:
        # kremory-http has no namespace-listing route yet — stub.
        return []


