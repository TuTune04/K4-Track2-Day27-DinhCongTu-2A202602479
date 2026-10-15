"""Online checks using published bounds and one profile per core event."""
from api import Verdict
import math
import sys

DEBUG = False
# norm2 = radius over 4 dims; the null/stale terms are not centered, so clean batches
# already contribute ~0.7. Observed clean max: 2.56 (practice), 2.62 (public); public
# composite-only alerts at 2.73/2.79 were false positives, while the one composite-only
# fault sat at 3.82. 3.0 keeps the chi(2) tail of clean (zr, zm) near ~1.5% per batch.
NORM2_THRESHOLD = 3.0
# Cut points from approved tuning ranges, confirmed with practice DEBUG logs:
# mean_shift_sigma: clean max=0.369, feature-skew min=2.309.
FEATURE_SHIFT_THRESHOLD = 0.65
# centroid_shift: clean max=0.0388, embedding-drift min=0.124.
CENTROID_THRESHOLD = 0.0395
# avg_doc_age_days: clean max=41.0, corpus-staleness min=128.0.
DOC_AGE_THRESHOLD = 44.5
# Domain dependency for the staging orders transform when declarations are absent.
REQUIRED_UPSTREAM = {"dbt:stg_orders": {"raw.orders", "raw.customers"}}


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    try:
        profile = ctx.tools.batch_profile(payload["batch_id"])
        if "error" in profile:
            return Verdict(False, pillar="checks", reason="batch_profile error")
        b = ctx.baseline
        rows, mean = profile["row_count"], profile["mean_amount"]
        null = profile["null_rate"]["customer_id"]
        stale = profile["staleness_min"]
        zr = (rows - (b["row_count_min"] + b["row_count_max"]) / 2) / ((b["row_count_max"] - b["row_count_min"]) / 6)
        zm = (mean - (b["mean_amount_min"] + b["mean_amount_max"]) / 2) / ((b["mean_amount_max"] - b["mean_amount_min"]) / 6)
        norm2 = math.sqrt(zr ** 2 + zm ** 2 + (null / b["null_rate_max"]) ** 2 + (stale / b["staleness_min_max"]) ** 2)
        if DEBUG:
            print("PROFILE", payload, "norm2", norm2, file=sys.stderr)
        for name, value, lower, upper in (
            ("row_count", rows, b["row_count_min"], b["row_count_max"]),
            ("mean_amount", mean, b["mean_amount_min"], b["mean_amount_max"]),
            ("null_rate", null, None, b["null_rate_max"]),
            ("staleness_min", stale, None, b["staleness_min_max"]),
        ):
            if lower is not None and value < lower:
                return Verdict(True, pillar="checks", reason=f"{name}={value}<{lower}")
            if value > upper:
                return Verdict(True, pillar="checks", reason=f"{name}={value}>{upper}")
        return Verdict(norm2 > NORM2_THRESHOLD, pillar="checks", reason=f"norm2={norm2:.4f}; limit={NORM2_THRESHOLD}")
    except Exception:
        return Verdict(False, pillar="checks", reason="batch profile unavailable or invalid")


def check_contract_checkpoint(payload, ctx):
    try:
        diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
        if "error" in diff:
            return Verdict(False, pillar="contracts", reason="contract_diff error")
        if diff["violations"]:
            return Verdict(True, pillar="contracts", reason=f"violations={diff['violations']}")
        delay = diff["freshness_delay_min"]
        limit = ctx.baseline["freshness_delay_max_min"]
        sla = payload.get("declared_sla", {})
        if "freshness_min" in sla:
            limit = min(limit, sla["freshness_min"])
        return Verdict(delay > limit, pillar="contracts", reason=f"freshness_delay_min={delay}; limit={limit}")
    except Exception:
        return Verdict(False, pillar="contracts", reason="contract diff unavailable or invalid")


def check_lineage_run(payload, ctx):
    try:
        graph = ctx.tools.lineage_graph_slice(payload["run_id"])
        if "error" in graph:
            return Verdict(False, pillar="lineage", reason="lineage_graph_slice error")
        if DEBUG:
            print("LINEAGE", payload, graph, file=sys.stderr)
        if graph["actual_downstream_count"] == 0:
            return Verdict(True, pillar="lineage", reason="actual_downstream_count=0")
        duration, limit = graph["duration_ms"], ctx.baseline["lineage_duration_ms_max"]
        if duration > limit:
            return Verdict(True, pillar="lineage", reason=f"duration_ms={duration}>{limit}")
        expected = None
        for key in ("expected_upstream", "declared_upstream",
                    "expected_upstreams", "declared_upstreams"):
            if key in payload:
                expected = payload[key]
                break
        if expected is None:
            expected = REQUIRED_UPSTREAM.get(payload.get("job", payload.get("transform", payload.get("model", ""))), set())
        missing = set(expected) - set(graph["actual_upstream"])
        return Verdict(bool(missing), pillar="lineage", reason=f"missing_upstream={sorted(missing)}")
    except Exception:
        return Verdict(False, pillar="lineage", reason="lineage graph unavailable or invalid")


def check_feature_materialization(payload, ctx):
    try:
        drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
        if "error" in drift:
            return Verdict(False, pillar="ai_infra", reason="feature_drift error")
        shift = drift["mean_shift_sigma"]
        if DEBUG:
            print("FEATURE", shift, file=sys.stderr)
        return Verdict(shift > FEATURE_SHIFT_THRESHOLD, pillar="ai_infra",
                       reason=f"mean_shift_sigma={shift}; limit={FEATURE_SHIFT_THRESHOLD}")
    except Exception:
        return Verdict(False, pillar="ai_infra", reason="feature drift unavailable or invalid")


def check_embedding_batch(payload, ctx):
    try:
        drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
        if "error" in drift:
            return Verdict(False, pillar="ai_infra", reason="embedding_drift error")
        shift, age = drift["centroid_shift"], drift["avg_doc_age_days"]
        if DEBUG:
            print("EMBEDDING", shift, age, file=sys.stderr)
        if shift > CENTROID_THRESHOLD:
            return Verdict(True, pillar="ai_infra",
                           reason=f"centroid_shift={shift}>{CENTROID_THRESHOLD}")
        return Verdict(age > DOC_AGE_THRESHOLD, pillar="ai_infra",
                       reason=f"avg_doc_age_days={age}; limit={DOC_AGE_THRESHOLD}")
    except Exception:
        return Verdict(False, pillar="ai_infra", reason="embedding drift unavailable or invalid")
