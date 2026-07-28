"""Stable schemas used by the preference review pack."""

PREFERENCE_VALUES = {"LEFT", "RIGHT", "TIE", "UNJUDGEABLE"}
DIMENSION_FIELDS = (
    "timing_preference",
    "event_coverage_preference",
    "audio_quality_preference",
    "unwanted_event_preference",
)
LABEL_FIELDS = (
    "pair_id",
    "case_id",
    "protocol_version",
    "review_session_id",
    "left_artifact",
    "right_artifact",
    "left_digest",
    "right_digest",
    "presentation_order",
    "is_hidden_repeat",
    "repeat_group_id",
    "is_audit_pair",
    "overall_preference",
    *DIMENSION_FIELDS,
    "confidence",
    "reason_codes",
    "free_text_reason",
    "reviewer_id",
    "reviewed_at",
    "submitted",
)

PUBLIC_ALLOWED_FIELDS = {
    "opaque_pair_id",
    "left_media",
    "right_media",
    "video_media",
    "block_id",
    "display_index",
}

BLIND_FORBIDDEN_TOKENS = (
    "strategy",
    "model",
    "control",
    "rerank",
    "repair",
    "proxy",
    "candidate",
    "source_commit",
    "artifact_path",
    "digest",
)


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_judgment(row):
    """Return validation failures for a submitted judgment."""
    if not parse_bool(row.get("submitted")):
        return []
    failures = []
    overall = row.get("overall_preference", "").strip().upper()
    if overall not in PREFERENCE_VALUES:
        failures.append("invalid_overall_preference")
    for field in DIMENSION_FIELDS:
        value = row.get(field, "").strip().upper()
        if value and value not in PREFERENCE_VALUES:
            failures.append(f"invalid_{field}")
    try:
        confidence = int(row.get("confidence", ""))
        if not 1 <= confidence <= 5:
            failures.append("confidence_out_of_range")
    except ValueError:
        failures.append("confidence_missing")
    if overall not in {"TIE", "UNJUDGEABLE"}:
        if not row.get("reason_codes", "").strip():
            failures.append("reason_codes_missing")
        if not row.get("free_text_reason", "").strip():
            failures.append("free_text_reason_missing")
    return failures
