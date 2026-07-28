"""Reject identity and grouping leakage before pairwise training."""

IDENTITY_FEATURES = {
    "case_id",
    "candidate_id",
    "strategy_id",
    "model_name",
    "path",
    "artifact_path",
    "digest",
    "left_digest",
    "right_digest",
    "publish_decision",
    "repair_decision",
}


def identity_features(feature_names):
    return sorted(set(feature_names) & IDENTITY_FEATURES)


def assert_feature_safe(feature_names):
    leaked = identity_features(feature_names)
    if leaked:
        raise ValueError("identity features are forbidden: " + ", ".join(leaked))


def group_leakage(train_rows, test_rows, group_key="case_id"):
    train = {row[group_key] for row in train_rows}
    test = {row[group_key] for row in test_rows}
    return sorted(train & test)
