"""Dependency-free L2 pairwise logistic regression and grouped evaluation."""

from __future__ import annotations

import math
import random
import statistics

from .leakage_guard import assert_feature_safe, group_leakage


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def fit_scaler(vectors):
    columns = list(zip(*vectors))
    means = [statistics.mean(column) for column in columns]
    scales = [statistics.pstdev(column) or 1.0 for column in columns]
    return means, scales


def scale(vector, means, scales):
    return [(value - mean) / width for value, mean, width in zip(vector, means, scales)]


def fit_logistic(vectors, labels, l2=1.0, learning_rate=0.08, epochs=1200):
    if not vectors or len(set(labels)) < 2:
        raise ValueError("pairwise logistic training requires two classes")
    weights = [0.0] * len(vectors[0])
    intercept = 0.0
    n = len(vectors)
    for epoch in range(epochs):
        grad = [l2 * value for value in weights]
        bias_grad = 0.0
        for vector, label in zip(vectors, labels):
            error = sigmoid(sum(w * x for w, x in zip(weights, vector)) + intercept) - label
            for index, value in enumerate(vector):
                grad[index] += error * value / n
            bias_grad += error / n
        step = learning_rate / math.sqrt(1.0 + epoch / 100.0)
        weights = [value - step * delta for value, delta in zip(weights, grad)]
        intercept -= step * bias_grad
    return weights, intercept


def predict(vector, weights, intercept):
    return sigmoid(sum(w * x for w, x in zip(weights, vector)) + intercept)


def reverse_augment(vectors, labels):
    return vectors + [[-value for value in vector] for vector in vectors], labels + [
        1 - label for label in labels
    ]


def leave_one_case_out(rows, feature_names, seed=20260728):
    del seed  # deterministic implementation; retained in the contract
    assert_feature_safe(feature_names)
    groups = sorted({row["case_id"] for row in rows})
    output = []
    for group in groups:
        train = [row for row in rows if row["case_id"] != group]
        test = [row for row in rows if row["case_id"] == group]
        if group_leakage(train, test):
            raise ValueError("case leakage detected")
        raw_train = [[float(row[name]) for name in feature_names] for row in train]
        train_labels = [int(row["label"]) for row in train]
        if len(set(train_labels)) < 2:
            for row in test:
                output.append({**row, "probability": 0.5, "fold_status": "ONE_CLASS_TRAIN"})
            continue
        means, scales = fit_scaler(raw_train)
        train_x = [scale(vector, means, scales) for vector in raw_train]
        train_x, train_labels = reverse_augment(train_x, train_labels)
        weights, intercept = fit_logistic(train_x, train_labels)
        for row in test:
            vector = scale([float(row[name]) for name in feature_names], means, scales)
            output.append(
                {
                    **row,
                    "probability": predict(vector, weights, intercept),
                    "fold_status": "OK",
                }
            )
    return output


def symmetry_error(vector, weights, intercept=0.0):
    return abs(predict(vector, weights, intercept) + predict([-x for x in vector], weights, intercept) - 1.0)


def cluster_bootstrap(values_by_case, resamples=1000, seed=0):
    if not values_by_case:
        return None
    rng = random.Random(seed)
    cases = sorted(values_by_case)
    samples = []
    for _ in range(resamples):
        chosen = [rng.choice(cases) for _ in cases]
        values = [value for case in chosen for value in values_by_case[case]]
        samples.append(statistics.mean(values))
    samples.sort()
    return [
        samples[int(0.025 * (resamples - 1))],
        samples[int(0.975 * (resamples - 1))],
    ]
