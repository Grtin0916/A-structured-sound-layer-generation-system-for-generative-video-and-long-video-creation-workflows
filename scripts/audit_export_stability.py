from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ONNX fixed-shape stability.")
    parser.add_argument(
        "--onnx",
        type=str,
        default="artifacts/onnx/baseline_v1.onnx",
        help="Path to ONNX model.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="artifacts/logs/week02_shape_audit.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = inp.shape

    cases = [
        ("valid_ref", (1, 1, 64, 313), True, "reference fixed shape"),
        ("valid_same_shape_alt", (1, 1, 64, 313), True, "same legal shape"),
        ("invalid_width_minus_1", (1, 1, 64, 312), False, "width mismatch"),
        ("invalid_width_plus_1", (1, 1, 64, 314), False, "width mismatch"),
        ("invalid_channel_2", (1, 2, 64, 313), False, "channel mismatch"),
        ("invalid_batch_2", (2, 1, 64, 313), False, "batch mismatch"),
    ]

    rows = []
    for case_name, shape, expected_pass, note in cases:
        x = np.random.randn(*shape).astype(np.float32)
        try:
            outputs = sess.run(None, {input_name: x})
            output_shape = tuple(outputs[0].shape) if outputs else None
            actual_pass = True
            error_msg = ""
        except Exception as e:
            output_shape = None
            actual_pass = False
            error_msg = str(e).replace("\n", " ")[:500]

        rows.append(
            {
                "case_name": case_name,
                "input_name": input_name,
                "model_declared_shape": str(input_shape),
                "test_shape": str(shape),
                "expected_pass": expected_pass,
                "actual_pass": actual_pass,
                "match_expectation": expected_pass == actual_pass,
                "output_shape": str(output_shape),
                "note": note,
                "error": error_msg,
            }
        )

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "input_name",
                "model_declared_shape",
                "test_shape",
                "expected_pass",
                "actual_pass",
                "match_expectation",
                "output_shape",
                "note",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {csv_path}")
    for row in rows:
        print(
            f"{row['case_name']}: expected={row['expected_pass']} "
            f"actual={row['actual_pass']} match={row['match_expectation']}"
        )


if __name__ == "__main__":
    main()