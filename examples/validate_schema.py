"""
Validate example outputs against the expected v2 schema.

Run from the clinical_pipelines_v2 directory:
    python examples/validate_schema.py

Reads from examples/outputs/all_examples.jsonl (produced by generate_examples.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EXAMPLES_FILE = Path(__file__).parent / "outputs" / "all_examples.jsonl"

# (field_path, weight) — field_path uses dot notation for nested fields
SCORED_FIELDS = [
    ("drug", 2.0),
    ("disease", 2.0),
    ("outcome.response", 3.0),
    ("patient_context.demographics", 1.0),
    ("patient_context.genomics", 1.0),
    ("patient_context.labs", 1.0),
    ("patient_context.patient_factors", 1.0),
    ("kg_features.concept_count", 1.0),  # must be > 0
]

REQUIRED_FIELDS = ["drug", "disease", "outcome", "patient_context"]


def _get_nested(obj: dict, path: str):
    """Return value at dot-delimited path, or None if missing."""
    parts = path.split(".")
    for part in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _is_populated(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value > 0
    return bool(str(value).strip())


def validate_record(record: dict) -> dict:
    issues = []

    # Required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            issues.append(f"Missing required field: {field}")

    # outcome.response specifically
    outcome = record.get("outcome") or {}
    if isinstance(outcome, dict) and not outcome.get("response"):
        issues.append("outcome.response is null or missing")

    # Score completeness
    score = 0.0
    total = sum(w for _, w in SCORED_FIELDS)
    for path, weight in SCORED_FIELDS:
        val = _get_nested(record, path)
        if _is_populated(val):
            score += weight

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "completeness": round(score / total, 2),
    }


def main():
    if not EXAMPLES_FILE.exists():
        print(f"No examples found at {EXAMPLES_FILE}")
        print("Run: python examples/generate_examples.py")
        sys.exit(1)

    records = []
    with open(EXAMPLES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("File is empty.")
        sys.exit(1)

    print(f"Validating {len(records)} records from {EXAMPLES_FILE}\n")

    total_completeness = 0.0
    valid_count = 0

    for record in records:
        pmid = record.get("pmid", "unknown")
        title = (record.get("title") or "")[:60]
        result = validate_record(record)

        status = "PASS" if result["valid"] else "FAIL"
        pct = f"{result['completeness']:.0%}"
        kg_count = (record.get("kg_features") or {}).get("concept_count", 0)

        print(f"[{status}] PMID {pmid} | completeness={pct} | kg_concepts={kg_count}")
        print(f"       {title}")
        for issue in result["issues"]:
            print(f"       ! {issue}")
        print()

        total_completeness += result["completeness"]
        if result["valid"]:
            valid_count += 1

    avg = total_completeness / len(records)
    print("-" * 60)
    print(f"Valid:               {valid_count}/{len(records)}")
    print(f"Average completeness: {avg:.0%}")


if __name__ == "__main__":
    main()
