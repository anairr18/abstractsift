import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PartitioningBenchmark:
    def run(self, abstracts: List[str], labels: List[int]) -> Dict[str, dict]:
        from pipeline1_patient_outcomes.partitioner import (
            TFIDFPartitioner,
            EmbeddingPartitioner,
            LLMPartitioner,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score

        _, abstracts_te, _, labels_te = train_test_split(
            abstracts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        results = {}

        tfidf = TFIDFPartitioner()
        try:
            tfidf.load()
            preds = tfidf.predict(abstracts_te)
            results["tfidf"] = {
                "precision": precision_score(labels_te, preds, zero_division=0),
                "recall": recall_score(labels_te, preds, zero_division=0),
                "f1": f1_score(labels_te, preds, zero_division=0),
            }
        except Exception as e:
            logger.warning("TFIDFPartitioner benchmark skipped: %s", e)
            results["tfidf"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        emb = EmbeddingPartitioner()
        try:
            emb.load()
            preds = emb.predict(abstracts_te)
            results["embedding"] = {
                "precision": precision_score(labels_te, preds, zero_division=0),
                "recall": recall_score(labels_te, preds, zero_division=0),
                "f1": f1_score(labels_te, preds, zero_division=0),
            }
        except Exception as e:
            logger.warning("EmbeddingPartitioner benchmark skipped: %s", e)
            results["embedding"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        llm = LLMPartitioner()
        try:
            preds = llm.predict(abstracts_te[:50])  # limit LLM calls
            results["llm"] = {
                "precision": precision_score(labels_te[:50], preds, zero_division=0),
                "recall": recall_score(labels_te[:50], preds, zero_division=0),
                "f1": f1_score(labels_te[:50], preds, zero_division=0),
            }
        except Exception as e:
            logger.warning("LLMPartitioner benchmark skipped: %s", e)
            results["llm"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        self._print_table(results)
        return results

    def _print_table(self, results: Dict[str, dict]):
        header = f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        print(header)
        print("-" * len(header))
        for method, m in results.items():
            print(
                f"{method:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}"
            )


class ExtractionBenchmark:
    TRACKED_FIELDS = ["drug", "disease", "outcome.response", "outcome.confidence"]

    def run(
        self, articles: List[Dict], manual_extractions: List[Dict]
    ) -> Dict[str, float]:
        from pipeline1_patient_outcomes.schematizer import LLMSchematizer

        schematizer = LLMSchematizer()
        llm_extractions = schematizer.schematize_batch(articles)

        field_matches: Dict[str, List[bool]] = {f: [] for f in self.TRACKED_FIELDS}

        for llm_rec, manual_rec in zip(llm_extractions, manual_extractions):
            for field in self.TRACKED_FIELDS:
                llm_val = _get_nested(llm_rec, field)
                man_val = _get_nested(manual_rec, field)
                # Normalize for comparison
                match = _normalize(llm_val) == _normalize(man_val)
                field_matches[field].append(match)

        accuracy = {}
        for field, matches in field_matches.items():
            if matches:
                accuracy[field] = sum(matches) / len(matches)
            else:
                accuracy[field] = 0.0

        self._print_table(accuracy)
        return accuracy

    def _print_table(self, accuracy: Dict[str, float]):
        print(f"\n{'Field':<35} {'Accuracy':>10}")
        print("-" * 47)
        for field, acc in accuracy.items():
            print(f"{field:<35} {acc:>10.3f}")


def _get_nested(record: dict, dotpath: str):
    parts = dotpath.split(".")
    val = record
    for part in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val


def _normalize(val) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()
