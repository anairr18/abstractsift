import csv
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class Pipeline1:
    def __init__(self):
        import config
        self.cfg = config
        self._setup_dirs()

    def _setup_dirs(self):
        for d in self.cfg.DIRS.values():
            Path(d).mkdir(parents=True, exist_ok=True)

    def run(self):
        from pipeline1_patient_outcomes.retrieval import PubMedRetriever
        from pipeline1_patient_outcomes.partitioner import BinaryPartitioner
        from pipeline1_patient_outcomes.schematizer import LLMSchematizer
        from pipeline1_patient_outcomes.kg_enrichment import KGEnricher
        from pipeline1_patient_outcomes.benchmarks import PartitioningBenchmark, ExtractionBenchmark
        from shared_utilities.utils import PerformanceTracker

        tracker = PerformanceTracker(self.cfg.DIRS["metrics"])

        # Retrieval
        retriever = PubMedRetriever(cache_dir=f"{self.cfg.DIRS['cache']}/pubmed")
        logger.info("Searching PubMed (target: %d abstracts)...", self.cfg.P1_TARGET_ABSTRACTS)

        pmids = []
        query = self.cfg.P1_PUBMED_QUERY
        for year in range(self.cfg.P1_MIN_YEAR, 2026):
            if len(pmids) >= self.cfg.P1_TARGET_ABSTRACTS:
                break
            year_query = f"{query} AND {year}[pdat]"
            batch = retriever.search(year_query, max_results=self.cfg.P1_MAX_RESULTS_PER_QUERY)
            pmids.extend(batch)

        pmids = list(dict.fromkeys(pmids))[: self.cfg.P1_TARGET_ABSTRACTS]
        logger.info("Fetching %d abstracts...", len(pmids))
        abstracts_data = retriever.fetch_abstracts(pmids)
        abstracts_data = [a for a in abstracts_data if a.get("abstract")]

        tracker.log("pipeline1", "retrieval", {"total_abstracts": len(abstracts_data)})

        # Build synthetic training labels if no labels exist
        # Heuristic: abstract contains case-report-like keywords → label=1
        texts = [a["abstract"] for a in abstracts_data]
        labels = [_heuristic_label(a["abstract"]) for a in abstracts_data]
        pos_count = sum(labels)
        logger.info("Heuristic labels: %d positive, %d negative", pos_count, len(labels) - pos_count)

        # Binary partitioning
        partitioner = BinaryPartitioner()
        train_metrics = partitioner.train(texts, labels)
        tracker.log("pipeline1", "partitioning_train", {k: v["f1"] for k, v in train_metrics.items()})

        if self.cfg.BENCHMARK_PARTITIONING:
            bench = PartitioningBenchmark()
            bench_results = bench.run(texts, labels)
            tracker.log("pipeline1", "partitioning_benchmark", {
                k: v["f1"] for k, v in bench_results.items()
            })

        # Predict and keep useful papers
        useful_mask = partitioner.predict(texts)
        useful_articles = [
            abstracts_data[i] for i, flag in enumerate(useful_mask) if flag
        ]
        logger.info("Useful articles after partitioning: %d", len(useful_articles))
        useful_articles = useful_articles[: self.cfg.P1_USEFUL_TARGET]

        # Fetch full text
        logger.info("Fetching full text for %d articles...", len(useful_articles))
        for article in useful_articles:
            ft = retriever.fetch_full_text(article["pmid"])
            article["full_text"] = ft or article.get("abstract", "")

        # LLM schematization
        schematizer = LLMSchematizer()
        logger.info("Schematizing %d articles...", len(useful_articles))
        structured_records = schematizer.schematize_batch(useful_articles)

        for i, record in enumerate(structured_records):
            record["pmid"] = useful_articles[i].get("pmid", "")
            record["title"] = useful_articles[i].get("title", "")
            record["year"] = useful_articles[i].get("year", 0)

        # KG enrichment
        enricher = KGEnricher()
        logger.info("Enriching %d records with KG features...", len(structured_records))
        enriched = []
        for record in structured_records:
            try:
                enriched.append(enricher.enrich_record(record))
            except Exception as e:
                logger.warning("KG enrichment failed for %s: %s", record.get("pmid"), e)
                record["kg_features"] = {"concepts": [], "subgraph_embedding": [], "concept_count": 0}
                enriched.append(record)

        # Extraction benchmark
        if self.cfg.BENCHMARK_EXTRACTION and len(enriched) >= 5:
            sample_size = min(self.cfg.MANUAL_EXTRACTION_SAMPLE_SIZE, len(enriched))
            sample_articles = useful_articles[:sample_size]
            # Use existing structured records as pseudo-gold-standard for self-consistency check
            pseudo_gold = structured_records[:sample_size]
            bench = ExtractionBenchmark()
            ext_results = bench.run(sample_articles, pseudo_gold)
            tracker.log("pipeline1", "extraction_benchmark", ext_results)

        # Save outputs
        self._save_outputs(enriched)
        tracker.save()
        logger.info("Pipeline 1 complete. %d records saved.", len(enriched))
        return enriched

    def _save_outputs(self, records: List[dict]):
        out_dir = Path(self.cfg.DIRS["outputs"])
        jsonl_path = out_dir / f"pipeline1_{self.cfg.RUN_ID}.jsonl"
        csv_path = out_dir / f"pipeline1_{self.cfg.RUN_ID}.csv"

        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        flat_records = [_flatten_record(r) for r in records]
        if flat_records:
            fieldnames = list(flat_records[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(flat_records)

        logger.info("Saved: %s", jsonl_path)
        logger.info("Saved: %s", csv_path)


def _heuristic_label(abstract: str) -> int:
    text = abstract.lower()
    positive_terms = [
        "case report", "we report", "we present", "a patient", "the patient",
        "case presentation", "n-of-1", "individual patient", "treated with",
        "year-old", "yo ", "y.o.", "a man", "a woman", "male patient", "female patient",
    ]
    return 1 if any(t in text for t in positive_terms) else 0


def _flatten_record(record: dict) -> dict:
    flat = {
        "pmid": record.get("pmid", ""),
        "title": record.get("title", ""),
        "year": record.get("year", ""),
        "drug": record.get("drug", ""),
        "disease": record.get("disease", ""),
    }
    outcome = record.get("outcome", {}) or {}
    flat["outcome_response"] = outcome.get("response", "")
    flat["outcome_confidence"] = outcome.get("confidence", "")
    flat["outcome_duration_months"] = outcome.get("duration_months", "")

    pc = record.get("patient_context", {}) or {}
    demo = pc.get("demographics", {}) or {}
    flat["age"] = demo.get("age", "")
    flat["sex"] = demo.get("sex", "")

    kg = record.get("kg_features", {}) or {}
    flat["kg_concept_count"] = kg.get("concept_count", 0)

    return flat
