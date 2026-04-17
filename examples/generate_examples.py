"""
Generate 3-5 example patient outcome records from known good case report PMIDs.

Reads from the existing PubMed cache in clinical_pipelines_complete — no new API
calls are made for cached PMIDs. Full text is fetched from PMC (network needed).

Run from the clinical_pipelines_v2 directory:
    python examples/generate_examples.py
"""

import json
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Case report PMIDs confirmed from cache:
#   32015975 - Olmesartan-induced autoimmune hepatitis (drug: olmesartan, disease: hepatitis)
#   32079429 - Ketoprofen hypersensitivity (drug: ketoprofen, disease: hypersensitivity)
#   32300505 - Lithium-induced cardiotoxicity (drug: lithium, disease: bipolar)
#   32594911 - Omeprazole-induced hyponatremia (drug: omeprazole, disease: hyponatremia)
EXAMPLE_PMIDS = ["32015975", "32079429", "32300505", "32594911"]

# Reuse the existing cache from the v1 run rather than re-fetching
V1_CACHE_DIR = str(
    Path(__file__).parent.parent.parent
    / "clinical_pipelines_complete"
    / "clinical_pipelines"
    / "cache"
    / "pubmed"
)


def main():
    from pipeline1_patient_outcomes.retrieval import PubMedRetriever
    from pipeline1_patient_outcomes.schematizer import LLMSchematizer
    from pipeline1_patient_outcomes.kg_enrichment import KGEnricher
    from shared_utilities.utils import EntityLinker

    retriever = PubMedRetriever(cache_dir=V1_CACHE_DIR)
    schematizer = LLMSchematizer()
    enricher = KGEnricher()
    entity_linker = EntityLinker()

    logger.info("Fetching %d abstracts (from cache where available)...", len(EXAMPLE_PMIDS))
    abstracts = retriever.fetch_abstracts(EXAMPLE_PMIDS)
    abstracts = [a for a in abstracts if a.get("abstract")]
    logger.info("Got %d abstracts with text", len(abstracts))

    logger.info("Fetching full text from PMC (may make network calls)...")
    for article in abstracts:
        ft = retriever.fetch_full_text(article["pmid"])
        article["full_text"] = ft or article.get("abstract", "")
        logger.info("  PMID %s: %d chars", article["pmid"], len(article["full_text"]))

    logger.info("Running LLM schematization...")
    structured = schematizer.schematize_batch(abstracts)

    for i, record in enumerate(structured):
        record["pmid"] = abstracts[i].get("pmid", "")
        record["title"] = abstracts[i].get("title", "")
        entity_linker.link(record)
        enricher.enrich_record(record)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_examples.jsonl"

    with open(all_path, "w") as f:
        for record in structured:
            f.write(json.dumps(record) + "\n")

    print("\n" + "=" * 70)
    for record in structured:
        print(f"\nPMID: {record.get('pmid')}  |  {record.get('title', '')[:80]}")
        print(json.dumps(record, indent=2))
        print("=" * 70)

    logger.info("Saved %d records → %s", len(structured), all_path)


if __name__ == "__main__":
    main()
