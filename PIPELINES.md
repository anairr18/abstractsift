# Clinical Pipelines — Architecture Overview

## What These Pipelines Do

Both pipelines feed a drug repurposing model that learns:

> **patient_context + drug + disease → outcome**

The goal is to accumulate high-quality structured training examples so a downstream ML model can predict whether a patient with specific biological characteristics will respond to a given drug.

---

## Pipeline 1: Individual Patient Outcomes

Extracts individual patient treatment outcomes from medical literature and enriches them with external biomedical knowledge graph context.

### How it works (4-stage flow)

**Stage 1 — Retrieval**
Runs a broad PubMed query targeting case reports and N-of-1 studies. Fetches up to 10,000 abstracts with per-PMID disk caching so re-runs skip already-fetched records.

**Stage 2 — Partitioning (Track 1, RF binary classifier)**
Filters the 10k abstracts down to ~500 useful papers. Three methods are trained and benchmarked head-to-head; the best by F1 score is used for final prediction:

| Method | Features | Loss |
|--------|----------|------|
| RF + TF-IDF | 500-feature unigram/bigram TF-IDF | binary cross-entropy |
| RF + Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | binary cross-entropy |
| LLM classifier | Direct yes/no prompt (Claude / GPT-4o) | — |

Label: `1` = useful case report, `0` = not useful. A heuristic pass pre-labels the corpus before classifier training.

**Stage 3 — Schematization (Track 2, LLM extraction)**
For each paper classified as useful, fetches full text from PMC and prompts an LLM to extract structured JSON matching this schema:

```json
{
  "drug": "pembrolizumab",
  "disease": "melanoma",
  "patient_context": {
    "demographics": { "age": 65, "sex": "male", "race": "caucasian" },
    "genomics": { "mutations": ["BRAF V600E"], "gene_expression": {"PD-L1": "high"}, "sequencing_type": "WGS" },
    "labs": { "LDH": {"value": 450, "unit": "U/L"} },
    "patient_factors": { "prior_treatments": ["ipilimumab"], "comorbidities": ["hypertension"], "ECOG": 1 }
  },
  "outcome": { "response": "CR", "confidence": 1.0, "duration_months": 18 }
}
```

**Stage 4 — KG Enrichment**
For each structured record, queries external ontologies (Monarch Initiative / MONDO / HPO) via CURIE IDs:
1. Resolves drug and disease names to CURIE identifiers via NCATS
2. Pulls related concepts from the Monarch Initiative graph
3. Filters out mutually exclusive concepts (e.g., conflicting gene variant alleles)
4. Pares subgraph to top-50 concepts by relevance score
5. Mean-pools embeddings of the subgraph into a `kg_features` vector

**Output:** JSONL of structured patient records + `kg_features` arrays ready for ML training.

### Benchmarking

Two benchmark suites run automatically when enabled in config:

- **Partitioning benchmark** — compares all three partitioner methods on a held-out split, reports precision / recall / F1 per method.
- **Extraction benchmark** — compares LLM-extracted fields against manual gold-standard annotations for `MANUAL_EXTRACTION_SAMPLE_SIZE` articles. Reports per-field accuracy.

---

## Pipeline 2: Failed Clinical Trials

Extracts structured records of failed drug trials from ClinicalTrials.gov as negative training signal for the drug repurposing model.

### How it works

1. Queries ClinicalTrials.gov API v2 for trials with status `TERMINATED`, `WITHDRAWN`, or `SUSPENDED` since `P2_MIN_YEAR`.
2. Fetches full trial details for each NCT ID (disk-cached).
3. Extracts per-(drug, disease) records — combination therapy trials produce one record per drug.
4. Each record includes:
   - **failure_category**: safety / efficacy / enrollment / administrative / other
   - **failure_strength**: weighted by phase (Phase 1 failure = 1.0, Phase 4 = 0.1) — earlier failure is a stronger negative signal
   - **failure_interpretation**: strong / moderate / weak negative signal
   - **usable_for_training**: False if the trial was likely stopped for business reasons rather than biological ones
   - **sponsor_type**: industry / academic / other
5. Links drug and disease to NCATS canonical identifiers.
6. Saves JSONL + CSV + GraphML knowledge graph.

**Output:** JSONL of failed trial records tagged with failure category, strength, and entity links.

---

## Shared Infrastructure

**`shared_utilities/utils.py`**

- `NameResolver` (ABC) → `NCATSResolver` → `NameResolverPlus` (stub for future replacement)
- `get_resolver(resolver_type)` factory — swap NCATS for `nameresolverplus` by changing one env var: `NAME_RESOLVER=nameresolverplus`
- `EntityLinker` — resolves drug/disease fields, flags low-confidence resolutions
- `EmbeddingGenerator` — OpenAI embeddings with TF-IDF fallback when no API key is set
- `PerformanceTracker` — appends metrics rows to a CSV per run

---

## What Changed from v1

| Area | v1 (clinical_pipelines_complete) | v2 (this version) |
|------|---------------------------------|-------------------|
| Classification | 4-way (single_patient / cohort / preclinical / adverse_event) | Binary (useful / not_useful) |
| Features for classifier | TF-IDF only | TF-IDF, OpenAI embeddings, or direct LLM — benchmarked |
| PubMed queries | 14 narrowly targeted queries | One broad query, ~10k abstracts |
| Extraction input | Abstract text only | Full text fetched from PMC |
| Extraction method | Regex rules | LLM with structured JSON schema |
| Knowledge graph | Built from scratch in-memory (NetworkX) | Queries external KGs (Monarch / MONDO / HPO) |
| KG output | Local GraphML of extracted entities | Per-record subgraph embeddings as `kg_features` |
| Entity resolution | Direct NCATSResolver import | Abstract `NameResolver` base class; swap via `get_resolver()` factory |
| Output schema | Flat fields + linked_entities | Nested `patient_context` with demographics, genomics, labs, patient_factors |
| API keys | Hardcoded in source | `os.getenv()` only |
| Benchmarking | None | Built-in partitioning + extraction benchmark harness |

### Key design decisions explained

**Binary vs. 4-way classification** — The original 4-way scheme (single_patient / cohort / preclinical / adverse_event) mixed the classification goal. The only question that matters for downstream training is: *does this paper contain an individual patient treatment outcome?* Binary labeling is simpler to label, easier to calibrate, and produces a cleaner training set.

**Embeddings over TF-IDF** — TF-IDF misses semantic similarity (e.g., "pembrolizumab" and "anti-PD-1 therapy" treated as unrelated). Embeddings capture this; the benchmark harness quantifies whether the improvement justifies the API cost.

**External KG over local** — Building a KG from extracted entities creates a closed world with no background biological knowledge. Querying MONDO/HPO/Monarch gives the model access to established ontological relationships (gene→pathway→disease) without manual curation.

**Abstract NameResolver base class** — The NCATS API is a stopgap; `nameresolverplus` (in development) will replace it. The factory pattern means switching is a one-line config change rather than a code refactor.
