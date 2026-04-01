import os
from datetime import datetime

RANDOM_SEED = 42
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

NCATS_API_URL = "https://name-resolution-sri.renci.org/lookup"
NAME_RESOLVER = os.getenv("NAME_RESOLVER", "ncats")

EXTERNAL_KG_SOURCE = os.getenv("EXTERNAL_KG_SOURCE", "mondo")
MONARCH_API_URL = "https://api.monarchinitiative.org/api/"

P1_PUBMED_QUERY = '("case report" OR "case presentation" OR "N-of-1") AND (patient OR case) AND treatment'
P1_TARGET_ABSTRACTS = 10000
P1_MAX_RESULTS_PER_QUERY = 1000
P1_USEFUL_TARGET = 500
P1_MIN_YEAR = 2020
P1_CLASSIFICATION_TYPE = "binary"
P1_SCHEMA = {
    "drug": None,
    "disease": None,
    "patient_context": {
        "demographics": {"age": None, "sex": None, "race": None},
        "genomics": {"mutations": [], "gene_expression": {}, "sequencing_type": None},
        "labs": {},
        "patient_factors": {"prior_treatments": [], "comorbidities": [], "ECOG": None}
    },
    "outcome": {"response": None, "confidence": None, "duration_months": None}
}

BENCHMARK_PARTITIONING = True
BENCHMARK_EXTRACTION = True
MANUAL_EXTRACTION_SAMPLE_SIZE = 50

P2_TARGET_TRIALS = 500
P2_MIN_YEAR = 2010
CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
P2_FAILURE_STATUSES = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
P2_FAILURE_CATEGORIES = {
    "safety": ["adverse event", "toxicity", "safety concern", "death"],
    "efficacy": ["lack of efficacy", "futility", "no benefit", "failed endpoint"],
    "enrollment": ["slow enrollment", "recruitment", "accrual"],
    "administrative": ["funding", "sponsor decision", "business"],
    "other": ["covid", "protocol", "regulatory"]
}

INTERACTIVE = os.getenv("INTERACTIVE", "true").lower() == "true"
USE_OPENAI = bool(OPENAI_API_KEY)

DIRS = {
    "cache": "cache",
    "embeddings": "embeddings_cache",
    "knowledge_graph": "knowledge_graph",
    "outputs": "outputs",
    "models": "models",
    "metrics": "performance_tracking"
}
