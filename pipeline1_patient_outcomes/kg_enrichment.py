import logging
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class KGEnricher:
    def __init__(self):
        from config import MONARCH_API_URL, NAME_RESOLVER
        from shared_utilities.utils import get_resolver, EmbeddingGenerator
        self.monarch_url = MONARCH_API_URL
        self.resolver = get_resolver(NAME_RESOLVER)
        self.embedder = EmbeddingGenerator()

    def enrich_record(self, record: dict) -> dict:
        concepts = []

        drug = record.get("drug")
        disease = record.get("disease")

        if drug:
            drug_concepts = self.query_concept(drug)
            concepts.extend(drug_concepts.get("related", []))

        if disease:
            disease_concepts = self.query_concept(disease)
            concepts.extend(disease_concepts.get("related", []))

        # Fallback: if external KG returned nothing, seed from extracted record fields
        if not concepts:
            if drug:
                concepts.append({"label": drug, "type": "drug", "source": "extracted"})
            if disease:
                concepts.append({"label": disease, "type": "disease", "source": "extracted"})
            genomics = record.get("patient_context", {}).get("genomics", {})
            for mutation in (genomics.get("mutations") or []):
                if mutation:
                    concepts.append({"label": mutation, "type": "mutation", "source": "extracted"})

        concepts = self.filter_mutually_exclusive(concepts, record)
        concepts = self.pare_down(concepts, max_concepts=50)

        embedding = self.embed_subgraph(concepts) if concepts else np.zeros(1536, dtype=np.float32)

        record["kg_features"] = {
            "concepts": concepts,
            "subgraph_embedding": embedding.tolist(),
            "concept_count": len(concepts),
        }
        return record

    def query_concept(self, concept: str) -> dict:
        resolved = self.resolver.resolve(concept)
        curie = resolved.get("curie", "")

        if not curie:
            return {"curie": "", "label": concept, "related": []}

        related = self._monarch_neighbors(curie)
        return {"curie": curie, "label": resolved.get("label", concept), "related": related}

    def _monarch_neighbors(self, curie: str) -> list:
        try:
            # Monarch v3 API (old /api/association/from/{curie} endpoint returns 404)
            url = "https://api.monarchinitiative.org/v3/api/association"
            resp = requests.get(url, params={"subject": curie, "limit": 20}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            results = []
            for item in items:
                label = item.get("object_label", "")
                curie_obj = item.get("object", "")
                if label:
                    results.append({"label": label, "curie": curie_obj, "score": 1.0})
            return results
        except Exception as e:
            logger.warning("Monarch query failed for %s: %s", curie, e)
            return []

    def filter_mutually_exclusive(self, concepts: list, patient_record: dict) -> list:
        """Remove concepts that conflict with known patient state."""
        known_mutations = set()
        genomics = patient_record.get("patient_context", {}).get("genomics", {})
        for m in genomics.get("mutations", []):
            if isinstance(m, str):
                known_mutations.add(m.upper())

        filtered = []
        for concept in concepts:
            label = concept.get("label", "") if isinstance(concept, dict) else str(concept)
            # Skip wild-type concepts when patient has known mutations
            if "wild-type" in label.lower() and known_mutations:
                continue
            filtered.append(concept)

        return filtered

    def pare_down(self, concepts: list, max_concepts: int = 50) -> list:
        if len(concepts) <= max_concepts:
            return concepts
        # Sort by score descending, keep top-N
        def _score(c):
            return c.get("score", 0.0) if isinstance(c, dict) else 0.0

        return sorted(concepts, key=_score, reverse=True)[:max_concepts]

    def embed_subgraph(self, concepts: list) -> np.ndarray:
        labels = []
        for c in concepts:
            if isinstance(c, dict):
                labels.append(c.get("label", ""))
            else:
                labels.append(str(c))
        labels = [l for l in labels if l]
        if not labels:
            from config import EMBEDDING_DIM
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vectors = self.embedder.generate(labels)
        return vectors.mean(axis=0)
