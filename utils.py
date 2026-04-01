import os
import csv
import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

NCATS_CONFIDENCE_THRESHOLD = 0.7


class NameResolver(ABC):
    @abstractmethod
    def resolve(self, name: str, categories: Optional[List[str]] = None) -> dict:
        """Resolve a name to a normalized entity dict."""


class NCATSResolver(NameResolver):
    def __init__(self, api_url: Optional[str] = None, cache_dir: str = "cache/ncats"):
        from config import NCATS_API_URL
        self.api_url = api_url or NCATS_API_URL
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, name: str, categories: Optional[List[str]]) -> Path:
        key = f"{name}|{','.join(categories or [])}"
        digest = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def resolve(self, name: str, categories: Optional[List[str]] = None) -> dict:
        cache_file = self._cache_path(name, categories)
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        params = {"string": name, "limit": 5}
        if categories:
            params["biolink_type"] = categories[0]

        result = {}
        try:
            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                top = data[0]
                score = top.get("score", 0.0)
                result = {
                    "curie": top.get("curie", ""),
                    "label": top.get("label", name),
                    "score": score,
                    "confident": score >= NCATS_CONFIDENCE_THRESHOLD,
                    "raw": top,
                }
        except Exception as e:
            logger.warning("NCATSResolver failed for '%s': %s", name, e)
            result = {"curie": "", "label": name, "score": 0.0, "confident": False}

        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result


class NameResolverPlus(NameResolver):
    def resolve(self, name: str, categories: Optional[List[str]] = None) -> dict:
        raise NotImplementedError("nameresolverplus not yet implemented")


def get_resolver(resolver_type: str = "ncats") -> NameResolver:
    if resolver_type == "ncats":
        return NCATSResolver()
    if resolver_type == "nameresolverplus":
        return NameResolverPlus()
    raise ValueError(f"Unknown resolver type: {resolver_type}")


class EntityLinker:
    def __init__(self, resolver: Optional[NameResolver] = None):
        self.resolver = resolver or get_resolver()

    def link(self, record: dict) -> dict:
        linked = {}
        quality_scores = []

        drug = record.get("drug")
        if drug:
            r = self.resolver.resolve(drug, categories=["biolink:Drug"])
            linked["drug"] = r
            quality_scores.append(r.get("score", 0.0))

        disease = record.get("disease")
        if disease:
            r = self.resolver.resolve(disease, categories=["biolink:Disease"])
            linked["disease"] = r
            quality_scores.append(r.get("score", 0.0))

        record["linked_entities"] = linked
        record["entity_linking_quality"] = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        return record


class EmbeddingGenerator:
    def __init__(self):
        from config import OPENAI_API_KEY, EMBEDDING_MODEL
        self.api_key = OPENAI_API_KEY
        self.model = EMBEDDING_MODEL
        self._tfidf = None
        self._use_tfidf = not bool(self.api_key)

    def generate(self, texts: List[str]) -> np.ndarray:
        if self._use_tfidf or not self.api_key:
            return self._tfidf_fallback(texts)
        return self._openai_embed(texts)

    def _openai_embed(self, texts: List[str]) -> np.ndarray:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            vectors = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(input=batch, model=self.model)
                vectors.extend([d.embedding for d in resp.data])
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            logger.warning("OpenAI embedding failed, falling back to TF-IDF: %s", e)
            return self._tfidf_fallback(texts)

    def _tfidf_fallback(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=1536, ngram_range=(1, 2))
            matrix = self._tfidf.fit_transform(texts)
        else:
            matrix = self._tfidf.transform(texts)
        arr = matrix.toarray().astype(np.float32)
        # Pad or truncate to EMBEDDING_DIM
        from config import EMBEDDING_DIM
        if arr.shape[1] < EMBEDDING_DIM:
            arr = np.pad(arr, ((0, 0), (0, EMBEDDING_DIM - arr.shape[1])))
        else:
            arr = arr[:, :EMBEDDING_DIM]
        return arr


class PerformanceTracker:
    def __init__(self, output_dir: str = "performance_tracking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[dict] = []

    def log(self, pipeline: str, stage: str, metrics: dict):
        from config import RUN_ID
        entry = {"run_id": RUN_ID, "pipeline": pipeline, "stage": stage, **metrics}
        self.records.append(entry)
        logger.info("[%s/%s] %s", pipeline, stage, metrics)

    def save(self, filename: str = "metrics.csv"):
        if not self.records:
            return
        path = self.output_dir / filename
        fieldnames = list(self.records[0].keys())
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(self.records)
        self.records.clear()
