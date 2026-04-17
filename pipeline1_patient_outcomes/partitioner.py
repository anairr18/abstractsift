import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


class TFIDFPartitioner:
    def __init__(self, model_path: str = "models/tfidf_partitioner.pkl"):
        self.model_path = Path(model_path)
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._fitted = False

    def train(self, texts: List[str], labels: List[int]) -> dict:
        X = self.vectorizer.fit_transform(texts)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.clf.fit(X_tr, y_tr)
        self._fitted = True
        self.save()
        preds = self.clf.predict(X_te)
        return _metrics(y_te, preds)

    def predict(self, texts: List[str]) -> List[int]:
        if not self._fitted:
            self.load()
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X).tolist()

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vectorizer, self.clf), f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.vectorizer, self.clf = pickle.load(f)
        self._fitted = True


class EmbeddingPartitioner:
    def __init__(self, model_path: str = "models/embedding_partitioner.pkl"):
        self.model_path = Path(model_path)
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._fitted = False
        from shared_utilities.utils import EmbeddingGenerator
        self.embedder = EmbeddingGenerator()

    def train(self, texts: List[str], labels: List[int]) -> dict:
        X = self.embedder.generate(texts)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.clf.fit(X_tr, y_tr)
        self._fitted = True
        self.save()
        preds = self.clf.predict(X_te)
        return _metrics(y_te, preds)

    def predict(self, texts: List[str]) -> List[int]:
        if not self._fitted:
            self.load()
        X = self.embedder.generate(texts)
        return self.clf.predict(X).tolist()

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.clf, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.clf = pickle.load(f)
        self._fitted = True


class LLMPartitioner:
    PROMPT_TEMPLATE = (
        "Is this a case report of an individual patient's treatment and outcome? "
        "Answer yes or no only.\n\n{abstract}"
    )

    def __init__(self):
        from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
        self._anthropic_key = ANTHROPIC_API_KEY
        self._openai_key = OPENAI_API_KEY

    def predict_single(self, text: str) -> int:
        prompt = self.PROMPT_TEMPLATE.format(abstract=text)
        answer = self._call_llm(prompt).strip().lower()
        return 1 if answer.startswith("yes") else 0

    def predict(self, texts: List[str]) -> List[int]:
        results = []
        for text in texts:
            try:
                results.append(self.predict_single(text))
            except Exception as e:
                logger.warning("LLMPartitioner failed on sample: %s", e)
                results.append(0)
        return results

    def evaluate(self, texts: List[str], labels: List[int]) -> dict:
        _, texts_te, _, labels_te = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        preds = self.predict(texts_te)
        return _metrics(labels_te, preds)

    def _call_llm(self, prompt: str) -> str:
        if self._anthropic_key:
            return self._call_anthropic(prompt)
        if self._openai_key:
            return self._call_openai(prompt)
        raise RuntimeError("No LLM API key configured for LLMPartitioner")

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        from config import LLM_MODEL
        client = anthropic.Anthropic(api_key=self._anthropic_key)
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self._openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class BinaryPartitioner:
    def __init__(self):
        self.tfidf = TFIDFPartitioner()
        self.embedding = EmbeddingPartitioner()
        self.llm = LLMPartitioner()
        self._best: Optional[str] = None
        self._best_model = None

    def train(self, texts: List[str], labels: List[int]) -> Dict[str, dict]:
        results = {}

        logger.info("Training TFIDFPartitioner...")
        results["tfidf"] = self.tfidf.train(texts, labels)

        logger.info("Training EmbeddingPartitioner...")
        results["embedding"] = self.embedding.train(texts, labels)

        logger.info("Evaluating LLMPartitioner (test split only)...")
        try:
            results["llm"] = self.llm.evaluate(texts, labels)
        except Exception as e:
            logger.warning("LLMPartitioner evaluation skipped: %s", e)
            results["llm"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        best_name = max(results, key=lambda k: results[k]["f1"])
        self._best = best_name
        self._best_model = {"tfidf": self.tfidf, "embedding": self.embedding, "llm": self.llm}[
            best_name
        ]
        logger.info("Best partitioner: %s (F1=%.3f)", best_name, results[best_name]["f1"])
        return results

    def predict(self, abstracts: List[str]) -> List[bool]:
        if self._best_model is None:
            self._best_model = self.tfidf
        preds = self._best_model.predict(abstracts)
        return [bool(p) for p in preds]
