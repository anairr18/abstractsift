import copy
import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _deep_keys(schema: dict) -> set:
    keys = set()
    for k, v in schema.items():
        keys.add(k)
        if isinstance(v, dict):
            keys.update(_deep_keys(v))
    return keys


def _validate_against_schema(data: dict, schema: dict) -> bool:
    if not isinstance(data, dict):
        return False
    for key in schema:
        if key not in data:
            return False
        if isinstance(schema[key], dict) and schema[key]:
            if not _validate_against_schema(data.get(key, {}), schema[key]):
                return False
    return True


class LLMSchematizer:
    SYSTEM_PROMPT = (
        "You are a clinical data extraction engine specializing in drug adverse event case reports. "
        "Extract structured information strictly about the DRUG ADVERSE EVENT described, not the patient's underlying disease being treated. "
        "Return only valid JSON matching the provided schema. "
        "Use null for missing values and empty lists/objects where appropriate. "
        "Critical rules:\n"
        "- drug: the drug that CAUSED the adverse event (not drugs used to treat it)\n"
        "- disease: the adverse event or drug-induced condition (e.g. drug-induced liver injury, cardiotoxicity, hyponatremia)\n"
        "- outcome.response: how the adverse event resolved (e.g. 'resolved after drug withdrawal', 'managed conservatively') — NOT the outcome of the patient's underlying treated disease\n"
        "- outcome.confidence: your confidence in the outcome classification (0.0-1.0)\n"
        "- outcome.duration_months: months until resolution of the adverse event, if stated\n"
        "- labs: extract ALL laboratory values mentioned (include units and reference ranges where given)\n"
        "- demographics: extract age (integer), sex, race if explicitly stated in the text\n"
        "- comorbidities: list ALL comorbid conditions mentioned, not just the primary one"
    )

    def __init__(self):
        from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_MODEL, P1_SCHEMA
        self._anthropic_key = ANTHROPIC_API_KEY
        self._openai_key = OPENAI_API_KEY
        self._model = LLM_MODEL
        self._schema = P1_SCHEMA

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip XML/HTML tags so clinical content is not buried under markup overhead."""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_prompt(self, full_text: str) -> str:
        schema_str = json.dumps(self._schema, indent=2)
        cleaned = self._clean_text(full_text)
        return (
            f"Extract the following fields from this clinical case report about a drug adverse event.\n\n"
            f"IMPORTANT DISAMBIGUATION RULES:\n"
            f"- 'drug' = the drug that CAUSED the adverse event (olmesartan caused liver injury; ketoprofen caused hypersensitivity)\n"
            f"- 'disease' = the drug-induced adverse condition, NOT the patient's pre-existing illness\n"
            f"- 'outcome.response' = how the ADVERSE EVENT resolved (e.g. 'resolved after drug withdrawal', 'managed conservatively') — NOT outcomes of the patient's underlying disease (e.g. cancer response).\n"
            f"- 'outcome.duration_months' = if the text states how many weeks/months until the adverse event resolved, convert to months (e.g. '2 months' = 2)\n"
            f"- 'outcome.confidence' = your confidence 0.0-1.0 that the outcome label is correct based on available evidence\n"
            f"- 'labs' = ALL lab values in the text (AST, ALT, bilirubin, INR, drug levels, electrolytes, WBC, etc.)\n"
            f"- 'demographics.age' = integer age of patient at time of adverse event\n"
            f"- 'comorbidities' = ALL comorbid conditions listed, not just the primary one\n\n"
            f"Schema to fill:\n{schema_str}\n\n"
            f"Case report text:\n{cleaned[:12000]}\n\n"
            f"Return only the filled JSON object."
        )

    def schematize(self, full_text: str) -> dict:
        prompt = self._build_prompt(full_text)
        raw = self._call_llm(prompt)
        parsed = self._parse_json(raw)
        if parsed and _validate_against_schema(parsed, self._schema):
            return parsed
        # Return schema with nulls on validation failure
        logger.warning("LLM extraction failed validation, returning empty schema")
        return copy.deepcopy(self._schema)

    def schematize_batch(self, articles: List[Dict]) -> List[dict]:
        results = []
        for article in articles:
            text = article.get("full_text") or article.get("abstract") or ""
            if not text:
                results.append(copy.deepcopy(self._schema))
                continue
            try:
                results.append(self.schematize(text))
            except Exception as e:
                logger.warning("Schematization failed for article %s: %s", article.get("pmid"), e)
                results.append(copy.deepcopy(self._schema))
        return results

    def _parse_json(self, raw: str) -> Optional[dict]:
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object within the text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def _call_llm(self, prompt: str) -> str:
        if self._anthropic_key:
            return self._call_anthropic(prompt)
        if self._openai_key:
            return self._call_openai(prompt)
        raise RuntimeError("No LLM API key configured for LLMSchematizer")

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._anthropic_key)
        msg = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self._openai_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
