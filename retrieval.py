import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_EFETCH = f"{EUTILS_BASE}/efetch.fcgi"
ESEARCH = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH = f"{EUTILS_BASE}/efetch.fcgi"


def _ncbi_params(extra: dict) -> dict:
    from config import ENTREZ_EMAIL, ENTREZ_API_KEY
    p = {"email": ENTREZ_EMAIL, "tool": "clinical_pipelines_v2"}
    if ENTREZ_API_KEY:
        p["api_key"] = ENTREZ_API_KEY
    p.update(extra)
    return p


def _with_backoff(fn, retries: int = 4, base_delay: float = 1.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, delay)
            time.sleep(delay)


class PubMedRetriever:
    def __init__(self, cache_dir: str = "cache/pubmed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._search_cache = self.cache_dir / "search_cache.json"
        self._search_index: dict = {}
        if self._search_cache.exists():
            with open(self._search_cache) as f:
                self._search_index = json.load(f)

    def search(self, query: str, max_results: int = 1000) -> List[str]:
        cache_key = f"{query}|{max_results}"
        if cache_key in self._search_index:
            return self._search_index[cache_key]

        pmids = []
        batch = 500
        for start in range(0, max_results, batch):
            retmax = min(batch, max_results - start)
            params = _ncbi_params({
                "db": "pubmed",
                "term": query,
                "retmax": retmax,
                "retstart": start,
                "retmode": "json",
                "usehistory": "n",
            })

            def _do():
                r = requests.get(ESEARCH, params=params, timeout=30)
                r.raise_for_status()
                return r.json()

            data = _with_backoff(_do)
            batch_ids = data.get("esearchresult", {}).get("idlist", [])
            pmids.extend(batch_ids)
            if len(batch_ids) < retmax:
                break
            time.sleep(0.34)

        self._search_index[cache_key] = pmids
        with open(self._search_cache, "w") as f:
            json.dump(self._search_index, f)
        return pmids

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        results = []
        uncached = []

        for pmid in pmids:
            cache_file = self.cache_dir / f"abstract_{pmid}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    results.append(json.load(f))
            else:
                uncached.append(pmid)

        batch_size = 200
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            params = _ncbi_params({
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            })

            def _do():
                r = requests.get(EFETCH, params=params, timeout=60)
                r.raise_for_status()
                return r.text

            xml_text = _with_backoff(_do)
            parsed = _parse_pubmed_xml(xml_text)

            for record in parsed:
                pmid = record.get("pmid", "")
                cache_file = self.cache_dir / f"abstract_{pmid}.json"
                with open(cache_file, "w") as f:
                    json.dump(record, f)
                results.append(record)

            time.sleep(0.34)

        return results

    def fetch_full_text(self, pmid: str) -> Optional[str]:
        cache_file = self.cache_dir / f"fulltext_{pmid}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        # Convert PMID to PMCID first
        params = _ncbi_params({
            "db": "pmc",
            "linkname": "pubmed_pmc",
            "id": pmid,
            "retmode": "json",
        })
        try:
            def _link():
                r = requests.get(f"{EUTILS_BASE}/elink.fcgi", params=params, timeout=20)
                r.raise_for_status()
                return r.json()

            link_data = _with_backoff(_link)
            linksets = link_data.get("linksets", [])
            pmc_ids = []
            for ls in linksets:
                for lsd in ls.get("linksetdbs", []):
                    if lsd.get("linkname") == "pubmed_pmc":
                        pmc_ids = lsd.get("links", [])
                        break

            if not pmc_ids:
                return None

            pmc_id = str(pmc_ids[0])
            fetch_params = _ncbi_params({
                "db": "pmc",
                "id": pmc_id,
                "rettype": "full",
                "retmode": "text",
            })

            def _fetch():
                r = requests.get(PMC_EFETCH, params=fetch_params, timeout=60)
                r.raise_for_status()
                return r.text

            text = _with_backoff(_fetch)
            cache_file.write_text(text, encoding="utf-8")
            return text

        except Exception as e:
            logger.warning("Full text fetch failed for PMID %s: %s", pmid, e)
            return None


def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    import xml.etree.ElementTree as ET
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return records

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(el.itertext()) for el in abstract_parts)

        year_el = article.find(".//PubDate/Year")
        year = int(year_el.text) if year_el is not None and year_el.text else 0

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
        })

    return records
