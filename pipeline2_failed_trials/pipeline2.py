import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

import config
from shared_utilities.utils import EntityLinker, get_resolver

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    def __init__(self, cache_dir: str = "cache/clinicaltrials"):
        self.api_url = config.CLINICALTRIALS_API
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_failed_trials(
        self,
        min_year: int = config.P2_MIN_YEAR,
        max_results: int = config.P2_TARGET_TRIALS,
        statuses: List[str] = None,
    ) -> List[str]:
        if statuses is None:
            statuses = config.P2_FAILURE_STATUSES

        nct_ids = []
        seen_ids = set()

        for status in statuses:
            if len(nct_ids) >= max_results:
                break

            params = {
                "query.cond": "",
                "query.term": "",
                "filter.overallStatus": status,
                "filter.advanced": f"AREA[StartDate]RANGE[{min_year}-01-01,MAX]",
                "pageSize": 100,
                "format": "json",
            }
            page_token = None

            while True:
                if page_token:
                    params["pageToken"] = page_token

                try:
                    import requests
                    resp = requests.get(self.api_url, params=params, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        for study in data.get("studies", []):
                            nct_id = (
                                study.get("protocolSection", {})
                                .get("identificationModule", {})
                                .get("nctId")
                            )
                            if nct_id and nct_id not in seen_ids:
                                seen_ids.add(nct_id)
                                nct_ids.append(nct_id)

                        page_token = data.get("nextPageToken")
                        if not page_token or len(nct_ids) >= max_results:
                            break
                    else:
                        break
                    time.sleep(0.5)
                except Exception as e:
                    logger.error("Error searching trials: %s", e)
                    break

        return nct_ids[:max_results]

    def fetch_trial_details(self, nct_id: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{nct_id}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        url = f"{self.api_url}/{nct_id}"
        try:
            import requests
            resp = requests.get(url, params={"format": "json"}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                return data
        except Exception as e:
            logger.error("Error fetching %s: %s", nct_id, e)
        return None


class TrialExtractor:
    PHASE_WEIGHTS = {
        "PHASE1": 1.0, "Phase 1": 1.0,
        "PHASE2": 0.7, "Phase 2": 0.7,
        "PHASE3": 0.4, "Phase 3": 0.4,
        "PHASE4": 0.1, "Phase 4": 0.1,
    }

    _INDUSTRY_KEYWORDS = [
        "pharma", "pharmaceutical", "therapeutics", "biosciences",
        "biotech", "laboratories", "labs", "inc", "ltd", "llc",
        "corporation", "corp", "company", "gmbh", "ag",
    ]
    _ACADEMIC_KEYWORDS = [
        "university", "college", "institute", "hospital",
        "medical center", "cancer center", "clinic",
    ]

    def __init__(self):
        self.failure_categories = config.P2_FAILURE_CATEGORIES

    def categorize_sponsor_type(self, sponsor_name: str) -> str:
        if not sponsor_name:
            return "unknown"
        s = sponsor_name.lower()
        if any(kw in s for kw in self._INDUSTRY_KEYWORDS):
            return "industry"
        if any(kw in s for kw in self._ACADEMIC_KEYWORDS):
            return "academic"
        return "other"

    def extract(self, trial_data: Dict) -> List[Dict]:
        if not trial_data:
            return []

        protocol = trial_data.get("protocolSection", {})

        id_module = protocol.get("identificationModule", {})
        nct_id = id_module.get("nctId")

        status_module = protocol.get("statusModule", {})
        overall_status = status_module.get("overallStatus")
        why_stopped = status_module.get("whyStopped", "")

        arms_module = protocol.get("armsInterventionsModule", {})
        interventions = arms_module.get("interventions", [])
        drugs = [
            i.get("name") for i in interventions
            if i.get("type") in ["DRUG", "BIOLOGICAL"]
        ]

        conditions_module = protocol.get("conditionsModule", {})
        conditions = conditions_module.get("conditions", [])

        design_module = protocol.get("designModule", {})
        phases = design_module.get("phases", [])

        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {}).get("name", "")

        design_info = design_module.get("enrollmentInfo", {})
        enrollment = design_info.get("count", 0)

        start_date = status_module.get("startDateStruct", {}).get("date", "")
        completion_date = status_module.get("completionDateStruct", {}).get("date", "")

        outcomes_module = protocol.get("outcomesModule", {})
        primary_outcomes = outcomes_module.get("primaryOutcomes", [])

        failure_category = self._categorize_failure(why_stopped)
        phase_str = phases[0] if phases else None
        failure_strength = self.PHASE_WEIGHTS.get(phase_str, 0.5)

        if failure_strength >= 0.7:
            failure_interpretation = "strong_negative_signal"
        elif failure_strength >= 0.4:
            failure_interpretation = "moderate_negative_signal"
        else:
            failure_interpretation = "weak_negative_signal"

        sponsor_type = self.categorize_sponsor_type(lead_sponsor)
        if failure_category in ("administrative", "other") and sponsor_type == "industry":
            usable_for_training = False
            exclusion_reason = "likely_business_decision_not_biological"
        else:
            usable_for_training = True
            exclusion_reason = None

        outcome_results = []
        results_section = trial_data.get("resultsSection", {})
        outcome_measures = results_section.get("outcomeMeasuresModule", {})
        for om in outcome_measures.get("outcomeMeasures", []):
            result = {
                "measure": om.get("title"),
                "description": om.get("description"),
                "time_frame": om.get("timeFrame"),
            }
            for group in om.get("groups", []):
                result[group.get("title")] = group.get("value")
            outcome_results.append(result)

        base = {
            "trial_id": nct_id,
            "all_drugs": drugs,
            "all_conditions": conditions,
            "phase": phase_str,
            "status": overall_status,
            "failure_reason": why_stopped,
            "failure_category": failure_category,
            "failure_strength": failure_strength,
            "failure_interpretation": failure_interpretation,
            "enrollment": enrollment,
            "start_date": start_date,
            "completion_date": completion_date,
            "sponsor": lead_sponsor,
            "sponsor_type": sponsor_type,
            "usable_for_training": usable_for_training,
            "exclusion_reason": exclusion_reason,
            "primary_outcomes": [o.get("measure") for o in primary_outcomes],
            "outcome_results": outcome_results if outcome_results else None,
            "source": "ClinicalTrials.gov",
            "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        }

        if not drugs or not conditions:
            return [{**base, "drug": None, "disease": None, "extractable": False}]

        return [
            {**base, "drug": drug, "disease": condition, "extractable": True}
            for drug in drugs
            for condition in conditions
        ]

    def _categorize_failure(self, reason: str) -> str:
        reason_lower = reason.lower()
        for category, keywords in self.failure_categories.items():
            for keyword in keywords:
                if keyword in reason_lower:
                    return category
        return "other"


def _build_kg(records: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for record in records:
        drug = record.get("drug")
        disease = record.get("disease")
        if not drug or not disease:
            continue
        if not G.has_node(drug):
            G.add_node(drug, node_type="drug")
        if not G.has_node(disease):
            G.add_node(disease, node_type="disease")
        G.add_edge(
            drug,
            disease,
            trial_id=record.get("trial_id") or "",
            failure_category=record.get("failure_category") or "",
            failure_strength=record.get("failure_strength") or 0.5,
            phase=record.get("phase") or "",
        )
    return G


class Pipeline2:
    def __init__(self):
        self.client = ClinicalTrialsClient()
        self.extractor = TrialExtractor()
        self.entity_linker = EntityLinker(resolver=get_resolver(config.NAME_RESOLVER), skip_on_failure=False)

    def run(self):
        logger.info("Pipeline 2: Failed Clinical Trials")

        logger.info("Searching ClinicalTrials.gov for failed trials...")
        nct_ids = self.client.search_failed_trials(
            min_year=config.P2_MIN_YEAR,
            max_results=config.P2_TARGET_TRIALS,
            statuses=config.P2_FAILURE_STATUSES,
        )
        logger.info("Found %d failed trials", len(nct_ids))

        logger.info("Extracting trial details...")
        extracted_records = []
        for nct_id in tqdm(nct_ids, desc="Processing trials"):
            from_cache = (self.client.cache_dir / f"{nct_id}.json").exists()
            trial_data = self.client.fetch_trial_details(nct_id)
            if trial_data:
                for record in self.extractor.extract(trial_data):
                    if record["extractable"]:
                        extracted_records.append(record)
            if not from_cache:
                time.sleep(0.5)

        logger.info("Extracted %d trial records", len(extracted_records))

        logger.info("Linking entities...")
        linked_records = []
        for record in tqdm(extracted_records, desc="Entity linking"):
            linked = self.entity_linker.link(record)
            linked_records.append(linked)
        extracted_records = linked_records

        self._save_outputs(extracted_records)
        self._print_summary(extracted_records)
        return extracted_records

    def _to_output_record(self, record: Dict) -> Dict:
        return {
            "drug": record["drug"],
            "disease": record["disease"],
            "failure_reason": record["failure_reason"],
            "failure_category": record["failure_category"],
            "failure_strength": record.get("failure_strength"),
            "failure_interpretation": record.get("failure_interpretation"),
            "phase": record["phase"],
            "sponsor_type": record.get("sponsor_type"),
            "usable_for_training": record.get("usable_for_training", True),
            "exclusion_reason": record.get("exclusion_reason"),
            "trial_id": record["trial_id"],
            "source": record["source_url"],
            "linked_entities": record.get("linked_entities", {}),
            "entity_linking_quality": record.get("entity_linking_quality"),
        }

    def _save_outputs(self, records: List[Dict]):
        Path(config.DIRS["outputs"]).mkdir(parents=True, exist_ok=True)
        Path(config.DIRS["knowledge_graph"]).mkdir(parents=True, exist_ok=True)

        jsonl_path = f"{config.DIRS['outputs']}/failed_trials_{config.RUN_ID}.jsonl"
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(self._to_output_record(record)) + "\n")

        kg = _build_kg(records)
        kg_path = f"{config.DIRS['knowledge_graph']}/trials_kg_{config.RUN_ID}.graphml"
        nx.write_graphml(kg, kg_path)

        df = pd.DataFrame(records)
        csv_path = f"{config.DIRS['outputs']}/failed_trials_{config.RUN_ID}.csv"
        df.to_csv(csv_path, index=False)

        logger.info("Saved: %s", jsonl_path)
        logger.info("Saved: %s", kg_path)
        logger.info("Saved: %s", csv_path)

    def _print_summary(self, records: List[Dict]):
        if not records:
            print("No records to summarize.")
            return

        df = pd.DataFrame(records)
        print(f"\nTotal records: {len(df)}")
        print("\nBy failure category:")
        print(df["failure_category"].value_counts().to_string())
        print("\nBy phase:")
        print(df["phase"].value_counts().to_string())
        print("\nTop diseases:")
        print(df["disease"].value_counts().head(10).to_string())
        print("\nTop drugs:")
        print(df["drug"].value_counts().head(10).to_string())
