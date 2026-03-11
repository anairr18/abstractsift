import subprocess
import sys

# Install dependencies
packages = [
    'biopython',
    'pandas',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'requests',
    'tqdm',
    'openpyxl',
    'scipy'
]

import os
INSTALL_DEPS = os.getenv("INSTALL_DEPS", "true").lower() == "true"

if INSTALL_DEPS:
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import json
import pickle
import time
import re
import warnings
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests
from Bio import Entrez
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, recall_score, 
    precision_score, f1_score, precision_recall_curve, auc
)

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

print("\n[1/18] Configuration")

ENTREZ_EMAIL = "anair@utexas.edu"
Entrez.email = ENTREZ_EMAIL
Entrez.tool = "ClinicalOutcomesDB"
Entrez.api_key = "your api key"

OPENAI_API_KEY = "your api key"
USE_OPENAI = False

if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
    try:
        test = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5},
            timeout=10
        )
        USE_OPENAI = (test.status_code == 200)
    except:
        USE_OPENAI = False

TARGET_EXTRACTION_COUNT = 250
FULL_PROJECT_TARGET = 625
MAX_RESULTS_PER_QUERY = 150
TARGET_LABELS = 200
MIN_POSITIVE_RATE = 0.35
TARGET_POSITIVE_RATE = 0.50
MIN_YEAR = 2020
DATE_RANGE = f"{MIN_YEAR}:{datetime.now().year}"
DEFAULT_THRESHOLD = 0.15
EXTRACTION_THRESHOLD = 0.50
EXTRACTION_MIN_CANDIDATES = 50
RECALL_REQUIREMENT = 0.95
INTERACTIVE = os.getenv("INTERACTIVE", "true").lower() == "true"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Email: {ENTREZ_EMAIL}")
print(f"Target labels: {TARGET_LABELS}, extractions: {TARGET_EXTRACTION_COUNT}")
print(f"Date range: {DATE_RANGE}, Interactive: {INTERACTIVE}")

# ============================================================================
# SECTION 2: QUERY STRATEGIES
# ============================================================================

print("\n[2/18] Query Strategies")

QUERY_STRATEGIES = {
    "age_specific_cases": (
        f'("year-old" OR "years old" OR "y/o") AND (diagnosed OR treated) '
        f'AND (mutation OR genetic OR biomarker) AND {DATE_RANGE}[Date - Publication]'
    ),
    "oncogene_specific": (
        f'(BRAF OR EGFR OR KRAS OR ALK OR ROS1 OR "HER2" OR BRCA) AND (patient OR case) '
        f'AND (treated OR therapy OR inhibitor) AND {DATE_RANGE}[Date - Publication]'
    ),
    "complete_response": (
        f'("complete response" OR "complete remission" OR "dramatic response") '
        f'AND (patient OR case) AND (cancer OR tumor) AND {DATE_RANGE}[Date - Publication]'
    ),
    "exceptional_responders": (
        f'("exceptional responder" OR "outlier response" OR "durable response") '
        f'AND (cancer OR tumor) AND (patient OR case) AND {DATE_RANGE}[Date - Publication]'
    ),
    "genomic_case_reports": (
        f'("whole genome sequencing" OR "WGS" OR "NGS") AND (patient OR case) '
        f'AND (treatment OR therapy) AND {DATE_RANGE}[Date - Publication]'
    ),
    "precision_medicine": (
        f'("precision medicine" OR "personalized medicine" OR "targeted therapy") '
        f'AND (mutation OR biomarker) AND (patient OR case) AND {DATE_RANGE}[Date - Publication]'
    ),
    "immunotherapy_cases": (
        f'(immunotherapy OR "checkpoint inhibitor" OR pembrolizumab OR nivolumab) '
        f'AND (patient OR case) AND (response OR outcome) AND {DATE_RANGE}[Date - Publication]'
    ),
    "rheumatoid_arthritis_cases": (
        f'("rheumatoid arthritis" OR "RA") AND (patient OR case) '
        f'AND (treatment OR remission OR flare OR outcome) AND {DATE_RANGE}[Date - Publication]'
    ),
    "systemic_lupus_cases": (
        f'("systemic lupus erythematosus" OR "SLE" OR lupus) AND (patient OR case) '
        f'AND (treatment OR outcome OR presentation) AND {DATE_RANGE}[Date - Publication]'
    ),
    "psoriatic_arthritis_cases": (
        f'("psoriatic arthritis" OR "PsA") AND (patient OR case) '
        f'AND (response OR outcome OR treatment) AND {DATE_RANGE}[Date - Publication]'
    ),
    "ibd_cases": (
        f'("inflammatory bowel disease" OR "IBD" OR "Crohn\'s disease" OR "ulcerative colitis") '
        f'AND (patient OR case) AND (response OR outcome OR treatment) AND {DATE_RANGE}[Date - Publication]'
    ),
    "multiple_sclerosis_cases": (
        f'("multiple sclerosis" OR "MS") AND (patient OR case) '
        f'AND (relapse OR progression OR treatment) AND {DATE_RANGE}[Date - Publication]'
    ),
    "ankylosing_spondylitis_cases": (
        f'("ankylosing spondylitis" OR "AS") AND (patient OR case) '
        f'AND (response OR outcome OR treatment) AND {DATE_RANGE}[Date - Publication]'
    ),
}

print(f"{len(QUERY_STRATEGIES)} query strategies defined")

# ============================================================================
# SECTION 3: PUBMED RETRIEVAL WITH CACHING
# ============================================================================

print("\n[3/18] PubMed Retrieval")

CACHE_DIR = "pubmed_cache"
QUERY_CACHE_FILE = "query_cache.json"
os.makedirs(CACHE_DIR, exist_ok=True)

query_cache = {}
if os.path.exists(QUERY_CACHE_FILE):
    with open(QUERY_CACHE_FILE, 'r') as f:
        query_cache = json.load(f)
    print(f"Loaded {len(query_cache)} cached queries")

def exponential_backoff(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + np.random.uniform(0, 1)
            time.sleep(wait_time)

def search_pubmed(query, max_results=200):
    query_key = f"{query.strip()}_{max_results}"
    
    if query_key in query_cache:
        return query_cache[query_key]['pmids'], query_cache[query_key]['total']
    
    def _search():
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance", usehistory="y")
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"], int(record["Count"])
    
    pmids, total = exponential_backoff(_search)
    query_cache[query_key] = {'pmids': pmids, 'total': total}
    with open(QUERY_CACHE_FILE, 'w') as f:
        json.dump(query_cache, f)
    
    return pmids, total

def fetch_abstracts(pmid_list, batch_size=100):
    abstracts = []
    
    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i:i+batch_size]
        cached, uncached = [], []
        
        for pmid in batch:
            cache_file = os.path.join(CACHE_DIR, f"{pmid}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached.append(json.load(f))
                except:
                    uncached.append(pmid)
            else:
                uncached.append(pmid)
        
        if uncached:
            def _fetch():
                handle = Entrez.efetch(db="pubmed", id=uncached, rettype="abstract", retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                return records
            
            try:
                records = exponential_backoff(_fetch)
                
                for article in records['PubmedArticle']:
                    try:
                        pmid = str(article['MedlineCitation']['PMID'])
                        title = str(article['MedlineCitation']['Article']['ArticleTitle'])
                        
                        abstract_parts = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
                        
                        if isinstance(abstract_parts, list):
                            abstract = ' '.join([str(p) for p in abstract_parts])
                        else:
                            abstract = str(abstract_parts)
                        
                        pub_date = article['MedlineCitation']['Article']['Journal']['JournalIssue'].get('PubDate', {})
                        year = pub_date.get('Year', pub_date.get('MedlineDate', 'Unknown'))
                        if isinstance(year, str) and len(year) > 4:
                            year = year[:4]
                        
                        journal = str(article['MedlineCitation']['Article']['Journal']['Title'])
                        
                        pmcid = None
                        ids = article['PubmedData'].get('ArticleIdList', [])
                        for id_obj in ids:
                            if hasattr(id_obj, 'attributes') and id_obj.attributes.get('IdType') == 'pmc':
                                pmcid = str(id_obj)
                        
                        record = {
                            'pmid': pmid,
                            'pmcid': pmcid,
                            'title': title,
                            'abstract': abstract,
                            'year': year,
                            'journal': journal,
                            'title_abstract': f"{title} {abstract}"
                        }
                        
                        cache_file = os.path.join(CACHE_DIR, f"{pmid}.json")
                        with open(cache_file, 'w') as f:
                            json.dump(record, f)
                        
                        abstracts.append(record)
                        
                    except Exception as e:
                        continue
                
                time.sleep(0.5)
                
            except Exception as e:
                continue
        
        abstracts.extend(cached)
    
    return abstracts

all_results = {}
query_stats = []

for idx, (strategy_name, query) in enumerate(QUERY_STRATEGIES.items(), 1):
    print(f"[{idx}/{len(QUERY_STRATEGIES)}] {strategy_name}...", end=' ')
    
    pmids, total_count = search_pubmed(query.strip(), max_results=MAX_RESULTS_PER_QUERY)
    
    if len(pmids) == 0:
        print("No results")
        query_stats.append({'strategy': strategy_name, 'total': 0, 'retrieved': 0})
        continue
    
    abstracts = fetch_abstracts(pmids)
    all_results[strategy_name] = abstracts
    
    query_stats.append({'strategy': strategy_name, 'total': total_count, 'retrieved': len(abstracts)})
    
    print(f"{len(abstracts)} retrieved")
    time.sleep(1)

all_articles = []
for strategy, articles in all_results.items():
    for article in articles:
        article['source_query'] = strategy
        all_articles.append(article)

df_all = pd.DataFrame(all_articles)
df_all = df_all.drop_duplicates(subset=['pmid'])
df_all['year_int'] = pd.to_numeric(df_all['year'], errors='coerce')
df_all = df_all[df_all['year_int'] >= MIN_YEAR]

print(f"Total: {len(df_all):,} unique articles")

df_query_stats = pd.DataFrame(query_stats)
query_eval_file = f'query_evaluation_{RUN_ID}.csv'
df_query_stats.to_csv(query_eval_file, index=False)
df_query_stats.to_csv('query_evaluation_latest.csv', index=False)

# ============================================================================
# SECTION 4: KEYWORD SCORING
# ============================================================================

print("\n[4/18] Keyword Scoring")

def calculate_relevance_score(text):
    text_lower = text.lower()
    scores = {}
    
    individual_keywords = [
        r'\d+\s*year[s]?[- ]old', r'\d+\s*y/?o', 
        'presented with', 'patient presented',
        'a patient', 'case report', 'we report', 'we describe', 
        'this report describes', 'index patient', 'index case', 
        'proband', 'single patient', 'n-of-1',
        r'in (his|her) \d{2}s', 
        r'(man|woman|male|female) in (his|her) \d{2}s'
    ]
    scores['individual_patient'] = sum(1 for p in individual_keywords if re.search(p, text_lower))
    
    genomic_keywords = ['mutation', 'variant', 'sequencing', 'wgs', 'ngs', 'genetic testing', 'braf', 'egfr', 'kras', 'alk', 'her2', 'brca']
    scores['genomic_context'] = sum(1 for kw in genomic_keywords if kw in text_lower)
    
    biomarker_keywords = ['biomarker', 'laboratory', 'lab result', 'serum level', 'immunohistochemistry', 'pd-l1', 'tmb', 'msi']
    scores['biomarker'] = sum(1 for kw in biomarker_keywords if kw in text_lower)
    
    vitals_keywords = ['blood pressure', r'\bBP\b', 'heart rate', 'weight', 'bmi', 'vital signs', 'performance status', 'ecog', 'karnofsky']
    scores['vitals'] = sum(1 for p in vitals_keywords if re.search(p, text_lower))
    
    outcome_keywords = ['response', 'remission', 'complete response', 'survival', 'improved', 'outcome']
    scores['outcome'] = sum(1 for kw in outcome_keywords if kw in text_lower)
    
    negative_keywords = [
        'cohort study', 'cohort analysis', 'retrospective cohort', 'systematic review', 'meta-analysis', 
        'literature review', 'case series', 'case-control', 'population-based', 'randomized', 
        r'randomized controlled trial', 'rct', r'phase\s+(i|ii|iii|iv)', 'double-blind', 'placebo-controlled',
        r'\bn\s*=\s*\d+', r'\d+\s*patients', r'patients\s*\(\s*n\s*=\s*\d+\)', r'n\s*=\s*\d+\s*patients',
        r'\d+\s*subjects', r'case series of \d+', 'in vitro', 'in vivo', 'animal model', 'mouse model',
        'cell line', 'cell culture', 'xenograft'
    ]
    scores['negative'] = sum(1 for p in negative_keywords if re.search(p, text_lower))
    
    weighted_score = (
        scores['individual_patient'] * 3.5 +
        scores['outcome'] * 3.0 +
        scores['genomic_context'] * 2.5 +
        scores['biomarker'] * 2.0 +
        scores['vitals'] * 2.0 +
        scores['negative'] * -4.0
    )
    scores['total_weighted'] = max(0, weighted_score)
    
    return scores

score_results = []
for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Scoring"):
    scores = calculate_relevance_score(row['title_abstract'])
    scores['pmid'] = row['pmid']
    score_results.append(scores)

df_scores = pd.DataFrame(score_results)
df_final = df_all.merge(df_scores, on='pmid', how='left')
df_final = df_final.sort_values('total_weighted', ascending=False)

# ============================================================================
# SECTION 5: LLM SCORING (OPTIONAL)
# ============================================================================

print("\n[5/18] LLM Scoring")

def score_with_llm(title, abstract):
    if not USE_OPENAI:
        return {"score": None, "reasoning": "No API"}
    
    prompt = f"""Rate this article 0-10 for individual patient clinical outcomes database.
Title: {title[:200]}
Abstract: {abstract[:800]}

10 = Individual patient + drug + disease + outcome + context
0 = Not relevant

Respond with JSON only: {{"score":<0-10>,"reasoning":"<brief explanation>"}}"""
    
    def _call():
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Medical expert. JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 80,
            },
            timeout=30
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        return json.loads(response.json()['choices'][0]['message']['content'].strip())
    
    try:
        return exponential_backoff(_call, max_retries=5)
    except Exception as e:
        return {"score": None, "reasoning": f"LLM failed: {str(e)[:80]}"}

if USE_OPENAI:
    num_to_score = min(300, len(df_final))
    llm_scores = []
    
    for idx, row in tqdm(df_final.head(num_to_score).iterrows(), total=num_to_score, desc="LLM Scoring"):
        result = score_with_llm(row['title'], row['abstract'])
        result['pmid'] = row['pmid']
        llm_scores.append(result)
        time.sleep(0.5)
    
    df_llm_scores = pd.DataFrame(llm_scores)
    
    for col in ['score', 'reasoning']:
        if col in df_final.columns:
            df_final = df_final.drop(columns=[col])
            
    df_final = df_final.merge(df_llm_scores, on='pmid', how='left')
else:
    print("Skipped (no API key)")
    df_final['score'] = None
    df_final['reasoning'] = None

# ============================================================================
# SECTION 6: BALANCED INTERACTIVE LABELING
# ============================================================================

print("\n[6/18] Interactive Labeling")

def balanced_labeling(df, target_labels=200, target_positive_rate=0.50, progress_file='labeling_progress.csv'):
    existing_labels, labeled_pmids = [], set()
    if os.path.exists(progress_file):
        df_existing = pd.read_csv(progress_file)
        existing_labels = df_existing.to_dict('records')
        labeled_pmids = set(df_existing['pmid'].astype(str).values)
        print(f"Resuming: {len(existing_labels)} existing labels")
    
    labeled_data = existing_labels.copy()
    
    current_positive = sum(1 for x in labeled_data if x.get('label') == 1)
    current_negative = sum(1 for x in labeled_data if x.get('label') == 0)
    total_labeled = current_positive + current_negative
    
    target_positives = int(target_labels * target_positive_rate)
    target_negatives = target_labels - target_positives
    
    remaining_positives = target_positives - current_positive
    remaining_negatives = target_negatives - current_negative
    
    if total_labeled > 0:
        current_pos_rate = current_positive / total_labeled
        print(f"Balance: {current_positive}+/{current_negative}- ({current_pos_rate*100:.1f}%)")
        print(f"Remaining: {remaining_positives}+/{remaining_negatives}-")
    else:
        print(f"Target: {target_positives}+/{target_negatives}-")
    
    if not INTERACTIVE:
        df_queue = df.copy()
        df_queue['_priority'] = (
            df_queue['total_weighted'].fillna(0) + 
            pd.to_numeric(df_queue.get('score', pd.Series(0, index=df_queue.index)), errors='coerce').fillna(0) * 2
        )
        df_queue = df_queue.nlargest(min(500, len(df_queue)), '_priority')
        queue_file = f'label_queue_{RUN_ID}.csv'
        df_queue.to_csv(queue_file, index=False)
        df_queue.to_csv('label_queue_latest.csv', index=False)
        print(f"Saved label queue: {queue_file}")
        return pd.DataFrame()
    
    print("Commands: y=relevant, n=not relevant, r=read abstract, s=skip, q=quit")
    input("Press ENTER to start...")
    
    skip_counts = {}
    
    if '_pos_hint' in df.columns:
        pos_pool = [row for _, row in df.iterrows() if row.get('_pos_hint', 0) > 0 and str(row['pmid']) not in labeled_pmids]
        neg_pool = [row for _, row in df.iterrows() if row.get('_pos_hint', 0) == 0 and str(row['pmid']) not in labeled_pmids]
    else:
        pos_pool = [row for _, row in df.iterrows() if str(row['pmid']) not in labeled_pmids]
        neg_pool = []

    while True:
        valid_labels = [x for x in labeled_data if x.get('label', -1) != -1]
        valid_positive = sum(1 for x in valid_labels if x['label'] == 1)
        valid_negative = sum(1 for x in valid_labels if x['label'] == 0)
        total_valid = valid_positive + valid_negative
        
        if total_valid >= target_labels:
            print(f"Target reached: {target_labels} labels")
            break
        
        remaining_pos = target_positives - valid_positive
        remaining_neg = target_negatives - valid_negative
        
        if remaining_pos <= 0 and remaining_neg <= 0:
            print("Both quotas filled")
            break
            
        current_pool = None
        if remaining_pos > 0 and pos_pool:
            if remaining_pos >= remaining_neg or not neg_pool:
                row = pos_pool.pop(0)
                current_pool = 'pos'
            else:
                row = neg_pool.pop(0)
                current_pool = 'neg'
        elif remaining_neg > 0 and neg_pool:
            row = neg_pool.pop(0)
            current_pool = 'neg'
        elif pos_pool:
            row = pos_pool.pop(0)
            current_pool = 'pos'
        elif neg_pool:
            row = neg_pool.pop(0)
            current_pool = 'neg'
        else:
            print("No more candidates")
            break
        
        current_rate = valid_positive / total_valid if total_valid > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Article {total_valid+1}/{target_labels} | {valid_positive}+/{valid_negative}- ({current_rate*100:.0f}%)")
        print(f"Remaining: {remaining_pos}+/{remaining_neg}-")
        print(f"\nPMID: {row['pmid']}, Year: {row['year']}")
        print(f"Title: {row['title']}")
        print(f"Score: {row['total_weighted']:.1f}")
        
        score_val = pd.to_numeric(pd.Series([row.get('score')]), errors='coerce').iloc[0]
        if pd.notna(score_val):
            print(f"LLM: {score_val:.1f}/10")
        
        while True:
            response = input("\nIndividual patient case? (y/n/r/s/q): ").strip().lower()
            
            if response == 'q':
                pd.DataFrame(labeled_data).to_csv(progress_file, index=False)
                pd.DataFrame(labeled_data).to_csv(f'labeling_progress_backup_{RUN_ID}.csv', index=False)
                return pd.DataFrame(labeled_data)[pd.DataFrame(labeled_data)['label'] != -1]
            
            elif response == 'r':
                print(f"\n{'='*70}")
                print(row['abstract'])
                print(f"{'='*70}")
                continue
            
            elif response == 's':
                pmid = str(row['pmid'])
                skip_counts[pmid] = skip_counts.get(pmid, 0) + 1
                
                if skip_counts[pmid] < 2:
                    if current_pool == 'pos':
                        pos_pool.append(row)
                    else:
                        neg_pool.append(row)
                break
            
            elif response == 'y':
                if remaining_pos > 0:
                    labeled_data.append({'pmid': str(row['pmid']), 'label': 1, 'method': 'manual'})
                    labeled_pmids.add(str(row['pmid']))
                    pd.DataFrame(labeled_data).to_csv(progress_file, index=False)
                    break
                else:
                    print("Positive quota full")
                    break
            
            elif response == 'n':
                if remaining_neg > 0:
                    labeled_data.append({'pmid': str(row['pmid']), 'label': 0, 'method': 'manual'})
                    labeled_pmids.add(str(row['pmid']))
                    pd.DataFrame(labeled_data).to_csv(progress_file, index=False)
                    break
                else:
                    print("Negative quota full")
                    break
            
            else:
                print("Invalid input")
    
    pd.DataFrame(labeled_data).to_csv(progress_file, index=False)
    pd.DataFrame(labeled_data).to_csv(f'labeling_progress_backup_{RUN_ID}.csv', index=False)
    final_labels = pd.DataFrame(labeled_data)[pd.DataFrame(labeled_data)['label'] != -1]
    
    if len(final_labels) > 0:
        final_pos = (final_labels['label'] == 1).sum()
        final_neg = (final_labels['label'] == 0).sum()
        final_rate = final_pos / len(final_labels)
        
        print(f"\nFinal: {final_pos}+/{final_neg}- ({final_rate*100:.1f}%)")
        
        if final_rate < MIN_POSITIVE_RATE:
            print(f"Warning: {final_rate*100:.1f}% below minimum {MIN_POSITIVE_RATE*100:.0f}%")
    
    return final_labels

existing_progress = pd.read_csv('labeling_progress.csv') if os.path.exists('labeling_progress.csv') else pd.DataFrame()

if len(existing_progress) > 0:
    existing_pos = (existing_progress['label'] == 1).sum()
    existing_neg = (existing_progress['label'] == 0).sum()
    existing_total = existing_pos + existing_neg
    existing_rate = existing_pos / existing_total if existing_total > 0 else 0
    
    print(f"Existing: {existing_pos}+/{existing_neg}- ({existing_rate*100:.1f}%)")
    
    if existing_rate < MIN_POSITIVE_RATE and existing_total >= 50:
        print("Rebalancing mode: sampling likely positives")
        
        likely_positives = df_final[
            (df_final['title'].str.lower().str.contains('case report', na=False)) |
            (df_final['title_abstract'].str.lower().str.contains(r'\d+\s*year[s]?[- ]old', na=False, regex=True)) |
            ((df_final['individual_patient'] >= 2) & (df_final['negative'] == 0))
        ].copy()
        
        labeled_pmids_existing = set(existing_progress['pmid'].astype(str).values)
        likely_positives = likely_positives[~likely_positives['pmid'].astype(str).isin(labeled_pmids_existing)]
        likely_positives = likely_positives.drop_duplicates('pmid')
        
        df_labeling = likely_positives.sample(min(200, len(likely_positives)), random_state=RANDOM_SEED)
    
    else:
        likely_pos = df_final[
            (df_final['title'].str.lower().str.contains('case report', na=False)) |
            (df_final['title_abstract'].str.lower().str.contains(r'\d+\s*year[s]?[- ]old', na=False, regex=True)) |
            (df_final['total_weighted'] > 18)
        ].copy()
        
        likely_neg = df_final[(df_final['negative'] > 2)].copy()
        
        likely_pos = likely_pos.drop_duplicates('pmid')
        likely_neg = likely_neg.drop_duplicates('pmid')
        
        pos_pmids = set(likely_pos['pmid'].astype(str))
        likely_neg = likely_neg[~likely_neg['pmid'].astype(str).isin(pos_pmids)]
        
        sampled_pos = likely_pos.sample(min(120, len(likely_pos)), random_state=RANDOM_SEED)
        sampled_neg = likely_neg.sample(min(80, len(likely_neg)), random_state=RANDOM_SEED)
        
        df_labeling = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=RANDOM_SEED)

else:
    likely_pos = df_final[
        (df_final['title'].str.lower().str.contains('case report', na=False)) |
        (df_final['total_weighted'] > 18)
    ].copy()
    
    likely_neg = df_final[(df_final['negative'] > 2)].copy()
    
    likely_pos = likely_pos.drop_duplicates('pmid')
    likely_neg = likely_neg.drop_duplicates('pmid')
    
    pos_pmids = set(likely_pos['pmid'].astype(str))
    likely_neg = likely_neg[~likely_neg['pmid'].astype(str).isin(pos_pmids)]
    
    sampled_pos = likely_pos.sample(min(120, len(likely_pos)), random_state=RANDOM_SEED)
    sampled_neg = likely_neg.sample(min(80, len(likely_neg)), random_state=RANDOM_SEED)
    
    df_labeling = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=RANDOM_SEED)

df_labeling = df_labeling.copy()
df_labeling["_pos_hint"] = (
    df_labeling["title"].str.lower().str.contains("case report", na=False).astype(int) +
    df_labeling["title_abstract"].str.lower().str.contains(r"\d+\s*year[s]?[- ]old", na=False, regex=True).astype(int) +
    (df_labeling["total_weighted"] > 18).astype(int)
)
df_labeling = df_labeling.sort_values("_pos_hint", ascending=False)

print(f"{len(df_labeling)} candidates prepared")
df_labels = balanced_labeling(df_labeling, TARGET_LABELS, TARGET_POSITIVE_RATE)
df_labeling = df_labeling.drop(columns=["_pos_hint"], errors="ignore")

if len(df_labels) == 0:
    print("No labels collected - exiting")
else:
    df_labels['pmid'] = df_labels['pmid'].astype(str)
    df_final['pmid'] = df_final['pmid'].astype(str)
    
    df_training = (
        df_final[df_final["pmid"].isin(df_labels["pmid"])]
        .merge(df_labels, on="pmid", how="inner")
    )
    
    if len(df_training) > 0:
        n_pos = (df_training['label']==1).sum()
        n_neg = (df_training['label']==0).sum()
        pos_rate = n_pos / len(df_training)
        
        print(f"\nTraining: {len(df_training)} labels")
        print(f"Balance: {n_pos}+/{n_neg}- = {pos_rate*100:.1f}%")
        
        if pos_rate < MIN_POSITIVE_RATE:
            print(f"BLOCKED: {pos_rate*100:.1f}% below {MIN_POSITIVE_RATE*100:.0f}%")
            print(f"Need ~{int((MIN_POSITIVE_RATE * len(df_training)) - n_pos)} more positive labels")
            
            df_training.to_csv(f'labeled_training_data_{RUN_ID}.csv', index=False)
            classifier_results = {'trained': False, 'reason': 'imbalanced', 'pos_rate': pos_rate}
            
        else:
            df_training.to_csv(f'labeled_training_data_{RUN_ID}.csv', index=False)
            classifier_results = None

if 'classifier_results' in locals() and classifier_results and classifier_results.get('trained') == False:
    print("\nSTOPPING: Dataset imbalanced")
    
else:
    # ========================================================================
    # SECTION 7: CLASSIFIER TRAINING
    # ========================================================================
    
    print("\n[7/18] Classifier Training")
    
    classifier_results = {
        'trained': False, 
        'best_threshold': DEFAULT_THRESHOLD,
        'selected_result': None
    }
    
    if 'df_training' in locals() and len(df_training) >= 50:
        df_train, df_test = train_test_split(
            df_training, 
            test_size=0.2, 
            stratify=df_training['label'], 
            random_state=RANDOM_SEED
        )
        
        print(f"Train: {len(df_train)}, Test: {len(df_test)}")
        
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2, stop_words='english')
        
        X_train = vectorizer.fit_transform(df_train['title_abstract'])
        y_train = df_train['label']
        X_test = vectorizer.transform(df_test['title_abstract'])
        y_test = df_test['label']
        
        base_clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
        clf = CalibratedClassifierCV(base_clf, cv=3, method='sigmoid')
        clf.fit(X_train, y_train)
        
        y_test_proba = clf.predict_proba(X_test)[:, 1]
        
        thresholds = np.linspace(0.05, 0.95, 91)
        test_results = []
        
        for threshold in thresholds:
            y_pred = (y_test_proba >= threshold).astype(int)
            
            test_results.append({
                'threshold': threshold,
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'cm': confusion_matrix(y_test, y_pred)
            })
        
        valid_thresholds = [r for r in test_results if r['recall'] >= RECALL_REQUIREMENT and r['precision'] >= 0.70]
        
        if valid_thresholds:
            best_result = max(valid_thresholds, key=lambda x: x['precision'])
            best_threshold = best_result['threshold']
        else:
            best_result = [r for r in test_results if abs(r['threshold'] - DEFAULT_THRESHOLD) < 0.01][0]
            best_threshold = DEFAULT_THRESHOLD
        
        print(f"Threshold: {best_threshold:.3f}")
        print(f"Recall: {best_result['recall']:.3f}, Precision: {best_result['precision']:.3f}, F1: {best_result['f1']:.3f}")
        
        classifier_results.update({
            'trained': True,
            'best_threshold': best_threshold,
            'selected_result': best_result,
            'test_results': test_results
        })
        
        threshold_df = pd.DataFrame([{
            'Threshold': r['threshold'],
            'Recall': r['recall'],
            'Precision': r['precision'],
            'F1': r['f1']
        } for r in test_results])
        thresh_file = f'threshold_analysis_{RUN_ID}.csv'
        threshold_df.to_csv(thresh_file, index=False)
        threshold_df.to_csv('threshold_analysis_latest.csv', index=False)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.heatmap(
            best_result['cm'], 
            annot=True, 
            fmt='d', 
            cmap='Greens', 
            ax=axes[0],
            xticklabels=['Not Relevant', 'Relevant'],
            yticklabels=['Not Relevant', 'Relevant']
        )
        axes[0].set_title(f'Test Set (t={best_threshold:.3f})')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')
        
        axes[1].plot(recall_vals, precision_vals, linewidth=2, label=f'AUC={pr_auc:.3f}')
        axes[1].scatter([best_result['recall']], [best_result['precision']], color='red', s=100, zorder=5)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('PR Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].plot([r['threshold'] for r in test_results], [r['recall'] for r in test_results], label='Recall', linewidth=2)
        axes[2].plot([r['threshold'] for r in test_results], [r['precision'] for r in test_results], label='Precision', linewidth=2)
        axes[2].plot([r['threshold'] for r in test_results], [r['f1'] for r in test_results], label='F1', linewidth=2)
        axes[2].axvline(best_threshold, color='red', linestyle='--')
        axes[2].axhline(RECALL_REQUIREMENT, color='orange', linestyle=':')
        axes[2].set_xlabel('Threshold')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Threshold Selection')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        cm_file = f'confusion_matrices_{RUN_ID}.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.savefig('confusion_matrices_latest.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        X_all = vectorizer.transform(df_final['title_abstract'])
        df_final['model_probability'] = clf.predict_proba(X_all)[:, 1]
        df_final['model_prediction'] = (df_final['model_probability'] >= best_threshold).astype(int)
        
        scored_file = f'scored_articles_{RUN_ID}.csv'
        df_final.to_csv(scored_file, index=False)
        df_final.to_csv('scored_articles_latest.csv', index=False)
        
        classifier_model_file = f'classifier_model_{RUN_ID}.pkl'
        with open(classifier_model_file, 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'classifier': clf, 'threshold': best_threshold}, f)
        with open('classifier_model_latest.pkl', 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'classifier': clf, 'threshold': best_threshold}, f)
        
        y_pred_final = (y_test_proba >= best_threshold).astype(int)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final, target_names=['Not Relevant', 'Relevant'], digits=3))
    
    else:
        print("Insufficient labels (need ≥50)")
    
    # ====================================================================
    # SECTION 8: ERROR ANALYSIS
    # ====================================================================
    
    print("\n[8/18] Error Analysis")
    
    if (classifier_results.get('trained') and 
        all(k in locals() for k in ["y_test", "y_test_proba", "df_test", "best_threshold"])):
        
        y_pred = (y_test_proba >= best_threshold).astype(int)
        
        fps = df_test[(y_test == 0) & (y_pred == 1)]['pmid'].tolist()
        fns = df_test[(y_test == 1) & (y_pred == 0)]['pmid'].tolist()
        
        error_buckets = {'false_positives': fps, 'false_negatives': fns}
        
        error_buckets_file = f'error_buckets_{RUN_ID}.json'
        with open(error_buckets_file, 'w') as f:
            json.dump(error_buckets, f, indent=2)
        with open('error_buckets_latest.json', 'w') as f:
            json.dump(error_buckets, f, indent=2)
        
        print(f"FP: {len(fps)}, FN: {len(fns)}")
    else:
        print("Skipped")
    
    # ====================================================================
    # SECTION 9: ONTOLOGY SETUP
    # ====================================================================
    
    print("\n[9/18] Ontology Setup")
    
    DRUG_ONTOLOGY = {
        'vemurafenib': 'DB08881', 'dabrafenib': 'DB08912', 'pembrolizumab': 'DB09037',
        'nivolumab': 'DB09035', 'ipilimumab': 'DB06186', 'rituximab': 'DB00073',
        'methotrexate': 'DB00563', 'adalimumab': 'DB00051', 'trastuzumab': 'DB00072',
        'erlotinib': 'DB00530', 'imatinib': 'DB00619', 'crizotinib': 'DB08865',
        'osimertinib': 'DB09330', 'alectinib': 'DB09526', 'infliximab': 'DB00065',
        'etanercept': 'DB00005', 'sulfasalazine': 'DB00795', 'hydroxychloroquine': 'DB01611',
        'tofacitinib': 'DB08895', 'baricitinib': 'DB11817', 'upadacitinib': 'DB14782',
        'tocilizumab': 'DB06273', 'sarilumab': 'DB11306', 'ustekinumab': 'DB05679',
        'secukinumab': 'DB09029',
    }
    
    DISEASE_ONTOLOGY = {
        'melanoma': 'DOID:1909', 'lung cancer': 'DOID:1324', 'breast cancer': 'DOID:1612',
        'colorectal cancer': 'DOID:9256', 'lung adenocarcinoma': 'DOID:3910',
        'non-small cell lung cancer': 'DOID:3908', 'glioblastoma': 'DOID:3068',
        'rheumatoid arthritis': 'DOID:7148', 'systemic lupus erythematosus': 'DOID:9074',
        'psoriatic arthritis': 'DOID:9008', 'ankylosing spondylitis': 'DOID:7147',
        "crohn's disease": 'DOID:8778', 'ulcerative colitis': 'DOID:8577',
        'multiple sclerosis': 'DOID:2377', 'myasthenia gravis': 'DOID:437',
        "sjogren's syndrome": 'DOID:45', 'systemic sclerosis': 'DOID:418',
        'vasculitis': 'DOID:865', 'polymyalgia rheumatica': 'DOID:9827',
    }
    
    GENE_ONTOLOGY = {
        'braf': 'HGNC:1097', 'egfr': 'HGNC:3236', 'kras': 'HGNC:6407',
        'alk': 'HGNC:427', 'ros1': 'HGNC:10261', 'met': 'HGNC:7029',
        'brca1': 'HGNC:1100', 'brca2': 'HGNC:1101',
    }
    
    def normalize_entity(text, ontology_dict, system_name):
        if not text:
            return None
        
        t = text.lower().strip()
        
        if t in ontology_dict:
            return {'text': text, 'id': ontology_dict[t], 'system': system_name, 'confidence': 1.0, 'method': 'exact'}
        
        for key, value in ontology_dict.items():
            if re.search(rf"\b{re.escape(key)}\b", t, flags=re.IGNORECASE):
                return {'text': text, 'id': value, 'system': system_name, 'confidence': 0.8, 'method': 'word_boundary'}
        
        return {'text': text, 'id': None, 'system': None, 'confidence': 0.0, 'method': 'unmapped'}
    
    print(f"Drugs: {len(DRUG_ONTOLOGY)}, Diseases: {len(DISEASE_ONTOLOGY)}, Genes: {len(GENE_ONTOLOGY)}")
    
    # ====================================================================
    # SECTION 10: EXTRACTION GATE
    # ====================================================================
    
    print("\n[10/18] Extraction Gate")
    
    def is_valid_individual_case(title, abstract, keyword_scores):
        text = f"{title} {abstract}".lower()
        
        hard_blocks = [
            r'phase\s+(i|ii|iii|iv|1|2|3)', 'randomized', 'double-blind', 'placebo-controlled',
            'systematic review', 'meta-analysis', 'retrospective', 'prospective', 'multicenter',
            'multi-center', r'\bn\s*=\s*\d{2,}', r'\d+\s*patients', r'cohort of \d+',
            r'\d+\s+hospitals', 'tertiary hospitals', 'dose-escalation', r'\bMTD\b', r'\bDLT\b',
        ]
        
        for pattern in hard_blocks:
            if re.search(pattern, text):
                return False, f"blocked:{pattern[:20]}"
        
        has_individual = (
            keyword_scores.get('individual_patient', 0) > 0 or
            re.search(r'\d+\s*year[s]?[- ]old', text) or
            'case report' in text or
            'we report' in text
        )
        
        if not has_individual:
            return False, "no_individual_indicators"
        
        return True, "valid"
    
    # ====================================================================
    # SECTION 11: EXTRACTION LOGIC
    # ====================================================================
    
    print("\n[11/18] Extraction Logic")
    
    FAILURE_PHRASES = ['no response', 'failed', 'did not respond', 'progression', 'progressive disease', 'no improvement', 'worsening']
    
    def extract_clinical_outcomes(title, abstract, pmid, pmcid=None):
        text = f"{title} {abstract}"
        text_lower = text.lower()
        
        record = {
            'extractable': False,
            'method': 'regex_fallback',
            'drug': None,
            'disease': None,
            'demographics': {},
            'clinical_notes': [],
            'context': [],
            'source': {'pmid': pmid, 'pmcid': pmcid, 'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}", 'title': title},
            'flags': []
        }
        
        age_match = re.search(r'(\d+)\s*year[s]?[- ]old', text_lower)
        if age_match:
            record['demographics']['age'] = int(age_match.group(1))
        
        if re.search(r'\b(male|man)\b', text_lower) and not re.search(r'\bfemale\b', text_lower):
            record['demographics']['sex'] = 'male'
        elif re.search(r'\b(female|woman)\b', text_lower):
            record['demographics']['sex'] = 'female'

        weight_match = re.search(r'\bweight.*?(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>kg|lbs)\b', text_lower)
        if weight_match:
            record['demographics']['weight'] = f"{weight_match.group('val')} {weight_match.group('unit')}"
            
        bmi_match = re.search(r'\bbmi.*?(?P<val>\d+(?:\.\d+)?)\b', text_lower)
        if bmi_match:
            record['demographics']['bmi'] = float(bmi_match.group('val'))
            
        bp_match = re.search(r'\bblood pressure.*?(\d{2,3}\s*/\s*\d{2,3})\b', text_lower)
        if bp_match:
            record['demographics']['blood_pressure'] = bp_match.group(1).replace(' ', '')
            
        hr_match = re.search(r'\bheart rate.*?(\d{2,3})\s*(bpm|beats per minute)\b', text_lower)
        if hr_match:
            record['demographics']['heart_rate'] = hr_match.group(1)
            
        ecog_match = re.search(r'\becog\s*(?:performance status|ps)?\s*(?:of)?\s*([0-4])\b', text_lower)
        if ecog_match:
            record['demographics']['ecog'] = int(ecog_match.group(1))
            
        kps_match = re.search(r'\bkarnofsky\s*(?:performance status|score)?\s*(?:of)?\s*(\d{2,3})\b', text_lower)
        if kps_match:
            record['demographics']['karnofsky'] = int(kps_match.group(1))
        
        for drug in DRUG_ONTOLOGY.keys():
            pattern = re.compile(rf'\b{re.escape(drug)}\b', re.IGNORECASE)
            if pattern.search(text):
                record['drug'] = drug
                break
        
        for disease in DISEASE_ONTOLOGY.keys():
            pattern = re.compile(rf'\b{re.escape(disease)}\b', re.IGNORECASE)
            if pattern.search(text):
                record['disease'] = disease
                break
        
        variant_patterns = [
            r"\bV600E\b", r"\bV600K\b", r"\bL858R\b", r"\bT790M\b",
            r"\bexon\s*\d+\s*(deletion|del|insertion|ins|mutation)\b",
            r"\bp\.[A-Za-z]{1,3}\d+[A-Za-z]{1,3}\b",
            r"\bc\.\d+[ACGT]>\s*[ACGT]\b",
            r"\bfusion\b",
        ]
        
        for gene in GENE_ONTOLOGY.keys():
            for var_pattern in variant_patterns:
                pattern = re.compile(rf'\b{gene}\b.{{0,40}}(?P<variant>{var_pattern})', re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    record['context'].append({
                        'type': 'genomic',
                        'gene': gene.upper(),
                        'variant': match.group('variant'),
                        'evidence': match.group()[:100],
                        'provenance': {'pmid': pmid, 'section': 'title_abstract'}
                    })
                    break
            
            for var_pattern in variant_patterns:
                pattern = re.compile(rf'(?P<variant>{var_pattern}).{{0,40}}\b{gene}\b', re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    record['context'].append({
                        'type': 'genomic',
                        'gene': gene.upper(),
                        'variant': match.group('variant'),
                        'evidence': match.group()[:100],
                        'provenance': {'pmid': pmid, 'section': 'title_abstract'}
                    })
                    break
        
        outcome_types = [
            ('complete response', 'CR'),
            ('partial response', 'PR'),
            ('complete remission', 'CR'),
            ('stable disease', 'SD'),
            ('improved', 'improvement'),
        ]
        
        outcome_extracted = False
        for phrase, outcome_type in outcome_types:
            pattern = re.compile(rf'(.{{0,60}}){re.escape(phrase)}(.{{0,60}})', re.IGNORECASE)
            match = pattern.search(text)
            
            if match:
                local_window = match.group(0).lower()
                has_local_negation = any(neg in local_window for neg in FAILURE_PHRASES)
                
                if not has_local_negation:
                    record['outcome'] = {
                        'type': outcome_type,
                        'response': phrase,
                        'evidence': match.group().strip()
                    }
                    outcome_extracted = True
                    break
                else:
                    record['flags'].append('outcome_negated_local')
        
        notes_patterns = [
            r'([^.]*\blab\b[^.]*)',
            r'([^.]*\blaboratory\b[^.]*)',
            r'([^.]*\bbiomarker\b[^.]*)',
            r'([^.]*\bprogression\b[^.]*)',
            r'([^.]*\bflare\b[^.]*)',
            r'([^.]*\bpresentation\b[^.]*)'
        ]
        
        seen_notes = set()
        for pattern in notes_patterns:
            for match in re.finditer(pattern, text_lower):
                sentence = match.group(1).strip()
                if len(sentence) > 10 and sentence not in seen_notes:
                    seen_notes.add(sentence)
                    orig_sentence = text[match.start(1):match.end(1)].strip()
                    record['clinical_notes'].append(orig_sentence)
        
        record['clinical_notes'] = " | ".join(record['clinical_notes']) if record['clinical_notes'] else None
        
        seen = set()
        deduped_context = []
        for ctx in record['context']:
            key = (ctx.get('type'), ctx.get('gene'), ctx.get('variant'), ctx.get('evidence'))
            if key not in seen:
                seen.add(key)
                deduped_context.append(ctx)
        record['context'] = deduped_context
        
        if record['drug'] and record['disease'] and outcome_extracted:
            record['extractable'] = True
            
            record['normalized'] = {}
            if record['drug']:
                drug_norm = normalize_entity(record['drug'], DRUG_ONTOLOGY, 'DrugBank')
                if drug_norm:
                    record['normalized']['drug'] = drug_norm
            
            if record['disease']:
                disease_norm = normalize_entity(record['disease'], DISEASE_ONTOLOGY, 'DOID')
                if disease_norm:
                    record['normalized']['disease'] = disease_norm
            
            for ctx in record['context']:
                if ctx['type'] == 'genomic':
                    gene_norm = normalize_entity(ctx['gene'], GENE_ONTOLOGY, 'HGNC')
                    if gene_norm:
                        ctx['normalized'] = {"gene": gene_norm}
        
        return record
    
    # ====================================================================
    # SECTION 12: BULK EXTRACTION
    # ====================================================================
    
    print("\n[12/18] Bulk Extraction")
    
    extracted_records = []
    extraction_stats = {'attempted': 0, 'gated_out': 0, 'successful': 0, 'failed': 0}
    
    if classifier_results.get('trained') and 'model_probability' in df_final.columns:
        candidates = df_final[df_final['model_probability'] > EXTRACTION_THRESHOLD]
        
        if len(candidates) < EXTRACTION_MIN_CANDIDATES:
            print(f"Fallback: top {EXTRACTION_MIN_CANDIDATES} (only {len(candidates)} above threshold)")
            candidates = df_final.nlargest(EXTRACTION_MIN_CANDIDATES, "model_probability")
        
        candidates = candidates.head(200)
        print(f"Candidates: {len(candidates)}")
        
        for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Extracting"):
            extraction_stats['attempted'] += 1
            
            is_valid, reason = is_valid_individual_case(row['title'], row['abstract'], {'individual_patient': row.get('individual_patient', 0)})
            
            if not is_valid:
                extraction_stats['gated_out'] += 1
                continue
            
            record = extract_clinical_outcomes(row['title'], row['abstract'], row['pmid'], row.get('pmcid'))
            
            if record.get('extractable'):
                extraction_stats['successful'] += 1
            else:
                extraction_stats['failed'] += 1
            
            extracted_records.append(record)
        
        print(f"Attempted: {extraction_stats['attempted']}, Gated: {extraction_stats['gated_out']}")
        print(f"Success: {extraction_stats['successful']}, Failed: {extraction_stats['failed']}")
        
        prog_file = f'extraction_progress_{RUN_ID}.jsonl'
        with open(prog_file, 'w') as f:
            for rec in extracted_records:
                f.write(json.dumps(rec) + '\n')
        with open('extraction_progress_latest.jsonl', 'w') as f:
            for rec in extracted_records:
                f.write(json.dumps(rec) + '\n')
        
    else:
        print("Skipped (no classifier)")
    
    # ====================================================================
    # SECTION 13: DATABASE CREATION
    # ====================================================================
    
    print("\n[13/18] Database Creation")
    
    if extracted_records:
        valid_records = [r for r in extracted_records if r.get('extractable')]
        print(f"Valid: {len(valid_records)}")
        
        if len(valid_records) > 0:
            out_file = f'clinical_outcomes_database_{RUN_ID}.jsonl'
            with open(out_file, 'w') as f:
                for rec in valid_records:
                    f.write(json.dumps(rec) + '\n')
            with open('clinical_outcomes_database_latest.jsonl', 'w') as f:
                for rec in valid_records:
                    f.write(json.dumps(rec) + '\n')
            
            sample = valid_records[0]
            print(f"Sample - PMID: {sample['source']['pmid']}, Drug: {sample.get('drug')}, Disease: {sample.get('disease')}")
        else:
            valid_records = []
    else:
        valid_records = []
    
    # ====================================================================
    # SECTION 14: QC REPORT
    # ====================================================================
    
    print("\n[14/18] QC Report")
    
    if len(valid_records) > 0:
        qc_stats = {
            'total_records': len(valid_records),
            'with_drug': sum(1 for r in valid_records if r.get('drug')),
            'with_disease': sum(1 for r in valid_records if r.get('disease')),
            'with_outcome': sum(1 for r in valid_records if r.get('outcome')),
            'with_context': sum(1 for r in valid_records if r.get('context')),
            'with_demographics': sum(1 for r in valid_records if r.get('demographics')),
            'drug_mapped': sum(1 for r in valid_records if r.get('normalized', {}).get('drug', {}).get('id')),
            'disease_mapped': sum(1 for r in valid_records if r.get('normalized', {}).get('disease', {}).get('id')),
        }
        
        for metric, value in qc_stats.items():
            pct = (value / qc_stats['total_records'] * 100) if qc_stats['total_records'] > 0 else 0
            print(f"{metric}: {value} ({pct:.1f}%)")
        
        qc_df = pd.DataFrame([qc_stats])
        qc_file = f'qc_report_{RUN_ID}.csv'
        qc_df.to_csv(qc_file, index=False)
        qc_df.to_csv('qc_report_latest.csv', index=False)
    
    # ====================================================================
    # SECTION 15: EXTRACTION EXAMPLES
    # ====================================================================
    
    print("\n[15/18] Extraction Examples")
    
    if len(valid_records) >= 3:
        examples = []
        for i, rec in enumerate(valid_records[:5]):
            examples.append({
                'pmid': rec['source']['pmid'],
                'url': rec['source']['url'],
                'title': rec['source']['title'],
                'extraction': rec,
                'manual_validation': {'drug': '', 'disease': '', 'outcome': '', 'notes': 'TO BE VALIDATED'}
            })
        
        ex_file = f'extraction_examples_{RUN_ID}.json'
        with open(ex_file, 'w') as f:
            json.dump(examples, f, indent=2)
        with open('extraction_examples_latest.json', 'w') as f:
            json.dump(examples, f, indent=2)
    
    # ====================================================================
    # SECTION 16: MODEL CARD
    # ====================================================================
    
    print("\n[16/18] Model Card")
    
    pos_rate_str = f"{pos_rate*100:.1f}%" if "pos_rate" in locals() and isinstance(pos_rate, (int, float)) else "N/A"
    
    t = classifier_results.get("best_threshold")
    t_str = f"{t:.3f}" if isinstance(t, (int, float)) else "N/A"
    
    r = classifier_results.get('selected_result', {}).get('recall')
    r_str = f"{r:.3f}" if isinstance(r, (int, float)) else "N/A"
    
    p = classifier_results.get('selected_result', {}).get('precision')
    p_str = f"{p:.3f}" if isinstance(p, (int, float)) else "N/A"
    
    model_card = f"""# Clinical Outcomes Pipeline - Model Card

## Model
- Type: Calibrated Logistic Regression
- Task: PubMed article triage
- Date: {datetime.now().strftime('%Y-%m-%d')}

## Training
- Labels: {len(df_training) if 'df_training' in locals() else 0} ({pos_rate_str} positive)
- Threshold: {t_str}
- Recall: {r_str}, Precision: {p_str}

## Extraction
- Valid records: {len(valid_records)}/{FULL_PROJECT_TARGET}
- Features: TF-IDF, ontology normalization, local negation

## Limitations
- Abstract-only analysis
- Limited ontology coverage

## Contact
Advaith Nair - anair@utexas.edu
"""
    
    mc_file = f'model_card_{RUN_ID}.txt'
    with open(mc_file, 'w') as f:
        f.write(model_card)
    with open('model_card_latest.txt', 'w') as f:
        f.write(model_card)
    
    # ====================================================================
    # SECTION 17: FINAL REPORT
    # ====================================================================
    
    print("\n[17/18] Final Report")
    
    completion_pct = (len(valid_records) / FULL_PROJECT_TARGET * 100) if FULL_PROJECT_TARGET > 0 else 0
    
    drug_pct = (sum(1 for r in valid_records if r.get('drug')) / len(valid_records) * 100) if valid_records else 0
    disease_pct = (sum(1 for r in valid_records if r.get('disease')) / len(valid_records) * 100) if valid_records else 0
    outcome_pct = (sum(1 for r in valid_records if r.get('outcome')) / len(valid_records) * 100) if valid_records else 0
    
    report = f"""
CLINICAL OUTCOMES PIPELINE - FINAL REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATUS
------
Records: {len(valid_records)}/{FULL_PROJECT_TARGET} ({completion_pct:.1f}%)
Target: {TARGET_EXTRACTION_COUNT}

CLASSIFIER
----------
Trained: {classifier_results.get('trained', False)}
Threshold: {t_str}, Recall: {r_str}, Precision: {p_str}

EXTRACTION
----------
Attempted: {extraction_stats.get('attempted', 0)}
Gated: {extraction_stats.get('gated_out', 0)}
Success: {extraction_stats.get('successful', 0)}

QUALITY
-------
Drug: {sum(1 for r in valid_records if r.get('drug'))} ({drug_pct:.1f}%)
Disease: {sum(1 for r in valid_records if r.get('disease'))} ({disease_pct:.1f}%)
Outcome: {sum(1 for r in valid_records if r.get('outcome'))} ({outcome_pct:.1f}%)
"""
    
    print(report)
    
    report_file = f'FINAL_REPORT_{RUN_ID}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    with open('FINAL_REPORT_latest.txt', 'w') as f:
        f.write(report)
    
    