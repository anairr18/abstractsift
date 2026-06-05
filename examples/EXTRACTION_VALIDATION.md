# Pipeline 1 — Extraction Validation Report
## Manual vs Automated Comparison (4 Case Reports)

**Date:** 2026-04-15  
**Validated by:** Advaith Nair  
**Pipeline:** LLM Schematizer (GPT-4o) + NCATS Entity Linking + Monarch KG

---

## Summary

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Checks passed | 16/25 (64%) | 24/25 (96%) |
| Critical failures | 3 | 0 |
| Structural issues | 2 | 0 |

---

## Record-by-Record Comparison

### PMID 32015975 — Olmesartan-induced Drug-Induced Liver Injury

| Field | Ground Truth | Before Fix | After Fix | Status |
|-------|-------------|------------|-----------|--------|
| drug | olmesartan | olmesartan | olmesartan | ✅ |
| disease | drug-induced liver injury | drug-induced liver injury | drug-induced liver injury | ✅ |
| outcome.response | resolved after olmesartan withdrawal | **"Complete response in December 2016"** | "resolved after drug withdrawal" | ✅ Fixed |
| outcome.duration_months | 2 months | null | 2 | ✅ Fixed |
| outcome.confidence | — | null | 0.9 | ✅ Fixed |
| age | 80 | 80 | 80 | ✅ |
| sex | female | female | female | ✅ |
| labs | AST 207, ALT 213, GGTP 21, ALP 116, bilirubin 0.5, INR 1.2 | AST, ALT only | AST, ALT, GGTP, ALP, bilirubin, INR | ✅ Fixed |
| comorbidities | 4 (HTN, dyslipidaemia, diverticulosis, adenocarcinoma) | 4 | 4 | ✅ |

**Root cause of failure:** The LLM extracted "Complete response in December 2016" from the text — but this referred to the patient's cancer response (serous papillary peritoneal adenocarcinoma), not the liver injury outcome. The DILI resolved 2 months after olmesartan withdrawal. Fixed with explicit prompt disambiguation: *"outcome.response refers to the drug adverse event resolution, not the underlying disease outcome."*

---

### PMID 32079429 — Ketoprofen-induced Hypersensitivity

| Field | Ground Truth | Before Fix | After Fix | Status |
|-------|-------------|------------|-----------|--------|
| drug | ketoprofen | ketoprofen | ketoprofen | ✅ |
| disease | hypersensitivity reaction | hypersensitivity reaction | hypersensitivity reaction | ✅ |
| outcome.response | resolved (not stated explicitly) | symptom description | "resolved after drug withdrawal" | ✅ |
| outcome.confidence | — | null | 0.9 | ✅ Fixed |
| age | 43 | 43 | 43 | ✅ |
| sex | male | male | male | ✅ |
| labs | none in abstract | {} | {} | ⚠️ Acceptable — abstract only |
| comorbidities | kidney lithiasis, NSAID hypersensitivity | both | all 3 | ✅ |

**Note:** No numerical labs available in cached abstract. Full PMC text not available for this PMID. Lab result note from paper: "laboratory parameters did not reveal any evidence of liver disease."

---

### PMID 32300505 — Lithium-induced Cardiotoxicity

| Field | Ground Truth | Before Fix | After Fix | Status |
|-------|-------------|------------|-----------|--------|
| drug | lithium | lithium | lithium | ✅ |
| disease | cardiotoxicity | cardiotoxicity | cardiotoxicity | ✅ |
| outcome.response | managed conservatively (ICU) | managed conservatively | managed conservatively | ✅ |
| outcome.confidence | — | null | 1.0 | ✅ Fixed |
| age | 57 | **null** | 57 | ✅ Fixed |
| sex | male | **null** | male | ✅ Fixed |
| labs | WBC 15,000; AST 59; lithium 1.8 mmol/L | **{}** | WBC, AST, lithium level | ✅ Fixed |
| comorbidities | hypothyroidism, mental retardation, seizure disorder, bipolar | ["bipolar disorder"] only | all 4 | ✅ Fixed |

**Root cause of failure:** The PMC XML full text (36KB) contains a large metadata header. Clinical content begins at character 10,013, but the pipeline truncated input to 8,000 characters — all patient data was cut off before the LLM ever saw it. Fixed by: (1) stripping XML tags before truncation, (2) increasing limit to 12,000 characters.

---

### PMID 32594911 — Omeprazole/Pantoprazole-induced Hyponatremia (SIADH)

| Field | Ground Truth | Before Fix | After Fix | Status |
|-------|-------------|------------|-----------|--------|
| drug | omeprazole, pantoprazole | omeprazole, pantoprazole | omeprazole, pantoprazole | ✅ |
| disease | hyponatremia (SIADH) | hyponatremia | hyponatremia | ✅ |
| outcome.response | sodium normalized after withdrawal | "serum sodium normalized..." | "resolved after drug withdrawal" | ✅ |
| outcome.confidence | — | null | 1.0 | ✅ Fixed |
| age | 67 | 67 | 67 | ✅ |
| sex | male | male | male | ✅ |
| labs | Na 127, osmolarity, urinary Na | 4 labs | 15 labs (full panel) | ✅ Enhanced |
| comorbidities | reflux esophagitis | 1 | 4 (full history) | ✅ Enhanced |

**Note on entity linking:** NCATS resolved disease to UMLS:C0268815 ("Hyponatremia with extracellular fluid depletion") — patient was described as euvolemic (SIADH), so this is the wrong subtype. This is a limitation of the NCATS resolver, not the LLM extraction. Flagged for manual review.

---

## Pipeline Fixes Applied

### 1. Schematizer System Prompt (`schematizer.py`)
Added explicit disambiguation rules:
- `outcome.response` = drug adverse event resolution only, NOT the outcome of the underlying disease
- `outcome.duration_months` = convert stated timeframes to months
- `outcome.confidence` = 0–1 confidence in the outcome label
- `labs` = extract ALL laboratory values with units and reference ranges
- `comorbidities` = list ALL comorbid conditions, not just the primary one

### 2. XML Stripping + Extended Context Window
- Added `_clean_text()` to strip XML/HTML tags before LLM input
- Increased text limit from 8,000 to 12,000 characters
- **Impact:** Fully resolved the Lithium record failure (demographics, labs, comorbidities all missed due to XML header overflow)

---

## Remaining Limitations

1. **SIADH subtype misclassification by NCATS** — The entity linker maps "hyponatremia" to the fluid-depletion subtype rather than SIADH. Requires post-processing disease subtype disambiguation or a custom ontology mapping step.

2. **Abstract-only records** — When full PMC text is unavailable, labs and outcome details are sparse. Applies to PMID 32079429.

3. **Labs format inconsistency** — Across runs, labs may return as flat strings ("207 IU/L") or structured dicts ({"value": 207, "unit": "IU/L"}). Schema should enforce structured format explicitly.

4. **Outcome.response is free text** — Not standardized to ML-usable labels (CR/PR/SD/resolved/unresolved). Downstream models will need a normalization layer.
