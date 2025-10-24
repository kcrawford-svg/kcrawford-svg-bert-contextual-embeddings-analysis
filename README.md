# BERT Contextual Embedding Analysis


**Analyzing how modern language models represent meaning.**  
This project explores how **BERT** encodes context-dependent word meaning through embeddings
---

## Summary

This project demonstrates end-to-end understanding of **contextual embeddings**, **semantic similarity**, and **vector retrieval**

It reproduces a NLP experiment and extends it to practical, industry-relevant use cases such as:
- Semantic search and document retrieval  
- Finetuning evaluation and embedding quality metrics  
- Domain-specific model benchmarking (FinBERT, SciBERT, LegalBERT)

---

## Key Highlights

- Built a reusable function `genBERTVector()` for token-level embedding extraction  
- **Measured semantic consistency** via cosine similarity across 8 words × 10 contexts  
- Designed automated experiments with visual histograms and statistical reports  
- **Compared domain-specific BERT models** (FinBERT, SciBERT, BioBERT, LegalBERT)  
- Produced data-driven insights on meaning preservation and contextual variance  

---

## Technical Overview

| Component | Description |
|------------|-------------|
| **Frameworks** | Python, PyTorch, Matplotlib, Pandas |
| **Core Task** | Extract contextual embeddings for target words and evaluate pairwise cosine similarity |
| **Metrics** | Cosine similarity, mean ± std deviation, histogram analysis |
| **Outputs** | CSV statistics, PNG histograms, Markdown report outline |
| **Reproducibility** | Fixed random seed (7322) and deterministic PyTorch mode |

---

## Experimental Design

### Same Word – Same Meaning  
- 8 distinct words (e.g., *algorithm*, *glacier*, *festival*).  
- 10 unique sentences per word (same sense, different context).  
- Compare all 45 pairwise cosine similarities.  
- Evaluate how consistently BERT encodes meaning across contexts.

### Different Words – Different Meaning  
- Randomly sample 360 cross-word vector pairs.  
- Validate that unrelated terms yield lower similarity values.  

### Synonyms – Different Words, Same Meaning  
- Select 8 WordNet synonym pairs (e.g., *car–automobile*, *kid–child*).  
- Generate 5 sentences per word and compare 25 cross-similarities per pair.  
- Examine how BERT captures semantic equivalence.

### Domain Models Analysis
- Run experimentation using specialized models:  
  - **FinBERT** (finance) **BioBERT** (biomedical)  
  - **SciBERT** (scientific) **LegalBERT** (legal text)  
- Quantify domain-specific improvements in embedding consistency.

---

## How to Run

```bash
pip install -r requirements.txt

# Base BERT (general English)
python run_experiments.py --model bert-base-uncased --out out/base

# Optional: domain-specific variants
python run_experiments.py --model ProsusAI/finbert --out out/finbert