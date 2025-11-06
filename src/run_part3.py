import json
import os
import itertools
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from transformers import BertModel, BertTokenizer

from src.hw2 import genBERTVector, load_BERT
from utils.similarity_utils import cosine, mean_std, plot_histogram

# ========================
# CONFIG
# ========================
OUT_DIR = os.path.join('out', 'part3')
DATA_JSON = os.path.join('data', 'part3_synset_pairs.json')
MODEL_NAME = 'bert-base-uncased'
SAVE_PER_PAIR_HISTS = False  # set True if you want 8 separate histograms
os.makedirs(OUT_DIR, exist_ok=True)

# ========================
# LOAD DATA + MODEL
# ========================
with open(DATA_JSON, 'r') as f:
    pairs = json.load(f)

print(f"Loaded {len(pairs)} synonym pairs from JSON.")

model, tokenizer = load_BERT(MODEL_NAME)

all_sims = []
stats_rows = []

for item in pairs:
    synset_id = item.get("synset", "")
    w1, w2 = item["w1"], item["w2"]
    sents1, sents2 = item["sents1"], item["sents2"]

    print(f"\nProcessing pair: {w1} ↔ {w2} ({synset_id})")

    # WordNet validation
    wn_ok = False
    for syn in wn.synsets(w1):
        if w2 in syn.lemma_names():
            wn_ok = True
            break

    if not wn_ok:
        print(f"{w1} and {w2} are not in the same WordNet synset")
    else:
        print(f"WordNet confirms: {w1} and {w2} belong to same synset.")

    # get contextual vectors
    vecs1 = genBERTVector(model, tokenizer, w1, sents1)
    vecs2 = genBERTVector(model, tokenizer, w2, sents2)

    # Compute 5×5 cross similarities
    sims = []
    for v1 in vecs1:
        for v2 in vecs2:
            c = cosine(v1, v2)
            if c == c:  # No NaN
                sims.append(float(c))

    if not sims:
        print(f" No valid similarities for {w1}/{w2}. Check tokenization.")
        mean = std = float('nan')
    else:
        mean, std = mean_std(sims)
        all_sims.extend(sims)

    if SAVE_PER_PAIR_HISTS and sims:
        out_png = os.path.join(OUT_DIR, f"{w1}_{w2}_hist.png")
        plot_histogram(sims, f"{w1} ↔ {w2} cosine sims (n={len(sims)})", out_png)

    stats_rows.append({
        "synset": synset_id,
        "w1": w1,
        "w2": w2,
        "mean": mean,
        "std": std,
        "n_pairs": len(sims),
        "wn_verified": wn_ok
    })

mean_all, std_all = mean_std(all_sims)
plot_histogram(all_sims,
               f"All synonym pairs (n={len(all_sims)})",
               os.path.join(OUT_DIR, "combined_hist.png"))

# build dataframe
df_stats = pd.DataFrame(stats_rows)

# add summary row with EXACT same columns
df_stats.loc[len(df_stats)] = {
    "synset": "ALL_PAIRS",
    "w1": "-",
    "w2": "-",
    "mean": mean_all,
    "std": std_all,
    "n_pairs": len(all_sims),
    "wn_verified": "-"
}

# round for readability
df_stats = df_stats.round(4)

# save CSV
csv_path = os.path.join(OUT_DIR, "stats_table.csv")
df_stats.to_csv(csv_path, index=False)

