import json
import os

import pandas as pd

from src.hw2 import load_BERT, genBERTVector
from utils.similarity_utils import (pairwise_cosines,
                                    mean_std, plot_histogram,
                                    sample_cross_word_pairs, cosine)

OUT_DIR = os.path.join('out', 'part2')
DATA_JSON = os.path.join('data', 'part1_sentences.json')
MODEL_NAME = 'bert-base-uncased'
NUM_PAIRS = 360

os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_JSON, 'r') as f:
    words_dict = json.load(f)

model, tokenizer = load_BERT(MODEL_NAME)

# Vectors per word
word_to_vecs = {}
for word, sentences in words_dict.items():
    word_to_vecs[word] = genBERTVector(model, tokenizer, word, sentences)

# get cross sampled pairs and compute similarities
rand_pair_sim = sample_cross_word_pairs(word_to_vecs, num_pairs=NUM_PAIRS)

sims = [cosine(a, b) for a, b in rand_pair_sim]

mean, std = mean_std(sims)

plot_histogram(sims, f"Cross-word cosine similarities (n={len(sims)})", os.path.join(OUT_DIR, "cross_hist.png"))

# Make DataFrame
df_stats = pd.DataFrame([{
    "mean": mean,
    "std": std,
    "num_pairs": NUM_PAIRS
}], index=["CROSS_WORDS"]).round(4)

# Save CSV
stats_csv_path = os.path.join(OUT_DIR, "cross_stats.csv")
df_stats.to_csv(stats_csv_path)
print(f"\nStats CSV saved to: {stats_csv_path}")
print(df_stats)


