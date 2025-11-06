import json
import os
import pandas as pd

from src.hw2 import load_BERT, genBERTVector
from utils.similarity_utils import (pairwise_cosines, mean_std, plot_histogram,
                                    sample_cross_word_pairs, cosine)

OUT_DIR = os.path.join('out', 'part2_scibert')
DATA_JSON = os.path.join('data', 'part1_sentences.json')
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
NUM_PAIRS = 360

os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_JSON, 'r') as f:
    words_dict = json.load(f)

model, tokenizer = load_BERT(MODEL_NAME)

# store vectors for each word
word_to_vecs = {w: genBERTVector(model, tokenizer, w, sents)
                for w, sents in words_dict.items()}

rand_pair_sim = sample_cross_word_pairs(word_to_vecs, num_pairs=NUM_PAIRS)
sims = [cosine(a, b) for a, b in rand_pair_sim]

mean, std = mean_std(sims)

plot_histogram(
    sims,
    f"Cross-word cosine similarities (SciBERT, n={len(sims)})",
    os.path.join(OUT_DIR, "cross_hist.png")
)

df_stats = pd.DataFrame(
    [{"mean": mean, "std": std, "num_pairs": len(sims)}],
    index=["SCIBERT"]
)
df_stats.to_csv(os.path.join(OUT_DIR, "stats_table.csv"))

print("\nSciBERT Part 2 stats:")
print(df_stats)
