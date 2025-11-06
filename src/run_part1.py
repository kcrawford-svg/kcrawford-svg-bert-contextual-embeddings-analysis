import json
import os

import pandas as pd

from src.hw2 import load_BERT, genBERTVector
from utils.similarity_utils import pairwise_cosines, mean_std, plot_histogram

# File path configs and default model
OUT_DIR = os.path.join('out', 'part1')
DATA_JSON = os.path.join('data', 'part1_sentences.json')
MODEL_NAME = 'bert-base-uncased'

os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_JSON, 'r') as f:
    word_dict = json.load(f)

# Load model and tokenizer
model, tokenizer = load_BERT(MODEL_NAME)

overall = {}
all_pairs = []

for word, sentences in word_dict.items():
    vectors = genBERTVector(model, tokenizer, word, sentences)
    sims = pairwise_cosines(vectors)
    mean, std = mean_std(sims)
    overall[word] = {"mean": mean, "std": std, "num_pairs": len(sims)}

    pd.DataFrame({"Sentence": sentences}).to_csv(os.path.join(OUT_DIR, f"{word}_sentences.csv"), index=False)

    plot_histogram(sims, f"{word} cosine similarities (n={len(sims)})", os.path.join(OUT_DIR, f"{word}_hist.png"))

    all_pairs.extend(sims)

# Histogram and OverallMean/Standard Dev
plot_histogram(all_pairs, f"All words combined (n={len(all_pairs)})", os.path.join(OUT_DIR, "combined_hist.png"))
mean_all, std_all = mean_std(all_pairs)

df_stats = pd.DataFrame.from_dict(overall, orient='index')
df_stats.index.name = "word"

# Add combined row
df_stats.loc["ALL_WORDS"] = [mean_all, std_all, len(all_pairs)]

# Round for readability (optional)
df_stats = df_stats.round(4)

# Save to CSV
stats_csv_path = os.path.join(OUT_DIR, "stats_table.csv")
df_stats.to_csv(stats_csv_path)

print(f"\nStats table saved to: {stats_csv_path}")
print(df_stats)

