import os
import numpy as np

from .hw2 import genBERTVector
from .utils.simularity_utils import cosine_pairwise, histogram

def analyze_same_word(model, tokenizer, word_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    overall, all_scores = {}, []

    for word, sent in word_dict.items():
        vectors = genBERTVector(model, tokenizer, word, sent)
        word_similarity = cosine_pairwise(vectors)
        means, std = np.nanmean(word_similarity), np.nanstd(word_similarity)
        overall[word] = (means, std)
        all_scores.extend(word_similarity)

        # Plot means, std dev, and sim scores in histogram

        # output to file for report, `clean up values`





