import numpy as np
import matplotlib.pyplot as plt
import random


def cosine(u, v):
    if len(u) == 0 or len(v) == 0:
        return float('nan')
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return float('nan')
    return float(np.dot(u, v) / denom)


def pairwise_cosines(vectors):
    sims = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine(vectors[i], vectors[j])
            if not np.isnan(c):
                sims.append(float(c))
    return sims


def mean_std(values):
    values = np.array(values, dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values))


def plot_histogram(values, title, out_png):
    # 10 bins: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
    bins = [i / 10.0 for i in range(11)]
    plt.figure()
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def sample_cross_word_pairs(word_to_vecs, num_pairs):
    # Randomly sample pairs where the two vectors come from different words.
    # flatten index by (word, idx)
    pool = []
    for w, vecs in word_to_vecs.items():
        for i, v in enumerate(vecs):
            if len(v) > 0:
                pool.append((w, i, v))
    pairs = []
    attempts = 0
    max_attempts = num_pairs * 10
    while len(pairs) < num_pairs and attempts < max_attempts:
        a, b = random.sample(pool, 2)
        if a[0] != b[0]:
            pairs.append((a[2], b[2]))
        attempts += 1
    return pairs
