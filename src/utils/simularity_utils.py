from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def cosine_sim(u, v):
    if len(u) == 0 or len(v) == 0:
        return np.NaN
    u, v = np.array(u), np.array(v)

    # check for zero magnitude vectors
    norm_vec = np.linalg.norm(u) * np.linalg.norm(v)
    if norm_vec == 0:
        return np.NaN
    return float(np.dot(u, v) / norm_vec)


def cosine_pairwise(vectors):
    return [cosine_sim(vectors[i], vectors[j])
            # all combinations of sentence embeddings
            for i, j in combinations(range(len(vectors)), 2)
            if len(vectors[i] and len(vectors[j]) == 2)]


def histogram(values, title, filename):
    plt.figure()
    plt.hist(values, bins=[i/10 for i in range(11)], edgecolor='black')
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()






