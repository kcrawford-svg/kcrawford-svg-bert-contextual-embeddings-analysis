import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
def genBERTVector(model, tokenizer, word, sentences):
    """
    Extract contextualized embeddings for a given word across sentences.
    Returns a list of 50-dimensional vectors (or [] if not found).
    """
    vectors = []

    # ensuring that all tensors live on the same hardware
    device = next(model.parameters()).device

    for sent in sentences:
        # Find the word not the substring word
        match = re.search(rf'\b{re.escape(word)}\b', sent, re.IGNORECASE)
        if not match:
            vectors.append([])  # append empty if no match
            continue

        inputs = tokenizer(sent, return_tensors='pt', return_offsets_mapping=True)
        offsets = inputs.pop('offset_mapping')[0].tolist()
        start, end = match.span()

        # getting token indices
        token_indices = [i for i, (s, e) in enumerate(offsets)
                        if s >= start and e <= end and not (s == e)]
        if not token_indices:
            vectors.append([])
            continue

        # Forward pass through the BERT model
        with torch.no_grad():
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            outputs = model(**inputs)
            # remove batch dim for better token row correspondence
            last_hidden_states = outputs.last_hidden_state.squeeze(0)

        vector = last_hidden_states[token_indices].mean(dim=0)[:50].cpu().numpy()
        vectors.append(vector.tolist())

    return vectors
















