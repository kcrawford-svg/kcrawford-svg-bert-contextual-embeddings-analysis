from typing import List, Optional
import re
import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA CUDA enabled GPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal GPU
    return torch.device("cpu")


@torch.no_grad()  # Only inference no gradient tracking
def genBERTVector(model,
                  tokenizer,
                  word,
                  sentences,
                  device: Optional[torch.device] = None,
                  take_dims: int = 50):
    if device is None:
        device = pick_device()

    outputs: List[List[float]] = []  # Stores on vector per sentence or empty []
    model.to(device)
    model.eval()  # BERT eval, prevents random embeddings

    # Get whole word not substrings (ex. just get dog from tothdog)
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)

    for sent in sentences:
        match = pattern.search(sent)
        if not match:
            outputs.append([])
            continue

        # index alignment of each word
        start_char, end_char = match.span()

        encoding = tokenizer(sent, return_tensors="pt",
                             return_offsets_mapping=True,
                             padding=True, truncation=True)

        # Character spans per token
        offsets = encoding['offset_mapping'][0].tolist()

        token_idx = []
        for idx, (start, end) in enumerate(offsets):
            if start == 0 == end == 0:
                continue
            if max(start, start_char) < max(end, end_char):
                token_idx.append(idx)
        if not token_idx:
            outputs.append([])
            continue

        encoding = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}
        out = model(**encoding)
        last_hidden_state = out.last_hidden_state[0]

        # Mean Pool or avg sub word vectors in single vector
        sub_word_vecs = [last_hidden_state[i] for i in token_idx]
        if not sub_word_vecs:
            outputs.append([])
            continue
        mean_vector = torch.stack(sub_word_vecs).mean(dim=0)
        vector_np = mean_vector.detach().cpu().numpy()
        outputs.append(vector_np[:take_dims].tolist())
    return outputs


def load_BERT(model_name: str = "bert-base-uncased"):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer
