from tqdm import tqdm
from typing import Optional
from transformers import BatchEncoding

import torch
import torch.nn as nn
from torch.nn import functional as F

from ncompass.internal.logging import ERROR
from ncompass.internal.exec.inference import iterate_windows

__all__ = ["sliding_window_perplexity"]

def run_model(**kwargs):
    model = kwargs["model"]
    input_ids = kwargs["input_ids"]
    window_end = kwargs["window_end"]
    prev_window_end = kwargs["prev_window_end"]
    
    trg_len = window_end - prev_window_end
    target_ids = input_ids.clone()
    target_ids[:-trg_len] = -100
    with torch.no_grad():
        output = model(input_ids, labels=target_ids)
        neg_log_likelihood = output.loss

    kwargs["output"] = output
    
    return neg_log_likelihood, kwargs
    
def sliding_window_perplexity(\
            model: nn.Module\
          , encodings: BatchEncoding\
          , stride: Optional[int] = None\
          , context_len: Optional[int] = None) -> float:
    ctx_len = model.config.n_positions if context_len is None else context_len
    _stride = model.config.n_positions if stride is None else stride
    num_tokens = encodings.input_ids.size(1)

    if (ctx_len == -1) or (ctx_len > model.config.n_positions):
        ERROR(f"ANN model cannot be run with provided context len {ctx_len} which is "\
              f"greater than model context len {model.config.n_positions}", ValueError)
    
    ppl = iterate_windows(
            model
            , encodings.input_ids.to(model.device)
            , num_tokens
            , _stride
            , ctx_len
            , False
            , run_model
            , {})
    return ppl
