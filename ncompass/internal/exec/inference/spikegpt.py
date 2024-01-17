from tqdm import tqdm
from typing import Optional
from transformers import BatchEncoding

import torch
import torch.nn as nn
from torch.nn import functional as F

from ncompass.internal.logging import ERROR
from ncompass.internal.exec.inference import iterate_windows

__all__ = ["sliding_window_perplexity"]

def run_warmup(model, tokens, num_tokens):
    state, mem1, mem2 = None, None, None
    iter_space = range(num_tokens-1)
    pbar = tqdm(iter_space, total=len(iter_space)) 
    pbar.set_description(f"Warming up spikegpt net with {num_tokens} tokens")
    for token_id in pbar:
        #TODO: consider what batch size (dim 1) implications are
        input_tokens = tokens[0, :token_id+1]
        state, mem1, mem2 = \
                model.forward(input_tokens, state, mem1, mem2, preprocess_only=True)
    return state, mem1, mem2 

def warmup_model(
            model: nn.Module
          , warmup_tokens) :
    num_tokens = warmup_tokens.size(1)
    state, mem1, mem2 = run_warmup(model, warmup_tokens, num_tokens)
    return model.forward(warmup_tokens[0,:], state, mem1, mem2)
            
def iterate_tokens(
        model
        , tokens
        , window_offset
        , window_end
        , state = None
        , mem1 = None
        , mem2 = None
        , init_out = None) :
    iter_space = range(window_offset, window_end)
    logits = [init_out] if init_out is not None else []
    labels = tokens[window_offset:] if init_out is not None else tokens[window_offset+1:]
    output, _state, _mem1, _mem2 = None, state, mem1, mem2
    for token_id in iter_space:
        inputs = tokens[:token_id+1]
        with torch.no_grad():
            output, _state, _mem1, _mem2 = model(inputs, _state, _mem1, _mem2)
            if token_id < (window_end-1):
                logits.append(output)
    torch_logits = torch.stack(logits)
    ce_loss = F.cross_entropy(torch_logits, labels.to(torch_logits.device))
    return output, _state, _mem1, _mem2, ce_loss

def run_single_window(
        model
        , input_ids
        , encoding_start
        , window_end
        , prev_window_end
        , state
        , mem1
        , mem2
        , init_out
        , **kwargs):
    init_out, state, mem1, mem2, loss = iterate_tokens(
                                              model
                                            , input_ids
                                            , prev_window_end - encoding_start
                                            , window_end - encoding_start
                                            , state
                                            , mem1
                                            , mem2
                                            , init_out)
    updated_vars = {}
    updated_vars["init_out"] = init_out
    updated_vars["state"] = state
    updated_vars["mem1"] = mem1
    updated_vars["mem2"] = mem2
    return loss, updated_vars 

def run_sliding_window(
          model
        , tokens
        , ctx_len
        , stride
        , state = None
        , mem1 = None
        , mem2 = None
        , init_out = None):
    infinite_ctx = True if ctx_len == -1 else False
    window_len = 1 if ctx_len == -1 else ctx_len
    num_tokens = tokens.size(1)

    run_args =\
            {"state": state
             , "mem1": mem1
             , "mem2": mem2
             , "init_out": init_out}
    ppl = iterate_windows(
            model
            , tokens
            , num_tokens
            , stride
            , window_len
            , infinite_ctx
            , run_single_window
            , run_args)
    return ppl
    
def sliding_window_perplexity(\
            model: nn.Module\
          , encodings: BatchEncoding\
          , stride: Optional[int] = None\
          , context_len: Optional[int] = None\
          , num_warmup_tokens: Optional[int] = None) -> float:
    _stride = 1 if stride is None else stride
    nwt = 1024 if num_warmup_tokens is None else num_warmup_tokens
    ctx_len = model.config.n_positions if context_len is None else context_len

    if nwt == 0:
        return run_sliding_window(model, encodings.input_ids, ctx_len, _stride)
     
    if num_warmup_tokens >= encodings.input_ids.size(1):
        ERROR(f"Num warmup tokens {num_warmup_tokens} >= sequence length "\
              f"{encodings.input_ids.size(1)}", ValueError)
    
    init_out, state, mem1, mem2 = \
            warmup_model(model, encodings.input_ids[:,:num_warmup_tokens])
    return run_sliding_window(
              model
            , encodings.input_ids[:,num_warmup_tokens:]
            , ctx_len
            , _stride
            , state
            , mem1
            , mem2
            , init_out)

