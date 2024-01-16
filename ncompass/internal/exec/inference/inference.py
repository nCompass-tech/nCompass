from tqdm import tqdm
from typing import Optional
from transformers import BatchEncoding

import torch
import torch.nn as nn
from torch.nn import functional as F

from ncompass.internal.logging import INFO, ERROR

def ann_sliding_window_perplexity(\
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

    prev_ctx_end = 0
    device = model.device
    iter_space = range(0, num_tokens, _stride)
    pbar = tqdm(iter_space, total=len(iter_space))
    neg_log_likelihood_list = []
    for ctx_start in pbar:
        ctx_end = min(ctx_start + ctx_len, num_tokens)
        encoding_start = ctx_start
        input_ids = encodings.input_ids[:, encoding_start:ctx_end].to(device)
        target_ids = input_ids.clone()
        trg_len = ctx_end - prev_ctx_end 
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            output = model(input_ids, labels = target_ids)
            neg_log_likelihood = output.loss
        neg_log_likelihood_list.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
        pbar.set_description(f"Perplexity = {ppl:.2f}")
        prev_ctx_end = ctx_end
    ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
    return float(ppl)

def spikegpt_run_warmup(model, tokens, num_tokens):
    state, mem1, mem2 = None, None, None
    pbar = tqdm(range(num_tokens-1), total=num_tokens-1) 
    pbar.set_description(f"Warming up spikegpt net with {num_tokens} tokens")
    for token_id in pbar:
        #TODO: consider what batch size (dim 1) implications are
        input_tokens = tokens[0, :token_id+1]
        state, mem1, mem2 = \
                model.forward(input_tokens, state, mem1, mem2, preprocess_only=True)
    return state, mem1, mem2 

def spikegpt_warmup_model(
            model: nn.Module
          , warmup_tokens) :
    num_tokens = warmup_tokens.size(1)
    state, mem1, mem2 = spikegpt_run_warmup(model, warmup_tokens, num_tokens)
    return model.forward(warmup_tokens[0,:], state, mem1, mem2)
            
def spikegpt_run_single_window(
        model
        , tokens
        , window_offset
        , window_end
        , state = None
        , mem1 = None
        , mem2 = None
        , init_out = None) :
    iter_space = range(window_offset, window_end)
    num_tokens = len(iter_space)
    pbar = tqdm(iter_space, total=num_tokens, leave=False, disable=True)
    logits = [init_out] if init_out is not None else []
    labels = tokens[window_offset:] if init_out is not None else tokens[window_offset+1:]
    output, _state, _mem1, _mem2 = None, state, mem1, mem2
    for token_id in pbar:
        inputs = tokens[:token_id+1]
        pbar.set_description(f"input size = {inputs.size(0)}")
        with torch.no_grad():
            output, _state, _mem1, _mem2 = model(inputs, _state, _mem1, _mem2)
            if token_id < (window_end-1):
                logits.append(output)
    torch_logits = torch.stack(logits)
    ce_loss = F.cross_entropy(torch_logits, labels.to(torch_logits.device))
    return output, _state, _mem1, _mem2, ce_loss

def spikegpt_run_sliding_window(
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
    prev_window_end = 0
    num_tokens = tokens.size(1)
    iter_space = range(0, num_tokens, stride)
    pbar = tqdm(iter_space, total=len(iter_space)) 
    neg_log_likelihood_list = []
    for window_start in pbar:
        window_end = min(window_start + window_len, num_tokens)
        encoding_start = 0 if infinite_ctx else window_start
        input_ids = tokens[0, encoding_start:window_end]
        init_out, state, mem1, mem2, loss = spikegpt_run_single_window(
                                                  model
                                                , input_ids
                                                , prev_window_end - encoding_start
                                                , window_end - encoding_start
                                                , state
                                                , mem1
                                                , mem2
                                                , init_out)
        neg_log_likelihood_list.append(loss)
        ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
        pbar.set_description(f"Perplexity = {ppl:.2f}")
        prev_window_end = window_end
    ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
    return float(ppl)

def spikegpt_sliding_window_perplexity(\
            model: nn.Module\
          , encodings: BatchEncoding\
          , stride: Optional[int] = None\
          , context_len: Optional[int] = None\
          , num_warmup_tokens: Optional[int] = None) -> float:
    _stride = 1 if stride is None else stride
    nwt = 1024 if num_warmup_tokens is None else num_warmup_tokens
    ctx_len = model.config.n_positions if context_len is None else context_len

    if nwt == 0:
        return spikegpt_run_sliding_window(model, encodings.input_ids, ctx_len, _stride)
     
    if num_warmup_tokens >= encodings.input_ids.size(1):
        ERROR(f"Num warmup tokens {num_warmup_tokens} >= sequence length "\
              f"{encodings.input_ids.size(1)}", ValueError)
    
    init_out, state, mem1, mem2 = \
            spikegpt_warmup_model(model, encodings.input_ids[:,:num_warmup_tokens])
    return spikegpt_run_sliding_window(
              model
            , encodings.input_ids[:,num_warmup_tokens:]
            , ctx_len
            , _stride
            , state
            , mem1
            , mem2
            , init_out)

