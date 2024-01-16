import torch.nn as nn
from typing import Optional
from transformers import BatchEncoding

from ncompass.internal.logging import ERROR
from ncompass.internal.config.exec import SLWConfig, ExecConfig
from ncompass.internal.exec.inference import \
          ann_sliding_window_perplexity\
        , spikegpt_sliding_window_perplexity

def get_slw_config(**kwargs) -> SLWConfig:
    return SLWConfig(**kwargs)

def get_exec_config(task: str, **kwargs) -> ExecConfig:
    return ExecConfig(task, **kwargs)

def _run_sliding_window_perplexity(
            model: nn.Module
          , encodings: BatchEncoding
          , model_type: str = "gpt2"
          , stride: Optional[int] = None
          , ctx_len: Optional[int] = None
          , num_warmup_tokens: Optional[int] = None) -> Optional[float]:
    if model_type == "gpt2":
        return ann_sliding_window_perplexity(
                                          model
                                        , encodings
                                        , stride
                                        , ctx_len)
    elif model_type == "spikegpt":
        return spikegpt_sliding_window_perplexity(
                                          model
                                        , encodings
                                        , stride
                                        , ctx_len
                                        , num_warmup_tokens)
    else:
        ERROR(f"Invalid model type {model_type}.", ValueError)

def run_sliding_window_perplexity(config: SLWConfig) -> Optional[float]:
    return _run_sliding_window_perplexity(
                                  config.model
                                , config.encodings
                                , config.nc_model_type
                                , config.stride
                                , config.ctx_len
                                , config.num_warmup_tokens)

