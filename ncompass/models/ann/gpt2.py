from typing import Optional

import torch.nn as nn

from transformers.pipelines.base import Pipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import ncompass.internal.logging as nclog
from ncompass.internal.utils import validate_arg
import ncompass.internal.utils.models as nc_model_utils

def get_hf_pipeline(\
          size: Optional[str] = None\
        , model: Optional[str] = None\
        , device: int = 0) \
        -> Optional[Pipeline]:
    if model != None:
        nclog.INFO(f"Calling generic hf model loader. If {model} is not a gpt2 variant, then the"\
                    " returned model will not be gpt2 based.")
        return nc_model_utils.get_model_hf_pipeline(model, device)
    elif size != None:
        if size == "medium":  return get_hf_pipeline_medium(device)
        elif size == "large": return get_hf_pipeline_large(device)
        elif size == "xl":    return get_hf_pipeline_xl(device)
        else: 
            nclog.ERROR("Invalid model size passed. Valid sizes are: medium, large, xl",\
                        ValueError)
    else:
        return get_hf_pipeline_base(device)
            
def get_hf_pipeline_base(device: int = 0) -> Pipeline:
    return nc_model_utils.get_model_hf_pipeline("gpt2", device)

def get_hf_pipeline_medium(device: int = 0) -> Pipeline:
    return nc_model_utils.get_model_hf_pipeline("gpt2-medium", device)

def get_hf_pipeline_large(device: int = 0) -> Pipeline:
    return nc_model_utils.get_model_hf_pipeline("gpt2-large", device)

def get_hf_pipeline_xl(device: int = 0) -> Pipeline:
    return nc_model_utils.get_model_hf_pipeline("gpt2-xl", device)

def get_hf_model(size: Optional[str] = None, model_type: str = "lm_head") -> Optional[nn.Module]:
    if model_type != "lm_head":
        nclog.ERROR(f"Invalid model type {model_type}.", ValueError)
    else:
        from transformers import GPT2LMHeadModel as gpt2_model
        if size is None: model_name = "gpt2"
        else:
            validate_arg(size, ["medium", "large", "xl"])
            model_name = f"gpt2-{size}"
        nclog.INFO(f"Loading model {model_name} from huggingface pretrained models")
        return gpt2_model.from_pretrained(model_name)

def get_hf_tokenizer(size: Optional[str] = None) -> PreTrainedTokenizerBase :
    from transformers import GPT2TokenizerFast as tokenizer
    if size is None: model_name = "gpt2"
    else:
        validate_arg(size, ["medium", "large", "xl"])
        model_name = f"gpt2-{size}"
    nclog.INFO(f"Loading hugginface pretrained tokenizer for {model_name}")
    return tokenizer.from_pretrained(model_name)


