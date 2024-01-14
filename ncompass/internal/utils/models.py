import torch
import logging as pylog
from transformers import pipeline
import ncompass.internal.logging as nclog

def __torch_num_params__(model: torch.nn.Module):
    return sum([torch.prod(torch.tensor(x.shape)) for x in model.parameters()])

def torch_model_size_in_parameters_M(model: torch.nn.Module):
    return __torch_num_params__(model) / 1e6

def get_model_hf_pipeline(model_name: str, device: int = 0, **kwargs):
    logger = pylog.getLogger(nclog.DETAILED)
    logger.info(f"Loading model {model_name}")
    pipe = pipeline(model=model_name, device=device, **kwargs)
    model_size = torch_model_size_in_parameters_M(pipe.model)
    logger.info(f"Loaded model has {model_size:.2f}M parameters")
    return pipe

