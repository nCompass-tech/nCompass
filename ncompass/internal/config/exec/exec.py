import torch.nn as nn
from typing import Optional

from transformers import BatchEncoding

from ncompass.internal.utils import validate_arg 
from ncompass.internal.config.base import NCBaseConfig

class SLWConfig():
    def __init__(
            self
            , model: nn.Module
            , encodings: BatchEncoding
            , model_type: str = "gpt2"
            , stride: Optional[int] = None
            , ctx_len: Optional[int] = None):
        self.__model = model
        self.__encodings = encodings
        self.__model_type = model_type
        self.__ctx_len = ctx_len
        self.__stride = stride
    
    @property
    def model(self): return self.__model
    @property
    def encodings(self): return self.__encodings
    @property
    def model_type(self): return self.__model_type
    @property
    def ctx_len(self): return self.__ctx_len
    @property
    def stride(self): return self.__stride


class ExecConfig(NCBaseConfig, SLWConfig):
    def __init__(\
              self\
            , task: str\
            , **kwargs):
        self.__valid_tasks = ["sliding_window_perplexity"]
        validate_arg(task, self.__valid_tasks)
        NCBaseConfig.__init__(self, **kwargs)
        SLWConfig.__init__(self, **kwargs)
