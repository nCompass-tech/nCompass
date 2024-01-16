from typing import Optional, TypeVar, List
from ncompass.internal.models import ModelConfig
from ncompass.internal.utils import validate_arg

Wordname_t = TypeVar("Wordname_t", str, list[str])
class SpikeGPTConfig(ModelConfig):
    def __init__(\
              self\
            , token_mode: str\
            , word_name: Wordname_t\
            , unknown_char: Optional[str]\
            , device_type: str = "cuda"\
            , valid_device_types: List[str] = ["cpu", "cuda"]\
            , device_id: int = 0\
            , float_mode: str = "fp32"\
            , ctx_len: Optional[int] = None\
            , rwkv_rescale_layer: int = 6
            , **kwargs) :
        validate_arg(device_type, valid_device_types)
        
        self.token_mode = token_mode
        self.word_name = word_name
        self.unknown_char = unknown_char
        self.ctx_len = ctx_len
        self.device_type = device_type
        self.device_id = device_id
        self.float_mode = float_mode
        self.rwkv_rescale_layer = rwkv_rescale_layer
        
        ModelConfig.__init__(self, **kwargs)
