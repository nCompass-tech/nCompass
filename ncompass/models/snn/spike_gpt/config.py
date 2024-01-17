from typing import Optional, TypeVar, List
from ncompass.internal.utils import validate_arg 
from ncompass.internal.config.model import ModelConfig

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
            , rwkv_rescale_layer: int = 6\
            , mix_mode: str = 'RWKV'\
            , **kwargs) :
        validate_arg(device_type, valid_device_types)
        
        # ncompass spikegpt model params
        self.__token_mode = token_mode
        self.__word_name = word_name
        self.__unknown_char = unknown_char
        self.__ctx_len = ctx_len
        self.__device_type = device_type
        self.__device_id = device_id
        self.__float_mode = float_mode
        self.__rwkv_rescale_layer = rwkv_rescale_layer
        self.__mix_mode = mix_mode
        
        # huggingface PretrainedConfig params
        self.n_positions = ctx_len

        ModelConfig.__init__(self, **kwargs)

    @property
    def token_mode(self): return self.__token_mode
    @property
    def word_name(self): return self.__word_name
    @property
    def unknown_char(self): return self.__unknown_char
    @property
    def ctx_len(self): return self.__ctx_len
    @property
    def device_type(self): return self.__device_type
    @property
    def device_id(self): return self.__device_id
    @property
    def float_mode(self): return self.__float_mode
    @property
    def rwkv_rescale_layer(self): return self.__rwkv_rescale_layer
    @property
    def mix_mode(self): return self.__mix_mode
