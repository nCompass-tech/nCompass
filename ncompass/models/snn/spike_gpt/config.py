from typing import Optional
from ncompass.models.config import ModelConfig

class SpikeGPTConfig(ModelConfig):
    def __init__(\
            self\
            , token_mode: str\
            , word_name: str\
            , unknown_char: Optional[str]\
            , ctx_len: Optional[int] = None\
            , **kwargs):
        self.token_mode = token_mode
        self.word_name = word_name
        self.unknown_char = unknown_char
        self.ctx_len = ctx_len
        ModelConfig.__init__(self, **kwargs)
