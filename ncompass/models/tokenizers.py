from typing import Type
from ncompass.models.config import ModelConfig
import ncompass.models.snn.spike_gpt as spike_gpt

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

#TODO: This is partly taken from ridcherchu/SpikeGPT -> probably will need to change
class NCTokenizer():
    def __init__(self):
        self.tokenizer: Optional[Type[PreTrainedTokenizerBase]] = None
        return

def get_tokenizer(config: Type[ModelConfig]) -> NCTokenizer:
    if config.tokenizer_class == "SpikeGPT":
        tok = spike_gpt.tokenizer.SpikeGPTTokenizer(config.word_name, config.unknown_char)
        spike_gpt.tokenizer_check(tok, config.token_mode)
        return tok
