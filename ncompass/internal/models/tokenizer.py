from typing import Type, Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

#TODO: This is partly taken from ridcherchu/SpikeGPT -> probably will need to change
class NCTokenizer():
    def __init__(self):
        self.tokenizer: Optional[Type[PreTrainedTokenizerBase]] = None
