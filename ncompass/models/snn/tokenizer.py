from typing import Optional

from ncompass.internal.models import NCTokenizer, ModelConfig
from ncompass.internal.logging import ERROR

import ncompass.models.snn.spike_gpt as spike_gpt

def _get_tokenizer(tokenizer_class: str,
                   word_name: str,
                   unknown_char: str,
                   token_mode: str) -> Optional[NCTokenizer]:
    if tokenizer_class == "SpikeGPT": spike_gpt._get_tokenizer(word_name, unknown_char, token_mode)
    else:                             ERROR("Invalid tokenizer_class: {tokenizer_class}", ValueError)

def get_tokenizer(config: ModelConfig) -> Optional[NCTokenizer]:
    return _get_tokenizer(config.tokenizer_class, config.word_name, config.unknown_char, config.token_mode)
