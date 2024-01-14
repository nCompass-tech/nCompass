from typing import Optional
import ncompass.models.snn.spike_gpt as spike_gpt
from ncompass.internal.models import NCTokenizer, ModelConfig

def get_tokenizer(config: ModelConfig) -> Optional[NCTokenizer]:
    if config.tokenizer_class == "SpikeGPT":
        tok = spike_gpt.tokenizer.SpikeGPTTokenizer(config.word_name, config.unknown_char)
        spike_gpt.tokenizer_check(tok, config.token_mode)
        return tok
