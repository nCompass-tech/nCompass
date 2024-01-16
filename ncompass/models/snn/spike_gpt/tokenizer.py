from ncompass.internal.models import NCTokenizer, ModelConfig
from ncompass.internal.models.snn.spike_gpt import SpikeGPTTokenizer, spikegpt_tokenizer_check

def _get_tokenizer(word_name: str,
                   unknown_char: str,
                   token_mode: str) -> NCTokenizer:
    tok = SpikeGPTTokenizer(word_name, unknown_char)
    spikegpt_tokenizer_check(tok, token_mode)
    return tok

def get_tokenizer(config: ModelConfig) -> NCTokenizer:
    return _get_tokenizer(config.word_name, config.unknown_char, config.token_mode)
