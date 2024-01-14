from transformers import PretrainedConfig
import ncompass.internal.logging as nclog

class ModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)
        if self.tokenizer_class is None:
            nclog.ERROR("We currently don't have a default tokenizer class. "\
                        "Please specify a tokenizer class to use.", AttributeError)
