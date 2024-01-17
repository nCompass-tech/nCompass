from ncompass.internal.logging import ERROR 
from ncompass.internal.config.base import NCBaseConfig

class ModelConfig(NCBaseConfig):
    def __init__(self, **kwargs):
        NCBaseConfig.__init__(self, **kwargs)
        if self.tokenizer_class is None:
            ERROR("We currently don't have a default tokenizer class. "\
                  "Please specify a tokenizer class to use.", AttributeError)

