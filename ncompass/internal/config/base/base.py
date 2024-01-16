from transformers import PretrainedConfig
from ncompass.internal.utils import ImmutableClass

class NCBaseConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)

