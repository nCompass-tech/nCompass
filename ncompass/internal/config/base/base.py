from transformers import PretrainedConfig

class NCBaseConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        PretrainedConfig.__init__(self, **kwargs)

