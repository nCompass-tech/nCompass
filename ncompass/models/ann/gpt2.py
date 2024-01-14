import typing
import ncompass.internal.logging as nclog
import ncompass.internal.utils.models as nc_model_utils

def get_hf_pipeline(\
        size: typing.Optional[str] = None\
        , model: typing.Optional[str] = None\
        , device: int = 0):
    if model != None:
        nclog.INFO(f"Calling generic hf model loader. If {model} is not a gpt2 variant, then the"\
                    " returned model will not be gpt2 based.")
        return nc_model_utils.get_model_hf_pipeline(model, device)
    elif size != None:
        if size == "medium":  return get_hf_pipeline_medium(device)
        elif size == "large": return get_hf_pipeline_large(device)
        elif size == "xl":    return get_hf_pipeline_xl(device)
        else: 
            nclog.ERROR("Invalid model size passed. Valid sizes are: medium, large, xl",\
                        ValueError)
    else:
        return get_hf_pipeline_base(device)
            
def get_hf_pipeline_base(device: int = 0):
    return nc_model_utils.get_model_hf_pipeline("gpt2", device)

def get_hf_pipeline_medium(device: int = 0):
    return nc_model_utils.get_model_hf_pipeline("gpt2-medium", device)

def get_hf_pipeline_large(device: int = 0):
    return nc_model_utils.get_model_hf_pipeline("gpt2-large", device)

def get_hf_pipeline_xl(device: int = 0):
    return nc_model_utils.get_model_hf_pipeline("gpt2-xl", device)
