
def error_msg(err):
    raise RuntimeError(f'{err}:\nPlease contact admin@ncompass.tech for resolution.') 

def api_key_not_set(custom_var = None):
    if custom_var is not None:
        raise RuntimeError(\
                f'Env variable {custom_var} not set to api key. Please set this and run again.')
    else:
        raise RuntimeError(\
                'Env variable NCOMPASS_API_KEY needs to be set to the api key. ' 
                'Alternatively, pass a custom variable.')
