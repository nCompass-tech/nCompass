
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

def model_not_started(api_key):
    msg = (f'Model {api_key} has not been started or is currently starting. '
            'If you have started it, please wait and try again. '
            'If not, please contact admin@ncompass.tech for support')
    raise RuntimeError(msg)
