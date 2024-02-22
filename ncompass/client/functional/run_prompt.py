
from ncompass.async_utils import get_sync_generator_from_async\
                                 , async_post 

def build_body_from_params(prompt
                           , miid
                           , max_tokens = 1
                           , temperature = 0
                           , top_p = 0.50
                           , stream = True):
    body = {'prompt':        prompt
            , 'miid':        miid
            , 'max_tokens':  max_tokens
            , 'temperature': temperature
            , 'top_p':       top_p
            , 'stream':      stream}
    return body

def stream(url: str
           , miid: str
           , prompt: str
           , params: dict
           ):
    _url = f'{url}/run_prompt'
    headers = {'Content-Type': "application/json"} 
    body = build_body_from_params(prompt, miid, **params) 
    return get_sync_generator_from_async(async_post(_url, headers, body))
