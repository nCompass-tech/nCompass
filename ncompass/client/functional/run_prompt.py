
from ncompass.async_utils import get_sync_generator_from_async\
                                 , async_post 

def build_body_from_params(prompt
                           , miid
                           , max_tokens
                           , temperature
                           , top_p
                           , stream):
    body = {'prompt':        prompt
            , 'miid':        miid
            , 'max_tokens':  max_tokens
            , 'temperature': temperature
            , 'top_p':       top_p
            , 'stream':      stream}
    return body

def stream_prompt(url: str
                  , miid: str
                  , prompt: str
                  , max_tokens: int
                  , temperature: float
                  , top_p: float
                  , stream: bool
                  ):
    _url = f'{url}/run_prompt'
    headers = {'Content-Type': "application/json", 'Authorization': miid} 
    body = build_body_from_params(prompt, miid, max_tokens, temperature, top_p, stream) 
    return get_sync_generator_from_async(async_post(_url, headers, body))
