import time
from datetime import datetime, timedelta

from .run_prompt import stream_prompt
from ncompass.network_utils import get
from .model_health import check_model_health
from ncompass.errors import model_not_started

def start_session(url, api_key):
    return get(f'{url}/start_session', {'Authorization': api_key})

def stop_session(url, api_key):
    return get(f'{url}/stop_session', {'Authorization': api_key})

def model_is_running(url, api_key):
    response = check_model_health(url, api_key)
    if (response.status_code == 404) or (response.status_code == 400):
        raise RuntimeError(response.text) 
    elif response.status_code == 200:
        return True
    elif response.status_code == 504:
        model_not_started(api_key)
    else:
        return False

def wait_until_model_running(url, api_key):
    break_loop = False
    wait_until = datetime.now() + timedelta(seconds=10)
    while not break_loop:
        if model_is_running(url, api_key): break_loop = True
        if wait_until < datetime.now(): 
            model_not_started(api_key)

def complete_prompt(url
                    , api_key
                    , prompt
                    , max_tokens
                    , temperature
                    , top_p
                    , stream):
    get_stream = lambda : stream_prompt(url=url
                                        , miid=api_key
                                        , prompt=prompt
                                       , max_tokens=max_tokens
                                       , temperature=temperature
                                       , top_p=top_p
                                       , stream=stream)

    if stream: return get_stream()
    else:      return ''.join(res for res in get_stream())
    
def print_prompt(response_iterator):
    ttft = -1
    start = time.time()
    for elem in response_iterator:
        if ttft == -1: ttft = time.time() 
        print(elem, end='', flush=True)
    print()
    return (ttft-start)
