import time
from .run_prompt import stream_prompt
from ncompass.network_utils import get
from .model_health import check_model_health

def start_model(url, api_key):
    print(url)
    print(api_key)
    return get(f'{url}/start_model', {'Authorization': api_key})

def stop_model(url, api_key):
    return get(f'{url}/stop_model', {'Authorization': api_key})

def model_is_running(url, api_key):
    return check_model_health(url, api_key).status_code == 200

def wait_until_model_running(url, api_key):
    while not model_is_running(url, api_key):
        continue

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
