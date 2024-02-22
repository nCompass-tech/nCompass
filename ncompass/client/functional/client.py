import time
from .run_prompt import stream_prompt
from .model_health import check_model_health

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
    return stream_prompt(url=url
                        , miid=api_key
                        , prompt=prompt
                        , max_tokens=max_tokens
                        , temperature=temperature
                        , top_p=top_p
                        , stream=stream)
    
def print_prompt(response_iterator):
    ttft = -1
    start = time.time()
    for elem in response_iterator:
        if ttft == -1: ttft = time.time() 
        print(elem, end='', flush=True)
    print()
    return (ttft-start)

