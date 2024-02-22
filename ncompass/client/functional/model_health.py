import httpx

def check_model_health(url, api_key):
    url = f'{url}/health'
    body = {'miid': api_key}
    req = httpx.post(url
                     , headers={'Content-Type': 'application/json'}
                     , json=body
                     , verify=False)
    return req
