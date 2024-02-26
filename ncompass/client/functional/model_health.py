from ncompass.network_utils import post_json

def check_model_health(url, api_key):
    url = f'{url}/health'
    body = {'miid': api_key}
    headers = {'Authorization': api_key}
    return post_json(url, body, headers)
