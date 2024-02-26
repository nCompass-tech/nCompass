import httpx
import aiohttp

def exec_url():
    return 'api.ncompass.tech'

def get(url, headers=None):
    client = httpx.Client(timeout=20, verify=False)
    return client.get(f'https://{url}', headers=headers)

def post_json(url, payload, headers=None):
    hdr = {'Content-Type': 'application/json'}
    if headers is not None: hdr.update(headers)
    client = httpx.Client(timeout=20, verify=False)
    return client.post(f'https://{url}', headers=hdr, json=payload)
