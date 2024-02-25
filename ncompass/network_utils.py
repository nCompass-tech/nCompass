import httpx
import aiohttp

def get(url, headers=None):
    client = httpx.Client(timeout=20, verify=False)
    return client.get(f'https://{url}', headers=headers)

def post_json(url, payload, headers=None):
    hdr = {'Content-Type': 'application/json'}
    if headers is not None: hdr.update(headers)
    client = httpx.Client(timeout=20, verify=False)
    return client.post(f'https://{url}', headers=hdr, json=payload)

async def async_streaming_post_json(url, payload, headers, handler):
    session = aiohttp.ClientSession()
    response = await session.post(f'https://{url}', headers=headers, json=payload, ssl=False)
    return (await handler(response))

