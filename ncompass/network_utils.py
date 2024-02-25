import httpx
import asyncio
import aiohttp

def get(url, headers=None):
    return httpx.get(f'https://{url}', headers=headers, verify=False)

def post_json(url, payload, headers=None):
    hdr = {'Content-Type': 'application/json'}
    if headers is not None: hdr.update(headers)
    return httpx.post(f'https://{url}', headers=hdr, json=payload, verify=False)

async def async_streaming_post_json(url, payload, headers, handler):
    session = aiohttp.ClientSession()
    response = await session.post(f'https://{url}', headers=headers, json=payload, ssl=False)
    return (await handler(response))

