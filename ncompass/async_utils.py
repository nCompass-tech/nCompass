import json
import aiohttp
import asyncio
from ncompass.errors import error_msg

async def async_get_next(ait):
    try:
        obj = await ait.__anext__()
        return False, obj
    except StopAsyncIteration:
        return True, None

def get_sync_generator_from_async(async_gen):
    ait = async_gen.__aiter__()
    loop = asyncio.get_event_loop()
    while True:
        done, res = loop.run_until_complete(async_get_next(ait))
        if done: break
        yield res

async def async_post(url, headers, body, stream=True, verify=False):
    async with aiohttp.ClientSession() as session:
        async with session.post(url
                                , headers=headers
                                , json=body
                                , ssl=verify) as response:
            if response.status == 200:
                response_len = 0
                async for chunk, _ in response.content.iter_chunks():
                    try:
                        data = json.loads(chunk.split(b'\0')[0].decode("utf-8"))
                        res = data['text'][0]
                        new_data = res[response_len:]
                        response_len = len(res)
                        yield new_data
                    except Exception as e:
                        error_msg(str(e))
            else:
                res = await response.json()
                error_msg(res['error'])

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
           , headers: dict):
    body = build_body_from_params(prompt, miid, **params) 
    return get_sync_generator_from_async(async_post(url, headers, body))
