import json
import aiohttp
import asyncio

from ncompass.errors import error_msg
from ncompass.async_utils import get_sync_generator_from_async

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

async def get_streamed_response(url, payload, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'https://{url}', headers=headers, json=payload, ssl=False)\
                as response:
            if response.status == 200:
                response_len = 0
                async for chunk, _ in response.content.iter_chunks():
                    try:
                        split_chunk = chunk.split(b'\0')
                        decoded_chunk = split_chunk[0].decode('utf-8')
                        try:
                            data = json.loads(decoded_chunk)
                            res = data['text'][0] 
                            new_data = res[response_len:]
                            response_len = len(res)
                            yield new_data
                        except Exception as e:
                            yield ''
                    except Exception as e:
                        error_msg(str(e))
            else:
                res = await response.json()
                error_msg(res['error'])

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
    return get_sync_generator_from_async(get_streamed_response(_url
                                                               , body
                                                               , headers))
