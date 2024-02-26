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
