import os
from typing import Optional, Union, Generator

import ncompass.client.functional as F
from ncompass.errors import api_key_not_set
from ncompass.network_utils import exec_url

class nCompass():
    def __init__(self
                 , api_key: Optional[str] = None
                 , custom_env_var: Optional[str] = None):
        assert not ((api_key is not None) and (custom_env_var is not None))\
            ,'Cannot have both api_key and custom_env_var set'

        # NOTE: nCompassOLOC only works as long as the only state that this class holds is api_key.
        
        self.api_key = None
        self.exec_url = exec_url()
        
        if api_key is not None:          self.api_key = api_key
        elif custom_env_var is not None: self.api_key = os.environ.get(custom_env_var)
        else:                            self.api_key = os.environ.get('NCOMPASS_API_KEY')
        
        if self.api_key is None: api_key_not_set(custom_env_var)

    def start_session(self):
        F.start_session(self.exec_url, self.api_key) 
        self.wait_until_model_running()
    
    def stop_session(self):
        return F.stop_session(self.exec_url, self.api_key) 

    def model_is_running(self):
        return F.model_is_running(self.exec_url, self.api_key)

    def wait_until_model_running(self) :
        F.wait_until_model_running(self.exec_url, self.api_key)

    def complete_prompt(self
                        , prompt
                        , max_tokens=300
                        , temperature=0.5
                        , top_p=0.9
                        , stream=True):
        return F.complete_prompt(self.exec_url
                                 , self.api_key
                                 , prompt
                                 , max_tokens
                                 , temperature
                                 , top_p
                                 , stream)

    def print_prompt(self, response_iterator):
        return F.print_prompt(response_iterator)

class nCompassOLOC():
    client = None

    @classmethod
    def start_client(cls, api_key):
        if (cls.client is None) or (cls.client.api_key != api_key):
            cls.client = nCompass(api_key = api_key)

    @classmethod
    def start_session(cls, api_key):
        cls.start_client(api_key)
        cls.client.start_session()
        cls.client.wait_until_model_running()
    
    @classmethod
    def stop_session(cls, api_key):
        cls.start_client(api_key)
        return cls.client.stop_session()
    
    @classmethod
    def complete_prompt(cls
                        , api_key
                        , prompt
                        , max_tokens=300
                        , temperature = 0.5
                        , top_p = 0.9
                        , stream = True
                        , pprint = False) -> Union[None, Generator]:
        cls.start_client(api_key)
        iterator = cls.client.complete_prompt(prompt, max_tokens, temperature, top_p, stream)
        if (stream and pprint): return F.print_prompt(iterator)
        else:                   return iterator # prompt in case of stream=false else iterator 

    @classmethod
    def print_response(cls, iterator):
        return F.print_prompt(iterator)
