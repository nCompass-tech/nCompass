import os
import time
import httpx
from typing import Optional

import ncompass.client.functional as F
from ncompass.errors import api_key_not_set

class nCompass():
    def __init__(self
                 , api_key: Optional[str] = None
                 , custom_env_var: Optional[str] = None):
        assert not ((api_key is not None) and (custom_env_var is not None))\
            ,'Cannot have both api_key and custom_env_var set'

        self.api_key = None
        self.exec_url = 'https://api.ncompass.tech' 
        
        if api_key is not None:          self.api_key = api_key
        elif custom_env_var is not None: self.api_key = os.environ.get(custom_env_var)
        else:                            self.api_key = os.environ.get('NCOMPASS_API_KEY')
        
        if self.api_key is None: api_key_not_set(custom_env_var)

    def model_is_running(self):
        return F.model_is_running(self.exec_url, self.api_key)

    def wait_until_model_running(self) :
        F.wait_until_model_running(self.exec_url, self.api_key)

    def complete_prompt(self, prompt, params):
        return F.complete_prompt(self.exec_url, self.api_key, prompt, params)

    def print_prompt(self, response_iterator):
        return F.print_prompt(response_iterator)

