import sys
import ncompass.client.functional as F
from ncompass.network_utils import exec_url

def start_session(api_key):
    response = F.start_session(exec_url(), api_key) 
    print(response.text, response.status_code)

def main():
    start_session(sys.argv[1])
