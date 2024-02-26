import sys
import ncompass.client.functional as F
from ncompass.network_utils import exec_url

def stop_session(api_key):
    response = F.stop_session(exec_url(), api_key)
    print(response.text, response.status_code)

def main():
    stop_session(sys.argv[1])
