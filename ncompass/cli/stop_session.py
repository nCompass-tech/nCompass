import sys
import ncompass.client.functional as F
from ncompass.network_utils import exec_url

def stop_session(api_key):
    F.stop_session(exec_url(), api_key)

def main():
    stop_session(sys.argv[1])
