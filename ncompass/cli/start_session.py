import sys
import ncompass.client.functional as F
from ncompass.network_utils import exec_url

def start_session(api_key):
    F.start_session(exec_url(), api_key) 

def main():
    start_session(sys.argv[1])
